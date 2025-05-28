"""
datasets.py

Lightweight PyTorch Dataset Definition for wrapping RLDS TFDS Pipeline; just defines transform from RLDS default
format to OpenVLA, IterableDataset shim.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple, Type, Callable, Optional
import json
import pickle
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, IterableDataset
from transformers import PreTrainedTokenizerBase

from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import ImageTransform
from prismatic.util.data_utils import tree_map
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import (
    ACTION_DIM,
    ACTION_PROPRIO_NORMALIZATION_TYPE,
    ACTION_TOKEN_BEGIN_IDX,
    IGNORE_INDEX,
    NUM_ACTIONS_CHUNK,
    PROPRIO_DIM,
    STOP_INDEX,
)
from prismatic.vla.datasets.rlds import make_interleaved_dataset, make_single_dataset
from prismatic.vla.datasets.rlds.oxe import OXE_NAMED_MIXTURES, get_oxe_dataset_kwargs_and_weights
from prismatic.vla.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata


@dataclass
class RLDSBatchTransform:
    action_tokenizer: ActionTokenizer
    base_tokenizer: PreTrainedTokenizerBase
    image_transform: ImageTransform
    prompt_builder_fn: Type[PromptBuilder]
    predict_stop_token: bool = True
    use_wrist_image: bool = False
    use_proprio: bool = False

    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Converts a RLDS batch to the format expected by the OpenVLA collator/models."""
        dataset_name, current_action = rlds_batch["dataset_name"], rlds_batch["action"][0]
        img = Image.fromarray(rlds_batch["observation"]["image_primary"][0])
        lang = rlds_batch["task"]["language_instruction"].decode().lower()
        actions = rlds_batch["action"]

        # Construct Chat-based Prompt =>> Input is default query + language instruction, output are the action tokens
        prompt_builder = self.prompt_builder_fn("openvla")

        # Get future action chunk
        future_actions = rlds_batch["action"][1:]
        future_actions_string = "".join(self.action_tokenizer(future_actions))

        # Get action chunk string
        current_action_string = self.action_tokenizer(current_action)
        action_chunk_string = current_action_string + future_actions_string
        action_chunk_len = len(action_chunk_string)

        conversation = [
            {"from": "human", "value": f"What action should the robot take to {lang}?"},
            {"from": "gpt", "value": action_chunk_string},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize (w/ `base_tokenizer`)
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)

        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        #   =>> IMPORTANT :: IF WE'RE USING HF LLM.forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        pixel_values = self.image_transform(img)

        # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
        labels[: -(action_chunk_len + 1)] = IGNORE_INDEX
        if not self.predict_stop_token:
            labels[-1] = IGNORE_INDEX

        return_dict = dict(
            pixel_values=pixel_values, input_ids=input_ids, labels=labels, dataset_name=dataset_name, actions=actions
        )

        # Add additional inputs
        if self.use_wrist_image:
            all_wrist_pixels = []
            for k in rlds_batch["observation"].keys():
                if "wrist" in k:
                    img_wrist = Image.fromarray(rlds_batch["observation"][k][0])
                    pixel_values_wrist = self.image_transform(img_wrist)
                    all_wrist_pixels.append(pixel_values_wrist)
            return_dict["pixel_values_wrist"] = torch.cat(all_wrist_pixels, dim=0)
        if self.use_proprio and "proprio" in rlds_batch["observation"]:
            proprio = rlds_batch["observation"]["proprio"]
            return_dict["proprio"] = proprio

        return return_dict


# DEBUG:add debug


class RLDSDataset(IterableDataset):
    def __init__(
        self,
        data_root_dir: Path,
        data_mix: str,
        batch_transform: Optional[RLDSBatchTransform],
        resize_resolution: Tuple[int, int],
        shuffle_buffer_size: int = 256_000,
        train: bool = True,
        image_aug: bool = False,
    ) -> None:
        """Lightweight wrapper around RLDS TFDS Pipeline for use with PyTorch/OpenVLA Data Loaders."""
        self.data_root_dir, self.data_mix, self.batch_transform = data_root_dir, data_mix, batch_transform

        # Configure RLDS Dataset(s)
        if self.data_mix in OXE_NAMED_MIXTURES:
            mixture_spec = OXE_NAMED_MIXTURES[self.data_mix]
        else:
            # Assume that passed "mixture" name is actually a single dataset -- create single-dataset "mix"
            mixture_spec = [(self.data_mix, 1.0)]

        # fmt: off
        if "aloha" in self.data_mix:
            load_camera_views = ("primary", "left_wrist", "right_wrist")
        else:
            load_camera_views = ("primary", "wrist")

        per_dataset_kwargs, weights = get_oxe_dataset_kwargs_and_weights(
            self.data_root_dir,
            mixture_spec,
            load_camera_views=load_camera_views,
            load_depth=False,
            load_proprio=True,
            load_language=True,
            action_proprio_normalization_type=ACTION_PROPRIO_NORMALIZATION_TYPE,
        )
        rlds_config = dict(
            traj_transform_kwargs=dict(
                window_size=1,                                      # If we wanted to feed / predict more than one step
                future_action_window_size=NUM_ACTIONS_CHUNK-1,      # For action chunking
                skip_unlabeled=True,                                # Skip trajectories without language labels
                goal_relabeling_strategy="uniform",                 # Goals are currently unused
            ),
            frame_transform_kwargs=dict(
                resize_size=resize_resolution,
                num_parallel_calls=16,                          # For CPU-intensive ops (decoding, resizing, etc.)
            ),
            dataset_kwargs_list=per_dataset_kwargs,
            shuffle_buffer_size=shuffle_buffer_size,
            sample_weights=weights,
            balance_weights=True,
            traj_transform_threads=len(mixture_spec),
            traj_read_threads=len(mixture_spec),
            train=train,
        )

        # If applicable, enable image augmentations
        if image_aug:
            rlds_config["frame_transform_kwargs"].update({"image_augment_kwargs" : dict(
                random_resized_crop=dict(scale=[0.9, 0.9], ratio=[1.0, 1.0]),
                random_brightness=[0.2],
                random_contrast=[0.8, 1.2],
                random_saturation=[0.8, 1.2],
                random_hue=[0.05],
                augment_order=[
                    "random_resized_crop",
                    "random_brightness",
                    "random_contrast",
                    "random_saturation",
                    "random_hue",
                ],
            )}),
        # fmt: on

        # Initialize RLDS Dataset
        # DEBUG: debug use
        # with open("/root/openvla-oft/test_tmp/dataset_cfg.pkl", "wb") as f:
        #     pickle.dump(rlds_config, f)
        # with open("/root/openvla-oft/test_tmp/batch_transform", "wb") as f:
        #     pickle.dump(batch_transform, f)
        self.dataset, self.dataset_length, self.dataset_statistics = self.make_dataset(rlds_config)

    def make_dataset(self, rlds_config):
        return make_interleaved_dataset(**rlds_config)

    def __iter__(self) -> Dict[str, Any]:
        """
        batch size 1 example output after batch_transform:
        data:
        {'pixel_values':torch.Size([6, 224, 224]),
        'input_ids':torch.Size([90]),
        'labels':torch.Size([90]),
        'dataset_name':b'ruijia_robot_grip_dataset',
        'actions':np.size([8,7]),
        'pixel_values_wrist':torch.Size([6, 224, 224]),
        'proprio':np.size([1,7])}
        """
        for rlds_batch in self.dataset.as_numpy_iterator():
            """
            batch size 1
            rlds_batch: 
            {'observation':{
                'image_primary':np.size([1, 224, 224, 3]),np.uint8,
                'image_wrist':np.size([1, 224, 224, 3]),np.uint8, 
                'proprio':array([[0.394, 0.707, -0.293, 0.314, 0.556, 0.848, 0.024]], dtype=float32),np.size([1, 7]),np.float32,
                'timestep':83,np.size([1]),int32},
                'pad_mask_dict':{
                    'image_primary': array([ True]), 
                    'image_wrist': array([ True]), 
                    'proprio': array([ True]), 
                    'timestep': array([ True])
                    }, 
                'pad_mask:array([ True])'
            },
            'task':{
                'language_instruction':b'pick up the red tool from the desk and move it to hang on the black hook' , 
                'pad_mask_dict':{
                    'language_instruction': True, 
                    'image_primary': True, 
                    'image_wrist': True, 
                    'proprio': True, 
                    'timestep': True}, 
                'image_primary':np.size([224, 224, 3]),np.uint8, 
                'image_wrist':np.size([224, 224, 3]),np.uint8, 
                'proprio':array([0.264, 0.046, 0.332, -0.342, 0.029, -0.331, 0.930], dtype=float32), 
                'timestep':155,int]
                },
            'action':np.size([8, 7]),np.float32,
            'dataset_name':b'ruijia_robot_grip_dataset',
            'absolute_action_mask':array([False, False, False, False, False, False,  True])}
            """
            if self.batch_transform is not None:
                yield self.batch_transform(rlds_batch)
            else:
                yield rlds_batch

    def __len__(self) -> int:
        return self.dataset_length

    # === Explicitly Unused ===
    def __getitem__(self, idx: int) -> None:
        raise NotImplementedError("IterableDataset does not implement map-style __getitem__; see __iter__ instead!")


class EpisodicRLDSDataset(RLDSDataset):
    """Returns full episodes as list of steps instead of individual transitions (useful for visualizations)."""

    def make_dataset(self, rlds_config):
        per_dataset_kwargs = rlds_config["dataset_kwargs_list"]
        assert len(per_dataset_kwargs) == 1, "Only support single-dataset `mixes` for episodic datasets."

        return make_single_dataset(
            per_dataset_kwargs[0],
            train=rlds_config["train"],
            traj_transform_kwargs=rlds_config["traj_transform_kwargs"],
            frame_transform_kwargs=rlds_config["frame_transform_kwargs"],
        )

    def __iter__(self) -> Dict[str, Any]:
        for rlds_batch in self.dataset.as_numpy_iterator():
            out = [
                self.batch_transform(tree_map(lambda x: x[i], rlds_batch))  # noqa: B023
                for i in range(rlds_batch["action"].shape[0])
            ]
            yield out


class DummyDataset(Dataset):
    def __init__(
        self,
        action_tokenizer: ActionTokenizer,
        base_tokenizer: PreTrainedTokenizerBase,
        image_transform: ImageTransform,
        prompt_builder_fn: Type[PromptBuilder],
    ) -> None:
        self.action_tokenizer = action_tokenizer
        self.base_tokenizer = base_tokenizer
        self.image_transform = image_transform
        self.prompt_builder_fn = prompt_builder_fn

        # Note =>> We expect the dataset to store statistics for action de-normalization. Specifically, we store the
        # per-dimension 1st and 99th action quantile. The values below correspond to "no normalization" for simplicity.
        self.dataset_statistics = {
            "dummy_dataset": {
                "action": {"q01": np.zeros((7,), dtype=np.float32), "q99": np.ones((7,), dtype=np.float32)}
            }
        }

    def __len__(self):
        # TODO =>> Replace with number of elements in your dataset!
        return 10000

    def __getitem__(self, idx):
        # TODO =>> Load image, action and instruction from disk -- we use dummy values
        image = Image.fromarray(np.asarray(np.random.rand(224, 224, 3) * 255.0, dtype=np.uint8))
        action = np.asarray(np.random.rand(7), dtype=np.float32)
        instruction = "do something spectacular"

        # Add instruction to VLA prompt
        prompt_builder = self.prompt_builder_fn("openvla")
        conversation = [
            {"from": "human", "value": f"What action should the robot take to {instruction}?"},
            {"from": "gpt", "value": self.action_tokenizer(action)},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize (w/ `base_tokenizer`)
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)

        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        #   =>> IMPORTANT :: IF WE'RE USING HF .forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        pixel_values = self.image_transform(image)

        # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
        labels[: -(len(action) + 1)] = IGNORE_INDEX

        return dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels)


class LeRobotIterDataset(IterableDataset):
    def __init__(
        self,
        repo_id: str,
        batch_transform: Optional[RLDSBatchTransform],
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[list[float]] | None = None,
        tolerance_s: float = 1e-4,
        revision: str | None = None,
        force_cache_sync: bool = False,
        download_videos: bool = True,
        video_backend: str | None = None,
        train: bool = True,
    ) -> None:
        """
        初始化 LeRobotIterDataset

        Args:
            repo_id: LeRobot 数据集的仓库 ID
            batch_transform: 数据转换器，用于将 LeRobot 格式转换为 OpenVLA 格式
            其他参数与 LeRobotDataset 相同
        """
        self.repo_id = repo_id
        self.batch_transform = batch_transform
        self.train = train

        # 创建内部的 LeRobotDataset 实例
        self.lerobot_dataset = LeRobotDataset(
            repo_id=repo_id,
            root=root,
            episodes=episodes,
            image_transforms=image_transforms,
            delta_timestamps=delta_timestamps,
            tolerance_s=tolerance_s,
            revision=revision,
            force_cache_sync=force_cache_sync,
            download_videos=download_videos,
            video_backend=video_backend,
        )

        # 存储数据集统计信息，用于动作去归一化
        self.dataset_statistics = {self.repo_id: self.lerobot_dataset.meta.stats}

        self.dataset_length = len(self.lerobot_dataset)

    def _convert_lerobot_to_rlds_format(self, lerobot_item: Dict[str, Any]) -> Dict[str, Any]:
        """
        将 LeRobot 数据格式转换为 RLDS 格式，以便 RLDSBatchTransform 可以处理

        Args:
            lerobot_item: LeRobot 数据集返回的单个数据项

        Returns:
            转换为 RLDS 格式的数据项
        """
        # 构造 RLDS 格式的数据结构
        rlds_batch = {
            "dataset_name": self.repo_id.encode(),  # 转换为 bytes
            "observation": {},
            "task": {},
            "action": None,
            "absolute_action_mask": None,
        }

        # 处理图像数据 - 查找相机视角
        camera_keys = self.lerobot_dataset.meta.camera_keys
        primary_found = False
        wrist_found = False

        for cam_key in camera_keys:
            if cam_key in lerobot_item:
                # 获取图像数据并转换格式
                img_data = lerobot_item[cam_key]
                if isinstance(img_data, torch.Tensor):
                    # 从 (C, H, W) 转换为 (1, H, W, C) 格式
                    if img_data.dim() == 3:
                        img_data = img_data.permute(1, 2, 0).unsqueeze(0).numpy()
                    elif img_data.dim() == 4:
                        img_data = img_data.permute(0, 2, 3, 1).numpy()

                # 根据相机名称映射到标准命名
                if not primary_found and (
                    "primary" in cam_key.lower() or "top" in cam_key.lower() or "front" in cam_key.lower()
                ):
                    rlds_batch["observation"]["image_primary"] = img_data
                    primary_found = True
                elif not wrist_found and (
                    "wrist" in cam_key.lower() or "hand" in cam_key.lower() or "gripper" in cam_key.lower()
                ):
                    rlds_batch["observation"]["image_wrist"] = img_data
                    wrist_found = True
                else:
                    # 如果没有明确的主相机，使用第一个相机作为主相机
                    if not primary_found:
                        rlds_batch["observation"]["image_primary"] = img_data
                        primary_found = True
                    elif not wrist_found:
                        rlds_batch["observation"]["image_wrist"] = img_data
                        wrist_found = True

        # 处理本体感知数据
        if "observation.state" in lerobot_item:
            proprio_data = lerobot_item["observation.state"]
            if isinstance(proprio_data, torch.Tensor):
                proprio_data = proprio_data.unsqueeze(0).numpy()  # 添加 batch 维度
            rlds_batch["observation"]["proprio"] = proprio_data

        # 处理时间戳
        if "timestamp" in lerobot_item:
            timestamp = lerobot_item["timestamp"]
            if isinstance(timestamp, torch.Tensor):
                timestamp = timestamp.item()
            rlds_batch["observation"]["timestep"] = timestamp

        # 处理动作数据
        if "action" in lerobot_item:
            action = lerobot_item["action"]
            if isinstance(action, torch.Tensor):
                action = action.numpy()

            # 构造动作序列（用于 action chunking）
            # 假设当前只有一个动作，需要构造 future actions
            action_dim = action.shape[-1] if action.ndim > 0 else len(action)
            future_actions = []

            # 尝试获取未来的动作（如果可用）
            current_idx = lerobot_item.get("index", 0)
            if isinstance(current_idx, torch.Tensor):
                current_idx = current_idx.item()

            # 构造动作序列（当前动作 + 未来动作）
            actions_sequence = [action]

            # 填充未来动作
            for i in range(1, NUM_ACTIONS_CHUNK):
                try:
                    future_idx = current_idx + i
                    if future_idx < len(self.lerobot_dataset):
                        future_item = self.lerobot_dataset[future_idx]
                        future_action = future_item["action"]
                        if isinstance(future_action, torch.Tensor):
                            future_action = future_action.numpy()
                        actions_sequence.append(future_action)
                    else:
                        # 如果没有更多动作，重复最后一个动作
                        actions_sequence.append(action)
                except:
                    # 如果获取失败，重复当前动作
                    actions_sequence.append(action)

            rlds_batch["action"] = np.array(actions_sequence[:NUM_ACTIONS_CHUNK])

        # 处理任务描述
        if "task" in lerobot_item:
            task_description = lerobot_item["task"]
            if isinstance(task_description, str):
                task_description = task_description.encode()
            rlds_batch["task"]["language_instruction"] = task_description

        # 添加必要的 pad_mask 信息
        rlds_batch["observation"]["pad_mask_dict"] = {
            "image_primary": np.array([True]),
            "image_wrist": np.array([True]),
            "proprio": np.array([True]),
            "timestep": np.array([True]),
        }

        rlds_batch["task"]["pad_mask_dict"] = {
            "language_instruction": True,
            "image_primary": True,
            "image_wrist": True,
            "proprio": True,
            "timestep": True,
        }

        # 添加绝对动作掩码（根据动作类型设置）
        if rlds_batch["action"] is not None:
            action_dim = rlds_batch["action"].shape[-1]
            # 假设最后一维是抓取器，设置为绝对值
            absolute_mask = [False] * (action_dim - 1) + [True]
            rlds_batch["absolute_action_mask"] = np.array(absolute_mask)

        return rlds_batch

    def __iter__(self) -> Dict[str, Any]:
        """
        迭代数据集中的所有样本，返回经过 batch_transform 处理的数据

        输出示例：
        {
            'pixel_values': torch.Size([6, 224, 224]),
            'input_ids': torch.Size([90]),
            'labels': torch.Size([90]),
            'dataset_name': b'your_lerobot_dataset',
            'actions': np.array([8, 7]),
            'pixel_values_wrist': torch.Size([6, 224, 224]),  # 如果使用手腕相机
            'proprio': np.array([1, 7])  # 如果使用本体感知
        }
        """
        for idx in range(len(self.lerobot_dataset)):
            # 获取 LeRobot 格式的数据
            lerobot_item = self.lerobot_dataset[idx]

            # 转换为 RLDS 格式
            rlds_batch = self._convert_lerobot_to_rlds_format(lerobot_item)

            # 应用 batch_transform 转换为 OpenVLA 格式
            if self.batch_transform is None:
                yield rlds_batch
            else:
                transformed_batch = self.batch_transform(rlds_batch)

                yield transformed_batch

    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, idx: int) -> None:
        raise NotImplementedError("IterableDataset does not implement map-style __getitem__; see __iter__ instead!")

    @property
    def fps(self) -> int:
        return self.lerobot_dataset.fps

    @property
    def num_episodes(self) -> int:
        return self.lerobot_dataset.num_episodes

    @property
    def features(self) -> dict:
        return self.lerobot_dataset.features

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({{\n"
            f"    Repository ID: '{self.repo_id}',\n"
            f"    Number of samples: '{len(self)}',\n"
            f"    Number of episodes: '{self.num_episodes}',\n"
            f"    FPS: '{self.fps}',\n"
            f"    Features: '{list(self.features.keys())}',\n"
            "}}"
        )


class EpisodicLeRobotIterDataset(LeRobotIterDataset):
    def __iter__(self) -> Dict[str, Any]:
        """
        按 episode 返回数据，每次 yield 一个完整的 episode
        """
        current_episode = []
        current_ep_idx = None

        for idx in range(len(self.lerobot_dataset)):
            lerobot_item = self.lerobot_dataset[idx]
            ep_idx = lerobot_item["episode_index"].item()

            # 如果是新的 episode
            if current_ep_idx is not None and ep_idx != current_ep_idx:
                # 返回上一个 episode 的所有步骤
                yield current_episode
                current_episode = []

            # 转换当前步骤
            rlds_batch = self._convert_lerobot_to_rlds_format(lerobot_item)
            transformed_step = self.batch_transform(rlds_batch)
            current_episode.append(transformed_step)
            current_ep_idx = ep_idx

        # 返回最后一个 episode
        if current_episode:
            yield current_episode
