"""
datasets.py

Lightweight PyTorch Dataset Definition for wrapping RLDS TFDS Pipeline; just defines transform from RLDS default
format to OpenVLA, IterableDataset shim.
"""

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple, Type, Callable, Optional, List, Union
import json
import pickle
from concurrent import futures
import multiprocessing as mp
from copy import deepcopy

from tqdm import tqdm
import pyarrow.parquet as pq
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, IterableDataset
from transformers import PreTrainedTokenizerBase
import torchvision.transforms as T
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
    NormalizationType,
    CLOSE_ACTION_NORM,
    CLOSE_SHUFFLE,
)
from prismatic.vla.datasets.rlds import make_interleaved_dataset, make_single_dataset
from prismatic.vla.datasets.rlds.oxe import OXE_NAMED_MIXTURES, get_oxe_dataset_kwargs_and_weights
from prismatic.vla.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
import prismatic.debug_tools as D


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


# *********TOOLS*************
def resize_image(image: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    """使用 Lanczos 插值法调整图像大小。输入和输出均为 uint8 类型。"""
    assert image.dtype == torch.uint8, "输入图像必须是 uint8 类型"
    transform = T.Compose([T.ToPILImage(), T.Resize(size, interpolation=T.InterpolationMode.LANCZOS), T.ToTensor()])
    image = transform(image)
    image = torch.clamp((image * 255).round(), 0, 255).byte()
    return image


# NOTE: not support img_aug,not suppport train
class LeRobotIterDataset(IterableDataset):
    def __init__(
        self,
        repo_id: str,
        batch_transform: Optional[RLDSBatchTransform],
        resize_resolution: Tuple[int, int],
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
        goal_relabeling_strategy: str = "uniform",  # 新增参数
        action_normalization_type: NormalizationType = NormalizationType.BOUNDS_Q99,
        action_norm_mask: Optional[Union[Tuple[bool, bool, bool, bool, bool, bool, bool], np.ndarray]] = (
            True,
            True,
            True,
            True,
            True,
            True,
            False,
        ),  # 用于动作去归一化
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
        self.dataset_statistics = self._get_dataset_statistics()
        self.dataset_length = len(self.lerobot_dataset)
        self.resize_resolution = resize_resolution

        # 如果启用目标重标记，需要按episode组织数据
        self.goal_relabeling_strategy = goal_relabeling_strategy
        if self.goal_relabeling_strategy:
            self._organize_episodes_data()
        self.action_normalization_type = action_normalization_type
        if isinstance(action_norm_mask, np.ndarray):
            self.action_norm_mask = action_norm_mask
        else:
            self.action_norm_mask = np.array(action_norm_mask, dtype=np.bool_)
        self.idx_list=list(range(len(self.lerobot_dataset)))


    def _norm_one_action(self, statistics: Dict[str, np.ndarray], data: np.ndarray) -> np.ndarray:
        if self.action_normalization_type == NormalizationType.NORMAL:
            mean = statistics["mean"]
            std = statistics["std"]
            # 对masked的维度进行归一化
            normalized_data = np.where(self.action_norm_mask, (data - mean) / (std + 1e-8), data)
            return normalized_data

        elif self.action_normalization_type in [NormalizationType.BOUNDS, NormalizationType.BOUNDS_Q99]:
            if self.action_normalization_type == NormalizationType.BOUNDS:
                low = statistics["min"]
                high = statistics["max"]
            else:  # BOUNDS_Q99
                low = statistics["q01"]
                high = statistics["q99"]

            # 归一化到[-1, 1]范围
            normalized_data = np.where(
                self.action_norm_mask, np.clip(2 * (data - low) / (high - low + 1e-8) - 1, -1, 1), data
            )

            # 将未使用的动作维度(min == max)映射为0
            zeros_mask = np.isclose(low, high)
            normalized_data = np.where(zeros_mask, 0.0, normalized_data)
            return normalized_data

    def _normalize_sample_action_and_proprio(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        使用numpy归一化sample中的action和proprio字段

        Args:
            sample: 包含action和observation字典的样本

        Returns:
            归一化后的sample
        """

        # 复制sample避免修改原始数据
        normalized_sample = sample.copy()
        normalized_sample["observation"] = sample["observation"].copy()

        # 定义需要归一化的键
        keys_to_normalize = {
            "action": ("action", sample["action"], self.dataset_statistics[self.repo_id]["action"]),
            "proprio": (
                "observation/proprio",
                sample["observation"]["proprio"],
                self.dataset_statistics[self.repo_id]["proprio"],
            ),
        }

        for key, (_, data, statistics) in keys_to_normalize.items():
            if key == "action":
                normalized_sample["action"] = self._norm_one_action(statistics, data)
            else:  # proprio
                normalized_sample["observation"]["proprio"] = self._norm_one_action(statistics, data)

        return normalized_sample

    def _get_dataset_statistics(self) -> None:
        dataset_statistics = {
            self.repo_id: {
                "action": self.lerobot_dataset.meta.stats["action"],
                "proprio": self.lerobot_dataset.meta.stats["observation.state"],
                "num_transitions": np.array(self.lerobot_dataset.meta.total_frames, dtype=np.int64),
                "num_trajectories": np.array(self.lerobot_dataset.meta.total_episodes, dtype=np.int64),
            }
        }

        # 这里就是为了得到狗日的q01，q99
        parquet_files = self._get_parquet_files()
        if not parquet_files:
            print("没有找到任何parquet文件，无法获取数据集统计信息")
            return dataset_statistics

        print(f"找到 {len(parquet_files)} 个parquet文件")

        # 使用多进程并行处理parquet文件
        num_processes = min(len(parquet_files), mp.cpu_count())

        # 收集结果
        all_actions = []
        all_proprio = []
        with futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
            # 将文件分配给不同进程
            tasks = []
            for file_path in parquet_files:
                future = executor.submit(self._process_parquet_file, file_path)
                tasks.append(future)

            for future in tqdm(
                futures.as_completed(tasks, timeout=300), total=len(tasks), desc="Processing parquet files"
            ):
                try:
                    stats = future.result()  # 5分钟超时
                    if stats is not None:
                        all_actions.append(stats[0])
                        all_proprio.append(stats[1])
                    # print(f"已处理文件 {i + 1}/{len(parquet_files)}")
                except Exception as e:
                    print(f"处理文件时出错: {e}")
        with D.timeblock(f"{self.repo_id} 拼接计算q01和q99 耗时:"):
            all_actions_array = (
                np.concatenate(all_actions, axis=0) if all_actions else np.empty((0, ACTION_DIM), dtype=np.float64)
            )
            all_proprio_array = (
                np.concatenate(all_proprio, axis=0) if all_proprio else np.empty((0, PROPRIO_DIM), dtype=np.float64)
            )
        with D.timeblock(f"Calculate quantiles for actions and proprioception data for {self.repo_id}"):
            actions_q01 = np.quantile(all_actions_array, 0.01, axis=0)
            actions_q99 = np.quantile(all_actions_array, 0.99, axis=0)
            proprio_q01 = np.quantile(all_proprio_array, 0.01, axis=0)
            proprio_q99 = np.quantile(all_proprio_array, 0.99, axis=0)

        dataset_statistics[self.repo_id]["action"]["q01"] = actions_q01.astype(np.float64)
        dataset_statistics[self.repo_id]["action"]["q99"] = actions_q99.astype(np.float64)
        dataset_statistics[self.repo_id]["proprio"]["q01"] = proprio_q01.astype(np.float64)
        dataset_statistics[self.repo_id]["proprio"]["q99"] = proprio_q99.astype(np.float64)

        return dataset_statistics

    def _process_parquet_file(self, file_path: Path) -> Dict[str, Any]:
        try:
            # 使用pyarrow读取parquet文件（更快）
            table = pq.read_table(file_path)

            # 转换为pandas DataFrame以便处理
            df = table.to_pandas()

            # print(f"处理文件 {file_path.name}, 包含 {len(df)} 行数据")

            # 处理action数据
            action_array = None
            if "action" in df.columns:
                action_data = df["action"].values
                # 处理可能的嵌套结构
                if len(action_data) > 0:
                    # 检查是否为数组类型
                    first_action = action_data[0]
                    if isinstance(first_action, (list, tuple, np.ndarray)):
                        # 将所有action堆叠成2D数组
                        action_array = np.stack([np.array(a) for a in action_data])
                    else:
                        action_array = action_data.reshape(-1, 1)
                    # print(f"  - Action数据形状: {action_array.shape}")

            proprio_array = None
            # 处理observation.state数据（proprio）
            if "observation.state" in df.columns:
                proprio_data = df["observation.state"].values
                if len(proprio_data) > 0:
                    first_proprio = proprio_data[0]
                    if isinstance(first_proprio, (list, tuple, np.ndarray)):
                        proprio_array = np.stack([np.array(p) for p in proprio_data])
                    else:
                        proprio_array = proprio_data.reshape(-1, 1)
                    # print(f"  - Proprio数据形状: {proprio_array.shape}")

        except Exception as e:
            print(f"读取文件 {file_path} 时出错: {e}")
        finally:
            if action_array is None:
                action_array = np.empty((0, ACTION_DIM), dtype=np.float32)
            if proprio_array is None:
                proprio_array = np.empty((0, PROPRIO_DIM), dtype=np.float32)
            return action_array, proprio_array

    def _get_parquet_files(self) -> List[Path]:
        """
        获取数据集中所有的parquet文件路径
        """
        parquet_files = []

        # 获取数据集根目录
        try:
            dataset_root = self.lerobot_dataset.root
            data_dir = dataset_root / "data"

            if not data_dir.exists():
                print(f"数据目录不存在: {data_dir}")
                return []

            # 如果指定了特定的episodes，只处理这些episodes
            if self.lerobot_dataset.episodes is not None:
                for ep_idx in self.lerobot_dataset.episodes:
                    parquet_path = dataset_root / self.lerobot_dataset.meta.get_data_file_path(ep_idx)
                    if parquet_path.exists():
                        parquet_files.append(parquet_path)
            else:
                # 递归查找所有parquet文件
                parquet_files = list(data_dir.rglob("*.parquet"))

            print(f"数据目录: {data_dir}")
            print(f"找到parquet文件数: {len(parquet_files)}")

        except Exception as e:
            print(f"获取parquet文件列表失败: {e}")
            return []

        return sorted(parquet_files)

    def _convert_lerobot_to_rlds_format(self, lerobot_item: Dict[str, Any]) -> Dict[str, Any]:
        """
        将 LeRobot 数据格式转换为 RLDS 格式，以便 RLDSBatchTransform 可以处理
        """
        # 构造 RLDS 格式的数据结构
        rlds_batch = {
            "dataset_name": self.repo_id.encode(),
            "observation": {},
            "task": {
                "language_instruction": None,  # 稍后填充
            },
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
                        img_data = (
                            resize_image(img_data, size=self.resize_resolution).permute(1, 2, 0).unsqueeze(0).numpy()
                        )
                    elif img_data.dim() == 4:
                        img_data = resize_image(img_data, size=self.resize_resolution).permute(0, 2, 3, 1).numpy()

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
            rlds_batch["observation"]["timestep"] = np.array(timestamp, dtype=np.int32)

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
        rlds_batch["observation"]["pad_mask"] = np.array([True])

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
            absolute_mask = [True] * (action_dim - 1) + [True]
            rlds_batch["absolute_action_mask"] = np.array(absolute_mask)

        return rlds_batch

    def _organize_episodes_data(self):
        """按episode组织数据，用于目标重标记"""
        self.episodes_data = {}
        ori_episode_len_infos = self.lerobot_dataset.meta.episodes
        current_data_idx = 0
        for episode_idx in range(len(ori_episode_len_infos)):
            episode_len = ori_episode_len_infos[episode_idx]["length"]
            self.episodes_data[episode_idx] = [x for x in range(current_data_idx, current_data_idx + episode_len)]
            current_data_idx += episode_len

    def _uniform_goal_relabeling(
        self, rlds_batch: Dict[str, Any], episode_indices: list, current_step_idx: int
    ) -> Dict[str, Any]:
        """
        实现均匀分布的目标重标记策略

        Args:
            rlds_batch: 当前的RLDS格式数据
            episode_indices: 当前episode中所有步骤的索引
            current_step_idx: 当前步骤在episode中的位置

        Returns:
            添加了目标信息的rlds_batch
        """
        traj_len = len(episode_indices)

        # 选择一个未来的随机索引 [current_step_idx + 1, traj_len)
        if current_step_idx + 1 < traj_len:
            # 随机选择未来的一个步骤作为目标
            future_step_idx = random.randint(current_step_idx + 1, traj_len - 1)
            goal_data_idx = episode_indices[future_step_idx]

            # 获取目标状态的观察数据
            goal_item = self.lerobot_dataset[goal_data_idx]

            # 将目标观察添加到task中
            for cam_key in self.lerobot_dataset.meta.camera_keys:
                if cam_key in goal_item:
                    img_data = goal_item[cam_key]
                    if isinstance(img_data, torch.Tensor):
                        if img_data.dim() == 3:
                            img_data = (
                                resize_image(img_data, size=self.resize_resolution).permute(1, 2, 0).unsqueeze(0).numpy()
                            )

                    # FIXME: 这里命名是靠拢libero的  后续要改
                    if "top" in cam_key.lower() or "primary" in cam_key.lower():
                        rlds_batch["task"]["image_primary"] = img_data
                    elif "right" in cam_key.lower() or "wrist" in cam_key.lower():
                        rlds_batch["task"]["image_wrist"] = img_data

            # 添加目标状态的本体感知信息
            if "observation.state" in goal_item:
                proprio_data = goal_item["observation.state"]
                if isinstance(proprio_data, torch.Tensor):
                    proprio_data = proprio_data.numpy()
                # rlds_batch["task"]["proprio"] = proprio_data
                rlds_batch["task"]["proprio"] = self._norm_one_action(
                    self.dataset_statistics[self.repo_id]["proprio"], proprio_data
                )

            rlds_batch["task"]["timestep"] = goal_data_idx
        return rlds_batch

    def __iter__(self) -> Dict[str, Any]:
        """
        迭代数据集中的所有样本，支持目标重标记
        """
        for idx in range(len(self.lerobot_dataset)) :
            # 获取 LeRobot 格式的数据
            if not CLOSE_SHUFFLE:
                idx=random.sample(self.idx_list, 1)[0]
            lerobot_item = self.lerobot_dataset[idx]

            # 转换为 RLDS 格式
            rlds_batch = self._convert_lerobot_to_rlds_format(lerobot_item)

            # 如果启用目标重标记
            if self.goal_relabeling_strategy == "uniform":
                ep_idx = lerobot_item["episode_index"].item()
                episode_indices = self.episodes_data[ep_idx]
                current_step_idx = episode_indices.index(idx)

                rlds_batch = self._uniform_goal_relabeling(rlds_batch, episode_indices, current_step_idx)

            if not CLOSE_ACTION_NORM:
                rlds_batch = self._normalize_sample_action_and_proprio(rlds_batch)

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
