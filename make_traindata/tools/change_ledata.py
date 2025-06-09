import torch
import numpy as np
from pathlib import Path
from typing import Callable, Optional
import logging
from tqdm import tqdm
from concurrent import futures

from prismatic.vla.datasets.lerobot_dataset.lerobot_dataset import LeRobotDataset


class DatasetProcessor:
    """用于处理 LeRobot 数据集的类，支持读取、修改和保存数据集"""

    def __init__(
        self,
        source_repo_id: str,
        target_repo_id: str,
        source_root: Optional[str] = None,
        target_root: Optional[str] = None,
        action_modifier: Optional[Callable] = None,
        state_modifier: Optional[Callable] = None,
    ):
        """
        初始化数据集处理器

        Args:
            source_repo_id: 源数据集的仓库ID
            target_repo_id: 目标数据集的仓库ID
            source_root: 源数据集的本地路径
            target_root: 目标数据集的本地路径
            action_modifier: 修改action的函数，接收action数组，返回修改后的action
            state_modifier: 修改state的函数，接收state数组，返回修改后的state
        """
        self.source_repo_id = source_repo_id
        self.target_repo_id = target_repo_id
        self.source_root = Path(source_root) if source_root else None
        self.target_root = Path(target_root) if target_root else None
        self.action_modifier = action_modifier
        self.state_modifier = state_modifier

        # 加载源数据集
        self.source_dataset = LeRobotDataset(
            repo_id=source_repo_id,
            root=source_root,
        )

        logging.info(f"已加载源数据集: {source_repo_id}")
        logging.info(f"数据集信息: {self.source_dataset}")

    def create_target_dataset(self) -> LeRobotDataset:
        """创建目标数据集"""
        target_dataset = LeRobotDataset.create(
            repo_id=self.target_repo_id,
            fps=self.source_dataset.fps,
            root=self.target_root,
            features=self.source_dataset.features,
            use_videos=len(self.source_dataset.meta.video_keys) > 0,
        )

        logging.info(f"已创建目标数据集: {self.target_repo_id}")
        return target_dataset

    def process_and_save_dataset(self):
        """处理整个数据集并保存到新位置"""
        target_dataset = self.create_target_dataset()

        # 按episode处理数据
        episode_indices = list(range(self.source_dataset.meta.total_episodes))

        for ep_idx in tqdm(episode_indices, desc="处理episodes"):
            self._process_episode(ep_idx, target_dataset)

        # 编码视频（如果有的话）
        if len(target_dataset.meta.video_keys) > 0:
            logging.info("开始编码视频...")
            target_dataset.encode_videos()

        logging.info(f"数据集处理完成！保存在: {target_dataset.root}")
        return target_dataset

    def _process_episode(self, episode_idx: int, target_dataset: LeRobotDataset):
        """处理单个episode"""
        # 获取该episode的所有帧
        ep_start = self.source_dataset.episode_data_index["from"][episode_idx]
        ep_end = self.source_dataset.episode_data_index["to"][episode_idx]

        # 为目标数据集创建episode buffer
        target_dataset.episode_buffer = target_dataset.create_episode_buffer(episode_idx)

        for frame_idx in range(ep_start, ep_end):
            frame_data = self.source_dataset[frame_idx]

            # 修改frame数据
            modified_frame = self._modify_frame(frame_data)

            # 添加到目标数据集
            target_dataset.add_frame(modified_frame)

        # 保存episode
        target_dataset.save_episode()

    def _modify_frame(self, frame_data: dict) -> dict:
        """修改单帧数据"""
        modified_frame = {}

        for key, value in frame_data.items():
            if key == "action" and self.action_modifier is not None:
                # 修改action
                if isinstance(value, torch.Tensor):
                    action_array = value.numpy()
                else:
                    action_array = np.array(value)

                modified_action = self.action_modifier(action_array)
                if not isinstance(modified_action, torch.Tensor):
                    modified_action = torch.tensor(modified_action)
                modified_frame[key] = modified_action

            elif key.startswith("observation.") and "state" in key and self.state_modifier is not None:
                # 修改state相关的观测
                if isinstance(value, torch.Tensor):
                    state_array = value.numpy()
                else:
                    state_array = np.array(value)

                modified_state = self.state_modifier(state_array)
                if not isinstance(modified_state, torch.Tensor):
                    modified_state = torch.tensor(modified_state)
                modified_frame[key] = modified_state

            elif key in {'episode_index', 'frame_index', 'index', 'task_index'}:
                    continue
            elif key.startswith("observation.images"):
                modified_frame[key]=value.permute(1,2,0)
            else:
                # 其他数据保持不变
                if isinstance(value, torch.Tensor):
                    if value.dim() == 0:
                        value.unsqueeze_(0)
                modified_frame[key] = value

        return modified_frame


def modify_action(action: np.ndarray) -> np.ndarray:
    noise = np.ones_like(action)
    modified_action = action + noise

    return modified_action


def modify_state(state: np.ndarray) -> np.ndarray:
    modified_state = np.ones_like(state) + state

    return modified_state


# 使用示例
def main():
    """主函数示例"""
    logging.basicConfig(level=logging.INFO)

    # 创建数据集处理器
    processor = DatasetProcessor(
        source_repo_id="ori",  # 替换为您的源数据集ID
        target_repo_id="change_src",  # 替换为您的目标数据集ID
        source_root="/data1/datasets/ur_grasp_db/ur_grasp_0421",  # 可选：源数据集本地路径
        target_root="/data1/tmp/test_ur_dataset",  # 可选：目标数据集本地路径
        action_modifier=modify_action,
        state_modifier=modify_state,
    )

    # 处理并保存数据集
    try:
        modified_dataset = processor.process_and_save_dataset()
        print(f"数据集修改完成！新数据集保存在: {modified_dataset.root}")

        # 验证新数据集
        print("\n验证新数据集:")
        print(f"原数据集episodes数量: {processor.source_dataset.num_episodes}")
        print(f"新数据集episodes数量: {modified_dataset.num_episodes}")
        print(f"原数据集frames数量: {processor.source_dataset.num_frames}")
        print(f"新数据集frames数量: {modified_dataset.num_frames}")

    except Exception as e:
        logging.error(f"处理数据集时出错: {e}")
        raise


if __name__ == "__main__":
    main()
