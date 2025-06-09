import os
import pyarrow.parquet as pq
import pyarrow as pa
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import Callable, Optional


def get_all_file_path(file_dir: str, filter_=(".parquet")) -> list:
    # 遍历文件夹下所有的file
    return [
        os.path.join(maindir, filename)
        for maindir, _, file_name_list in os.walk(file_dir)
        for filename in file_name_list
        if os.path.splitext(filename)[1] in filter_
    ]


def modify_parquet_data(
    input_datapath: str,
    output_datapath: str,
    state_modifier: Optional[Callable] = None,
    action_modifier: Optional[Callable] = None,
    sort_files: bool = True,
):
    """
    解析parquet文件，修改observation.state和action，然后保存到新的parquet文件

    参数:
    input_datapath (str): 输入parquet文件夹路径
    output_datapath (str): 输出parquet文件夹路径
    state_modifier (Callable): 修改state的函数，接收state数组，返回修改后的state
    action_modifier (Callable): 修改action的函数，接收action数组，返回修改后的action
    sort_files (bool): 是否对文件进行排序
    """

    # 创建输出文件夹
    os.makedirs(output_datapath, exist_ok=True)

    # 获取所有parquet文件
    filenames = [f for f in os.listdir(input_datapath) if f.endswith(".parquet")]

    if sort_files:
        filenames = sort_files_by_part(filenames)

    print(f"找到 {len(filenames)} 个parquet文件")

    for filename in tqdm(filenames, desc="处理parquet文件"):
        input_file_path = os.path.join(input_datapath, filename)
        output_file_path = os.path.join(output_datapath, filename)

        # 读取parquet文件
        df = pd.read_parquet(input_file_path)

        print(f"处理文件: {filename}, 行数: {len(df)}")
        print(f"列名: {list(df.columns)}")

        # 修改数据
        modified_df = modify_dataframe(df, state_modifier, action_modifier)

        # 保存到新的parquet文件
        modified_df.to_parquet(output_file_path, index=False)

        print(f"已保存修改后的文件: {output_file_path}")


def modify_dataframe(
    df: pd.DataFrame, state_modifier: Optional[Callable] = None, action_modifier: Optional[Callable] = None
) -> pd.DataFrame:
    """
    修改DataFrame中的state和action数据

    参数:
    df (pd.DataFrame): 原始DataFrame
    state_modifier (Callable): 修改state的函数
    action_modifier (Callable): 修改action的函数

    返回:
    pd.DataFrame: 修改后的DataFrame
    """

    modified_df = df.copy()

    # 根据您的代码，假设列的顺序是：action(第0列), observation.state(第1列), timestamp(第2列)
    column_names = list(df.columns)

    # 修改action数据（第0列）
    if action_modifier is not None and len(column_names) > 0:
        action_column = column_names[0]
        print(f"修改action列: {action_column}")

        # 对每行的action进行修改
        for idx in range(len(modified_df)):
            original_action = modified_df.iloc[idx, 0]

            # 如果action是列表形式，转换为numpy数组
            if isinstance(original_action, list):
                action_array = np.array(original_action)
            else:
                action_array = np.array(original_action)

            # 应用修改函数
            modified_action = action_modifier(action_array)

            # 将修改后的action转换回原格式
            modified_df.at[idx, action_column] = modified_action.tolist()

    # 修改observation.state数据（第1列）
    if state_modifier is not None and len(column_names) > 1:
        state_column = column_names[1]
        print(f"修改state列: {state_column}")

        # 对每行的state进行修改
        for idx in range(len(modified_df)):
            original_state = modified_df.iloc[idx, 1]

            # 如果state是列表形式，转换为numpy数组
            if isinstance(original_state, list):
                state_array = np.array(original_state)
            else:
                state_array = np.array(original_state)

            # 应用修改函数
            modified_state = state_modifier(state_array)

            # 将修改后的state转换回原格式
            modified_df.at[idx, state_column] = modified_state.tolist()

    return modified_df


def sort_files_by_part(file_names):
    """
    按文件名中的特定部分排序（复用您的代码）
    """

    def extract_sort_key(file_name):
        """从文件名中提取排序键"""
        keyname = str(file_name).split("_")[-1].split(".")[0]
        try:
            return int(keyname)
        except ValueError:
            return keyname

    return sorted(file_names, key=extract_sort_key)


import rtde_control
from scipy.spatial.transform import Rotation


class TCPTransformer:
    def __init__(self, robot_ip="192.168.0.205"):
        self.rtde_c = rtde_control.RTDEControlInterface(robot_ip)
        self.tcpoffset = self.rtde_c.getTCPOffset()

    def __call__(self, action_value: np.ndarray) -> np.ndarray:
        """
        将action_value转换为TCP坐标系下的位姿
        action_value: [x, y, z, rx, ry, rz]
        返回: [x, y, z, rx, ry, rz] 在TCP坐标系下的位姿
        """
        tcp = self.rtde_c.getFowardKinematics(action_value[:6].tolist(), self.tcpoffset)
        rpy = self._axis_angle_to_rpy(tcp[3:])
        return np.array([*tcp[:3], *rpy, action_value[-1]], dtype=action_value.dtype)

    def _axis_angle_to_rpy(self, axis_angle: np.ndarray, rot_axis="xyz") -> np.ndarray:
        """将轴角表示转换为RPY格式。

        Args:
            axis_angle (np.ndarray): 轴角表示 [Rx, Ry, Rz]。

        Returns:
            np.ndarray: RPY格式 [Roll, Pitch, Yaw]。
        """
        rotation = Rotation.from_rotvec(axis_angle)  # 从旋转向量创建旋转
        rpy = rotation.as_euler(rot_axis, degrees=False)  # 转换为RPY（弧度制）
        return rpy


# 使用示例
if __name__ == "__main__":
    # 基本使用示例
    input_path = "/data1/datasets/can_remove/hq_workspace/test_dataset/ori_data/data/chunk-000"
    output_path = "/data1/datasets/can_remove/hq_workspace/test_dataset/change_par"

    print("开始修改parquet数据...")

    modifier = TCPTransformer("192.168.0.205")
    modify_parquet_data(
        input_datapath=input_path,
        output_datapath=output_path,
        state_modifier=modifier,
        action_modifier=modifier,
    )

    print("数据修改完成！")
