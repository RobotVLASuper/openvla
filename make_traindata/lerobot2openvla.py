import os
import cv2
import pyarrow.parquet as pq
import json
import re

def sort_files_by_part(file_names):
    """
    按文件名中的特定部分排序

    参数:
    file_names (list): 包含文件名的列表
    pattern (str): 用于匹配的正则表达式模式
    case_sensitive (bool): 是否区分大小写，默认为True

    返回:
    list: 排序后的文件名列表
    """

    def extract_sort_key(file_name):
        """从文件名中提取排序键"""
        keyname = str(file_name).split('_')[-1].split('.')[0]
        return keyname

    # 使用提取的部分作为排序键
    return sorted(file_names, key=extract_sort_key)

def extract_frames(input_folder, output_folder, interval, captype):
    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        if filename.endswith(".mp4"):
            file_path = os.path.join(input_folder, filename)
            keyname = os.path.splitext(filename)[0]
            output_subfolder = os.path.join(output_folder, os.path.splitext(filename)[0])
            os.makedirs(output_subfolder, exist_ok=True)

            # 打开视频文件
            cap = cv2.VideoCapture(file_path)
            frame_count = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # 按照设定的间隔保存帧
                if frame_count % interval == 0:
                    output_filename = os.path.join(output_subfolder, f"{keyname}_{captype}_{frame_count}.png")
                    cv2.imwrite(output_filename, frame)

                frame_count += 1

            cap.release()


def parse_paquet_data(datapath,output_folder):

    alldatadict = {"steps":{}}
    filenames = os.listdir(datapath)
    filenames = sort_files_by_part(filenames)
    for filename in filenames:
        if filename.endswith(".parquet"):
            file_path = os.path.join(datapath, filename)
            keyname = os.path.splitext(filename)[0]
            output_subfolder = os.path.join(output_folder, os.path.splitext(filename)[0])
            topath = os.path.join(output_subfolder, 'robot_motion_data.json')
            # 打开Parquet文件
            parquet_file = pq.ParquetFile(file_path)

            # 读取整个文件内容
            table = parquet_file.read()
            num_rows = len(table)
            # 获取列名称
            column_names = table.column_names

            # 逐行读取文件
            # parquet_file = pq.ParquetFile(datapath)
            # 遍历每个row_group
            is_first = False
            is_last = False
            is_terminate = False

            for i in range(parquet_file.num_row_groups):
                row_group = parquet_file.read_row_group(i)
                row_group = row_group.to_pandas()
                # 遍历每一行

                for idx, row in row_group.iterrows():
                    if idx==0:
                        is_first = True
                    if idx == num_rows-1:
                        is_last = True
                        is_terminate = True
                    timestamp = row.iloc[2]
                    natural_language_instruction = "pick up the red tool from the desk and move it to hang on the black hook"
                    front_image_filename = keyname +'_top_' + str(idx) +'.png'
                    right_image_filename = keyname +'_right_' + str(idx) +'.png'
                    obs_state = row.iloc[1].tolist()
                    action = row.iloc[0].tolist()
                    observationdict = {}
                    observationdict['front_image_filename'] = front_image_filename
                    observationdict['right_image_filename'] = right_image_filename
                    observationdict['obs_state'] = obs_state
                    observationdict['action'] = action
                    print('file_path,obs_state is ',idx,file_path,obs_state)
                    print('file_path,action is ', idx,file_path,action)
                    alldatadict['steps'][str(idx)] = {}
                    alldatadict['steps'][str(idx)]['is_first'] = is_first
                    alldatadict['steps'][str(idx)]['is_last'] = is_last
                    alldatadict['steps'][str(idx)]['timestamp'] = timestamp
                    alldatadict['steps'][str(idx)]['natural_language_instruction'] = natural_language_instruction
                    alldatadict['steps'][str(idx)]['is_terminate'] = is_terminate
                    alldatadict['steps'][str(idx)]['observation'] = observationdict
                    # print('==========================')
            print('==========================',topath)
            with open(topath, 'w', encoding='utf-8') as f:
                json.dump(alldatadict, f, indent=4)






if __name__=="__main__":

    # input_folder_top = "/data1/workspace/wxl/data/tarindata/ur_grasp_0428_gello/videos/chunk-000/observation.images.0_top"
    output_folder_top = "/data1/workspace/wxl/data/tarindata/examples"
    # interval = 1
    # captype = 'top'
    # # 调用函数进行帧解析
    # extract_frames(input_folder_top, output_folder_top, interval, captype)
    #
    # input_folder_top = "/data1/workspace/wxl/data/tarindata/ur_grasp_0428_gello/videos/chunk-000/observation.images.1_right"
    # output_folder_top = "/data1/workspace/wxl/data/tarindata/examples"
    # interval = 1
    # captype = 'right'
    # # 调用函数进行帧解析
    # extract_frames(input_folder_top, output_folder_top, interval, captype)


    parquetdatapath = '/data1/workspace/wxl/data/tarindata/ur_grasp_0428_gello/data/chunk-000'

    parse_paquet_data(parquetdatapath,output_folder_top)

