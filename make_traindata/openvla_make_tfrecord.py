import os, cv2, json
import tensorflow as tf
from tqdm import tqdm


def json_load(json_file):
    with open(json_file, 'r') as fr:
        episode_infos = json.load(fr)
    return episode_infos['steps']


def parse_json(json_file, resize_imgW, resize_imgH):
    language_instruction = []
    obs_image_1 = []  # obs_image_1.tobytes()
    obs_image_2 = []  # obs_image_2.tobytes()
    obs_state = []
    action = []

    episode_infos = json_load(json_file)
    json_file_split = os.path.split(json_file)[0]
    episode_keys_length = len(episode_infos.keys())
    episode_length = episode_keys_length
    for idx in tqdm(range(episode_keys_length)):
        idx_info = episode_infos[str(idx)]
        ## process imgs
        image_1_path = os.path.join(json_file_split, idx_info['observation']['front_image_filename'].split('/')[-1])
        image_2_path = os.path.join(json_file_split, idx_info['observation']['right_image_filename'].split('/')[-1])
        if not os.path.isfile(image_1_path) or not os.path.isfile(
                image_2_path):  # img is not exists, episode_length - 1
            episode_length = episode_length - 1
            continue
        print('====== image_1_path  ==',image_1_path)
        image_1 = tf.io.read_file(image_1_path)
        image_1_tensor = tf.image.decode_png(image_1, channels=3)
        image_1_resize_tensor = tf.image.resize(image_1_tensor, (resize_imgH, resize_imgW))  # tf size(H,W)
        image_1_resize_tensor = tf.cast(image_1_resize_tensor * 255, tf.uint8)
        image_1_resize_string = tf.image.encode_png(image_1_resize_tensor)
        print(image_1_resize_string)
        # exit(0)
        obs_image_1.append(image_1_resize_string.numpy())

        image_2 = tf.io.read_file(image_2_path)
        image_2_tensor = tf.image.decode_png(image_2, channels=3)
        image_2_resize_tensor = tf.image.resize(image_2_tensor, (resize_imgH, resize_imgW))  # tf size(H,W)
        image_2_resize_tensor = tf.cast(image_2_resize_tensor * 255, tf.uint8)
        image_2_resize_string = tf.image.encode_png(image_2_resize_tensor)
        obs_image_2.append(image_2_resize_string.numpy())

        # process language
        if LanguageTask == "":
            language_instruction.append(idx_info['natural_language_instruction'].encode('utf-8'))
        else:
            language_instruction.append(LanguageTask.encode('utf-8'))
        # process obs_state
        # Gripper position must be between 0/255.0 and 255/255.0 (open==0.0~close==1.0 => open==1,close==0)
        # idx_gripper = 1 if idx_info['observation']['joint_positions'][-1] < 0.35 else 0
        #
        # if idx == len(episode_infos.keys()) - 1:
        #     idx_gripper = 1  # (open)
        if DataForm == 'grip':
            # obs_state.append(idx_info['observation']['end_effector_pose']['position'] + \
            #                  idx_info['observation']['end_effector_pose']['orientation_rpy'] + \
            #                  [idx_gripper])
            obs_state.append(idx_info['observation']['obs_state'])
            action.append(idx_info['observation']['obs_state'])
        elif DataForm == 'joint':
            obs_state.append(idx_info['observation']['joint_positions'][:-1] + [idx_gripper])
        else:
            print('DataForm must be joint or grip')
            import sys
            sys.exit(0)

    # guqiang process action
    # for idx in range(len(obs_state) - 1):
    #     # delta x y z r p y or delta joint_6
    #     idx_action = [state_next - state_curr for state_next, state_curr in
    #                   zip(obs_state[idx + 1][:-1], obs_state[idx][:-1])]
    #     idx_action.append(obs_state[idx + 1][-1])  # append state_next gripper_status
    #     action.append(idx_action)
    # action.append([0.0] * 6 + [obs_state[-1][-1]])  # append last action , gripper_status is open



    obs_state_scalar = []
    action_scalar = []
    for idx in range(len(obs_state)):
        obs_state_scalar.extend(obs_state[idx])
        action_scalar.extend(action[idx])

    return language_instruction, obs_image_1, obs_image_2, obs_state_scalar, action_scalar, episode_length


def serialize_example(json_file):
    #### images info ####
    resize_imgW = 640  # 1920
    resize_imgH = 400  # 1200

    #### parse key value ####
    language_instruction, obs_image_1, obs_image_2, obs_state, action, episode_length = parse_json(json_file,
                                                                                                   resize_imgW,
                                                                                                   resize_imgH)

    #### set other value ####
    is_first = [1] + [0] * (episode_length - 1)
    is_last = [0] * (episode_length - 1) + [1]
    is_terminal = [0] * (episode_length - 1) + [1]
    reward = [0.0] * (episode_length - 1) + [1.0]
    discount = [1.0] * episode_length
    file_path = json_file.encode('utf-8')

    #### serialize ####

    example = tf.train.Example(features=tf.train.Features(feature={
        'steps/observation/state': tf.train.Feature(float_list=tf.train.FloatList(value=obs_state)),
        'steps/observation/image': tf.train.Feature(bytes_list=tf.train.BytesList(value=obs_image_1)),
        'steps/observation/wrist_image': tf.train.Feature(bytes_list=tf.train.BytesList(value=obs_image_2)),
        'steps/language_instruction': tf.train.Feature(bytes_list=tf.train.BytesList(value=language_instruction)),
        'steps/action': tf.train.Feature(float_list=tf.train.FloatList(value=action)),
        'steps/is_first': tf.train.Feature(int64_list=tf.train.Int64List(value=is_first)),
        'steps/is_last': tf.train.Feature(int64_list=tf.train.Int64List(value=is_last)),
        'steps/is_terminal': tf.train.Feature(int64_list=tf.train.Int64List(value=is_terminal)),
        'steps/reward': tf.train.Feature(float_list=tf.train.FloatList(value=reward)),
        'steps/discount': tf.train.Feature(float_list=tf.train.FloatList(value=discount)),

        'episode_metadata/file_path': tf.train.Feature(bytes_list=tf.train.BytesList(value=[file_path])),

    }))

    return example.SerializeToString()


def dataset_info(dsname, trainval, numBytes, shardLengths):
    dataset_info = {
        "citation": "// TODO(example_dataset): BibTeX citation",
        "description": "TODO(example_dataset): Markdown description of your dataset.\nDescription is **formatted** as markdown.\n\nIt should also contain any processing which has been applied (if any),\n(e.g. corrupted example skipped, images cropped,...):",
        "fileFormat": "tfrecord",
        "moduleName": f"{dsname}.{dsname}",
        "name": f"{dsname}",
        "releaseNotes": {
            f"{DataVesion}": "Initial release."
        },
        "splits": [
            {
                "filepathTemplate": "{DATASET}-{SPLIT}.{FILEFORMAT}-{SHARD_X_OF_Y}",
                "name": trainval,  # "train" or "val"
                "numBytes": str(numBytes),  # "3474742"
                "shardLengths": shardLengths  # ["6", "7"]
            }
        ],
        "version": f"{DataVesion}"
    }
    return dataset_info


if __name__ == "__main__":
    DataForm = 'grip'  # 'grip' or 'joint'
    DataVesion = '1.3.1'  # 1.0.x 帧率高=750ms,初始位置不固定; 1.1.x 帧率高=750ms,初始位置固定; 1.2.x 帧率310ms,初始位置固定
    # dataset_root = '/home/edavio/RecordUR/RecordUR/20250303'
    dataset_root = '/data1/workspace/wxl/data/tarindata/examples'
    # save_dir = f'./ruijia_dataset/ruijia_robot_{DataForm}_dataset/{DataVesion}'
    save_dir = f'/data1/workspace/wxl/data/tarindata/tfrecorddata/ruijia_robot_{DataForm}_data/{DataVesion}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    LanguageTask = 'pick up the red tool from the desk and move it to hang on the black hook'

    BatchSize = 10
    DatasetName = save_dir.split('/')[-2]  # 'ruijia_robot_dataset'
    TrainVal = 'train'
    BytesNum = 0  # Bytes of all tfrecords
    LengthsShard = []  # episode_length of every tfrecord

    episodes_json_path = []
    for episode_dir in sorted(os.listdir(dataset_root)):
        json_path = os.path.join(dataset_root, episode_dir, 'robot_motion_data.json')

        episodes_json_path.append(json_path)
    # for episode_dir in sorted(os.listdir(dataset_root2)):
    #     json_path = os.path.join(dataset_root2, episode_dir, 'robot_motion_data.json')
    #     episodes_json_path.append(json_path)

    BatchNum = len(episodes_json_path) // BatchSize
    BatchNum = BatchNum + 1 if len(episodes_json_path) % BatchSize > 0 else BatchNum  # nums of tfrecord files
    batch_episodes_json_path = [episodes_json_path[i:i + BatchSize] for i in
                                range(0, len(episodes_json_path), BatchSize)]
    print('batch_episodes_json_path is ', batch_episodes_json_path)
    for i, batch in enumerate(batch_episodes_json_path):
        writer_path = os.path.join(save_dir, f'{DatasetName}-{TrainVal}.tfrecord-{i:05d}-of-{BatchNum:05d}')
        print(f'\nprocessing {i + 1} batch, save to {writer_path}')
        with tf.io.TFRecordWriter(writer_path) as writer:
            for episode_json_path in batch:
                print(f'loading from {episode_json_path}')
                writer.write(serialize_example(episode_json_path))

        BytesNum += os.path.getsize(episode_json_path)
        LengthsShard.append(str(len(batch)))

    with open(os.path.join(save_dir, 'dataset_info.json'), 'w') as fw:
        json.dump(dataset_info(DatasetName, TrainVal, BytesNum, LengthsShard), fw, indent=4)
