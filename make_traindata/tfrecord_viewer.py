import tensorflow as tf
import numpy as np

'''
    Definitions
'''
FIRST_N_RECORDS = 2
BYTES_LIST_PRINT_START_ELEMENTS = 2
BYTES_LIST_PRINT_TRAILING_ELEMENTS = 2
BYTES_LIST_MAX_BINARY_PRINT_LEN = 32
FLOAT_LIST_START_PRINT_LEN = 8
FLOAT_LIST_TRAILING_PRINT_LEN = 5
INT_LIST_START_PRINT_LEN = 8
INT_LIST_TRAILING_PRINT_LEN = 5

TFRECORD_PATH = 'tfrecord_examples/test.tfrecord'
# TFRECORD_PATH = 'tfrecord_examples/mnist.tfrecord'
# TFRECORD_PATH = 'tfrecord_examples/nyu_rot.tfrecord'
# TFRECORD_PATH = 'E:/data/temp/libero_spatial_no_noops/1.0.0/libero_spatial-train.tfrecord-00000-of-00016'
TFRECORD_PATH = 'E:/data/temp/test/ruijia_robot_grip_dataset/1.0.0/ruijia_robot_grip_dataset-train.tfrecord-00000-of-00002'
# TFRECORD_PATH = './ruijia_robot_grip_dataset-train.tfrecord-00000-of-00500'
# TFRECORD_PATH = 'E:/data/temp/test/le2tf/pick up the red tool from the desk and move it to hang on the black hook-train.tfrecord-00000-of-00032'
TFRECORD_PATH = '/data1/datasets/can_remove/hq_workspace/test_dataset/final_data/ruijia_robot_grip_dataset/1.3.1/ruijia_robot_grip_dataset-train.tfrecord-00000-of-00001'

raw_dataset = tf.data.TFRecordDataset([TFRECORD_PATH])


'''
    1st method to list all features of raw_dataset
'''
# raw_record = next(iter(raw_dataset))
# example = tf.train.Example()
# example.ParseFromString(raw_record.numpy())
# print(f"All Features: {list(example.features.feature.keys())}")


'''
    2nd method to list all features of raw_dataset
'''
# for raw_record in raw_dataset.take(1):
#     example = tf.train.Example()
#     example.ParseFromString(raw_record.numpy())
#     print(f"All Features: {list(example.features.feature.keys())}")


'''
    1st method to traverse whole dataset
'''


def bytes_to_str(bytes_array: bytes):
    is_asc = True
    for byte in bytes_array[:128]:
        if byte >= 32 and byte <= 126:  # 32-126 是可打印的 ASCII 字符范围
            pass
        else:
            if byte == 9:  # 排除制表符
                pass
            else:
                is_asc = False
                break

    if is_asc:
        return str(bytes_array)
    else:
        if len(bytes_array) <= BYTES_LIST_MAX_BINARY_PRINT_LEN:
            return str(bytes_array)
        else:
            return str(bytes_array[:BYTES_LIST_MAX_BINARY_PRINT_LEN]) + '...'


for record_idx, raw_record in enumerate(raw_dataset.take(FIRST_N_RECORDS)):
    print(f"Record {record_idx + 1}:")
    if record_idx!=0:
        continue
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())

    actions_list = []
    states_list = []

    # sort example.features.feature.keys() by asc
    sorted_keys = sorted(example.features.feature.keys())
    print('sorted_keys is ',sorted_keys)
    for feature_name in sorted_keys:
        feature = example.features.feature[feature_name]
        if feature_name not in ['steps/action','steps/observation/state']:
            continue

        if feature_name =='steps/action':
            value = np.array(feature.float_list.value)
            print(type(value),value.reshape(-1,7).shape)
            np.savetxt('actions.csv',value.reshape(-1,7),delimiter=",")
        if feature_name =='steps/observation/state':
            value = np.array(feature.float_list.value)
            print(type(value),value.reshape(-1,7).shape)
            np.savetxt('states.csv',value.reshape(-1,7),delimiter=",")

        if feature.HasField('bytes_list') and False:
            array_len = len(feature.bytes_list.value)

            if array_len == 1:  # if only 1 bytes, print in just 1 line, no wrap
                sub_array = feature.bytes_list.value[0]
                sub_array_len = len(sub_array)
                print(f"  {feature_name}: ({sub_array_len}) {bytes_to_str(sub_array)}")
            elif array_len > BYTES_LIST_PRINT_START_ELEMENTS + BYTES_LIST_PRINT_TRAILING_ELEMENTS:  # if too many bytes objects, print only start & trailing bytes objects
                print(f"  {feature_name}({array_len}): ")

                for sub_array_idx in range(BYTES_LIST_PRINT_START_ELEMENTS):
                    sub_array = feature.bytes_list.value[sub_array_idx]
                    sub_array_len = len(sub_array)
                    print(f"    ({sub_array_len}) {bytes_to_str(sub_array)}")

                print(f"    ...")

                for sub_array_idx in range(array_len - BYTES_LIST_PRINT_TRAILING_ELEMENTS, array_len):
                    sub_array = feature.bytes_list.value[sub_array_idx]
                    sub_array_len = len(sub_array)
                    print(f"    ({sub_array_len}) {bytes_to_str(sub_array)}")
            elif array_len > 0:  # if just a few bytes objects, print all
                print(f"  {feature_name}({array_len}): ")

                for sub_array_idx in range(array_len):
                    sub_array = feature.bytes_list.value[sub_array_idx]
                    sub_array_len = len(sub_array)
                    print(f"    ({sub_array_len}) {bytes_to_str(sub_array)}")
            else:
                print(f"  {feature_name}(0): []")
        elif feature.HasField('float_list'):
            array_len = len(feature.float_list.value)
            if array_len <= FLOAT_LIST_START_PRINT_LEN + FLOAT_LIST_TRAILING_PRINT_LEN:
                # rounded_float_list = [f"{value:.2f}" for value in feature.float_list.value]
                rounded_float_list = [f"{value}" for value in feature.float_list.value]
            else:
                rounded_float_list = []
                # rounded_float_list.extend([f"{value:.2f}" for value in feature.float_list.value[:FLOAT_LIST_START_PRINT_LEN + 1]])
                rounded_float_list.extend([f"{value}" for value in feature.float_list.value[:FLOAT_LIST_START_PRINT_LEN + 1]])
                rounded_float_list.append('...')
                # rounded_float_list.extend([f"{value:.2f}" for value in feature.float_list.value[-FLOAT_LIST_TRAILING_PRINT_LEN:]])
                rounded_float_list.extend([f"{value}" for value in feature.float_list.value[-FLOAT_LIST_TRAILING_PRINT_LEN:]])

            print(f"  {feature_name}({array_len}): [{', '.join(rounded_float_list)}]")
        elif feature.HasField('int64_list') and False:
            array_len = len(feature.int64_list.value)

            if array_len <= INT_LIST_START_PRINT_LEN + INT_LIST_TRAILING_PRINT_LEN:
                new_int_list = [f"{value}" for value in feature.int64_list.value]
            else:
                new_int_list = []
                new_int_list.extend([f"{value}" for value in feature.int64_list.value[:INT_LIST_START_PRINT_LEN + 1]])
                new_int_list.append('...')
                new_int_list.extend([f"{value}" for value in feature.int64_list.value[-INT_LIST_TRAILING_PRINT_LEN:]])

            print(f"  {feature_name}({array_len}): [{', '.join(new_int_list)}]")
        else:
            print(f"  {feature_name}: Unknown Feature Type")

    print('')

'''
    2nd method to traverse whole dataset using "feature_desc"
'''
# feature_desc = {
#     'my_bool': tf.io.FixedLenFeature([], tf.int64, default_value=0),
#     'my_int': tf.io.FixedLenFeature([], tf.int64, default_value=-1),
#     'my_string': tf.io.FixedLenFeature([], tf.string, default_value=''),
#     'my_float': tf.io.FixedLenFeature([], tf.float32, default_value=-1.0)
# }

# def _parse_function(example_proto):
#     return tf.io.parse_single_example(example_proto, feature_desc)

# parsed_dataset = raw_dataset.map(_parse_function)

# for parsed_record in parsed_dataset.take(2):
#     print(repr(parsed_record))