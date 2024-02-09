# import sys
# import os

# # Get the current script's directory
# current_dir = os.path.dirname(os.path.abspath(__file__))

# # Add the parent directory to the sys.path
# parent_dir = os.path.join(current_dir, '..')
# sys.path.append(parent_dir)
# print(parent_dir)

# # Add the subfolder to the sys.path
# subfolder_dir = os.path.join(parent_dir, 'pupil',  'pupil_src', 'shared_modules')
# sys.path.append(subfolder_dir)

# print(subfolder_dir)

# import file_methods

# # file_methods.PLData_Writer 
# # file_methods.load_pldata_file() 
# inputDir = '/Users/azamatkaibaldiyev/GREYC_project/Pupil_labs/pupil/recordings/2024_01_16/001'

# #pupil_data_path = os.path.join(inputDir, 'pupil_data')

# data, data_ts, topics = file_methods.load_pldata_file(inputDir, 'gaze') 

# print(len(data))
# print(len(data_ts))
# print(len(topics))

# print(data[0])


# data, data_ts, topics = file_methods.load_pldata_file(inputDir, 'pupil') 

# print(len(data))
# print(len(data_ts))
# print(len(topics))

# print(data[0])
# print("#")
# print(data[1])
# print("#")
# print(data[2])
# print("#")
# print(data[3])

########################################################################################################################################

import os
import collections

import msgpack
import numpy as np


PLData = collections.namedtuple("PLData", ["data", "timestamps", "topics"])


def serialized_dict_from_msgpack_bytes(data):
    return msgpack.unpackb(
        data, raw=False, use_list=False, ext_hook=msgpack_unpacking_ext_hook,
    )


def msgpack_unpacking_ext_hook(self, code, data):
    SERIALIZED_DICT_MSGPACK_EXT_CODE = 13
    if code == SERIALIZED_DICT_MSGPACK_EXT_CODE:
        return serialized_dict_from_msgpack_bytes(data)
    return msgpack.ExtType(code, data)


def load_pldata_file(directory, topic):
    ts_file = os.path.join(directory, topic + "_timestamps.npy")
    msgpack_file = os.path.join(directory, topic + ".pldata")
    try:
        data = []
        topics = []
        data_ts = np.load(ts_file)
        with open(msgpack_file, "rb") as fh:
            for topic, payload in msgpack.Unpacker(fh, raw=False, use_list=False):
                datum = serialized_dict_from_msgpack_bytes(payload)
                data.append(datum)
                topics.append(topic)
    except FileNotFoundError:
        data = []
        data_ts = []
        topics = []

    return PLData(data, data_ts, topics)


if __name__ == "__main__":

    # edit `path` s.t. it points to your recording
    #path = "/Users/me/recordings/2020_06_19/001"
    path = '/Users/azamatkaibaldiyev/GREYC_project/Pupil_labs/pupil/recordings/2024_01_16/001'

    # Read "gaze.pldata" and "gaze_timestamps.npy" data
    get_data = load_pldata_file(path, "gaze")

    data_gaze = get_data.data
    data_ts = get_data.timestamps
    topics = get_data.topics

    import pprint

    p = pprint.PrettyPrinter(indent=4)

    print(">>> FIRST GAZE TIMESTAMP:")
    p.pprint(data_ts[0])
    print()

    print(">>> FIRST GAZE DATUM:")
    p.pprint(data_gaze[0])
    print()

