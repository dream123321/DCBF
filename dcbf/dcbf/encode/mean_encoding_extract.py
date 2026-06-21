import os
import pickle
import zlib

import numpy as np

from ..mtp import normalize_mtp_type
from .mlp_encoding_extract import iter_descriptor_structures, save_compressed_pickle

def decode(data_pkl):
    with open(data_pkl, 'rb') as f:
        compressed_data = f.read()
        decompressed_data = zlib.decompress(compressed_data)
        loaded_list = pickle.loads(decompressed_data)
    return loaded_list

def array2index(array_2d):
    # 获取行数
    num_rows = array_2d.shape[0]

    # 创建一个包含行号的数组，形状与二维数组的行数相同
    row_indices = np.arange(num_rows).reshape(-1, 1)  # -1 表示自动计算行数，使其与原数组匹配
    #print(row_indices)

    # 将行号数组添加到原二维数组的最后
    array_2d_with_row_indices = np.hstack((array_2d, row_indices))
    return array_2d_with_row_indices

def compress_save(data, filename):
    save_compressed_pickle(data, filename)

#获取two_body,three_body......的对应的列表
def mtp_many_body_list(mtp_type):
    mtp_type = normalize_mtp_type(mtp_type)
    if mtp_type == 'l2k2':
        dic = {'<0>':[0,10],'<11>':[27,28,29],'<22>': [30,31,32],
               '<211>':[72,73,80,81],'<222>':[91,92,99,100]}
    elif mtp_type == 'l2k3':
        dic = {'<0>': [0, 10, 20], '<11>': [46, 47, 48, 49, 50, 51], '<22>': [52, 53, 54, 55, 56, 57],
               '<211>': [178,179,180,187,188,189,196,197,198], '<222>': [211,212,213,220,221,222,229,230,231]}
    else:
        raise ValueError("mtp_type does not exist! If you want to add, modify the program!")

    two_body = []
    three_body = []
    four_body = []

    for key,value in dic.items():
        if len(key)-1 == 2:
            two_body += value
        if len(key)-1 == 3:
            three_body += value
        if len(key)-1 == 4:
            four_body += value
    return two_body,three_body,four_body

def alpha_moment_mapping(hyx_mtp_path):
    with open(hyx_mtp_path,'r') as f:
        lines = f.readlines()
    alpha_moment_mapping_list = []
    for line in lines:
        if 'alpha_moment_mapping' in line:
            alpha_moment_mapping_str = line
            start = alpha_moment_mapping_str.index('{') + 1
            end = alpha_moment_mapping_str.index('}')
            content = alpha_moment_mapping_str[start:end]
            # 将内容转换为列表
            alpha_moment_mapping_list = [int(num.strip()) for num in content.split(',')]
    if len(alpha_moment_mapping_list) == 0:
        raise ValueError("len(alpha_moment_mapping_list) = 0!")
    return alpha_moment_mapping_list

def extract_mtp_many_body_index(mtp_type,hyx_mtp_path):
    two_body,three_body,four_body = mtp_many_body_list(mtp_type)
    tt = alpha_moment_mapping(hyx_mtp_path)
    two_body = [tt.index(a) for a in two_body]
    three_body = [tt.index(a) for a in three_body]
    four_body = [tt.index(a) for a in four_body]
    return two_body,three_body,four_body

def mean_des_out2pkl(des_out_path, prefix, num_ele, mtp_type, hyx_mtp_path, body_name_list,out_path):
    t1, t2, t3 = extract_mtp_many_body_index(mtp_type, hyx_mtp_path)
    select_list = np.asarray(t1 + t2 + t3, dtype=np.int64)
    stru = []
    for stru_index, atoms in iter_descriptor_structures(des_out_path):
        if not atoms:
            continue
        selected = np.vstack([descriptors[select_list] for _, descriptors in atoms])
        mean_atoms = np.mean(selected, axis=0).tolist()
        stru.append(mean_atoms + [stru_index])
    compress_save([stru], os.path.join(out_path, prefix + '_mean_coding_zlib.pkl'))






