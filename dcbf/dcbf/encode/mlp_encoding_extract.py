import os
import pickle
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import zlib

import numpy as np

from ..mtp import normalize_mtp_type


PICKLE_PROTOCOL = pickle.HIGHEST_PROTOCOL
ZLIB_COMPRESS_LEVEL = 1


@lru_cache(maxsize=None)
def _decode_cached(data_pkl, mtime_ns):
    with open(data_pkl, 'rb') as f:
        compressed_data = f.read()
    decompressed_data = zlib.decompress(compressed_data)
    return pickle.loads(decompressed_data)


def decode(data_pkl):
    data_pkl = os.path.abspath(data_pkl)
    mtime_ns = os.stat(data_pkl).st_mtime_ns
    return _decode_cached(data_pkl, mtime_ns)


def save_compressed_pickle(data, filename, compress_level=ZLIB_COMPRESS_LEVEL):
    serialized_data = pickle.dumps(data, protocol=PICKLE_PROTOCOL)
    compressed_data = zlib.compress(serialized_data, level=compress_level)
    with open(filename, "wb") as handle:
        handle.write(compressed_data)

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


def iter_descriptor_structures(des_out_path):
    structure_index = 0
    with open(des_out_path, "r", encoding="utf-8") as handle:
        line_iter = iter(handle)
        for line in line_iter:
            if "#start" not in line:
                continue
            atom_num = int(line.split()[1])
            atoms = []
            for _ in range(atom_num):
                atom_line = next(line_iter)
                parsed = np.fromstring(atom_line, sep=" ")
                if parsed.size == 0:
                    continue
                atom_type = int(parsed[0])
                descriptors = parsed[1:]
                atoms.append((atom_type, descriptors))
            yield structure_index, atoms
            structure_index += 1


def des_out2pkl(des_out_path, prefix, num_ele, mtp_type, hyx_mlp_path, body_name_list,out_path):
    total_list = [[] for _ in range(num_ele)]
    two_body_list = [[] for _ in range(num_ele)]
    three_body_list = [[] for _ in range(num_ele)]
    four_body_list = [[] for _ in range(num_ele)]
    two_body, three_body, four_body = extract_mtp_many_body_index(mtp_type, hyx_mlp_path)
    two_body = np.asarray(two_body, dtype=np.int64)
    three_body = np.asarray(three_body, dtype=np.int64)
    four_body = np.asarray(four_body, dtype=np.int64)

    for structure_index, atoms in iter_descriptor_structures(des_out_path):
        for atom_type, descriptors in atoms:
            if atom_type < 0 or atom_type >= num_ele:
                continue
            total_row = descriptors.tolist()
            total_row.append(structure_index)
            total_list[atom_type].append(total_row)

            two_row = descriptors[two_body].tolist()
            two_row.append(structure_index)
            two_body_list[atom_type].append(two_row)

            three_row = descriptors[three_body].tolist()
            three_row.append(structure_index)
            three_body_list[atom_type].append(three_row)

            four_row = descriptors[four_body].tolist()
            four_row.append(structure_index)
            four_body_list[atom_type].append(four_row)

    body_list = [two_body_list,three_body_list,four_body_list]
    body_name = [prefix+'_two_body_',prefix+'_three_body_',prefix+'_four_body_']
    dic = {'two':0,'three':1,'four':2}
    body_index = [dic[a] for a in body_name_list]

    selected_payloads = [
        (body, os.path.join(out_path, name + "coding_zlib.pkl"))
        for index, (body, name) in enumerate(zip(body_list, body_name))
        if index in body_index
    ]
    with ThreadPoolExecutor(max_workers=len(selected_payloads) or 1) as executor:
        futures = [executor.submit(save_compressed_pickle, body, filename) for body, filename in selected_payloads]
        for future in futures:
            future.result()

if __name__ == '__main__':
    hyx_mtp_path = 'hyx.mtp'
    mtp_type = 'l2k2'
    two_body, three_body, four_body = extract_mtp_many_body_index(mtp_type, hyx_mtp_path)
    print(two_body,three_body,four_body)
    des_out_path = 'md.out'
    prefix = 'md'
    ele = ['O','1','2']
    body_list = ['two']
    out_path = os.getcwd()
    des_out2pkl(des_out_path, prefix, len(ele), mtp_type, hyx_mtp_path, body_list,out_path)


