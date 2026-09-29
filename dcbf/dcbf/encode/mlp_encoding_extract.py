import os
import pickle
import tempfile
import zlib

import numpy as np

from ..mtp import normalize_mtp_type


PICKLE_PROTOCOL = pickle.HIGHEST_PROTOCOL
ZLIB_COMPRESS_LEVEL = 1


def _decode_cached(data_pkl, mtime_ns):
    from ..memory_guard import require_memory
    decoder = zlib.decompressobj()
    size = 0
    with tempfile.SpooledTemporaryFile(max_size=8 * 1024 ** 2) as temporary:
        with open(data_pkl, 'rb') as handle:
            for chunk in iter(lambda: handle.read(1024 ** 2), b''):
                pending = chunk
                while pending:
                    block = decoder.decompress(pending, 8 * 1024 ** 2)
                    pending = decoder.unconsumed_tail
                    temporary.write(block)
                    size += len(block)
        if not decoder.eof:
            raise RuntimeError(f'Truncated compressed descriptor cache: {data_pkl}')
        # A legacy pickle can contain many Python objects. Refuse unsafe fallback.
        require_memory(size * 8)
        temporary.seek(0)
        return pickle.load(temporary)


def decode(data_pkl):
    if not isinstance(data_pkl, (str, os.PathLike)):
        return data_pkl
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


def compact_column_layout(two_body, three_body, four_body):
    """Columns worth parsing out of a descriptor row, and where each body sits.

    The kept set is the union of all three bodies and never depends on the
    requested body_list, because the mean flows consume two+three+four even when
    the caller only asked for two. ``columns`` are positions in the full
    descriptor row (the leading atom-type column excluded); ``positions`` maps
    each body to its columns' places inside the compact row.
    """
    columns = list(dict.fromkeys(int(c) for c in list(two_body) + list(three_body) + list(four_body)))
    place = {column: index for index, column in enumerate(columns)}
    positions = {
        'two': [place[int(c)] for c in two_body],
        'three': [place[int(c)] for c in three_body],
        'four': [place[int(c)] for c in four_body],
    }
    return np.asarray(columns, dtype=np.int64), positions


def descriptor_column_layout(mtp_type, hyx_mtp_path):
    """compact_column_layout for callers that only hold the potential path."""
    two_body, three_body, four_body = extract_mtp_many_body_index(mtp_type, hyx_mtp_path)
    return compact_column_layout(two_body, three_body, four_body)


def iter_descriptor_structures(des_out_path, columns=None):
    from ..memory_guard import current_guard, require_memory
    structure_index = 0
    if columns is None:
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
                    if not atoms and current_guard() is not None:
                        require_memory(atom_num * (parsed.size * 8 + 128))
                    atom_type = int(parsed[0])
                    descriptors = parsed[1:]
                    atoms.append((atom_type, descriptors))
                yield structure_index, atoms
                structure_index += 1
        return
    # Selective parse. A descriptor row carries 175 components but the pipeline
    # reads at most 34 of them, so converting the rest is pure waste. The row is
    # tab separated, so the wanted fields are addressed by index and only those
    # are converted; the per-structure block is allocated once and each yielded
    # row is a view into it.
    field_indexes = [0] + [int(column) + 1 for column in columns]
    last_field = field_indexes[-1]
    width = len(field_indexes)
    with open(des_out_path, "rb") as handle:
        line_iter = iter(handle)
        for line in line_iter:
            if b"#start" not in line:
                continue
            atom_num = int(line.split()[1])
            block = np.empty((atom_num, width), dtype=np.float64)
            atoms = []
            for row in range(atom_num):
                fields = next(line_iter).split(b"\t")
                if len(fields) <= last_field:
                    continue
                block[row] = [float(fields[index]) for index in field_indexes]
                if not atoms and current_guard() is not None:
                    require_memory(atom_num * (width * 8 + 128))
                atoms.append((int(block[row, 0]), block[row, 1:]))
            yield structure_index, atoms
            structure_index += 1


def des_out2pkl(des_out_path, prefix, num_ele, mtp_type, hyx_mlp_path, body_name_list,out_path,
                column_subset=False):
    two_body_list = [[] for _ in range(num_ele)]
    three_body_list = [[] for _ in range(num_ele)]
    four_body_list = [[] for _ in range(num_ele)]
    if column_subset:
        columns, positions = descriptor_column_layout(mtp_type, hyx_mlp_path)
        two_body = np.asarray(positions['two'], dtype=np.int64)
        three_body = np.asarray(positions['three'], dtype=np.int64)
        four_body = np.asarray(positions['four'], dtype=np.int64)
    else:
        columns = None
        two_body, three_body, four_body = extract_mtp_many_body_index(mtp_type, hyx_mlp_path)
        two_body = np.asarray(two_body, dtype=np.int64)
        three_body = np.asarray(three_body, dtype=np.int64)
        four_body = np.asarray(four_body, dtype=np.int64)

    # Shards hold disjoint, consecutive frame ranges, so the frame index keeps
    # counting across them exactly as it would over a single merged file.
    if isinstance(des_out_path, (str, os.PathLike)):
        descriptor_paths = [des_out_path]
    else:
        descriptor_paths = list(des_out_path)
    frame_offset = 0
    for descriptor_path in descriptor_paths:
        local_frames = 0
        for structure_index, atoms in iter_descriptor_structures(descriptor_path, columns):
            local_frames = structure_index + 1
            global_index = frame_offset + structure_index
            for atom_type, descriptors in atoms:
                if atom_type < 0 or atom_type >= num_ele:
                    continue
                if 'two' in body_name_list:
                    two_body_list[atom_type].append(descriptors[two_body].tolist() + [global_index])
                if 'three' in body_name_list:
                    three_body_list[atom_type].append(descriptors[three_body].tolist() + [global_index])
                if 'four' in body_name_list:
                    four_body_list[atom_type].append(descriptors[four_body].tolist() + [global_index])
        frame_offset += local_frames

    body_list = [two_body_list,three_body_list,four_body_list]
    body_name = [prefix+'_two_body_',prefix+'_three_body_',prefix+'_four_body_']
    dic = {'two':0,'three':1,'four':2}
    body_index = [dic[a] for a in body_name_list]

    selected_payloads = [
        (body, os.path.join(out_path, name + "coding_zlib.pkl"))
        for index, (body, name) in enumerate(zip(body_list, body_name))
        if index in body_index
    ]
    for body, filename in selected_payloads:
        save_compressed_pickle(body, filename)

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


