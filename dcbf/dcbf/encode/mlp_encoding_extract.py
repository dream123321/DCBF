import os
import pickle
import tempfile
import zlib

import numpy as np

from .model_metadata import read_sus2_descriptor_metadata, validate_descriptor_output_width


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
def mtp_many_body_list(*_args, **_kwargs):
    raise ValueError(
        "Legacy MTP fixed-column descriptor parsing was removed; use a SUS2 model"
    )

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

def extract_mtp_many_body_index(_model_kind, model_path):
    if not str(model_path).lower().endswith(".sus2"):
        raise ValueError(
            "SUS2-only descriptor parsing requires a .sus2 model"
        )
    metadata = read_sus2_descriptor_metadata(model_path)
    return [list(metadata.body_columns[name]) for name in ("two", "three", "four")]


def compact_column_layout(two_body, three_body, four_body):
    """Columns worth parsing out of a descriptor row, and where each body sits.

    ``columns`` are positions in the full
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
    last_field = max(field_indexes)
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
                fields = next(line_iter).split()
                if len(fields) <= last_field:
                    raise ValueError(f"Short descriptor row in {des_out_path}")
                block[row] = [float(fields[index]) for index in field_indexes]
                if not atoms and current_guard() is not None:
                    require_memory(atom_num * (width * 8 + 128))
                atoms.append((int(block[row, 0]), block[row, 1:]))
            yield structure_index, atoms
            structure_index += 1


def iter_descriptor_blocks(
    des_out_path,
    columns,
    expected_width=None,
    species_count=None,
):
    """Yield one descriptor frame as compact NumPy blocks.

    This is the hot-path counterpart of ``iter_descriptor_structures``.  It
    keeps the same field selection and row order, but avoids creating one
    Python tuple for every atom before the descriptor store consumes it.
    Invalid rows fail rather than changing the frame/atom alignment.
    """
    field_indexes = [0] + [int(column) + 1 for column in columns]
    last_field = max(field_indexes)
    width = len(field_indexes)
    structure_index = 0
    with open(des_out_path, "rb") as handle:
        line_iter = iter(handle)
        for line in line_iter:
            if b"#start" not in line:
                continue
            atom_num = int(line.split()[1])
            atom_types = np.empty(atom_num, dtype=np.int64)
            block = np.empty((atom_num, width - 1), dtype=np.float64)
            valid_rows = 0
            for _ in range(atom_num):
                fields = next(line_iter).split()
                if len(fields) <= last_field:
                    raise ValueError(f"Short descriptor row in {des_out_path}")
                if expected_width is not None and len(fields) - 1 != int(expected_width):
                    raise ValueError(
                        f"Descriptor column count mismatch for {des_out_path}: "
                        f"output={len(fields) - 1}, model_metadata={int(expected_width)}"
                    )
                atom_type = int(fields[0])
                if atom_type < 0 or (
                    species_count is not None and atom_type >= int(species_count)
                ):
                    raise ValueError(
                        f"Descriptor atom type {atom_type} is outside the model mapping"
                    )
                atom_types[valid_rows] = atom_type
                block[valid_rows] = [float(fields[index]) for index in field_indexes[1:]]
                valid_rows += 1
            yield (
                structure_index,
                atom_types[:valid_rows],
                block[:valid_rows],
            )
            structure_index += 1


def des_out2pkl(des_out_path, prefix, num_ele, mtp_type, hyx_mlp_path, body_name_list,out_path,
                column_subset=False):
    metadata = read_sus2_descriptor_metadata(hyx_mlp_path)
    if not body_name_list:
        raise ValueError("body_list must not be empty")
    for body in body_name_list:
        if body not in metadata.body_columns or not metadata.body_columns[body]:
            raise ValueError(f"SUS2 metadata has no valid columns for requested body {body!r}")
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
    validate_descriptor_output_width(descriptor_paths, metadata.output_width, metadata.species_count)
    frame_offset = 0
    for descriptor_path in descriptor_paths:
        local_frames = 0
        for structure_index, atoms in iter_descriptor_structures(descriptor_path, columns):
            local_frames = structure_index + 1
            global_index = frame_offset + structure_index
            for atom_type, descriptors in atoms:
                if atom_type < 0 or atom_type >= num_ele:
                    raise ValueError(f"Descriptor atom type {atom_type} is outside the model element mapping")
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



