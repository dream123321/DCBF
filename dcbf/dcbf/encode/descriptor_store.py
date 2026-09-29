"""Typed, frame-ordered descriptor storage used by the sampling pipeline."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid

import numpy as np

from ..memory_guard import MIB, atomic_json, require_memory, stage_progress, work_memory

STORE_VERSION = 1
BLOCK_BYTES = 64 * MIB


class DescriptorRows:
    """Read-only views, including disjoint frame ranges, without matrix copies."""
    def __init__(self, parts, dimensions):
        self.parts = tuple((values, indices) for values, indices in parts if len(indices))
        self.dimensions = int(dimensions)
        self.shape = (sum(len(indices) for _, indices in self.parts), self.dimensions)

    def __len__(self):
        return self.shape[0]

    def column(self, dimension):
        if not self.parts:
            return np.empty(0, dtype=np.float64)
        if len(self.parts) == 1:
            # Drop the memmap subclass, not its storage: Python min/max would
            # otherwise pay memmap.__getitem__ overhead for every scalar.
            return np.asarray(self.parts[0][0])[:, dimension]
        require_memory(len(self) * 8)
        return np.concatenate([values[:, dimension] for values, _ in self.parts])

    def indices(self):
        if not self.parts:
            return np.empty(0, dtype=np.int64)
        if len(self.parts) == 1:
            return self.parts[0][1]
        require_memory(len(self) * 8)
        return np.concatenate([indices for _, indices in self.parts])

    def select_frames(self, structure_indices):
        wanted = np.asarray(sorted(set(int(i) for i in structure_indices)), dtype=np.int64)
        if not len(wanted):
            return DescriptorRows([], self.dimensions)
        breaks = np.flatnonzero(np.diff(wanted) != 1) + 1
        runs = np.split(wanted, breaks)
        parts = []
        for values, indices in self.parts:
            for run in runs:
                begin = int(np.searchsorted(indices, run[0], side='left'))
                end = int(np.searchsorted(indices, run[-1], side='right'))
                if end > begin:
                    parts.append((values[begin:end], indices[begin:end]))
        return DescriptorRows(parts, self.dimensions)

    def select_frame_range(self, start, stop):
        start = int(start)
        stop = int(stop)
        if stop <= start:
            return DescriptorRows([], self.dimensions)
        parts = []
        for values, indices in self.parts:
            begin = int(np.searchsorted(indices, start, side='left'))
            end = int(np.searchsorted(indices, stop, side='left'))
            if end > begin:
                parts.append((values[begin:end], indices[begin:end]))
        return DescriptorRows(parts, self.dimensions)

    def __iter__(self):
        # Compatibility for inspection only; hot paths use column()/indices().
        for values, indices in self.parts:
            for row, index in zip(values, indices):
                yield row.tolist() + [int(index)]


def values_and_indices(rows):
    if isinstance(rows, DescriptorRows):
        return rows, rows.indices()
    matrix = np.asarray(rows, dtype=np.float64)
    if not len(matrix):
        return np.empty((0, 0)), np.empty(0, dtype=np.int64)
    return matrix[:, :-1], matrix[:, -1]


def concatenate_rows(rows):
    rows = [item for item in rows if item is not None]
    if not rows:
        return DescriptorRows([], 0)
    dimensions = rows[0].dimensions
    if any(item.dimensions != dimensions for item in rows):
        raise ValueError('Cannot concatenate descriptor rows with different dimensions')
    return DescriptorRows(
        [part for item in rows for part in item.parts],
        dimensions,
    )


def column(data, index):
    return data.column(index) if isinstance(data, DescriptorRows) else data[:, index]


def numeric_data(data):
    return data if isinstance(data, DescriptorRows) else np.asarray(data)


def file_fingerprint(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {'path': str(path), 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


class DescriptorStore:
    def __init__(self, path, ram_limit=None):
        self.path = Path(path)
        self.manifest = json.loads((self.path / 'manifest.json').read_text())
        if self.manifest.get('version') != STORE_VERSION or not self.manifest.get('complete'):
            raise RuntimeError(f'Incomplete or incompatible descriptor cache: {self.path}')
        self.ram_limit = min(256 * MIB, work_memory() // 8) if ram_limit is None else max(0, int(ram_limit))
        self.ram_used = 0
        self._indices = {}
        self._bodies = {}
        self.cache_reused = False
        for name, size in self.manifest['files'].items():
            if (self.path / name).stat().st_size != size:
                raise RuntimeError(f'Truncated descriptor cache file: {self.path / name}')

    def _array(self, name, dtype, shape):
        size = int(np.prod(shape)) * np.dtype(dtype).itemsize
        if not size:
            return np.empty(shape, dtype=dtype)
        mapped = np.memmap(self.path / name, dtype=dtype, mode='r', shape=shape)
        if size + self.ram_used <= self.ram_limit:
            require_memory(size)
            result = np.array(mapped)
            self.ram_used += size
            return result
        return mapped

    def body(self, name):
        if name in self._bodies:
            return self._bodies[name]
        dimensions = len(self.manifest['body_columns'][name])
        result = []
        for element, count in enumerate(self.manifest['element_counts']):
            if element not in self._indices:
                self._indices[element] = self._array(f'indices_{element}.bin', np.int64, (count,))
            values = self._array(f'{name}_{element}.bin', np.float64, (count, dimensions))
            result.append(DescriptorRows([(values, self._indices[element])], dimensions))
        self._bodies[name] = result
        return result


def _normalize_descriptor_inputs(des_out_path):
    if isinstance(des_out_path, (str, os.PathLike)):
        return [Path(des_out_path)]
    return [Path(path) for path in des_out_path]


def descriptor_store_signature(des_out_path, elements, mtp_type, model, bodies,
                               mean_enabled=False, source_fingerprint=None):
    from .mlp_encoding_extract import extract_mtp_many_body_index
    descriptor_inputs = _normalize_descriptor_inputs(des_out_path)
    body_names = list(dict.fromkeys(bodies))
    mapping = dict(zip(('two', 'three', 'four'), extract_mtp_many_body_index(mtp_type, model)))
    for name in body_names:
        if name not in mapping:
            raise ValueError(f'Unknown descriptor body {name!r}')
    input_signature = (
        {'source': source_fingerprint}
        if source_fingerprint is not None
        else {'descriptor_outputs': [file_fingerprint(path) for path in descriptor_inputs]}
    )
    return {
        'input': input_signature,
        'model_sha256': hashlib.sha256(Path(model).read_bytes()).hexdigest(),
        'elements': list(elements),
        'mtp_type': mtp_type,
        'body_columns': {body: mapping[body] for body in body_names},
        'mean_enabled': bool(mean_enabled),
    }


def load_descriptor_store(out_path, prefix, signature, ram_limit=None):
    path = Path(out_path) / f'{prefix}_descriptor_store'
    try:
        manifest = json.loads((path / 'manifest.json').read_text())
        if manifest.get('signature') != signature:
            return None
        store = DescriptorStore(path, ram_limit=ram_limit)
        store.cache_reused = True
        return store
    except (OSError, ValueError, RuntimeError):
        return None


def build_descriptor_store(des_out_path, prefix, elements, mtp_type, model, bodies, out_path,
                           mean_enabled=False, block_bytes=BLOCK_BYTES, ram_limit=None,
                           source_fingerprint=None, index_offset=0):
    from .mlp_encoding_extract import (
        compact_column_layout,
        extract_mtp_many_body_index,
        iter_descriptor_structures,
        save_compressed_pickle,
    )
    descriptor_inputs = _normalize_descriptor_inputs(des_out_path)
    body_names = list(dict.fromkeys(bodies))
    mapping = dict(zip(('two', 'three', 'four'), extract_mtp_many_body_index(mtp_type, model)))
    for name in body_names:
        if name not in mapping:
            raise ValueError(f'Unknown descriptor body {name!r}')
    # Only the columns some body actually needs are parsed out of the .out text,
    # so every slice below must use `positions` (compact row) and never `mapping`
    # (full descriptor row).
    columns, positions = compact_column_layout(mapping['two'], mapping['three'], mapping['four'])
    descriptor_fingerprints = [file_fingerprint(path) for path in descriptor_inputs]
    signature = descriptor_store_signature(
        descriptor_inputs,
        elements,
        mtp_type,
        model,
        body_names,
        mean_enabled=mean_enabled,
        source_fingerprint=source_fingerprint,
    )
    path = Path(out_path) / f'{prefix}_descriptor_store'
    mean_path = Path(out_path) / f'{prefix}_mean_coding_zlib.pkl'
    try:
        existing = json.loads((path / 'manifest.json').read_text())
        if existing.get('signature') == signature:
            store = DescriptorStore(path, ram_limit=ram_limit)
            if not mean_enabled or mean_path.is_file():
                store.cache_reused = True
                return store
    except (OSError, ValueError, RuntimeError):
        pass
    # New data is only published after every array and the manifest are complete.
    temporary = path.with_name(path.name + '.partial-' + uuid.uuid4().hex)
    temporary.mkdir(parents=True)
    block_bytes = max(4096, min(int(block_bytes), max(4096, work_memory() // 16)))
    counts = [0] * len(elements)
    buffers = {}
    files = {}
    buffered = 0
    means = []
    frames = 0
    mean_columns = np.asarray(positions['two'] + positions['three'] + positions['four'], dtype=np.int64)
    estimated = int(sum(path.stat().st_size for path in descriptor_inputs) * 1.5) + 64 * MIB
    if shutil.disk_usage(temporary).free < estimated:
        raise OSError(28, f'Insufficient descriptor cache disk space: need at least {estimated} bytes')

    def append(name, array):
        nonlocal buffered
        buffers.setdefault(name, []).append(array)
        buffered += array.nbytes

    def flush():
        nonlocal buffered
        require_memory(buffered)
        for name, arrays in buffers.items():
            total = sum(a.nbytes for a in arrays)
            if total > shutil.disk_usage(temporary).free:
                raise OSError(28, 'Insufficient disk space for descriptor block')
            with (temporary / name).open('ab') as handle:
                for array in arrays:
                    handle.write(memoryview(array).cast('B'))
            files[name] = files.get(name, 0) + total
        buffers.clear()
        buffered = 0

    try:
        frame_base = int(index_offset)
        for descriptor_input in descriptor_inputs:
            frame_offset = frames
            local_frames = 0
            for frame_index, atoms in iter_descriptor_structures(descriptor_input, columns):
                global_frame_index = frame_base + frame_offset + frame_index
                local_frames = frame_index + 1
                if not atoms:
                    continue
                # Preserve the old per-frame mean operation, including all mapped bodies.
                if mean_enabled:
                    selected = np.vstack([descriptor[mean_columns] for _, descriptor in atoms])
                    means.append(np.mean(selected, axis=0).tolist() + [global_frame_index])
                by_element = {}
                for element, descriptor in atoms:
                    if 0 <= element < len(elements):
                        by_element.setdefault(element, []).append(descriptor)
                for element, rows in by_element.items():
                    raw = np.vstack(rows)
                    counts[element] += len(rows)
                    append(
                        f'indices_{element}.bin',
                        np.full(len(rows), global_frame_index, dtype=np.int64),
                    )
                    for name in body_names:
                        append(f'{name}_{element}.bin', np.ascontiguousarray(raw[:, positions[name]]))
                if buffered >= block_bytes:
                    stage_progress('descriptor_conversion', global_frame_index + 1, descriptor_input)
                    flush()
            frames += local_frames
        flush()
        if [file_fingerprint(path) for path in descriptor_inputs] != descriptor_fingerprints:
            raise RuntimeError('Descriptor input changed during cache construction')
        for element in range(len(elements)):
            for name in [f'indices_{element}.bin'] + [f'{b}_{element}.bin' for b in body_names]:
                if name not in files:
                    (temporary / name).touch()
                    files[name] = 0
        manifest = dict(version=STORE_VERSION, complete=True, signature=signature,
                        element_counts=counts, frame_count=frames, body_columns=signature['body_columns'],
                        descriptor_dtype='float64', index_dtype='int64', files=files)
        atomic_json(temporary / 'manifest.json', manifest, durable=False)
        if mean_enabled:
            mean_tmp = mean_path.with_name(mean_path.name + '.partial')
            save_compressed_pickle([means], mean_tmp)
            os.replace(mean_tmp, mean_path)
        if path.exists():
            # Keep earlier generations of this new-format cache for inspection.
            os.replace(path, path.with_name(path.name + '.previous-' + uuid.uuid4().hex))
        os.replace(temporary, path)
        stage_progress('descriptor_conversion_complete', frames, des_out_path)
        return DescriptorStore(path, ram_limit=ram_limit)
    except Exception:
        try:
            atomic_json(temporary / 'failed.json', {'complete': False, 'frames': frames})
        except OSError:
            pass
        raise
