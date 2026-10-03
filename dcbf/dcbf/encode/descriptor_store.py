"""Typed, frame-ordered descriptor storage used by the sampling pipeline."""
from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import time
import uuid
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from ..memory_guard import MIB, atomic_json, require_memory, stage_progress, work_memory

STORE_VERSION = 1
BLOCK_BYTES = 64 * MIB


def _parse_worker_cpu_limit():
    """Return a conservative CPU limit for Reduce parser workers."""
    if hasattr(os, "sched_getaffinity"):
        affinity = len(os.sched_getaffinity(0))
    else:
        affinity = os.cpu_count() or 1
    values = [max(1, int(affinity))]
    for key in ("SLURM_CPUS_ON_NODE", "LSB_DJOB_NUMPROC"):
        raw = os.environ.get(key)
        if raw:
            try:
                values.append(max(1, int(str(raw).split("(", 1)[0])))
            except ValueError:
                pass
    return min(values)


def _flush_parse_worker_buffers(buffers, output_dir, files):
    for name, arrays in buffers.items():
        if not arrays:
            continue
        path = Path(output_dir) / name
        with path.open("ab") as handle:
            for array in arrays:
                handle.write(memoryview(array).cast("B"))
                files[name] = files.get(name, 0) + int(array.nbytes)
    buffers.clear()


def _parse_descriptor_shard_worker(payload):
    """Parse one .out shard into compact binary files.

    The worker returns metadata only.  Descriptor arrays stay on disk, so the
    parent process never receives a full shard through multiprocessing IPC.
    """
    from .mlp_encoding_extract import iter_descriptor_blocks

    started = time.perf_counter()
    output_dir = Path(payload["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    body_names = tuple(payload["body_names"])
    positions = {
        name: np.asarray(values, dtype=np.int64)
        for name, values in payload["positions"].items()
    }
    element_count = int(payload["element_count"])
    block_bytes = max(4096, int(payload.get("block_bytes", BLOCK_BYTES)))
    buffers = {}
    files = {}
    counts = [0] * element_count
    buffered = 0
    frames = 0

    def append(name, array):
        nonlocal buffered
        buffers.setdefault(name, []).append(array)
        buffered += int(array.nbytes)

    for local_index, atom_types, descriptors in iter_descriptor_blocks(
        payload["descriptor_path"], payload["columns"]
    ):
        frames = local_index + 1
        global_index = int(payload["index_offset"]) + int(local_index)
        if len(atom_types) == 0:
            continue
        for element in np.unique(atom_types):
            element = int(element)
            if element < 0 or element >= element_count:
                continue
            selected = atom_types == element
            raw = descriptors[selected]
            row_count = len(raw)
            if row_count == 0:
                continue
            counts[element] += row_count
            append(
                f"indices_{element}.bin",
                np.full(row_count, global_index, dtype=np.int64),
            )
            for body in body_names:
                append(
                    f"{body}_{element}.bin",
                    np.ascontiguousarray(raw[:, positions[body]]),
                )
            if buffered >= block_bytes:
                _flush_parse_worker_buffers(buffers, output_dir, files)
                buffered = 0
    _flush_parse_worker_buffers(buffers, output_dir, files)
    for element in range(element_count):
        for name in [f"indices_{element}.bin"] + [f"{body}_{element}.bin" for body in body_names]:
            if name not in files:
                (output_dir / name).touch()
                files[name] = 0
    return {
        "counts": counts,
        "frames": frames,
        "files": files,
        "output_dir": str(output_dir),
        "elapsed_seconds": time.perf_counter() - started,
    }


def _parse_descriptor_shards_parallel(descriptor_inputs, temporary, columns, positions,
                                      body_names, element_count, index_offset,
                                      parse_workers, block_bytes):
    if parse_workers is None or len(descriptor_inputs) < 2:
        return None
    # Process startup and spawn imports dominate small reductions.  Keep those
    # on the compact serial path; parallel parsing is intended for real shard
    # workloads, not a handful of tiny test files.
    total_bytes = sum(path.stat().st_size for path in descriptor_inputs)
    if total_bytes < 128 * MIB:
        return None
    requested = max(1, int(parse_workers))
    available = _parse_worker_cpu_limit()
    effective = min(len(descriptor_inputs), requested, available)
    if effective <= 1:
        return None
    # Each worker holds one current frame plus its bounded write buffers.  Keep
    # a simple memory cap so a large encoding_cores value cannot oversubscribe
    # a small login-node allocation.
    effective = min(effective, max(1, work_memory() // (128 * MIB)))
    if effective <= 1:
        return None
    shard_root = temporary / ".parsed_shards"
    shard_root.mkdir(parents=True, exist_ok=True)
    payloads = []
    for shard_index, descriptor_path in enumerate(descriptor_inputs):
        payloads.append(
            {
                "descriptor_path": str(descriptor_path),
                "output_dir": str(shard_root / f"shard_{shard_index:04d}"),
                "columns": [int(value) for value in columns],
                "positions": {name: [int(value) for value in values] for name, values in positions.items()},
                "body_names": list(body_names),
                "element_count": int(element_count),
                # Workers use local frame numbers.  The parent applies the
                # cumulative offset while merging, so no pre-scan is needed.
                "index_offset": 0,
                "block_bytes": int(block_bytes),
            }
        )
    started = time.perf_counter()
    context = multiprocessing.get_context("spawn")
    results = [None] * len(payloads)
    try:
        with ProcessPoolExecutor(max_workers=effective, mp_context=context) as executor:
            futures = [executor.submit(_parse_descriptor_shard_worker, payload) for payload in payloads]
            for index, future in enumerate(futures):
                results[index] = future.result()
    except Exception:
        shutil.rmtree(shard_root, ignore_errors=True)
        raise
    return {
        "shards": results,
        "index_offset": int(index_offset),
        "requested_workers": requested,
        "effective_workers": effective,
        "elapsed_seconds": time.perf_counter() - started,
        "backend": "spawn_parallel",
    }


def _merge_parsed_descriptor_shards(parsed, temporary, body_names, element_count):
    counts = [0] * int(element_count)
    files = {}
    frames = 0
    frame_offset = int(parsed.get("index_offset", 0))
    for shard in parsed["shards"]:
        frames += int(shard["frames"])
        counts = [left + right for left, right in zip(counts, shard["counts"])]
        source_dir = Path(shard["output_dir"])
        for name in [f"indices_{element}.bin" for element in range(element_count)] + [
            f"{body}_{element}.bin" for body in body_names for element in range(element_count)
        ]:
            source = source_dir / name
            if not source.exists() or source.stat().st_size == 0:
                continue
            target = temporary / name
            with source.open("rb") as source_handle, target.open("ab") as target_handle:
                if name.startswith("indices_"):
                    while True:
                        raw = source_handle.read(8 * MIB)
                        if not raw:
                            break
                        values = np.frombuffer(raw, dtype=np.int64).copy()
                        values += frame_offset
                        target_handle.write(memoryview(values).cast("B"))
                else:
                    shutil.copyfileobj(source_handle, target_handle, length=8 * MIB)
            files[name] = files.get(name, 0) + source.stat().st_size
        frame_offset += int(shard["frames"])
    return counts, frames, files


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
                           source_fingerprint=None, index_offset=0, parse_workers=None):
    from .mlp_encoding_extract import (
        compact_column_layout,
        extract_mtp_many_body_index,
        iter_descriptor_blocks,
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
    if mean_enabled:
        # Mean descriptors intentionally keep the historical two+three+four
        # layout and reduction order.
        columns, positions = compact_column_layout(mapping['two'], mapping['three'], mapping['four'])
    else:
        requested_mapping = {
            name: mapping[name] if name in body_names else []
            for name in ('two', 'three', 'four')
        }
        columns, positions = compact_column_layout(
            requested_mapping['two'], requested_mapping['three'], requested_mapping['four']
        )
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
    parse_meta = {
        "backend": "serial_blocks",
        "requested_workers": int(parse_workers) if parse_workers is not None else 1,
        "effective_workers": 1,
        "shard_count": len(descriptor_inputs),
        "elapsed_seconds": 0.0,
    }
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
        parallel_started = time.perf_counter()
        parallel = None
        if not mean_enabled:
            parallel = _parse_descriptor_shards_parallel(
                descriptor_inputs,
                temporary,
                columns,
                positions,
                body_names,
                len(elements),
                int(index_offset),
                parse_workers,
                block_bytes,
            )
        if parallel is not None:
            counts, frames, files = _merge_parsed_descriptor_shards(
                parallel, temporary, body_names, len(elements)
            )
            parse_meta = {
                "backend": parallel["backend"],
                "requested_workers": parallel["requested_workers"],
                "effective_workers": parallel["effective_workers"],
                "shard_count": len(descriptor_inputs),
                "elapsed_seconds": time.perf_counter() - parallel_started,
                "worker_seconds": sum(
                    float(item["elapsed_seconds"]) for item in parallel["shards"]
                ),
            }
            shutil.rmtree(temporary / ".parsed_shards", ignore_errors=True)
        else:
            frame_base = int(index_offset)
            serial_started = time.perf_counter()
            for descriptor_input in descriptor_inputs:
                frame_offset = frames
                local_frames = 0
                if mean_enabled:
                    iterator = iter_descriptor_structures(descriptor_input, columns)
                    for frame_index, atoms in iterator:
                        global_frame_index = frame_base + frame_offset + frame_index
                        local_frames = frame_index + 1
                        if not atoms:
                            continue
                        # Preserve the old per-frame mean operation, including
                        # all mapped bodies and its reduction order.
                        selected = np.vstack([descriptor[mean_columns] for _, descriptor in atoms])
                        means.append(np.mean(selected, axis=0).tolist() + [global_frame_index])
                        by_element = {}
                        for element, descriptor in atoms:
                            if 0 <= element < len(elements):
                                by_element.setdefault(element, []).append(descriptor)
                        for element, rows in by_element.items():
                            raw = np.vstack(rows)
                            counts[element] += len(rows)
                            append(f'indices_{element}.bin', np.full(len(rows), global_frame_index, dtype=np.int64))
                            for name in body_names:
                                append(f'{name}_{element}.bin', np.ascontiguousarray(raw[:, positions[name]]))
                        if buffered >= block_bytes:
                            stage_progress('descriptor_conversion', global_frame_index + 1, descriptor_input)
                            flush()
                else:
                    # The block iterator keeps descriptor rows in their
                    # original frame/atom order while allowing NumPy to select
                    # each element in one mask operation.
                    for frame_index, atom_types, descriptors in iter_descriptor_blocks(descriptor_input, columns):
                        global_frame_index = frame_base + frame_offset + frame_index
                        local_frames = frame_index + 1
                        if len(atom_types) == 0:
                            continue
                        for element in np.unique(atom_types):
                            element = int(element)
                            if element < 0 or element >= len(elements):
                                continue
                            mask = atom_types == element
                            raw = descriptors[mask]
                            if len(raw) == 0:
                                continue
                            counts[element] += len(raw)
                            append(f'indices_{element}.bin', np.full(len(raw), global_frame_index, dtype=np.int64))
                            for name in body_names:
                                append(f'{name}_{element}.bin', np.ascontiguousarray(raw[:, positions[name]]))
                        if buffered >= block_bytes:
                            stage_progress('descriptor_conversion', global_frame_index + 1, descriptor_input)
                            flush()
                frames += local_frames
            flush()
            parse_meta["elapsed_seconds"] = time.perf_counter() - serial_started
        if [file_fingerprint(path) for path in descriptor_inputs] != descriptor_fingerprints:
            raise RuntimeError('Descriptor input changed during cache construction')
        for element in range(len(elements)):
            for name in [f'indices_{element}.bin'] + [f'{b}_{element}.bin' for b in body_names]:
                if name not in files:
                    (temporary / name).touch()
                    files[name] = 0
        manifest = dict(version=STORE_VERSION, complete=True, signature=signature,
                        element_counts=counts, frame_count=frames, body_columns=signature['body_columns'],
                        descriptor_dtype='float64', index_dtype='int64', files=files,
                        parse=parse_meta)
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
