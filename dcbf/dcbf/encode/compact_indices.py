"""Frame-index buckets with legacy iteration order and Python integer values."""
from __future__ import annotations
import os
from pathlib import Path
import uuid
import numpy as np
from .selection_core import locate_intervals
from ..memory_guard import MIB, current_guard, require_memory, work_memory


class IndexBucket:
    def __init__(self, values):
        self.values = values

    def __len__(self):
        return len(self.values)

    def __iter__(self):
        return map(int, self.values)

    def __getitem__(self, index):
        result = self.values[index]
        return IndexBucket(result) if isinstance(index, slice) else int(result)

    def __reduce__(self):
        values = self.values
        if isinstance(values, np.memmap):
            base = values
            while isinstance(base.base, np.memmap):
                base = base.base
            offset = int(values.ctypes.data - base.ctypes.data) + int(base.offset)
            return _mapped_bucket, (str(base.filename), offset, len(values))
        return IndexBucket, (values,)


def count_selected_indices(bucket, selected, chunk_bytes=8 * MIB):
    """Count selected frame indices without expanding a bucket to Python ints."""
    selected = np.asarray(selected, dtype=np.int64)
    if selected.ndim != 1:
        selected = selected.reshape(-1)
    if len(selected) == 0 or len(bucket) == 0:
        return np.zeros(len(selected), dtype=np.int64)
    if len(selected) > 1 and np.any(selected[1:] <= selected[:-1]):
        raise ValueError("selected indices must be strictly increasing")

    values = bucket.values if isinstance(bucket, IndexBucket) else bucket
    chunk_size = max(1, int(chunk_bytes) // np.dtype(np.int64).itemsize)
    counts = np.zeros(len(selected), dtype=np.int64)
    lower = int(selected[0])
    upper = int(selected[-1])
    span = upper - lower + 1

    # For ordinary frame indices the selected span is small. Bincount performs
    # the complete multiplicity calculation in C and preserves multi-cover
    # semantics. Limit the dense temporary so sparse, very large IDs remain safe.
    dense_limit = max(1, int(chunk_bytes) // np.dtype(np.intp).itemsize)
    selected_offsets = selected - lower if span <= dense_limit else None

    for start in range(0, len(values), chunk_size):
        chunk = np.asarray(values[start:start + chunk_size], dtype=np.int64).reshape(-1)
        if len(chunk) == 0:
            continue
        if selected_offsets is not None:
            in_range = (chunk >= lower) & (chunk <= upper)
            if np.any(in_range):
                dense_counts = np.bincount(chunk[in_range] - lower, minlength=span)
                counts += dense_counts[selected_offsets]
            continue

        positions = np.searchsorted(selected, chunk)
        valid = positions < len(selected)
        if not np.any(valid):
            continue
        positions = positions[valid]
        matched = selected[positions] == chunk[valid]
        if np.any(matched):
            counts += np.bincount(positions[matched], minlength=len(selected))
    return counts


def find_multi_cover_set_compact(classes, targets):
    """Exact greedy multi-cover using compact integer adjacency arrays."""
    if not classes:
        return []
    if len(classes) != len(targets):
        raise ValueError("Internal error: classes and targets length mismatch")
    remaining = np.asarray([max(0, int(value)) for value in targets], dtype=np.int64)
    require_memory(sum(len(bucket) for bucket in classes) * 40 + len(classes) * 128)

    members = []
    structure_parts = []
    class_parts = []
    count_parts = []
    for class_index, bucket in enumerate(classes):
        raw = bucket.values if isinstance(bucket, IndexBucket) else bucket
        values = np.asarray(raw, dtype=np.int64).reshape(-1)
        if len(values):
            structure_indices, counts = np.unique(values, return_counts=True)
            counts = counts.astype(np.int64, copy=False)
        else:
            structure_indices = np.empty(0, dtype=np.int64)
            counts = np.empty(0, dtype=np.int64)
        members.append((structure_indices, counts))
        if len(structure_indices):
            structure_parts.append(structure_indices)
            class_parts.append(np.full(len(structure_indices), class_index, dtype=np.int64))
            count_parts.append(counts)

    if not structure_parts:
        return []
    all_structures = np.concatenate(structure_parts)
    all_classes = np.concatenate(class_parts)
    all_counts = np.concatenate(count_parts)
    order = np.argsort(all_structures, kind="mergesort")
    all_structures = all_structures[order]
    all_classes = all_classes[order]
    all_counts = all_counts[order]
    structure_count = int(all_structures[-1]) + 1
    adjacency_counts = np.bincount(all_structures, minlength=structure_count)
    adjacency_offsets = np.empty(structure_count + 1, dtype=np.int64)
    adjacency_offsets[0] = 0
    np.cumsum(adjacency_counts, out=adjacency_offsets[1:])

    scores = np.zeros(structure_count, dtype=np.int64)
    for class_index, (structure_indices, counts) in enumerate(members):
        if remaining[class_index] > 0 and len(structure_indices):
            scores[structure_indices] += np.minimum(counts, remaining[class_index])

    selected = []
    while np.any(remaining > 0):
        best_index = int(np.argmax(scores))
        if scores[best_index] <= 0:
            break
        selected.append(best_index)
        begin = int(adjacency_offsets[best_index])
        end = int(adjacency_offsets[best_index + 1])
        for position in range(begin, end):
            class_index = int(all_classes[position])
            old_remaining = int(remaining[class_index])
            if old_remaining <= 0:
                continue
            new_remaining = max(0, old_remaining - int(all_counts[position]))
            if new_remaining == old_remaining:
                continue
            structure_indices, counts = members[class_index]
            scores[structure_indices] -= (
                np.minimum(counts, old_remaining) - np.minimum(counts, new_remaining)
            )
            remaining[class_index] = new_remaining
        scores[best_index] = -1
    return selected


def _mapped_bucket(path, offset, count):
    return IndexBucket(np.memmap(path, mode='r', dtype=np.int64, offset=offset, shape=(count,)))


def group_compact_indices(structure_indices, values, intervals):
    groups = [[] for _ in intervals]
    if not intervals or len(structure_indices) == 0:
        return groups
    require_memory(len(values) * 64)
    indices = np.asarray(structure_indices, dtype=np.int64)
    bucket_ids = locate_intervals(values, intervals)
    valid = bucket_ids >= 0
    if not np.any(valid):
        return groups
    filtered_buckets = bucket_ids[valid]
    filtered_indices = indices[valid]
    order = np.argsort(filtered_buckets, kind='mergesort')
    filtered_buckets = filtered_buckets[order]
    filtered_indices = filtered_indices[order]
    breaks = np.flatnonzero(np.diff(filtered_buckets)) + 1
    guard = current_guard()
    if guard is not None and filtered_indices.nbytes > min(64 * MIB, work_memory() // 16):
        directory = guard.workspace / '.descriptor_tasks'
        directory.mkdir(exist_ok=True)
        path = directory / (uuid.uuid4().hex + '.bin')
        filtered_indices.tofile(path)
        filtered_indices = np.memmap(path, dtype=np.int64, mode='r', shape=filtered_indices.shape)
    starts = np.r_[0, breaks]
    ends = np.r_[breaks, len(filtered_indices)]
    for begin, end in zip(starts, ends):
        groups[int(filtered_buckets[begin])] = IndexBucket(filtered_indices[begin:end])
    return groups


def group_descriptor_rows_by_interval(parts, dimension, intervals):
    """Group disjoint descriptor-row parts without concatenating float columns.

    Each part is already ordered by global frame index.  Merging the integer
    buckets in part order therefore preserves the historical stable order.
    """
    merged = [[] for _ in intervals]
    for values, indices in parts:
        values = np.asarray(values)
        local = group_compact_indices(indices, values[:, int(dimension)], intervals)
        for bucket_index, bucket in enumerate(local):
            if bucket:
                merged[bucket_index].append(bucket)

    output = []
    for buckets in merged:
        if not buckets:
            output.append([])
            continue
        if len(buckets) == 1:
            output.append(buckets[0])
            continue
        arrays = [
            np.asarray(bucket.values if isinstance(bucket, IndexBucket) else bucket, dtype=np.int64)
            for bucket in buckets
        ]
        output.append(IndexBucket(np.concatenate(arrays)))
    return output
