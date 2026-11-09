from __future__ import annotations

from collections import Counter
import heapq

import numpy as np

try:
    from ._min_cover_exact import exact_min_cover_ints as _exact_min_cover_ints
except ImportError:
    _exact_min_cover_ints = None


def locate_intervals(values, intervals):
    values = np.asarray(values, dtype=float)
    if values.size == 0 or not intervals:
        return np.full(values.shape, -1, dtype=np.int64)

    interval_array = np.asarray(intervals, dtype=float)
    starts = interval_array[:, 0]
    ends = interval_array[:, 1]

    positions = np.searchsorted(starts, values, side="right") - 1
    valid = positions >= 0
    if not np.any(valid):
        return np.full(values.shape, -1, dtype=np.int64)

    safe_positions = positions.copy()
    safe_positions[~valid] = 0
    right_edge = (safe_positions == len(ends) - 1) & (values == ends[safe_positions])
    valid &= (values >= starts[safe_positions]) & ((values < ends[safe_positions]) | right_edge)

    result = np.full(values.shape, -1, dtype=np.int64)
    result[valid] = safe_positions[valid]
    return result


def group_structure_indices_by_interval(stru_indexs, data_list, intervals):
    groups = [[] for _ in intervals]
    if not intervals or len(stru_indexs) == 0:
        return groups

    indices = np.asarray(stru_indexs, dtype=np.int64)
    values = np.asarray(data_list, dtype=float)
    bucket_ids = locate_intervals(values, intervals)
    valid_mask = bucket_ids >= 0
    if not np.any(valid_mask):
        return groups

    filtered_buckets = bucket_ids[valid_mask]
    filtered_indices = indices[valid_mask]
    order = np.argsort(filtered_buckets, kind="mergesort")
    filtered_buckets = filtered_buckets[order]
    filtered_indices = filtered_indices[order]

    split_points = np.flatnonzero(np.diff(filtered_buckets)) + 1
    bucket_groups = np.split(filtered_buckets, split_points)
    index_groups = np.split(filtered_indices, split_points)
    for bucket_group, index_group in zip(bucket_groups, index_groups):
        groups[int(bucket_group[0])] = index_group.tolist()
    return groups


def label_covered_values(values, zero_freq_intervals, max_min):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return np.zeros(0, dtype=bool)

    lower_bound = max_min[1]
    upper_bound = max_min[0]
    intervals = [[-np.inf, lower_bound], *zero_freq_intervals, [upper_bound, np.inf]]
    bucket_ids = locate_intervals(values, intervals)
    covered = bucket_ids == -1
    covered |= values == lower_bound
    covered |= values == upper_bound
    return covered


def frequency_counter(lists):
    return Counter(element for subset in lists for element in subset)


def _ordered_elements_by_legacy_frequency(lists):
    counter = Counter()
    first_seen = {}
    for index, subset in enumerate(lists):
        for element in subset:
            counter[element] += 1
            first_seen.setdefault(element, (index, len(first_seen)))
    return sorted(counter, key=lambda element: (-counter[element], first_seen[element][1]))


def _can_use_compiled_backend(lists):
    if _exact_min_cover_ints is None:
        return False
    for subset in lists:
        for element in subset:
            if not isinstance(element, int) or isinstance(element, bool):
                return False
    return True


def _greedy_cover_elements_python(lists):
    sets = [set(items) for items in lists]
    if not sets:
        return []

    ordered_elements = _ordered_elements_by_legacy_frequency(lists)
    if not ordered_elements:
        return []

    rank_by_element = {element: rank for rank, element in enumerate(ordered_elements)}
    set_to_ranks = [[] for _ in sets]
    element_to_sets = [[] for _ in ordered_elements]

    for set_index, item_set in enumerate(sets):
        for element in item_set:
            rank = rank_by_element[element]
            set_to_ranks[set_index].append(rank)
            element_to_sets[rank].append(set_index)

    current_count = [len(set_ids) for set_ids in element_to_sets]
    active = [count > 0 for count in current_count]
    remaining = [True] * len(sets)
    remaining_count = len(sets)

    buckets = [[] for _ in range(len(sets) + 1)]
    max_count = 0
    for rank, count in enumerate(current_count):
        if count > 0:
            heapq.heappush(buckets[count], rank)
            if count > max_count:
                max_count = count

    cover = []
    while remaining_count > 0 and max_count > 0:
        while max_count > 0:
            bucket = buckets[max_count]
            while bucket:
                rank = bucket[0]
                if not active[rank] or current_count[rank] != max_count:
                    heapq.heappop(bucket)
                    continue
                break
            if bucket:
                break
            max_count -= 1

        if max_count == 0:
            break

        best_rank = heapq.heappop(buckets[max_count])
        if not active[best_rank] or current_count[best_rank] != max_count:
            continue

        cover.append(ordered_elements[best_rank])
        for set_index in element_to_sets[best_rank]:
            if not remaining[set_index]:
                continue
            remaining[set_index] = False
            remaining_count -= 1
            for rank in set_to_ranks[set_index]:
                if not active[rank]:
                    continue
                new_count = current_count[rank] - 1
                current_count[rank] = new_count
                if new_count == 0:
                    active[rank] = False
                else:
                    heapq.heappush(buckets[new_count], rank)

    legacy_cover = set()
    for element in cover:
        legacy_cover.add(element)
    return list(legacy_cover)


def greedy_cover_elements(lists):
    if _can_use_compiled_backend(lists):
        return _exact_min_cover_ints(lists)
    return _greedy_cover_elements_python(lists)
