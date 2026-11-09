from __future__ import annotations

import bisect
from collections import Counter, defaultdict

import numpy as np


def _is_in_interval(value, interval):
    return interval[0] <= value < interval[1]


def legacy_group_structure_indices_by_interval(stru_indexs, data_list, intervals):
    groups = [[] for _ in intervals]
    if not intervals:
        return groups

    end_value = intervals[-1][1]
    end_index = len(intervals) - 1
    starts = [interval[0] for interval in intervals]

    for stru_index, value in zip(stru_indexs, data_list):
        index = bisect.bisect_left(starts, value)
        interval_index = None
        if index < len(intervals) and _is_in_interval(value, intervals[index]):
            interval_index = index
        elif index > 0 and _is_in_interval(value, intervals[index - 1]):
            interval_index = index - 1
        elif value == end_value:
            interval_index = end_index
        if interval_index is not None:
            groups[interval_index].append(stru_index)
    return groups


def legacy_label_covered_values(values, zero_freq_intervals, max_min):
    intervals = [[-float("inf"), max_min[1]], *zero_freq_intervals, [max_min[0], float("inf")]]
    starts = [interval[0] for interval in intervals]
    result = []
    for value in values:
        index = bisect.bisect_left(starts, value)
        if value == intervals[0][1] or value == intervals[-1][0]:
            result.append(True)
            continue
        in_zero = False
        if index < len(intervals) and _is_in_interval(value, intervals[index]):
            in_zero = True
        elif index > 0 and _is_in_interval(value, intervals[index - 1]):
            in_zero = True
        result.append(not in_zero)
    return np.asarray(result, dtype=bool)


def legacy_find_min_cover_set(lists):
    cover_set = set()
    sets = [set(items) for items in lists]

    element_to_sets = defaultdict(set)
    for index, item_set in enumerate(sets):
        for element in item_set:
            element_to_sets[element].add(index)

    flattened = []
    for item_list in lists:
        flattened += item_list
    counts = Counter(flattened)
    sorted_elements = sorted(counts, key=counts.get, reverse=True)
    element_to_sets = {element: element_to_sets[element] for element in sorted_elements}

    remaining_sets = set(range(len(sets)))
    while remaining_sets:
        best_elem = None
        best_cover_count = 0
        for element, covered_sets in element_to_sets.items():
            cover_count = len(covered_sets & remaining_sets)
            if cover_count > best_cover_count:
                best_cover_count = cover_count
                best_elem = element
        cover_set.add(best_elem)
        remaining_sets -= element_to_sets[best_elem]
        for element in list(element_to_sets.keys()):
            if remaining_sets.isdisjoint(element_to_sets[element]):
                del element_to_sets[element]

    return list(cover_set)
