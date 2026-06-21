from __future__ import annotations

import cProfile
import io
import json
from pathlib import Path
import pstats
import time

import numpy as np

from .core import _exact_min_cover_ints, greedy_cover_elements, group_structure_indices_by_interval, label_covered_values
from .legacy import (
    legacy_find_min_cover_set,
    legacy_group_structure_indices_by_interval,
    legacy_label_covered_values,
)


class SelectionBenchmark:
    def __init__(self, seed=7):
        self.rng = np.random.default_rng(seed)

    def _time_callable(self, func, *args, repeat=3):
        durations = []
        result = None
        for _ in range(repeat):
            start = time.perf_counter()
            result = func(*args)
            durations.append(time.perf_counter() - start)
        return {
            "seconds": min(durations),
            "repeat": repeat,
        }, result

    def _profile_callable(self, func, *args, lines=12):
        profiler = cProfile.Profile()
        profiler.enable()
        func(*args)
        profiler.disable()
        buffer = io.StringIO()
        stats = pstats.Stats(profiler, stream=buffer).sort_stats("cumtime")
        stats.print_stats(lines)
        return buffer.getvalue()

    def _make_interval_payload(self, n_structures=160_000, n_intervals=320):
        stru_indexs = np.arange(n_structures, dtype=np.int64)
        data_list = self.rng.normal(loc=0.0, scale=3.0, size=n_structures)
        edges = np.linspace(data_list.min() - 0.5, data_list.max() + 0.5, n_intervals + 1)
        intervals = [[float(edges[i]), float(edges[i + 1])] for i in range(n_intervals)]
        return stru_indexs.tolist(), data_list.tolist(), intervals

    def _make_coverage_payload(self, n_values=220_000, n_zero_intervals=280):
        values = self.rng.normal(loc=0.0, scale=4.0, size=n_values)
        max_min = [10.0, -10.0]
        edges = np.linspace(max_min[1], max_min[0], n_zero_intervals + 1)
        zero_freq_intervals = []
        for index in range(0, n_zero_intervals, 5):
            zero_freq_intervals.append([float(edges[index]), float(edges[index + 1])])
        return values.tolist(), zero_freq_intervals, max_min

    def _make_cover_payload(self, n_classes=45_000, universe_size=22_000, avg_cardinality=14):
        lists = []
        low = max(3, avg_cardinality - 5)
        high = avg_cardinality + 5
        for _ in range(n_classes):
            size = int(self.rng.integers(low, high))
            values = self.rng.choice(universe_size, size=size, replace=False)
            lists.append(values.tolist())
        return lists

    @staticmethod
    def _covers_all_classes(selected, lists):
        selected = set(selected)
        return all(selected.intersection(item_list) for item_list in lists if item_list)

    def run(self):
        interval_args = self._make_interval_payload()
        coverage_args = self._make_coverage_payload()
        cover_args = self._make_cover_payload()
        profile_cover_args = self._make_cover_payload(n_classes=12_000, universe_size=9_000, avg_cardinality=14)

        legacy_interval, legacy_interval_result = self._time_callable(
            legacy_group_structure_indices_by_interval, *interval_args
        )
        current_interval, current_interval_result = self._time_callable(
            group_structure_indices_by_interval, *interval_args
        )

        legacy_coverage, legacy_coverage_result = self._time_callable(
            legacy_label_covered_values, *coverage_args
        )
        current_coverage, current_coverage_result = self._time_callable(
            label_covered_values, *coverage_args
        )

        legacy_cover, legacy_cover_result = self._time_callable(
            legacy_find_min_cover_set, cover_args, repeat=1
        )
        current_cover, current_cover_result = self._time_callable(
            greedy_cover_elements, cover_args, repeat=1
        )

        interval_total_legacy = legacy_interval["seconds"]
        interval_total_current = current_interval["seconds"]
        coverage_total_legacy = legacy_coverage["seconds"]
        coverage_total_current = current_coverage["seconds"]
        cover_total_legacy = legacy_cover["seconds"]
        cover_total_current = current_cover["seconds"]

        legacy_total = interval_total_legacy + coverage_total_legacy + cover_total_legacy
        current_total = interval_total_current + coverage_total_current + cover_total_current

        results = {
            "dataset_shape": {
                "interval_grouping": {
                    "structures": len(interval_args[0]),
                    "intervals": len(interval_args[2]),
                },
                "coverage_labeling": {
                    "values": len(coverage_args[0]),
                    "zero_intervals": len(coverage_args[1]),
                },
                "min_cover": {
                    "classes": len(cover_args),
                    "universe_hint": 22_000,
                },
            },
            "correctness": {
                "interval_grouping_same_non_empty_buckets": sum(bool(item) for item in legacy_interval_result)
                == sum(bool(item) for item in current_interval_result),
                "coverage_same_true_count": int(np.sum(legacy_coverage_result)) == int(np.sum(current_coverage_result)),
                "cover_exact_match": legacy_cover_result == current_cover_result,
                "legacy_cover_valid": self._covers_all_classes(legacy_cover_result, cover_args),
                "current_cover_valid": self._covers_all_classes(current_cover_result, cover_args),
                "current_cover_not_larger": len(set(current_cover_result)) <= len(set(legacy_cover_result)),
            },
            "backend": {
                "min_cover": "compiled" if _exact_min_cover_ints is not None else "python",
            },
            "timings": {
                "legacy": {
                    "interval_grouping": interval_total_legacy,
                    "coverage_labeling": coverage_total_legacy,
                    "min_cover": cover_total_legacy,
                    "total": legacy_total,
                },
                "current": {
                    "interval_grouping": interval_total_current,
                    "coverage_labeling": coverage_total_current,
                    "min_cover": cover_total_current,
                    "total": current_total,
                },
            },
            "speedups": {
                "interval_grouping": interval_total_legacy / interval_total_current,
                "coverage_labeling": coverage_total_legacy / coverage_total_current,
                "min_cover": cover_total_legacy / cover_total_current,
                "total": legacy_total / current_total,
            },
            "shares": {
                "legacy": {
                    "interval_grouping": interval_total_legacy / legacy_total,
                    "coverage_labeling": coverage_total_legacy / legacy_total,
                    "min_cover": cover_total_legacy / legacy_total,
                },
                "current": {
                    "interval_grouping": interval_total_current / current_total,
                    "coverage_labeling": coverage_total_current / current_total,
                    "min_cover": cover_total_current / current_total,
                },
            },
            "profiles": {
                "legacy": {
                    "interval_grouping": self._profile_callable(legacy_group_structure_indices_by_interval, *interval_args),
                    "coverage_labeling": self._profile_callable(legacy_label_covered_values, *coverage_args),
                    "min_cover": self._profile_callable(legacy_find_min_cover_set, profile_cover_args),
                },
                "current": {
                    "interval_grouping": self._profile_callable(group_structure_indices_by_interval, *interval_args),
                    "coverage_labeling": self._profile_callable(label_covered_values, *coverage_args),
                    "min_cover": self._profile_callable(greedy_cover_elements, profile_cover_args),
                },
            },
        }
        return results

    def write_report(self, output_path):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results = self.run()
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)
        return output_path, results
