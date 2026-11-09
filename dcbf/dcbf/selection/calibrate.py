from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .core import _greedy_cover_elements_python, greedy_cover_elements
from .legacy import legacy_find_min_cover_set


class SelectionCalibrator:
    def __init__(self, seed=11):
        self.rng = np.random.default_rng(seed)

    def _make_case(self, case_index):
        n_classes = int(self.rng.integers(30, 200))
        universe_size = int(self.rng.integers(20, 140))
        lists = []
        for _ in range(n_classes):
            size = int(self.rng.integers(1, min(18, universe_size)))
            values = self.rng.choice(universe_size, size=size, replace=False).tolist()
            duplicate_count = int(self.rng.integers(0, 4))
            if duplicate_count:
                dup_values = self.rng.choice(values, size=duplicate_count, replace=True).tolist()
                values += dup_values
            lists.append(values)
        return {"case_index": case_index, "classes": n_classes, "universe_size": universe_size, "lists": lists}

    @staticmethod
    def _cover_valid(selected, lists):
        selected = set(selected)
        return all(selected.intersection(item) for item in lists if item)

    def run(self, cases=120):
        failures = []
        compared = 0
        mismatched = 0
        for case_index in range(cases):
            print(f"[calibrate] {case_index + 1}/{cases}", flush=True)
            case = self._make_case(case_index)
            lists = case["lists"]
            legacy = legacy_find_min_cover_set(lists)
            python_exact = _greedy_cover_elements_python(lists)
            active = greedy_cover_elements(lists)
            # Cover membership is a set property: the three implementations may
            # return the same indices in a different order, which is not a
            # correctness failure.  Compare sets, never the iteration order.
            legacy_set = set(legacy)
            python_set = set(python_exact)
            active_set = set(active)
            compared += 1
            if not (legacy_set == python_set == active_set):
                mismatched += 1
                failures.append(
                    {
                        "case_index": case_index,
                        "legacy": sorted(legacy_set),
                        "python_exact": sorted(python_set),
                        "active": sorted(active_set),
                        "same_size": len(legacy_set) == len(python_set) == len(active_set),
                        "only_legacy": sorted(legacy_set - python_set - active_set),
                        "only_python_exact": sorted(python_set - legacy_set - active_set),
                        "only_active": sorted(active_set - legacy_set - python_set),
                    }
                )
        print(
            f"[calibrate] done: cases={cases} compared={compared} mismatched={mismatched}",
            flush=True,
        )
        return {
            "schema_version": 1,
            "cases": cases,
            "compared": compared,
            "mismatched": mismatched,
            "passed": not failures,
            "failures": failures,
        }

    def write_report(self, output_path, cases=120):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        report = self.run(cases=cases)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        return output_path, report
