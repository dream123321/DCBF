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
        for case_index in range(cases):
            case = self._make_case(case_index)
            lists = case["lists"]
            legacy = legacy_find_min_cover_set(lists)
            python_exact = _greedy_cover_elements_python(lists)
            active = greedy_cover_elements(lists)
            if legacy != python_exact or legacy != active:
                failures.append(
                    {
                        "case_index": case_index,
                        "legacy": legacy,
                        "python_exact": python_exact,
                        "active": active,
                    }
                )

        return {
            "cases": cases,
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
