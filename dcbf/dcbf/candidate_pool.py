from __future__ import annotations

import math
from pathlib import Path

from ase.io import iread

from .das.file_conversion import write_normalized_extxyz


CANDIDATE_POOL_FILENAME = "candidate_pool.xyz"
GENERATION_SELECTION_FILENAME = "candidate_selected.xyz"
GENERATION_SELECTION_ADDED_MARKER = "__candidate_selected_added__"


def count_xyz_structures(path):
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return 0
    return sum(1 for _ in iread(str(path), index=":"))


def count_cfg_structures(path):
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return 0
    count = 0
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.strip() == "BEGIN_CFG":
                count += 1
    return count


def generation_selection_cache_path(workspace):
    return Path(workspace).resolve() / GENERATION_SELECTION_FILENAME


def generation_selection_added_marker_path(workspace):
    return Path(workspace).resolve() / GENERATION_SELECTION_ADDED_MARKER


def resolve_candidate_trigger(trigger_spec, base_structure_count):
    if trigger_spec is None:
        trigger_spec = 10

    if isinstance(trigger_spec, (int, float)):
        threshold = int(trigger_spec)
        if threshold <= 0:
            raise ValueError("parameter.candidate_trigger must be greater than 0")
        return threshold, f"{threshold}"

    raw_text = str(trigger_spec).strip()
    if not raw_text:
        raise ValueError("parameter.candidate_trigger cannot be empty")

    if raw_text.endswith("%"):
        percent_text = raw_text[:-1].strip()
        try:
            percent_value = float(percent_text)
        except ValueError as exc:
            raise ValueError(f"Invalid percentage candidate_trigger: {trigger_spec}") from exc
        if percent_value <= 0:
            raise ValueError("parameter.candidate_trigger percentage must be greater than 0")
        threshold = max(1, int(math.ceil(float(base_structure_count) * percent_value / 100.0)))
        return threshold, f"{percent_value:g}% of {base_structure_count}"

    try:
        threshold = int(raw_text)
    except ValueError as exc:
        raise ValueError(f"Invalid candidate_trigger value: {trigger_spec}") from exc
    if threshold <= 0:
        raise ValueError("parameter.candidate_trigger must be greater than 0")
    return threshold, raw_text


class CandidatePool:
    def __init__(self, run_dir):
        self.run_dir = Path(run_dir).resolve()
        self.path = self.run_dir / CANDIDATE_POOL_FILENAME

    def count(self):
        return count_xyz_structures(self.path)

    def load_atoms(self):
        if self.count() == 0:
            return []
        return list(iread(str(self.path), index=":"))

    def append_atoms(self, atoms):
        atoms = list(atoms)
        if not atoms:
            return 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        write_normalized_extxyz(self.path, atoms, append=self.path.exists() and self.path.stat().st_size > 0)
        return len(atoms)

    def clear(self):
        if self.path.exists():
            self.path.unlink()
