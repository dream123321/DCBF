from __future__ import annotations

import json
import math
from pathlib import Path
import warnings

from ase.io import read

try:
    from .path_names import MD_WORK_DIR
except ImportError:  # pragma: no cover - direct module import
    from path_names import MD_WORK_DIR


DEFAULT_NPT_MAX_CELL_VOLUME_FILTER_FACTOR = 1.5
MIN_NPT_MAX_CELL_VOLUME_FILTER_FACTOR = 1.1
NPT_VOLUME_FILTER_REPORT = "npt_cell_volume_filter.json"
NPT_VOLUME_FILTER_STAGE = "md_frame_intake"
TIMESTEP_INFO_KEY = "timestep"


def normalize_npt_max_cell_volume_filter_factor(value):
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(
            "npt_max_cell_volume_filter_factor must be null or a finite number"
        )
    try:
        factor = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "npt_max_cell_volume_filter_factor must be null or a finite number"
        ) from exc
    if not math.isfinite(factor):
        raise ValueError(
            "npt_max_cell_volume_filter_factor must be null or a finite number"
        )
    if factor < MIN_NPT_MAX_CELL_VOLUME_FILTER_FACTOR:
        warnings.warn(
            f"npt_max_cell_volume_filter_factor={value} is below "
            f"{MIN_NPT_MAX_CELL_VOLUME_FILTER_FACTOR}; using default "
            f"{DEFAULT_NPT_MAX_CELL_VOLUME_FILTER_FACTOR}",
            RuntimeWarning,
        )
        return DEFAULT_NPT_MAX_CELL_VOLUME_FILTER_FACTOR
    return factor


def _finite_positive_volume(atoms, description):
    try:
        volume = float(atoms.get_volume())
    except Exception as exc:
        raise RuntimeError(f"Cannot read cell volume for {description}") from exc
    if not math.isfinite(volume) or volume <= 0.0:
        raise RuntimeError(
            f"Invalid cell volume for {description}: {volume!r}"
        )
    return volume


def _volume_within_factor(current_volume, seed_volume, factor):
    volume_factor = current_volume / seed_volume
    return volume_factor <= factor or math.isclose(
        volume_factor,
        factor,
        rel_tol=1.0e-12,
        abs_tol=1.0e-12,
    )


def npt_seed_volume(case_dir):
    case_path = Path(case_dir)
    structure_name = case_path.parent.parent.name
    seed_path = case_path / f"{structure_name}.vasp"
    if not seed_path.is_file():
        raise RuntimeError(
            "NPT cell-volume filter cannot find the corresponding seed VASP: "
            f"{seed_path}"
        )
    try:
        seed_atoms = read(str(seed_path), index=0)
    except Exception as exc:
        raise RuntimeError(
            f"NPT cell-volume filter cannot read seed VASP: {seed_path}"
        ) from exc
    return _finite_positive_volume(seed_atoms, f"seed VASP {seed_path}")


def structure_name_from_case_dir(case_dir):
    """Structure name of a case dir, in the ``build_configuration_groups`` vocabulary."""
    return Path(case_dir).parent.parent.name


def case_dir_context(case_dir):
    """Return ``(structure_name, case_name)`` for a case dir below ``MD_WORK_DIR``.

    Matches the context used by the LAMMPS error log lines, e.g.
    ``structure=V_Mo_6.vasp | case=npt/1100``.
    """
    case_path = Path(case_dir)
    path_parts = case_path.parts
    structure_name = f"{case_path.name}.vasp"
    case_name = case_path.name

    work_indices = [index for index, part in enumerate(path_parts) if part == MD_WORK_DIR]
    if work_indices:
        work_index = work_indices[-1]
        if work_index + 1 < len(path_parts):
            structure_name = f"{path_parts[work_index + 1]}.vasp"
        if work_index + 2 < len(path_parts):
            case_name = "/".join(path_parts[work_index + 2:])

    return structure_name, case_name


def npt_frame_volume_guard(case_dir, factor):
    """Volume guard ``(seed_volume, factor)`` for the MD frames of one case dir.

    Returns ``None`` when the frames are not checked: the factor is disabled, or the
    trajectory came from NVT (constant volume, seeded from an already scaled cell).
    """
    if factor is None:
        return None
    case_path = Path(case_dir)
    ensemble = case_path.parent.name.lower()
    if ensemble not in {"npt", "nvt"}:
        raise RuntimeError(
            "NPT cell-volume filter cannot identify MD ensemble from case path: "
            f"{case_path}"
        )
    if ensemble != "npt":
        return None
    return npt_seed_volume(case_path), float(factor)


def new_frame_volume_detail():
    return {
        "total": 0,
        "kept": 0,
        "dropped": 0,
        "max_ratio": 0.0,
        "first_failed_step": None,
        "first_failed_ratio": None,
        "first_failed_reason": None,
        "truncated": False,
    }


def _frame_timestep(atoms):
    try:
        timestep = atoms.info.get(TIMESTEP_INFO_KEY)
    except AttributeError:
        return None
    if timestep is None:
        return None
    try:
        return int(timestep)
    except (TypeError, ValueError):
        return None


def classify_frame_volume(atoms, guard, detail):
    """Count one MD frame and return whether it stays within its volume guard.

    Over-expanded frames return ``False``; the caller drops them. Frames are counted in
    ``detail`` either way, so a disabled guard still reports the trajectory length.
    """
    if guard is None:
        detail["total"] += 1
        detail["kept"] += 1
        return True

    seed_volume, factor = guard
    volume = _finite_positive_volume(atoms, "MD trajectory frame")
    ratio = volume / seed_volume
    detail["total"] += 1
    if ratio > detail["max_ratio"]:
        detail["max_ratio"] = ratio

    # Once a trajectory first exceeds the limit, its suffix is intentionally
    # invalid even if a later cell contracts again.  Keep counting the suffix
    # for reporting, but never let it re-enter the sampling pipeline.
    if detail.get("truncated", False):
        detail["dropped"] += 1
        return False

    if _volume_within_factor(volume, seed_volume, factor):
        detail["kept"] += 1
        return True

    detail["dropped"] += 1
    detail["truncated"] = True
    detail["first_failed_reason"] = "cell_volume"
    if detail["first_failed_ratio"] is None:
        detail["first_failed_ratio"] = ratio
        detail["first_failed_step"] = _frame_timestep(atoms)
    return False


def build_volume_intake_stats(volume_details):
    """Collapse per-case intake details into generation counters and a per-case breakdown.

    ``original_selected_count`` is the number of candidate frames that entered the
    volume check, ``kept_count`` the frames that stayed within the limit and
    ``removed_count`` the truncated ones. The key names are kept from the previous
    post-selection filter so ``generation.py`` can still tell "everything was
    truncated" (``kept_count == 0``) from "nothing was selected".
    """
    stats = {
        "original_selected_count": 0,
        "kept_count": 0,
        "removed_count": 0,
        "dropped_configurations": [],
        "dropped_cases": {},
    }
    configurations = []
    for case_dir, detail in volume_details.items():
        dropped = int(detail["dropped"])
        stats["original_selected_count"] += int(detail["total"])
        stats["kept_count"] += int(detail["kept"])
        stats["removed_count"] += dropped
        if not dropped:
            continue
        stats["dropped_cases"][str(case_dir)] = {
            "dropped": dropped,
            "total": int(detail["total"]),
            "max_volume_ratio": float(detail["max_ratio"]),
            "first_failed_step": detail["first_failed_step"],
            "first_failed_ratio": detail["first_failed_ratio"],
            "first_failed_reason": detail.get("first_failed_reason"),
        }
        structure_name = structure_name_from_case_dir(case_dir)
        if structure_name not in configurations:
            configurations.append(structure_name)
    stats["dropped_configurations"] = configurations
    return stats


def lammps_error_brief(message):
    """Strip the ``LAMMPS error | structure=... | case=... |`` prefix for log lines."""
    text = (message or "").strip()
    if not text:
        return "no_error"
    parts = [part.strip() for part in text.split("|")]
    if len(parts) > 3 and parts[0] == "LAMMPS error":
        return " | ".join(parts[3:])
    return text


def format_volume_truncation_message(
    structure_name,
    case_name,
    detail,
    factor,
    lammps_message="",
):
    step_text = "unknown" if detail["first_failed_step"] is None else str(detail["first_failed_step"])
    ratio_text = (
        "unknown"
        if detail["first_failed_ratio"] is None
        else f"{float(detail['first_failed_ratio']):.4f}"
    )
    return (
        "NPT cell-volume truncation | "
        f"structure={structure_name} | case={case_name} | "
        f"dropped={int(detail['dropped'])}/{int(detail['total'])} frames | "
        f"V/V0 limit={float(factor):.4f} max={float(detail['max_ratio']):.4f} | "
        f"first_failed_step={step_text} V/V0={ratio_text} | "
        f"lammps={lammps_error_brief(lammps_message)}"
    )


def format_volume_summary_message(factor, stats):
    structures = sorted(set(stats.get("dropped_configurations", [])))
    suffix = f" | structures={', '.join(structures)}" if structures else ""
    return (
        f"NPT cell-volume filter: factor={float(factor):.3f} "
        f"dropped={int(stats['removed_count'])} of "
        f"{int(stats['original_selected_count'])} MD frames"
        + suffix
    )


def write_npt_volume_filter_report(workspace, factor, stats):
    report = {
        "enabled": factor is not None,
        "factor": factor,
        "stage": NPT_VOLUME_FILTER_STAGE,
        "original_selected_count": int(stats["original_selected_count"]),
        "kept_count": int(stats["kept_count"]),
        "removed_count": int(stats["removed_count"]),
        "dropped_configurations": list(stats.get("dropped_configurations", [])),
        "dropped_cases": dict(stats.get("dropped_cases", {})),
    }
    report_path = Path(workspace) / NPT_VOLUME_FILTER_REPORT
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def load_npt_volume_filter_report(workspace):
    report_path = Path(workspace) / NPT_VOLUME_FILTER_REPORT
    if not report_path.is_file():
        return None
    try:
        return json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Cannot read NPT cell-volume filter report: {report_path}"
        ) from exc


def truncated_frame_count(report):
    """Number of truncated MD frames in a report loaded by ``load_npt_volume_filter_report``.

    A non-zero count means the generation lost frame material to cell expansion, which
    keeps the affected structures in the sampling loop.
    """
    if not report or not report.get("enabled"):
        return 0
    try:
        return max(0, int(report.get("removed_count", 0)))
    except (TypeError, ValueError):
        return 0
