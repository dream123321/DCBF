"""SUS2-SH PCA and descriptor coverage analysis.

This command is deliberately an analysis utility.  It does not participate in
the sampling selector and it does not restore the removed DAS/MTP paths.  The
descriptor columns are obtained from the same SUS2 metadata-backed
``DescriptorStore`` used by Reduce.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import contextlib
import csv
import json
import math
import importlib.util
import os
import random
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .cli_examples import RawDefaultsHelpFormatter, command_example_epilog


DEFAULT_BODY_LIST = ["two", "three"]
BODY_CHOICES = ("all", "two", "three", "four", "five", "six")
DEFAULT_COVERAGE_GRID = "last-loop"
DEFAULT_QUERY_NAME = "sus2md_1000.xyz"
DEFAULT_LAMMPS_TIMEOUT_HOURS = 24.0
DEFAULT_LAMMPS_POLL_SECONDS = 15


def _missing_dir_chain(path: Path) -> List[Path]:
    """Directories that ``mkdir(parents=True)`` would create, deepest first."""
    missing: List[Path] = []
    current = Path(path)
    while not current.exists():
        missing.append(current)
        parent = current.parent
        if parent == current:
            break
        current = parent
    return missing


def _remove_created_empty_dirs(dirs: Sequence[Path]) -> None:
    """Undo ``mkdir(parents=True)`` after a failure, never deleting results."""
    for directory in dirs:
        try:
            os.rmdir(directory)
        except OSError:
            # Non-empty (partial results are kept) or already removed.
            break


def _atomic_write_text(path: Path, write_callback) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", dir=str(path.parent),
        prefix=f".{path.name}.", suffix=".tmp", delete=False,
    )
    temporary = Path(handle.name)
    try:
        write_callback(handle)
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
        os.replace(temporary, path)
    finally:
        with contextlib.suppress(Exception):
            handle.close()
        if temporary.exists():
            with contextlib.suppress(OSError):
                temporary.unlink()


@dataclass
class QueryStructureSource:
    index: int
    label: str
    path: Path
    frame_index: int
    atoms: object


@dataclass
class PcaModel:
    mean: np.ndarray
    std: np.ndarray
    components: np.ndarray
    explained: float


@dataclass
class CoverageResult:
    dataset: str
    element: str
    element_index: int
    ref_pca: np.ndarray
    query_pca: np.ndarray
    coverage_2d: float
    coverage_1d: float
    covered_2d: np.ndarray
    covered_1d_score: np.ndarray
    explained: float
    body_coverage: Dict[str, float]


def _parse_loop_select(value: str) -> Optional[List[int]]:
    text = str(value or "all").strip()
    if text in {"all", "middle-half", "uniform-half"}:
        return None
    text = text.strip("[]")
    try:
        return [int(item.strip()) for item in text.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError(
            "--loop-select must be all, middle-half, uniform-half, or a comma-separated integer list"
        ) from exc


def _select_loop_labels(labels: Sequence[int], selector: str) -> List[int]:
    available = sorted(set(int(item) for item in labels))
    requested = _parse_loop_select(selector)
    if requested is not None:
        missing = [item for item in requested if item not in available]
        if missing:
            raise ValueError(f"--loop-select requested values not present: {missing}; available={available}")
        return list(dict.fromkeys(requested))
    if selector == "all" or len(available) <= 3:
        return available
    initial = [-1] if -1 in available else []
    loops = [item for item in available if item >= 0]
    if selector == "middle-half":
        start = int(math.ceil(0.25 * (len(loops) - 1)))
        end = int(math.floor(0.75 * (len(loops) - 1)))
        return list(dict.fromkeys(initial + loops[start:end + 1] + loops[-1:]))
    if selector == "uniform-half":
        count = max(1, int(math.ceil(len(loops) / 2)))
        positions = np.linspace(0, len(loops) - 1, count).round().astype(int)
        return list(dict.fromkeys(initial + [loops[index] for index in positions] + loops[-1:]))
    raise ValueError(f"Unsupported --loop-select: {selector}")


def _fit_data(ref_arrays: Sequence[np.ndarray], query: np.ndarray, source: str) -> np.ndarray:
    if source == "query":
        return query
    if source == "input":
        return np.vstack(ref_arrays)
    if source == "combined":
        return np.vstack([*ref_arrays, query])
    raise ValueError(f"Unsupported --pca-fit-source: {source}")


def fit_pca_model(data: np.ndarray, components: int = 2) -> PcaModel:
    data = np.asarray(data, dtype=float)
    if data.ndim != 2 or data.shape[0] == 0 or data.shape[1] == 0:
        raise ValueError("PCA requires a non-empty two-dimensional descriptor array")
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    std[std == 0] = 1.0
    scaled = (data - mean) / std
    _, singular_values, vh = np.linalg.svd(scaled, full_matrices=False)
    available = min(int(components), vh.shape[0])
    basis = np.zeros((int(components), data.shape[1]), dtype=float)
    basis[:available] = vh[:available]
    variance = singular_values ** 2
    total = float(np.sum(variance))
    explained = float(np.sum(variance[:available]) / total) if total > 0 else 0.0
    return PcaModel(mean=mean, std=std, components=basis, explained=explained)


def transform_pca(data: np.ndarray, model: PcaModel) -> np.ndarray:
    return ((np.asarray(data, dtype=float) - model.mean) / model.std) @ model.components.T


def _fd_bins(data: np.ndarray, factor: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.asarray(data, dtype=float)
    mins = np.min(data, axis=0)
    maxs = np.max(data, axis=0)
    bins = np.ones(data.shape[1], dtype=int)
    widths = np.zeros(data.shape[1], dtype=float)
    for dim in range(data.shape[1]):
        span = float(maxs[dim] - mins[dim])
        if span <= 0:
            continue
        iqr = float(np.percentile(data[:, dim], 75) - np.percentile(data[:, dim], 25))
        width = 2.0 * iqr / (len(data) ** (1.0 / 3.0)) * float(factor)
        if not np.isfinite(width) or width <= 0:
            width = span
        bins[dim] = max(1, int(span / width))
        widths[dim] = span / bins[dim]
    return mins, maxs, widths


def _bin_ids(data: np.ndarray, mins: np.ndarray, maxs: np.ndarray, widths: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    data = np.asarray(data, dtype=float)
    ids = np.full(data.shape, -1, dtype=np.int64)
    valid = np.all(np.isfinite(data), axis=1)
    for dim in range(data.shape[1]):
        if widths[dim] == 0:
            current = np.isclose(data[:, dim], mins[dim])
            ids[current, dim] = 0
        else:
            current = (data[:, dim] >= mins[dim]) & (data[:, dim] < maxs[dim])
            ids[current, dim] = np.floor((data[current, dim] - mins[dim]) / widths[dim]).astype(np.int64)
        valid &= current
    return ids, valid


def fd_coverage(ref: np.ndarray, query: np.ndarray, factor: float, grid: np.ndarray) -> Tuple[float, np.ndarray]:
    if len(ref) == 0 or len(query) == 0:
        return 0.0, np.zeros(len(query), dtype=bool)
    mins, maxs, widths = _fd_bins(np.asarray(grid, dtype=float), factor)
    ref_ids, ref_valid = _bin_ids(ref, mins, maxs, widths)
    query_ids, query_valid = _bin_ids(query, mins, maxs, widths)
    occupied = {tuple(item) for item in ref_ids[ref_valid]}
    covered = query_valid & np.asarray([tuple(item) in occupied for item in query_ids], dtype=bool)
    return float(np.mean(covered)), covered


def mean_1d_coverage(ref: np.ndarray, query: np.ndarray, factor: float, grid: np.ndarray) -> Tuple[float, np.ndarray]:
    scores = []
    rates = []
    for dim in range(ref.shape[1]):
        rate, covered = fd_coverage(ref[:, [dim]], query[:, [dim]], factor, grid[:, [dim]])
        rates.append(rate)
        scores.append(covered.astype(float))
    if not rates:
        return 0.0, np.zeros(len(query), dtype=float)
    return float(np.mean(rates)), np.mean(np.vstack(scores), axis=0)


def _resolve_run_path(run_dir: Optional[Path], value: Optional[str], default: Optional[str] = None) -> Optional[Path]:
    if value:
        path = Path(value).expanduser()
        return path.resolve() if path.is_absolute() else (run_dir / path).resolve() if run_dir else path.resolve()
    if run_dir and default:
        candidate = (run_dir / default).resolve()
        if candidate.exists():
            return candidate
    return None


def _latest_model(run_dir: Path) -> Optional[Path]:
    candidates = list(run_dir.glob("main_*/gen_*/**/current_*.sus2"))
    candidates += list(run_dir.glob("current_*.sus2"))
    candidates += list(run_dir.glob("**/final.sus2"))
    candidates = [path for path in candidates if path.is_file()]
    return sorted(candidates, key=lambda path: path.stat().st_mtime_ns)[-1] if candidates else None


def _load_runtime_run_defaults(run_dir: Path) -> Tuple[Optional[Path], Optional[Path], Optional[List[str]]]:
    input_path = None
    query_path = None
    elements = None
    runtime_path = run_dir / "dcbf.runtime.json"
    if runtime_path.is_file():
        try:
            runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
            workflow = dict(runtime.get("workflow") or {})
            input_path = _resolve_run_path(run_dir, workflow.get("output_xyz_name"), "sampled_training_dataset.xyz")
        except (OSError, ValueError, TypeError):
            pass
    if input_path is None:
        input_path = _resolve_run_path(run_dir, None, "sampled_training_dataset.xyz")
    for relative in ("coverage_query_lammps/query.xyz", "npt.xyz", "query.xyz", "sus2md_1000.xyz"):
        candidate = run_dir / relative
        if candidate.is_file():
            query_path = candidate.resolve()
            break
    parameter_path = run_dir / "init" / "parameter.yaml"
    if parameter_path.is_file():
        try:
            import yaml
            parameter = yaml.safe_load(parameter_path.read_text(encoding="utf-8")) or {}
            if parameter.get("ele"):
                elements = list(parameter["ele"])
        except Exception:
            pass
    return input_path, query_path, elements


def _query_structure_files(run_dir: Path) -> List[Path]:
    structure_dir = run_dir / "stru"
    if not structure_dir.is_dir():
        raise FileNotFoundError(
            f"Structure directory not found for automatic LAMMPS query generation: {structure_dir}"
        )
    allowed_names = {"POSCAR", "CONTCAR"}
    allowed_suffixes = {".vasp", ".cif", ".xyz", ".extxyz"}
    files = []
    for path in sorted(structure_dir.iterdir(), key=lambda item: item.name):
        if not path.is_file():
            continue
        if path.name in allowed_names or path.suffix.lower() in allowed_suffixes:
            files.append(path)
    if not files:
        raise FileNotFoundError(
            f"No POSCAR/CONTCAR/*.vasp/*.cif/*.xyz files found in {structure_dir}"
        )
    return files


def _safe_query_structure_label(text: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in "._-" else "_" for char in text)
    return cleaned.strip("._") or "structure"


def _find_query_structure_sources(run_dir: Path) -> List[QueryStructureSource]:
    from ase.io import iread

    sources: List[QueryStructureSource] = []
    for path in _query_structure_files(run_dir):
        try:
            frames = list(iread(str(path), index=":"))
        except Exception as exc:
            raise ValueError(f"Could not read query structure source {path}: {exc}") from exc
        if not frames:
            continue
        base = _safe_query_structure_label(path.stem or path.name)
        for frame_index, atoms in enumerate(frames):
            label = base if len(frames) == 1 else f"{base}_f{frame_index}"
            sources.append(QueryStructureSource(len(sources), label, path, frame_index, atoms))
    if not sources:
        raise ValueError(f"No structures could be read from {run_dir / 'stru'}")
    return sources


def _normalize_query_structure_selectors(raw_selectors) -> List[str]:
    if raw_selectors is None:
        return ["all"]
    if isinstance(raw_selectors, str):
        raw_selectors = [raw_selectors]
    values: List[str] = []
    for value in raw_selectors:
        values.extend(item.strip() for item in str(value).split(",") if item.strip())
    return values or ["all"]


def _select_query_structure_sources(
    sources: Sequence[QueryStructureSource], raw_selectors
) -> List[QueryStructureSource]:
    selectors = _normalize_query_structure_selectors(raw_selectors)
    if "all" in selectors:
        return list(sources)
    selected: List[QueryStructureSource] = []
    for selector in selectors:
        if selector == "first":
            candidates = list(sources[:1])
        elif selector.startswith("index:"):
            try:
                wanted = int(selector.split(":", 1)[1])
            except ValueError as exc:
                raise ValueError(f"Invalid query structure selector {selector!r}") from exc
            candidates = [item for item in sources if item.index == wanted]
        else:
            candidates = [
                item
                for item in sources
                if item.label == selector
                or item.path.name == selector
                or item.path.stem == selector
                or item.path.match(selector)
            ]
        if not candidates:
            raise ValueError(
                f"Invalid query structure selector {selector!r}; "
                f"available={[item.label for item in sources]}"
            )
        for item in candidates:
            if item not in selected:
                selected.append(item)
    return selected


def _flatten_temperature_values(values) -> List[float]:
    if values is None:
        return []
    if isinstance(values, (list, tuple)):
        result: List[float] = []
        for value in values:
            result.extend(_flatten_temperature_values(value))
        return result
    try:
        return [float(values)]
    except (TypeError, ValueError):
        return []


def _choose_lammps_temperature(runtime_config: Dict) -> float:
    workflow = dict(runtime_config.get("workflow") or {})
    values = _flatten_temperature_values(workflow.get("main_loop_npt"))
    if not values:
        values = _flatten_temperature_values(workflow.get("main_loop_nvt"))
    return values[-1] if values else 300.0


def _parse_repeat_size(value) -> Tuple[int, int, int]:
    if isinstance(value, (list, tuple)) and len(value) == 3:
        return tuple(max(1, int(item)) for item in value)
    text = str(value or "(1, 1, 1)").strip().strip("()[]")
    parts = [item.strip() for item in text.split(",") if item.strip()]
    if len(parts) != 3:
        raise ValueError(f"parameter.size must contain three integers, got {value!r}")
    return tuple(max(1, int(item)) for item in parts)


def _load_lammps_template(run_dir: Path, ensemble: str, temperature: float) -> str:
    template_path = run_dir / "init" / "lmp_in.py"
    if not template_path.is_file():
        raise FileNotFoundError(f"LAMMPS template not found: {template_path}")
    spec = importlib.util.spec_from_file_location("dcbf_coverage_lmp_in", template_path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise ImportError(f"Could not load LAMMPS template: {template_path}")
    spec.loader.exec_module(module)
    if not hasattr(module, ensemble):
        raise ValueError(f"{template_path} does not define {ensemble!r} LAMMPS template")
    random_number = random.randint(1000, 1000000)
    return (
        f"variable T equal {temperature}\n"
        f"variable random equal {random_number}\n"
        + getattr(module, ensemble)
    )


def _inject_sus2_pair_style(script_text: str, model_filename: str) -> str:
    lines = script_text.splitlines()
    normalized: List[str] = []
    inserted = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("# variable mlip_ini"):
            continue
        if stripped.startswith("pair_style") and any(
            token in stripped for token in ("mlip", "sus2mtp", "sus2/embedded")
        ):
            continue
        if stripped == "pair_coeff * *":
            continue
        normalized.append(line)
        if not inserted and stripped.startswith("read_data data.in"):
            normalized.extend(
                [
                    "",
                    f"pair_style sus2/embedded {model_filename} radial_mode direct",
                    "pair_coeff * *",
                ]
            )
            inserted = True
    if not inserted:
        raise ValueError("Could not find 'read_data data.in' in init/lmp_in.py template")
    return "\n".join(normalized) + "\n"


def _lammps_box_to_cell(box_lines: Sequence[str]) -> np.ndarray:
    rows = [[float(item) for item in line.split()] for line in box_lines]
    if len(rows) != 3:
        return np.eye(3)
    if len(rows[0]) >= 3:
        xlo_bound, xhi_bound, xy = rows[0][:3]
        ylo_bound, yhi_bound, xz = rows[1][:3]
        zlo_bound, zhi_bound, yz = rows[2][:3]
        xlo = xlo_bound - min(0.0, xy, xz, xy + xz)
        xhi = xhi_bound - max(0.0, xy, xz, xy + xz)
        ylo = ylo_bound - min(0.0, yz)
        yhi = yhi_bound - max(0.0, yz)
        return np.array(
            [[xhi - xlo, 0.0, 0.0], [xy, yhi - ylo, 0.0], [xz, yz, zhi_bound - zlo_bound]],
            dtype=float,
        )
    return np.diag([rows[0][1] - rows[0][0], rows[1][1] - rows[1][0], rows[2][1] - rows[2][0]])


def _read_lammps_custom_dump(dump_path: Path, elements: Sequence[str]) -> List:
    from ase import Atoms
    from ase.calculators.singlepoint import SinglePointCalculator

    lines = dump_path.read_text(encoding="utf-8", errors="replace").splitlines()
    frames = []
    cursor = 0
    type_to_element = {index + 1: element for index, element in enumerate(elements)}
    while cursor < len(lines):
        if not lines[cursor].startswith("ITEM: TIMESTEP"):
            cursor += 1
            continue
        cursor += 2
        if cursor >= len(lines) or not lines[cursor].startswith("ITEM: NUMBER OF ATOMS"):
            raise ValueError(f"Unexpected LAMMPS dump near {dump_path}")
        natoms = int(lines[cursor + 1].strip())
        cursor += 2
        if cursor >= len(lines) or not lines[cursor].startswith("ITEM: BOX BOUNDS"):
            raise ValueError(f"Missing BOX BOUNDS in {dump_path}")
        cell = _lammps_box_to_cell(lines[cursor + 1 : cursor + 4])
        cursor += 4
        if cursor >= len(lines) or not lines[cursor].startswith("ITEM: ATOMS"):
            raise ValueError(f"Missing ATOMS section in {dump_path}")
        columns = lines[cursor].split()[2:]
        cursor += 1
        column_index = {name: index for index, name in enumerate(columns)}
        required = ["id", "type", "x", "y", "z"]
        missing = [name for name in required if name not in column_index]
        if missing:
            raise ValueError(f"LAMMPS dump is missing {missing}: {dump_path}")
        rows = []
        for _ in range(natoms):
            rows.append(lines[cursor].split())
            cursor += 1
        rows.sort(key=lambda row: int(float(row[column_index["id"]])))
        symbols = []
        positions = np.zeros((natoms, 3), dtype=float)
        forces = np.zeros((natoms, 3), dtype=float)
        per_atom_energy = np.zeros(natoms, dtype=float)
        for row_index, row in enumerate(rows):
            type_id = int(float(row[column_index["type"]]))
            if type_id not in type_to_element:
                raise ValueError(
                    f"LAMMPS atom type {type_id} is outside the configured element order {list(elements)}"
                )
            symbols.append(type_to_element[type_id])
            positions[row_index] = [float(row[column_index[axis]]) for axis in ("x", "y", "z")]
            if all(axis in column_index for axis in ("fx", "fy", "fz")):
                forces[row_index] = [float(row[column_index[axis]]) for axis in ("fx", "fy", "fz")]
            if "c_pe" in column_index:
                per_atom_energy[row_index] = float(row[column_index["c_pe"]])
        atoms = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True)
        atoms.calc = SinglePointCalculator(
            atoms, energy=float(np.sum(per_atom_energy)), forces=forces
        )
        frames.append(atoms)
    if not frames:
        raise ValueError(f"No frames found in LAMMPS dump: {dump_path}")
    return frames


def _apply_lammps_scheduler_overrides(scheduler: Dict, args: argparse.Namespace) -> Dict:
    normalized = dict(scheduler)
    if getattr(args, "lammps_exe", None):
        normalized["lmp_exe"] = str(args.lammps_exe)
    if getattr(args, "lammps_env", None):
        normalized["lmp_env"] = str(args.lammps_env)
    if getattr(args, "lammps_cores", None) is not None:
        cores = int(args.lammps_cores)
        normalized["lmp_cores"] = cores
        normalized["lmp_ptile"] = min(int(normalized.get("lmp_ptile", cores)), cores)
    return normalized


def _tail_file(path: Path, max_lines: int = 80) -> str:
    if not path.is_file():
        return ""
    return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-max_lines:])


def _lammps_failure_context(work_dir: Path) -> str:
    candidates = [work_dir / "jobs_mlip_0.ini.out", work_dir / "bsub.lsf"]
    candidates.extend(sorted(work_dir.glob("*.err")))
    candidates.extend(sorted(work_dir.glob("*.out")))
    blocks = []
    seen = set()
    for path in candidates:
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        text = _tail_file(path)
        if text:
            blocks.append(f"--- {path.name} ---\n{text}")
    return "\n\n".join(blocks)


def _lammps_timeout_seconds(args: argparse.Namespace) -> int:
    if getattr(args, "lammps_timeout", None) is not None:
        return max(1, int(float(args.lammps_timeout)))
    hours = float(getattr(args, "lammps_timeout_hours", DEFAULT_LAMMPS_TIMEOUT_HOURS))
    return max(1, int(hours * 3600))


def _run_lammps_query_local(work_dir: Path, scheduler: Dict, args: argparse.Namespace) -> None:
    lmp_exe = str(getattr(args, "lammps_exe", None) or scheduler.get("lmp_exe") or "").strip()
    if not lmp_exe:
        raise ValueError("No LAMMPS executable found; set scheduler.lmp_exe or --lammps-exe")
    lmp_env = str(getattr(args, "lammps_env", None) or scheduler.get("lmp_env") or "").strip()
    cores = int(getattr(args, "lammps_cores", None) or scheduler.get("lmp_cores") or 1)
    command = f"export NP={cores}; "
    if lmp_env:
        command += lmp_env.rstrip() + "; "
    command += f"{lmp_exe} -in lmp.in -var out_dump_file force.0.nc"
    subprocess.run(
        ["bash", "-lc", command],
        cwd=str(work_dir),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def _run_lammps_query_scheduler(work_dir: Path, scheduler: Dict, args: argparse.Namespace) -> None:
    from .runtime_config import build_scheduler_spec

    scheduler = _apply_lammps_scheduler_overrides(scheduler, args)
    if not scheduler.get("lmp_exe"):
        raise ValueError("Scheduler LAMMPS mode requires scheduler.lmp_exe or --lammps-exe")
    spec = build_scheduler_spec(scheduler)
    for stale_name in (
        "__start__",
        "__ok__",
        "__fail__",
        "force.0.dump",
        "force.0.nc",
        "jobs_mlip_0.ini.out",
    ):
        stale = work_dir / stale_name
        if stale.exists():
            stale.unlink()
    env_text = f"{str(spec.lmp_env).strip()}\n" if str(spec.lmp_env).strip() else ""
    script = (
        "#!/bin/bash\n"
        f"{spec.bsub_script_lmp_job_name}coverage_query_lammps\n"
        f"{spec.bsub_script_lmp}\n"
        "set -e\n"
        "trap 'touch __fail__' ERR\n"
        "touch __start__\n"
        f"{env_text}"
        'COMMAND="${COMMAND_0} -in lmp.in -var out_dump_file force.0.nc"\n'
        "$COMMAND > jobs_mlip_0.ini.out 2>&1\n"
        "touch __ok__\n"
    )
    script_path = work_dir / "bsub.lsf"
    script_path.write_text(script, encoding="utf-8")
    script_path.chmod(0o755)
    subprocess.run(spec.task_submission_method, cwd=str(work_dir), shell=True, check=True)
    deadline = time.time() + _lammps_timeout_seconds(args)
    while True:
        if (work_dir / "__ok__").is_file():
            return
        if (work_dir / "__fail__").is_file():
            raise RuntimeError(
                f"Scheduler LAMMPS query failed in {work_dir}.\n{_lammps_failure_context(work_dir)}"
            )
        if time.time() > deadline:
            raise TimeoutError(
                f"Timed out waiting for scheduler LAMMPS query in {work_dir}.\n"
                f"{_lammps_failure_context(work_dir)}"
            )
        time.sleep(DEFAULT_LAMMPS_POLL_SECONDS)


def _run_lammps_query_for_structure(
    run_dir: Path,
    args: argparse.Namespace,
    work_dir: Path,
    atoms,
    size: Tuple[int, int, int],
    specorder: Sequence[str],
    scheduler: Dict,
    ensemble: str,
    temperature: float,
    model_path: Path,
) -> List:
    from ase.io import write

    work_dir.mkdir(parents=True, exist_ok=True)
    repeated = atoms.copy().repeat(size)
    write(
        str(work_dir / "data.in"),
        repeated,
        format="lammps-data",
        masses=True,
        specorder=list(specorder),
        force_skew=True,
    )
    local_model = work_dir / "current_0.sus2"
    shutil.copy2(model_path, local_model)
    lmp_text = _inject_sus2_pair_style(
        _load_lammps_template(run_dir, ensemble, temperature), local_model.name
    )
    (work_dir / "lmp.in").write_text(lmp_text, encoding="utf-8")
    run_mode = str(getattr(args, "lammps_run_mode", "scheduler") or "scheduler").lower()
    if run_mode == "local":
        _run_lammps_query_local(work_dir, scheduler, args)
    elif run_mode == "scheduler":
        _run_lammps_query_scheduler(work_dir, scheduler, args)
    else:
        raise ValueError(f"Unsupported --lammps-run-mode: {run_mode}")
    dump_path = work_dir / "force.0.dump"
    if not dump_path.is_file():
        raise FileNotFoundError(f"LAMMPS finished but did not write {dump_path}")
    return _read_lammps_custom_dump(dump_path, specorder)


def run_lammps_query_generation(
    run_dir: Path, args: argparse.Namespace, output_xyz: Path, model_path: Path
) -> Path:
    from ase.data import atomic_numbers, chemical_symbols
    from ase.io import write

    runtime_path = run_dir / "dcbf.runtime.json"
    runtime_config = (
        json.loads(runtime_path.read_text(encoding="utf-8")) if runtime_path.is_file() else {}
    )
    scheduler = dict(runtime_config.get("scheduler") or {})
    parameter_path = run_dir / "init" / "parameter.yaml"
    if not parameter_path.is_file():
        raise FileNotFoundError(f"Missing generated parameter file: {parameter_path}")
    try:
        import yaml
        parameter = yaml.safe_load(parameter_path.read_text(encoding="utf-8")) or {}
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required for automatic query generation; pass --query explicitly"
        ) from exc
    elements = list(parameter.get("ele") or getattr(args, "elements", None) or [])
    if not elements:
        raise ValueError("Automatic query generation requires the model element order")
    sort_ele = bool(parameter.get("sort_ele", True))
    size = _parse_repeat_size(parameter.get("size", "(1, 1, 1)"))
    workflow = dict(runtime_config.get("workflow") or {})
    ensemble = "npt" if workflow.get("main_loop_npt") is not None else "nvt"
    temperature = _choose_lammps_temperature(runtime_config)
    specorder = (
        [chemical_symbols[number] for number in sorted(atomic_numbers[element] for element in elements)]
        if sort_ele
        else elements
    )
    sources = _select_query_structure_sources(
        _find_query_structure_sources(run_dir),
        getattr(args, "query_structures", ["all"]),
    )
    query_root = output_xyz.parent
    query_root.mkdir(parents=True, exist_ok=True)
    all_frames = []
    manifest = []
    for source in sources:
        work_dir = query_root if len(sources) == 1 else query_root / "runs" / source.label
        frames = _run_lammps_query_for_structure(
            run_dir,
            args,
            work_dir,
            source.atoms,
            size,
            specorder,
            scheduler,
            ensemble,
            temperature,
            model_path,
        )
        for frame in frames:
            frame.info["coverage_query_structure"] = source.label
            frame.info["coverage_query_source"] = str(source.path)
            frame.info["coverage_query_source_frame"] = int(source.frame_index)
        all_frames.extend(frames)
        manifest.append(
            {
                "index": source.index,
                "label": source.label,
                "source": str(source.path),
                "source_frame": source.frame_index,
                "work_dir": str(work_dir),
                "frames": len(frames),
            }
        )
    if not all_frames:
        raise ValueError("LAMMPS query generation produced no frames")
    if output_xyz.exists():
        output_xyz.unlink()
    write(str(output_xyz), all_frames, format="extxyz")
    (query_root / "query_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return output_xyz


def _ensure_lammps_query(run_dir: Path, args: argparse.Namespace, model_path: Path) -> Path:
    output_xyz = run_dir / "coverage_query_lammps" / "query.xyz"
    if output_xyz.is_file() and not getattr(args, "force_query", False):
        print(f"Using existing coverage LAMMPS query xyz: {output_xyz}")
        return output_xyz.resolve()
    for candidate_name in ("npt.xyz", "query.xyz", DEFAULT_QUERY_NAME):
        candidate = run_dir / candidate_name
        if candidate.is_file() and not getattr(args, "force_query", False):
            print(f"Using existing query xyz from run directory: {candidate}")
            return candidate.resolve()
    print(f"No existing query xyz found under {run_dir}; generating {output_xyz}")
    return run_lammps_query_generation(run_dir, args, output_xyz, model_path).resolve()


def _split_input_by_main(input_path: Path, main_key: str, selector: str, temporary_root: Path) -> List[Tuple[str, Path]]:
    from ase.io import iread, write

    frames = list(iread(str(input_path), index=":"))
    if not frames:
        raise ValueError(f"No structures found in {input_path}")
    labels = [frame.info.get(main_key) for frame in frames]
    if not any(label is not None for label in labels):
        return [("input", input_path)]
    # Match the historical PCA splitter: frames without the loop marker are
    # excluded from the loop datasets rather than silently assigned to loop 0.
    # Silent assignment contaminates cumulative coverage when a mixed xyz file
    # contains unlabeled query or auxiliary frames.
    labeled_frames = [(frame, label) for frame, label in zip(frames, labels) if label is not None]
    if not labeled_frames:
        return [("input", input_path)]
    try:
        numeric = [int(round(float(label))) for _, label in labeled_frames]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Frame info field {main_key!r} must be integer-like for loop splitting") from exc
    chosen = _select_loop_labels(numeric, selector)
    # The sampling plot is cumulative: the dataset shown at loop N contains
    # init and every loop through N, matching the historical DCBF splitter.
    temporary_root.mkdir(parents=True, exist_ok=True)
    result = []
    for label in chosen:
        selected = [
            frame for (frame, _), value in zip(labeled_frames, numeric)
            if value <= label
        ]
        if not selected:
            continue
        path = temporary_root / f"loop_{label}.xyz"
        write(str(path), selected, format="extxyz")
        result.append(("init" if label == -1 else f"loop-{label + 1}", path))
    if not result:
        raise ValueError("No input frames remain after --loop-select")
    return result


def _body_rows_to_arrays(rows, element_count: int) -> Dict[int, np.ndarray]:
    if isinstance(rows, (str, Path)):
        from .encode.mlp_encoding_extract import decode
        decoded = decode(rows)
        result = {}
        for element, raw in enumerate(decoded):
            array = np.asarray(raw, dtype=float)
            result[element] = array[:, :-1] if array.ndim == 2 and array.shape[1] else np.empty((0, 0))
        return result
    result = {}
    for element in range(element_count):
        row_set = rows[element]
        if not getattr(row_set, "parts", None):
            result[element] = np.empty((0, int(getattr(row_set, "dimensions", 0))), dtype=float)
            continue
        chunks = [np.asarray(values, dtype=float) for values, _ in row_set.parts]
        result[element] = chunks[0] if len(chunks) == 1 else np.vstack(chunks)
    return result


def _encode_dataset(reader, path: Path, label: str, scratch: Path):
    source = reader._encode_xyz_to_store(path)
    if source is None:
        fallback = scratch / label
        fallback.mkdir(parents=True, exist_ok=True)
        source = reader._descriptor_source(path, label, fallback)
    body_arrays = {}
    for body in reader.body_list:
        body_source = reader._descriptor_body(source, body)
        if body_source is None:
            raise ValueError(f"Descriptor body {body!r} is unavailable for {path}")
        body_arrays[body] = _body_rows_to_arrays(body_source, len(reader.elements))
    return body_arrays


def _combine_body_arrays(body_arrays: Dict[str, Dict[int, np.ndarray]], element: int, body_list: Sequence[str]):
    selected = []
    for body in body_list:
        array = body_arrays[body].get(element)
        if array is None or len(array) == 0:
            continue
        selected.append(array)
    if not selected:
        return np.empty((0, 0), dtype=float)
    rows = selected[0].shape[0]
    if any(array.shape[0] != rows for array in selected):
        raise ValueError(f"Descriptor body row counts do not agree for element index {element}")
    return np.hstack(selected)


def _write_points(path: Path, ref: np.ndarray, query: np.ndarray, covered: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path.with_name(path.stem + "_A.txt"),
               np.column_stack([ref[:, :2], np.ones(len(ref), dtype=int)]),
               fmt="%.8f %.8f %d", header="PC1 PC2 Is_Covered - reference", comments="")
    np.savetxt(path.with_name(path.stem + "_B.txt"),
               np.column_stack([query[:, :2], covered.astype(int)]),
               fmt="%.8f %.8f %d", header="PC1 PC2 Is_Covered - query", comments="")


def _coverage_for_element(element: str, index: int, query_bodies, input_bodies,
                          input_sets, last_label: str, body_list: Sequence[str],
                          pca_fit_source: str, coverage_grid: str,
                          width_factor: float, coverage_mode: str,
                          out_dir: Path):
    """Compute one element's coverage independently.

    This is a thread task rather than a process task deliberately: descriptor
    arrays are already disk-backed/owned by the caller, and threads avoid
    copying those arrays into every worker.  Output paths are disjoint by
    dataset/element, so the task is safe to run concurrently.
    """
    query = _combine_body_arrays(query_bodies, index, body_list)
    refs = {
        label: _combine_body_arrays(data, index, body_list)
        for label, data in input_bodies.items()
    }
    if len(query) == 0 or any(len(ref) == 0 for ref in refs.values()):
        return [], []

    pca = fit_pca_model(_fit_data(list(refs.values()), query, pca_fit_source), 2)
    query_pca = transform_pca(query, pca)
    transformed = {label: transform_pca(ref, pca) for label, ref in refs.items()}
    if coverage_grid == "query":
        grids = {label: query_pca for label in refs}
        raw_grids = {label: query for label in refs}
    elif coverage_grid == "current-loop":
        grids = transformed
        raw_grids = refs
    else:
        grids = {label: transformed[last_label] for label in refs}
        raw_grids = {label: refs[last_label] for label in refs}

    results = []
    summary_rows = []
    for label, ref in refs.items():
        ref_pca = transformed[label]
        coverage_2d, covered_2d = fd_coverage(
            ref_pca, query_pca, float(width_factor), grids[label]
        )
        coverage_1d, covered_1d_score = mean_1d_coverage(
            ref, query, float(width_factor), raw_grids[label]
        )
        body_coverage = {}
        for body in body_list:
            body_ref = input_bodies[label][body].get(index, np.empty((0, 0)))
            body_query = query_bodies[body].get(index, np.empty((0, 0)))
            if len(body_ref) and len(body_query):
                body_pca = fit_pca_model(
                    _fit_data([body_ref], body_query, pca_fit_source), 2
                )
                body_coverage[body] = fd_coverage(
                    transform_pca(body_ref, body_pca),
                    transform_pca(body_query, body_pca),
                    float(width_factor),
                    transform_pca(body_ref, body_pca),
                )[0]
        result = CoverageResult(
            label, element, index, ref_pca, query_pca, coverage_2d,
            coverage_1d, covered_2d, covered_1d_score, pca.explained,
            body_coverage,
        )
        results.append(result)
        _write_points(
            out_dir / "pca_txt" / label / element,
            ref_pca,
            query_pca,
            covered_2d if coverage_mode == "2d" else covered_1d_score > 0,
        )
        summary_rows.append({
            "dataset": label, "element": element, "element_index": index,
            "coverage_2d": f"{coverage_2d:.10f}",
            "coverage_2d_percent": f"{coverage_2d * 100:.6f}",
            "coverage_1d": f"{coverage_1d:.10f}",
            "coverage_1d_percent": f"{coverage_1d * 100:.6f}",
            "query_count": len(query), "input_count": len(ref),
            "explained_variance_2d": f"{pca.explained:.10f}",
            "body_coverage": json.dumps(body_coverage, sort_keys=True),
        })
    return results, summary_rows


def _downsample_indices(length: int, max_points: int, seed: int) -> np.ndarray:
    if max_points <= 0 or length <= max_points:
        return np.arange(length)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(length, size=max_points, replace=False))


def _finite_xy_limits(arrays: Sequence[np.ndarray], padding_fraction: float = 0.05):
    valid = [np.asarray(array, dtype=float)[:, :2] for array in arrays if len(array)]
    if not valid:
        return None
    data = np.vstack(valid)
    data = data[np.all(np.isfinite(data), axis=1)]
    if not len(data):
        return None
    mins = np.min(data, axis=0)
    maxs = np.max(data, axis=0)
    ranges = maxs - mins
    pads = np.where(ranges > 0, ranges * max(0.0, float(padding_fraction)), 1.0)
    return ((float(mins[0] - pads[0]), float(maxs[0] + pads[0])),
            (float(mins[1] - pads[1]), float(maxs[1] + pads[1])))


def _coverage_text(item: CoverageResult, mode: str) -> str:
    if mode == "1d":
        return f"{item.dataset} ({item.coverage_1d * 100:.2f}%)"
    return f"{item.dataset} ({item.coverage_2d * 100:.2f}%)"


def _plot(results: Sequence[CoverageResult], out_path: Path, coverage_mode: str, show_input: bool, max_points: int, dpi: int, show_ticks: bool) -> None:
    """Keep the historical DCBF PCA presentation while using SUS2 results.

    The colors, typography, point sizes and shared row limits intentionally
    follow the old deployment.  The descriptor backend changed to SUS2, but
    the figure is a compatibility output and must remain visually comparable.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.lines import Line2D

    if not results:
        return
    elements = list(dict.fromkeys(item.element for item in results))
    datasets = list(dict.fromkeys(item.dataset for item in results))
    try:
        font_manager.findfont("Times New Roman", fallback_to_default=False)
        plot_font = "Times New Roman"
    except Exception:
        plot_font = "DejaVu Serif"
    plt.rcParams.update({
        "font.family": plot_font,
        "font.size": 16,
        "axes.linewidth": 1.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
    })
    fig = plt.figure(figsize=(max(4.8 * len(datasets), 6), max(4.2 * len(elements), 4)),
                     constrained_layout=True)
    axes = fig.subplots(len(elements), len(datasets), squeeze=False)
    legend_items = [
        Line2D([0], [0], marker="o", linestyle="None", markersize=9,
               markerfacecolor="white", markeredgecolor="#2E7D32",
               markeredgewidth=1.3, label="Current loop"),
        Line2D([0], [0], marker="o", linestyle="None", markersize=9,
               markerfacecolor="white", markeredgecolor="#2B6CB0",
               markeredgewidth=1.3, label="Query covered"),
        Line2D([0], [0], marker="o", linestyle="None", markersize=9,
               markerfacecolor="white", markeredgecolor="#C53030",
               markeredgewidth=1.3, label="Query uncovered"),
    ]
    query_indices = {}
    limits_by_element = {}
    for row, element in enumerate(elements):
        arrays = []
        for item in results:
            if item.element != element:
                continue
            query_indices[element] = _downsample_indices(len(item.query_pca), max_points, 2000 + row * 17)
            arrays.append(item.query_pca)
            if show_input:
                arrays.append(item.ref_pca)
        limits = _finite_xy_limits(arrays)
        if limits is not None:
            limits_by_element[element] = limits
    for row, element in enumerate(elements):
        for col, dataset in enumerate(datasets):
            ax = axes[row][col]
            item = next((value for value in results if value.element == element and value.dataset == dataset), None)
            if item is None:
                ax.axis("off")
                continue
            ref = item.ref_pca
            query = item.query_pca
            covered = item.covered_2d if coverage_mode == "2d" else item.covered_1d_score > 0
            ref_idx = _downsample_indices(len(ref), max_points, 1000 + row * 17 + col)
            query_idx = query_indices[element]
            if show_input and len(ref):
                xy = ref[ref_idx]
                ax.scatter(xy[:, 0], xy[:, 1], s=24, facecolors="white",
                           edgecolors="#2E7D32", linewidths=0.9, alpha=0.8,
                           zorder=2, rasterized=True)
            xy = query[query_idx]
            mask = covered[query_idx]
            if np.any(~mask):
                bad = xy[~mask]
                ax.scatter(bad[:, 0], bad[:, 1], s=24, facecolors="white",
                           edgecolors="#C53030", linewidths=0.9, alpha=0.8,
                           zorder=1, rasterized=True)
            if np.any(mask):
                good = xy[mask]
                ax.scatter(good[:, 0], good[:, 1], s=24, facecolors="white",
                           edgecolors="#2B6CB0", linewidths=0.9, alpha=0.8,
                           zorder=3, rasterized=True)
            if col == 0:
                ax.text(0.04, 0.94, element, transform=ax.transAxes,
                        ha="left", va="top", fontsize=16, fontweight="bold")
            ax.text(0.04, 0.06, _coverage_text(item, coverage_mode),
                    transform=ax.transAxes, ha="left", va="bottom",
                    fontsize=14, fontweight="bold",
                    bbox={"boxstyle": "round,pad=0.25", "facecolor": "none",
                          "edgecolor": "black", "linewidth": 1.0, "alpha": 1.0})
            limits = limits_by_element.get(element)
            if limits is not None:
                ax.set_xlim(*limits[0])
                ax.set_ylim(*limits[1])
            ax.set_xlabel("")
            ax.set_ylabel("")
            if not show_ticks:
                ax.tick_params(axis="both", which="both", bottom=False, left=False,
                               labelbottom=False, labelleft=False)
    legend = axes[0][0].legend(handles=legend_items if show_input else legend_items[1:],
                               loc="upper right", fontsize=11, frameon=True)
    for text_item in legend.get_texts():
        text_item.set_fontweight("bold")
    legend.get_frame().set_alpha(0.0)
    legend.get_frame().set_linewidth(0.0)
    try:
        fig.supxlabel("PC 1", fontweight="bold")
        fig.supylabel("PC 2", fontweight="bold")
    except AttributeError:
        fig.text(0.5, 0.02, "PC 1", ha="center", va="center", fontweight="bold")
        fig.text(0.02, 0.5, "PC 2", ha="center", va="center", rotation="vertical", fontweight="bold")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image_format = (out_path.suffix.lstrip(".") or "png").lower()
    handle = tempfile.NamedTemporaryFile(
        "wb", dir=str(out_path.parent), prefix=f".{out_path.name}.", suffix=".png", delete=False
    )
    temporary = Path(handle.name)
    handle.close()
    try:
        fig.savefig(temporary, format=image_format, dpi=int(dpi),
                    bbox_inches="tight", facecolor="white")
        os.replace(temporary, out_path)
    finally:
        plt.close(fig)
        if temporary.exists():
            with contextlib.suppress(OSError):
                temporary.unlink()


def _make_reader(config_dir: Path, input_path: Path, model: str, elements: Optional[Sequence[str]], body_list: Sequence[str], workers: int, out_dir: Path, mlp_exe: Optional[str] = None, config_path: Optional[Path] = None):
    from .reduce import DCBFReducer

    config = {
        "parameter": {"body_list": list(body_list), "dq_width_method": "Freedman_Diaconis"},
        "reduce": {
            "selection_mode": "candidate_only",
            "candidate_pool_xyz": str(input_path),
            "starting_model_path": str(model),
            "work_dir": str(out_dir / "descriptor_work"),
            "selected_structures_xyz": str(out_dir / "unused.xyz"),
            "remaining_candidates_xyz": None,
            "report_json": str(out_dir / "unused.json"),
            "encoding_cores": int(workers),
            "dimension_min_cover_workers": 0,
            "xyz_io_mode": "auto",
            "keep_intermediate": False,
        },
        "scheduler": ({"sus2_mlp_exe": str(mlp_exe)} if mlp_exe else {}),
    }
    if elements:
        config["parameter"]["ele"] = list(elements)
        config["parameter"]["sort_ele"] = False
    # The config is an internal wiring file; callers pass a temporary location so
    # absolute scratch paths never leak into the persistent output directory.
    if config_path is None:
        config_path = out_dir / "coverage_descriptor_config.json"
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return DCBFReducer(config_path)


def add_coverage_pca_subparser(subparsers):
    parser = subparsers.add_parser(
        "coverage-pca",
        help="plot SUS2 descriptor PCA coverage for input datasets against a query dataset",
        description=(
            "Compute SUS2 metadata-backed descriptors, fit a shared PC1/PC2 model, "
            "calculate 1D/2D Freedman-Diaconis coverage, and write PCA figures."
        ),
        formatter_class=RawDefaultsHelpFormatter,
        epilog=command_example_epilog("coverage-pca"),
    )
    parser.add_argument("--input", help="input xyz/extxyz; frames with --main-key may be split into loop datasets")
    parser.add_argument(
        "--query",
        help=(
            "query xyz/extxyz dataset; when omitted with --run-dir, an existing query is reused "
            "or generated from run_dir/stru with the configured SUS2 LAMMPS model"
        ),
    )
    parser.add_argument("--run-dir", help="prepared DCBF workspace used to resolve input/query/model defaults")
    parser.add_argument("--model", default="builtin:sus2_87", help="SUS2 model path or builtin:sus2_87")
    parser.add_argument("--elements", nargs="+", help="complete model element order; omit only for a registered bundled model")
    parser.add_argument(
        "--body-list",
        "--data-modes",
        dest="body_list",
        nargs="+",
        choices=list(BODY_CHOICES),
        default=DEFAULT_BODY_LIST,
        help="outer SUS2 body orders to combine; --data-modes is a compatibility alias",
    )
    parser.add_argument("--main-key", default="main", help="frame info key used for loop splitting")
    parser.add_argument("--loop-select", default="all", help="all, middle-half, uniform-half, or comma-separated main values")
    parser.add_argument(
        "--out-dir",
        default="xyz_pca_coverage_results",
        help=(
            "directory for CSV, PCA text, and figure outputs; a relative path is "
            "resolved against --run-dir when --run-dir is given, otherwise against "
            "the current working directory"
        ),
    )
    parser.add_argument("--pca-fit-source", choices=["query", "input", "combined"], default="query")
    parser.add_argument("--coverage-grid", choices=["query", "current-loop", "last-loop"], default=DEFAULT_COVERAGE_GRID)
    parser.add_argument(
        "--coverage-mode",
        "--coverage-label",
        dest="coverage_mode",
        choices=["1d", "2d"],
        default="2d",
        help="primary displayed coverage; --coverage-label is a compatibility alias",
    )
    parser.add_argument("--width-factor", type=float, default=1.0, help="Freedman-Diaconis width multiplier")
    parser.add_argument("--descriptor-workers", type=int, default=4, help="calc-descriptors/descriptor parsing workers")
    parser.add_argument("--coverage-workers", type=int, default=4, help="parallel PCA/coverage calculation workers")
    parser.add_argument("--mlp-exe", default=None, help="optional mlp-sus2 executable override")
    parser.add_argument(
        "--query-structures",
        nargs="+",
        default=["all"],
        help="structures under run_dir/stru: all, first, index:N, file/stem/label, or glob",
    )
    parser.add_argument(
        "--lammps-run-mode",
        choices=["scheduler", "local"],
        default="scheduler",
        help="automatic query generation backend; local is intended for explicit debugging",
    )
    parser.add_argument(
        "--lammps-timeout-hours",
        type=float,
        default=DEFAULT_LAMMPS_TIMEOUT_HOURS,
        help="hours to wait for scheduler LAMMPS query generation",
    )
    parser.add_argument("--lammps-timeout", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--lammps-exe", default=None, help="LAMMPS command for automatic query generation")
    parser.add_argument("--lammps-env", default=None, help="shell setup command for automatic query generation")
    parser.add_argument(
        "--lammps-cores",
        type=int,
        default=None,
        help="LAMMPS core count for automatic query generation; defaults to scheduler.lmp_cores",
    )
    parser.add_argument(
        "--force-query",
        action="store_true",
        help="regenerate coverage_query_lammps/query.xyz instead of reusing an existing query",
    )
    parser.add_argument("--plot-elements", nargs="+", default=None)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--show-input", action="store_true")
    parser.add_argument("--max-plot-points", type=int, default=200000)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--show-ticks", action="store_true")
    return parser


def _resolve_args(args: argparse.Namespace) -> Tuple[Path, Optional[Path], str, Optional[List[str]], Path]:
    """Resolve input/query/model/out-dir without touching the filesystem.

    ``query_path`` is None when it still has to be generated from ``run_dir/stru``;
    main_from_args does that only after every cheap argument check has passed so
    a bad worker/body/element argument never creates ``--out-dir`` or starts a
    scheduler LAMMPS job.
    """
    run_dir = Path(args.run_dir).expanduser().resolve() if args.run_dir else None
    default_input, default_query, runtime_elements = _load_runtime_run_defaults(run_dir) if run_dir else (None, None, None)
    input_path = _resolve_run_path(run_dir, args.input) or default_input
    query_path = _resolve_run_path(run_dir, args.query) or default_query
    model = args.model
    if run_dir and model in ("builtin:sus2_87", "builtin:NEP_data"):
        latest = _latest_model(run_dir)
        if latest is not None:
            model = str(latest)
    if input_path is None or not input_path.is_file():
        raise ValueError("coverage-pca requires --input, or --run-dir with sampled_training_dataset.xyz")
    if query_path is None or not query_path.is_file():
        if run_dir and args.query is None:
            query_path = None
        else:
            raise ValueError(
                "coverage-pca requires --query, or --run-dir with an existing/generated query"
            )
    elements = list(args.elements) if args.elements else runtime_elements
    out_dir = Path(args.out_dir).expanduser()
    out_dir = (run_dir / out_dir).resolve() if run_dir and not out_dir.is_absolute() else out_dir.resolve()
    return input_path, query_path, str(model), elements, out_dir


def main_from_args(args: argparse.Namespace) -> int:
    from .builtin_models import registered_elements, resolve_model_path

    input_path, query_path, raw_model, elements, out_dir = _resolve_args(args)
    run_dir = Path(args.run_dir).expanduser().resolve() if args.run_dir else None
    model = resolve_model_path(str(raw_model), run_dir or Path.cwd())
    registered = registered_elements(model, elements)
    if registered is not None:
        elements = registered
    if elements is None:
        raise ValueError("A custom SUS2 model requires the complete --elements order")
    if int(args.descriptor_workers) <= 0:
        raise ValueError("--descriptor-workers must be > 0")
    if int(args.coverage_workers) <= 0:
        raise ValueError("--coverage-workers must be > 0")
    if not args.body_list:
        raise ValueError("--body-list must not be empty")
    invalid_bodies = [body for body in args.body_list if body not in BODY_CHOICES]
    if invalid_bodies:
        raise ValueError(f"Unsupported --body-list entries: {invalid_bodies}")

    cpu_count = os.cpu_count() or 1
    descriptor_workers = min(int(args.descriptor_workers), cpu_count)
    coverage_workers = min(int(args.coverage_workers), cpu_count)

    if query_path is None:
        if run_dir is None:
            raise ValueError("coverage-pca requires --query when --run-dir is omitted")
        query_path = _ensure_lammps_query(run_dir, args, Path(model))

    created_dirs = _missing_dir_chain(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        return _run_coverage(args, input_path, query_path, model, elements, out_dir,
                             descriptor_workers, coverage_workers)
    except BaseException:
        _remove_created_empty_dirs(created_dirs)
        raise


def _run_coverage(args, input_path, query_path, model, elements, out_dir,
                  descriptor_workers, coverage_workers) -> int:
    summary_rows: List[Dict] = []
    results: List[CoverageResult] = []
    with tempfile.TemporaryDirectory(prefix="coverage_pca_", dir=str(out_dir)) as temp_name:
        temporary_root = Path(temp_name)
        input_sets = _split_input_by_main(input_path, args.main_key, args.loop_select, temporary_root)
        reader = _make_reader(
            out_dir,
            input_sets[0][1],
            str(model),
            elements,
            args.body_list,
            descriptor_workers,
            out_dir,
            args.mlp_exe,
            config_path=temporary_root / "coverage_descriptor_config.json",
        )
        scratch = temporary_root / "fallback"
        scratch.mkdir()
        query_bodies = _encode_dataset(reader, query_path, "query", scratch)
        input_bodies = {label: _encode_dataset(reader, path, label.replace("/", "_"), scratch) for label, path in input_sets}

        try:
            element_index = {element: index for index, element in enumerate(elements)}
            plot_elements = list(args.plot_elements or elements)
            unknown = [element for element in plot_elements if element not in element_index]
            if unknown:
                raise ValueError(f"Unknown --plot-elements: {unknown}")
            last_label = input_sets[-1][0]
            element_jobs = [
                (element, element_index[element]) for element in plot_elements
            ]
            worker_count = min(int(coverage_workers), max(1, len(element_jobs)))
            def compute(job):
                element, index = job
                return _coverage_for_element(
                    element, index, query_bodies, input_bodies, input_sets,
                    last_label, args.body_list, args.pca_fit_source,
                    args.coverage_grid, args.width_factor, args.coverage_mode,
                    out_dir,
                )
            if worker_count == 1:
                computed = [compute(job) for job in element_jobs]
            else:
                with ThreadPoolExecutor(max_workers=worker_count) as executor:
                    computed = list(executor.map(compute, element_jobs))
            for element_results, element_rows in computed:
                results.extend(element_results)
                summary_rows.extend(element_rows)
        finally:
            for source in list(getattr(reader, "_descriptor_sources", {}).values()):
                if hasattr(source, "release_all"):
                    source.release_all()

    fields = ["dataset", "element", "element_index", "coverage_2d", "coverage_2d_percent", "coverage_1d", "coverage_1d_percent", "query_count", "input_count", "explained_variance_2d", "body_coverage"]

    def write_summary(handle):
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    _atomic_write_text(out_dir / "coverage_summary.csv", write_summary)
    _atomic_write_text(
        out_dir / "coverage_remark.txt",
        lambda handle: handle.write(
            "PCA/coverage uses SUS2 sh_scalar_info body columns through the shared DescriptorStore. "
            "It is an analysis plot and does not control sampling selection.\n"
        ),
    )
    if not args.no_plot:
        _plot(results, out_dir / "pca_coverage.png", args.coverage_mode, args.show_input, args.max_plot_points, args.dpi, args.show_ticks)
    print(f"SUS2 PCA coverage completed: datasets={len(input_sets)} results={len(results)} output={out_dir}")
    return 0


def handle_coverage_pca_command(args: argparse.Namespace) -> int:
    return main_from_args(args)


def run_coverage_pca_from_config(config: dict, run_dir: Path) -> int:
    """Run the automatic post-sampling PCA/coverage analysis for a workspace."""
    coverage = dict(config.get("coverage_plot") or {})
    if not coverage.get("enabled", False):
        return 0
    namespace = argparse.Namespace(
        input=coverage.get("input"), query=coverage.get("query"), run_dir=str(run_dir),
        model=coverage.get("model", "builtin:sus2_87"), elements=coverage.get("elements"),
        body_list=coverage.get("body_list", coverage.get("data_modes", DEFAULT_BODY_LIST)),
        main_key=coverage.get("main_key", "main"),
        loop_select=coverage.get("loop_select", "middle-half"), out_dir=coverage.get("output_dir", "xyz_pca_coverage_results"),
        pca_fit_source=coverage.get("pca_fit_source", "query"), coverage_grid=coverage.get("coverage_grid", DEFAULT_COVERAGE_GRID),
        coverage_mode=coverage.get("coverage_mode", coverage.get("coverage_label", "2d")),
        width_factor=float(coverage.get("width_factor", 1.0)),
        descriptor_workers=int(coverage.get("descriptor_workers", 4)),
        coverage_workers=int(coverage.get("coverage_workers", 4)),
        mlp_exe=coverage.get("mlp_exe"),
        plot_elements=coverage.get("plot_elements"), no_plot=bool(coverage.get("no_plot", False)),
        show_input=bool(coverage.get("show_input", False)), max_plot_points=int(coverage.get("max_plot_points", 200000)),
        dpi=int(coverage.get("dpi", 300)), show_ticks=bool(coverage.get("show_ticks", False)),
        query_structures=coverage.get("query_structures", ["all"]),
        lammps_run_mode=coverage.get("lammps_run_mode", "scheduler"),
        lammps_timeout_hours=float(coverage.get("lammps_timeout_hours", DEFAULT_LAMMPS_TIMEOUT_HOURS)),
        lammps_timeout=coverage.get("lammps_timeout"),
        lammps_exe=coverage.get("lammps_exe"), lammps_env=coverage.get("lammps_env"),
        lammps_cores=coverage.get("lammps_cores"), force_query=bool(coverage.get("force_query", False)),
    )
    return main_from_args(namespace)


__all__ = ["add_coverage_pca_subparser", "handle_coverage_pca_command", "run_coverage_pca_from_config"]
