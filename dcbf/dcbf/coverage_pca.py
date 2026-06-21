import argparse
import ast
import csv
import fnmatch
import glob
import importlib.util
import json
import math
import os
import pickle
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
import zlib
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import yaml
except ImportError:  # pragma: no cover - standalone installs may not need --run-dir mode.
    yaml = None


DEFAULT_DATA_MODES = ["two", "three"]
DEFAULT_MLP_EXE = None
DEFAULT_MTP = "current.mtp"
DEFAULT_QUERY = "sus2md_1000.xyz"
DEFAULT_DESCRIPTOR_WORKERS = 8
DEFAULT_COVERAGE_WORKERS = 8
DEFAULT_COVERAGE_GRID = "last-loop"
DEFAULT_WIDTH_FACTOR_1D = 1.0
DEFAULT_WIDTH_FACTOR_2D = 2.0
DEFAULT_LAMMPS_RUN_MODE = "scheduler"
DEFAULT_LAMMPS_TIMEOUT_HOURS = 24.0
DEFAULT_LAMMPS_POLL_SECONDS = 15


@dataclass
class DatasetSpec:
    label: str
    source: Path
    descriptor_dir: Path
    converted: bool


@dataclass
class CoverageResult:
    dataset_label: str
    element: str
    ele_type: int
    coverage_2d: float
    coverage_1d: float
    coverage_1d_by_mode: Dict[str, float]
    covered_2d_count: int
    covered_1d_mean_count: float
    covered_1d_mean_count_by_mode: Dict[str, float]
    query_count: int
    input_count: int
    explained_variance_2d: float
    ref_pca: np.ndarray
    query_pca: np.ndarray
    covered_2d_bool: np.ndarray
    covered_1d_scores: np.ndarray
    covered_1d_display_bool: Optional[np.ndarray] = None


@dataclass
class PcaModel:
    mean: np.ndarray
    std: np.ndarray
    components: np.ndarray
    explained: float


@dataclass
class QueryStructureSource:
    index: int
    label: str
    path: Path
    frame_index: int
    atoms: object


def parse_label_path(text: str) -> Tuple[str, Path]:
    if "=" in text:
        label, path_text = text.split("=", 1)
        label = label.strip()
        if not label:
            raise ValueError(f"Empty label in dataset argument: {text}")
        return label, Path(path_text.strip())

    path = Path(text)
    return path.stem if path.suffix else path.name, path


def parse_label_value(text: str) -> Tuple[str, str]:
    if "=" not in text:
        raise ValueError(f"Expected LABEL=VALUE, got: {text}")
    label, value = text.split("=", 1)
    label = label.strip()
    value = value.strip()
    if not label:
        raise ValueError(f"Empty label in split mapping: {text}")
    if not value:
        raise ValueError(f"Empty value in split mapping: {text}")
    return label, value


def parse_label_values(items: Sequence[str]) -> List[Tuple[str, str]]:
    expanded: List[str] = []
    for item in items:
        expanded.extend(part.strip() for part in item.split(",") if part.strip())
    if not expanded:
        raise ValueError("No LABEL=VALUE mappings were provided.")
    return [parse_label_value(item) for item in expanded]


def build_display_label_map(dataset_labels: Sequence[str]) -> Dict[str, str]:
    return {label: label for label in dataset_labels}


def safe_filename(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip())
    return cleaned or "dataset"


def scalar_to_string(value) -> str:
    arr = np.asarray(value)
    if arr.shape == ():
        value = arr.item()
    elif arr.size == 1:
        value = arr.reshape(-1)[0].item()
    return str(value).strip()


def info_value_matches(actual, expected: str) -> bool:
    if actual is None:
        return False

    actual_text = scalar_to_string(actual)
    expected_text = str(expected).strip()

    try:
        return np.isclose(float(actual_text), float(expected_text))
    except (TypeError, ValueError):
        return actual_text == expected_text


def atoms_info_get(atoms, key: str):
    if key in atoms.info:
        return atoms.info[key]
    key_lower = key.lower()
    for info_key, value in atoms.info.items():
        if str(info_key).lower() == key_lower:
            return value
    return None


def sort_elements_by_atomic_number(elements: Iterable[str]) -> List[str]:
    from ase.data import atomic_numbers

    unique = {str(element) for element in elements if str(element).strip()}
    return sorted(unique, key=lambda element: (atomic_numbers.get(element, 10**6), element))


def collect_elements_from_xyz(source: Path, script_dir: Path) -> List[str]:
    from ase.io import iread

    source = resolve_dataset_path(source, script_dir)
    if not source.exists() or not source.is_file():
        return []

    elements = set()
    try:
        for atoms in iread(str(source), index=":"):
            elements.update(atoms.get_chemical_symbols())
    except Exception:
        return []
    return sort_elements_by_atomic_number(elements)


def resolve_path(path: Path, base_dir: Path) -> Path:
    if path.exists():
        return path.resolve()
    base_path = base_dir / path
    if base_path.exists():
        return base_path.resolve()
    return path


def resolve_default_mlp_exe(script_dir: Path) -> Path:
    for env_name in ("DCBF_V3_ROOT", "DCBF_DEPLOYMENT_ROOT"):
        root = os.environ.get(env_name)
        if root:
            candidate = Path(root) / "runtime" / "bin" / "mlp-sus2"
            if candidate.exists():
                return candidate.resolve()

    for root in (script_dir, *script_dir.parents):
        for candidate in (
            root / "runtime" / "bin" / "mlp-sus2",
            root / "bin" / "mlp-sus2" if root.name == "runtime" else None,
        ):
            if candidate is not None and candidate.exists():
                return candidate.resolve()

    path_exe = shutil.which("mlp-sus2")
    if path_exe:
        return Path(path_exe).resolve()
    return Path("mlp-sus2")


def resolve_dataset_path(path: Path, base_dir: Path) -> Path:
    resolved = resolve_path(path, base_dir)
    if resolved.exists():
        return resolved

    if path.suffix:
        return resolved

    for suffix in (".xyz", ".traj"):
        candidate = resolve_path(Path(f"{path}{suffix}"), base_dir)
        if candidate.exists():
            return candidate
    return resolved


def decode_zlib_pickle(filepath: Path):
    with filepath.open("rb") as f:
        return pickle.loads(zlib.decompress(f.read()))


def encode_zlib_pickle(obj, filepath: Path) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with filepath.open("wb") as f:
        f.write(zlib.compress(pickle.dumps(obj)))


def has_descriptor_pickles(path: Path, data_modes: Sequence[str]) -> bool:
    if not path.is_dir():
        return False
    return all(glob.glob(str(path / f"*{mode}*coding_zlib.pkl")) for mode in data_modes)


def split_xyz_by_info_label(
    source: Path,
    info_key: str,
    label_values: Sequence[Tuple[str, str]],
    out_dir: Path,
    script_dir: Path,
    selected_labels: Optional[Sequence[str]] = None,
) -> List[Tuple[str, Path]]:
    from ase.io import iread, write

    source = resolve_dataset_path(source, script_dir)
    if not source.exists():
        raise FileNotFoundError(f"Split source xyz/traj not found: {source}")
    if not source.is_file():
        raise ValueError(f"Split source must be a single xyz/traj file: {source}")

    selected_label_set = set(selected_labels or [label for label, _ in label_values])
    unknown_labels = [label for label in selected_label_set if label not in {item[0] for item in label_values}]
    if unknown_labels:
        raise ValueError(f"Selected labels are not present in split mappings: {unknown_labels}")

    grouped = {label: [] for label, _ in label_values}
    missing_key_count = 0
    unmatched_count = 0
    matched_unique_count = 0
    total_count = 0

    for atoms in iread(str(source)):
        total_count += 1
        actual = atoms_info_get(atoms, info_key)
        if actual is None:
            missing_key_count += 1
            continue

        matched_index = None
        for index, (_, expected) in enumerate(label_values):
            if info_value_matches(actual, expected):
                matched_index = index
                break
        if matched_index is None:
            unmatched_count += 1
            continue

        matched_unique_count += 1
        for label, _ in label_values[matched_index:]:
            grouped[label].append(atoms)

    split_dir = out_dir / "split_xyz"
    split_dir.mkdir(parents=True, exist_ok=True)

    dataset_items: List[Tuple[str, Path]] = []
    for index, (label, expected) in enumerate(label_values):
        if label not in selected_label_set:
            continue
        frames = grouped[label]
        if not frames:
            raise ValueError(
                f"No frames matched cumulative {info_key} values through {expected!r} "
                f"for label {label!r} in {source}"
            )
        split_path = split_dir / f"{safe_filename(label)}.xyz"
        write(str(split_path), frames, format="extxyz")
        dataset_items.append((label, split_path))
        included_values = ", ".join(value for _, value in label_values[: index + 1])
        print(
            f"Split {label}: {len(frames)} cumulative frame(s), "
            f"{info_key} in [{included_values}], file={split_path}"
        )

    print(
        f"Split source summary: total={total_count}, "
        f"matched_unique={matched_unique_count}, "
        f"missing_{info_key}={missing_key_count}, unmatched={unmatched_count}"
    )
    return dataset_items


def main_value_to_loop_label(value: int) -> str:
    if value == -1:
        return "init"
    if value >= 0:
        return f"loop-{value + 1}"
    return f"main-{value}"


def canonical_loop_label(label: str) -> str:
    text = str(label).strip()
    if text == "init":
        return text
    match = re.fullmatch(r"loop[-_]?n?(\d+)", text)
    if match:
        return f"loop-n{int(match.group(1))}"
    return text


def loop_label_to_main_value(label: str) -> int:
    label = canonical_loop_label(label)
    if label == "init":
        return -1
    match = re.fullmatch(r"loop-n(\d+)", label)
    if not match:
        raise ValueError(f"Expected loop label like init or loop-n1, got: {label}")
    loop_index = int(match.group(1))
    if loop_index < 1:
        raise ValueError(f"Loop label index must be >= 1, got: {label}")
    return loop_index - 1


def discover_main_label_values(source: Path, info_key: str, script_dir: Path) -> List[Tuple[str, str]]:
    from ase.io import iread

    source = resolve_dataset_path(source, script_dir)
    if not source.exists():
        raise FileNotFoundError(f"Input xyz/traj not found: {source}")

    discovered = set()
    missing_key_count = 0
    for atoms in iread(str(source)):
        value = atoms_info_get(atoms, info_key)
        if value is None:
            missing_key_count += 1
            continue
        text = scalar_to_string(value)
        try:
            numeric = float(text)
        except ValueError as exc:
            raise ValueError(f"Auto loop discovery requires numeric {info_key}=... values, got {text!r}") from exc
        rounded = int(round(numeric))
        if not np.isclose(numeric, rounded):
            raise ValueError(f"Auto loop discovery requires integer-like {info_key}=... values, got {text!r}")
        discovered.add(rounded)

    if not discovered:
        raise ValueError(f"No frames with info key {info_key!r} were found in {source}; pass --main-key if the loop field has another name.")

    ordered_values = []
    if -1 in discovered:
        ordered_values.append(-1)
    ordered_values.extend(value for value in sorted(discovered) if value != -1)
    label_values = [(main_value_to_loop_label(value), str(value)) for value in ordered_values]
    print(
        f"Discovered {info_key} labels from {source}: "
        f"{[f'{label}={value}' for label, value in label_values]} "
        f"(missing_{info_key}={missing_key_count})"
    )
    return label_values


def parse_manual_main_values(text: Optional[str]) -> Optional[List[int]]:
    if not text:
        return None
    stripped = str(text).strip()
    if stripped in {"all", "middle-half", "uniform-half"}:
        return None
    if stripped.startswith("[") and stripped.endswith("]"):
        stripped = stripped[1:-1]
    values = []
    for chunk in re.split(r"[\s,]+", stripped):
        chunk = chunk.strip()
        if not chunk:
            continue
        lowered = chunk.lower()
        if lowered == "init":
            values.append(-1)
            continue
        match = re.fullmatch(r"loop[-_]?n?(\d+)", lowered)
        if match:
            values.append(int(match.group(1)) - 1)
            continue
        try:
            values.append(int(chunk))
        except ValueError as exc:
            raise ValueError(
                "--loop-select must be one of all/middle-half/uniform-half or a main-value list like [-1,0,2,8]"
            ) from exc
    if not values:
        raise ValueError("--loop-select manual list is empty.")
    return values


def unique_preserve_order(items: Sequence[str]) -> List[str]:
    seen = set()
    result = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def select_loop_labels(
    label_values: Sequence[Tuple[str, str]],
    loop_select: str,
) -> List[str]:
    value_to_label: Dict[int, str] = {}
    for label, value in label_values:
        try:
            main_value = int(round(float(value)))
        except ValueError as exc:
            raise ValueError(f"Expected integer-like main value for {label}: {value!r}") from exc
        value_to_label[main_value] = label

    explicit_values = parse_manual_main_values(loop_select)
    if explicit_values is not None:
        missing = [value for value in explicit_values if value not in value_to_label]
        if missing:
            available = sorted(value_to_label)
            raise ValueError(f"--loop-select requested main values not present: {missing}; available={available}")
        return [value_to_label[value] for value in unique_preserve_order(explicit_values)]

    loop_select = str(loop_select).strip()
    if loop_select not in {"all", "middle-half", "uniform-half"}:
        raise ValueError(
            "--loop-select must be one of all/middle-half/uniform-half or a main-value list like [-1,0,2,8]"
        )

    init_label = value_to_label.get(-1)
    loop_labels = [value_to_label[value] for value in sorted(value for value in value_to_label if value >= 0)]
    if loop_select == "all" or len(loop_labels) <= 3:
        selected = ([init_label] if init_label else []) + loop_labels
        return [item for item in selected if item is not None]

    if loop_select == "middle-half":
        last_position = len(loop_labels) - 1
        start = int(math.ceil(0.25 * last_position))
        end = int(math.floor(0.75 * last_position))
        selected = ([init_label] if init_label else []) + loop_labels[start : end + 1] + [loop_labels[-1]]
        return unique_preserve_order([item for item in selected if item is not None])

    if loop_select == "uniform-half":
        keep_count = max(1, int(math.ceil(len(loop_labels) / 2)))
        positions = np.linspace(0, len(loop_labels) - 1, keep_count).round().astype(int).tolist()
        selected = ([init_label] if init_label else []) + [loop_labels[index] for index in positions] + [loop_labels[-1]]
        return unique_preserve_order([item for item in selected if item is not None])

    raise ValueError(f"Unsupported --loop-select value: {loop_select}")


def read_json_if_exists(path: Path) -> Dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_parameter_yaml(path: Path) -> Dict:
    if yaml is None:
        raise RuntimeError(f"PyYAML is required to read {path}; install pyyaml or pass --query explicitly.")
    if not path.exists():
        raise FileNotFoundError(f"parameter.yaml not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def parse_repeat_size(value) -> Tuple[int, int, int]:
    if value is None:
        return (1, 1, 1)
    if isinstance(value, (list, tuple)):
        parts = [int(item) for item in value]
    else:
        try:
            parsed = ast.literal_eval(str(value))
            if isinstance(parsed, (list, tuple)):
                parts = [int(item) for item in parsed]
            else:
                parts = [int(parsed)]
        except Exception:
            parts = [int(item) for item in re.findall(r"-?\d+", str(value))]
    if len(parts) == 1:
        parts = parts * 3
    if len(parts) != 3:
        raise ValueError(f"Could not parse repeat size as three integers: {value!r}")
    return tuple(parts)  # type: ignore[return-value]


def numeric_child_dirs(parent: Path, prefix: str) -> List[Tuple[int, Path]]:
    if not parent.exists():
        return []
    result = []
    for child in parent.iterdir():
        if not child.is_dir() or not child.name.startswith(prefix):
            continue
        suffix = child.name[len(prefix) :]
        if suffix.isdigit():
            result.append((int(suffix), child))
    return sorted(result, key=lambda item: item[0])


def find_default_mtp(run_dir: Path) -> Optional[Path]:
    generation_dirs = []
    for _, main_dir in numeric_child_dirs(run_dir, "main_"):
        for _, gen_dir in numeric_child_dirs(main_dir, "gen_"):
            generation_dirs.append(gen_dir)
    for gen_dir in reversed(generation_dirs):
        for rel in ("sus2/current_0.mtp", "train_mlp/current_0.mtp", "current_0.mtp"):
            candidate = gen_dir / rel
            if candidate.exists():
                return candidate.resolve()
    for candidate in (run_dir / "current.mtp", run_dir / "current_0.mtp"):
        if candidate.exists():
            return candidate.resolve()
    return None


def _query_structure_files(run_dir: Path) -> List[Path]:
    stru_dir = run_dir / "stru"
    if not stru_dir.exists():
        raise FileNotFoundError(f"Structure directory not found for LAMMPS query generation: {stru_dir}")
    preferred = [stru_dir / "POSCAR", stru_dir / "CONTCAR"]
    preferred.extend(sorted(stru_dir.glob("*.vasp")))
    preferred.extend(sorted(stru_dir.glob("*.cif")))
    preferred.extend(sorted(stru_dir.glob("*.xyz")))
    return [candidate for candidate in preferred if candidate.exists() and candidate.is_file()]


def _safe_query_structure_label(text: str) -> str:
    label = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("._")
    return label or "structure"


def find_structure_sources(run_dir: Path) -> List[QueryStructureSource]:
    from ase.io import iread

    structure_files = _query_structure_files(run_dir)
    if not structure_files:
        raise FileNotFoundError(f"No POSCAR/CONTCAR/*.vasp/*.cif/*.xyz found in {run_dir / 'stru'}")

    sources: List[QueryStructureSource] = []
    seen_labels: Dict[str, int] = {}
    for path in structure_files:
        frames = [atoms.copy() for atoms in iread(str(path), index=":")]
        if not frames:
            continue
        for frame_index, atoms in enumerate(frames):
            base_label = _safe_query_structure_label(path.stem if len(frames) == 1 else f"{path.stem}_f{frame_index}")
            duplicate_index = seen_labels.get(base_label, 0)
            seen_labels[base_label] = duplicate_index + 1
            label = base_label if duplicate_index == 0 else f"{base_label}_{duplicate_index}"
            sources.append(
                QueryStructureSource(
                    index=len(sources),
                    label=label,
                    path=path,
                    frame_index=frame_index,
                    atoms=atoms,
                )
            )
    if not sources:
        raise ValueError(f"No readable structures were found in {run_dir / 'stru'}")
    return sources


def normalize_query_structure_selectors(raw_selectors) -> List[str]:
    if raw_selectors is None:
        return ["all"]
    if isinstance(raw_selectors, str):
        items = [raw_selectors]
    else:
        items = [str(item) for item in raw_selectors]
    selectors: List[str] = []
    for item in items:
        for part in str(item).split(","):
            part = part.strip()
            if part:
                selectors.append(part)
    return selectors or ["all"]


def select_query_structure_sources(sources: List[QueryStructureSource], raw_selectors) -> List[QueryStructureSource]:
    selectors = normalize_query_structure_selectors(raw_selectors)
    lowered = [item.lower() for item in selectors]
    if "all" in lowered:
        return list(sources)
    if lowered == ["first"]:
        return [sources[0]]

    selected: List[QueryStructureSource] = []
    selected_indices = set()
    available = []
    for source in sources:
        available.append(
            f"{source.index}:{source.label}({source.path.name}"
            + (f":frame{source.frame_index}" if source.frame_index else "")
            + ")"
        )

    for selector in selectors:
        matches: List[QueryStructureSource] = []
        selector_lower = selector.lower()
        if selector_lower == "first":
            matches = [sources[0]]
        elif selector_lower.startswith("index:"):
            try:
                wanted_index = int(selector.split(":", 1)[1])
            except ValueError as exc:
                raise ValueError(f"Invalid query structure selector {selector!r}; index selectors use index:N") from exc
            matches = [source for source in sources if source.index == wanted_index]
        else:
            for source in sources:
                identifiers = {
                    source.label,
                    source.path.name,
                    source.path.stem,
                    f"{source.path.name}:{source.frame_index}",
                    f"{source.label}:{source.frame_index}",
                }
                if any(selector == value for value in identifiers):
                    matches.append(source)
                    continue
                if any(fnmatch.fnmatchcase(value, selector) for value in identifiers):
                    matches.append(source)
        if not matches:
            raise ValueError(
                f"Query structure selector {selector!r} matched nothing. "
                f"Available structures: {', '.join(available)}"
            )
        for source in matches:
            if source.index not in selected_indices:
                selected.append(source)
                selected_indices.add(source.index)
    return selected


def find_structure_source(run_dir: Path) -> Path:
    # Backward-compatible helper for older callers: return the first selected source file.
    for candidate in _query_structure_files(run_dir):
        return candidate
    raise FileNotFoundError(f"No POSCAR/CONTCAR/*.vasp/*.cif/*.xyz found in {run_dir / 'stru'}")


def flatten_temperature_values(values) -> List[float]:
    if values is None:
        return []
    result = []
    if isinstance(values, (list, tuple)):
        for item in values:
            result.extend(flatten_temperature_values(item))
        return result
    try:
        result.append(float(values))
    except (TypeError, ValueError):
        pass
    return result


def choose_lammps_temperature(runtime_config: Dict) -> float:
    workflow = dict(runtime_config.get("workflow") or {})
    temps = flatten_temperature_values(workflow.get("main_loop_npt"))
    if not temps:
        temps = flatten_temperature_values(workflow.get("main_loop_nvt"))
    return temps[-1] if temps else 300.0


def inject_sus2_pair_style(script_text: str, mtp_filename: str) -> str:
    lines = script_text.splitlines()
    normalized = []
    inserted = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("# variable mlip_ini"):
            continue
        if stripped.startswith("pair_style") and ("mlip" in stripped or "sus2mtp" in stripped):
            continue
        if stripped == "pair_coeff * *":
            continue
        normalized.append(line)
        if not inserted and stripped.startswith("read_data data.in"):
            normalized.append("")
            normalized.append(f"pair_style sus2mtp {mtp_filename}")
            normalized.append("pair_coeff * *")
            inserted = True
    if not inserted:
        raise ValueError("Could not find 'read_data data.in' in init/lmp_in.py template")
    return "\n".join(normalized) + "\n"


def load_lammps_template(run_dir: Path, ensemble: str, temp: float) -> str:
    template_path = run_dir / "init" / "lmp_in.py"
    if not template_path.exists():
        raise FileNotFoundError(f"LAMMPS template not found: {template_path}")
    spec = importlib.util.spec_from_file_location("dcbf_coverage_lmp_in", template_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    if not hasattr(module, ensemble):
        raise ValueError(f"{template_path} does not define {ensemble!r} LAMMPS template")
    random_number = random.randint(1000, 1000000)
    return f"variable T equal {temp}\nvariable random equal {random_number}\n" + getattr(module, ensemble)


def lammps_box_to_cell(box_lines: Sequence[str]) -> np.ndarray:
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
        zlo, zhi = zlo_bound, zhi_bound
        return np.array([[xhi - xlo, 0.0, 0.0], [xy, yhi - ylo, 0.0], [xz, yz, zhi - zlo]], dtype=float)
    xlo, xhi = rows[0][:2]
    ylo, yhi = rows[1][:2]
    zlo, zhi = rows[2][:2]
    return np.diag([xhi - xlo, yhi - ylo, zhi - zlo]).astype(float)


def read_lammps_custom_dump(dump_path: Path, elements: Sequence[str]) -> List:
    from ase import Atoms
    from ase.calculators.singlepoint import SinglePointCalculator

    frames = []
    type_to_element = {index + 1: element for index, element in enumerate(elements)}
    lines = dump_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    index = 0
    while index < len(lines):
        if not lines[index].startswith("ITEM: TIMESTEP"):
            index += 1
            continue
        index += 2
        if index >= len(lines) or not lines[index].startswith("ITEM: NUMBER OF ATOMS"):
            raise ValueError(f"Unexpected LAMMPS dump format near timestep in {dump_path}")
        natoms = int(lines[index + 1].strip())
        index += 2
        if index >= len(lines) or not lines[index].startswith("ITEM: BOX BOUNDS"):
            raise ValueError(f"Missing BOX BOUNDS section in {dump_path}")
        box_lines = lines[index + 1 : index + 4]
        cell = lammps_box_to_cell(box_lines)
        index += 4
        if index >= len(lines) or not lines[index].startswith("ITEM: ATOMS"):
            raise ValueError(f"Missing ATOMS section in {dump_path}")
        columns = lines[index].split()[2:]
        index += 1

        col_index = {name: pos for pos, name in enumerate(columns)}
        required = ["id", "type", "x", "y", "z"]
        missing = [name for name in required if name not in col_index]
        if missing:
            raise ValueError(f"LAMMPS dump is missing required columns {missing}: {dump_path}")

        atom_rows = []
        for _ in range(natoms):
            values = lines[index].split()
            index += 1
            atom_rows.append(values)
        atom_rows.sort(key=lambda row: int(float(row[col_index["id"]])))

        symbols = []
        positions = np.zeros((natoms, 3), dtype=float)
        forces = np.zeros((natoms, 3), dtype=float)
        pe_values = np.zeros(natoms, dtype=float)
        for row_index, row in enumerate(atom_rows):
            type_id = int(float(row[col_index["type"]]))
            symbols.append(type_to_element.get(type_id, elements[min(type_id - 1, len(elements) - 1)]))
            positions[row_index] = [float(row[col_index[axis]]) for axis in ("x", "y", "z")]
            if all(axis in col_index for axis in ("fx", "fy", "fz")):
                forces[row_index] = [float(row[col_index[axis]]) for axis in ("fx", "fy", "fz")]
            if "c_pe" in col_index:
                pe_values[row_index] = float(row[col_index["c_pe"]])

        atoms = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True)
        atoms.calc = SinglePointCalculator(atoms, energy=float(np.sum(pe_values)), forces=forces)
        frames.append(atoms)
    if not frames:
        raise ValueError(f"No frames found in LAMMPS dump: {dump_path}")
    return frames


def _apply_lammps_scheduler_overrides(scheduler: Dict, args: argparse.Namespace) -> Dict:
    scheduler = dict(scheduler)
    lammps_exe = getattr(args, "lammps_exe", None)
    lammps_env = getattr(args, "lammps_env", None)
    lammps_cores = getattr(args, "lammps_cores", None)
    if lammps_exe:
        scheduler["lmp_exe"] = str(lammps_exe)
    if lammps_env:
        scheduler["lmp_env"] = str(lammps_env)
    if lammps_cores is not None:
        cores = int(lammps_cores)
        scheduler["lmp_cores"] = cores
        try:
            ptile = int(scheduler.get("lmp_ptile", cores))
        except (TypeError, ValueError):
            ptile = cores
        scheduler["lmp_ptile"] = min(ptile, cores)
    return scheduler


def _tail_file(path: Path, max_lines: int = 80) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-max_lines:])


def _lammps_failure_context(work_dir: Path) -> str:
    candidates = [work_dir / "jobs_mlip_0.ini.out", work_dir / "bsub.lsf"]
    candidates.extend(sorted(work_dir.glob("*.err")))
    candidates.extend(sorted(work_dir.glob("*.out")))
    blocks = []
    seen = set()
    for path in candidates:
        resolved = str(path.resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        text = _tail_file(path)
        if text:
            blocks.append(f"--- {path.name} ---\n{text}")
    return "\n\n".join(blocks)


def _lammps_timeout_seconds(args: argparse.Namespace) -> int:
    legacy_seconds = getattr(args, "lammps_timeout", None)
    if legacy_seconds is not None:
        return max(1, int(float(legacy_seconds)))
    timeout_hours = getattr(args, "lammps_timeout_hours", DEFAULT_LAMMPS_TIMEOUT_HOURS)
    return max(1, int(float(timeout_hours) * 3600))


def _run_lammps_query_local(work_dir: Path, scheduler: Dict, args: argparse.Namespace) -> None:
    lmp_exe = str(getattr(args, "lammps_exe", None) or scheduler.get("lmp_exe") or "")
    if not lmp_exe:
        raise ValueError("No LAMMPS executable found; set --lammps-exe or scheduler.lmp_exe, or pass --query.")
    lmp_env = str(getattr(args, "lammps_env", None) or scheduler.get("lmp_env") or "").strip()
    cores = int(getattr(args, "lammps_cores", None) or scheduler.get("lmp_cores") or 1)
    command_parts = [f"export NP={cores}"]
    if lmp_env:
        command_parts.append(lmp_env)
    command_parts.append(f"{lmp_exe} -in lmp.in -var out_dump_file force.0.nc")
    command = "; ".join(command_parts).replace("$NP", str(cores))
    subprocess.run(["bash", "-lc", command], cwd=str(work_dir), check=True)


def _run_lammps_query_scheduler(work_dir: Path, scheduler: Dict, args: argparse.Namespace) -> None:
    try:
        from .runtime_config import build_scheduler_spec
    except ImportError:  # pragma: no cover - direct script execution fallback.
        from runtime_config import build_scheduler_spec

    scheduler = _apply_lammps_scheduler_overrides(scheduler, args)
    if not scheduler.get("lmp_exe"):
        raise ValueError("Scheduler LAMMPS mode requires scheduler.lmp_exe or --lammps-exe; pass --query to skip generation.")
    try:
        scheduler_spec = build_scheduler_spec(scheduler)
    except KeyError as exc:
        raise ValueError(
            "Scheduler LAMMPS mode requires a full scheduler block from dcbf.runtime.json. "
            "Use --lammps-run-mode local only for explicit local debugging."
        ) from exc

    for stale_name in ("__start__", "__ok__", "__fail__", "force.0.dump", "force.0.nc", "jobs_mlip_0.ini.out"):
        stale_path = work_dir / stale_name
        if stale_path.exists():
            stale_path.unlink()

    lmp_env = str(scheduler_spec.lmp_env or "").strip()
    lmp_env_text = f"{lmp_env}\n" if lmp_env else ""
    script_text = f"""#!/bin/bash
{scheduler_spec.bsub_script_lmp_job_name}coverage_query_lammps
{scheduler_spec.bsub_script_lmp}
set -e
trap 'touch __fail__' ERR
touch __start__
{lmp_env_text}COMMAND="${{COMMAND_0}} -in lmp.in -var out_dump_file force.0.nc"
$COMMAND > jobs_mlip_0.ini.out 2>&1
touch __ok__
"""
    script_path = work_dir / "bsub.lsf"
    script_path.write_text(script_text, encoding="utf-8")
    script_path.chmod(0o755)

    subprocess.run(scheduler_spec.task_submission_method, cwd=str(work_dir), shell=True, check=True)

    timeout = _lammps_timeout_seconds(args)
    deadline = time.time() + timeout
    while True:
        if (work_dir / "__ok__").exists():
            return
        if (work_dir / "__fail__").exists():
            context = _lammps_failure_context(work_dir)
            raise RuntimeError(f"Scheduler LAMMPS query job failed in {work_dir}.\n{context}")
        if time.time() > deadline:
            context = _lammps_failure_context(work_dir)
            timeout_hours = timeout / 3600
            raise TimeoutError(
                f"Timed out waiting for scheduler LAMMPS query job in {work_dir} "
                f"after {timeout_hours:g} h ({timeout}s).\n{context}"
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
    temp: float,
    mtp_path: Path,
) -> List:
    from ase.io import write

    work_dir.mkdir(parents=True, exist_ok=True)
    atoms = atoms.copy()
    atoms = atoms.repeat(size)
    write(str(work_dir / "data.in"), atoms, format="lammps-data", masses=True, specorder=specorder, force_skew=True)

    shutil.copy2(mtp_path, work_dir / "current_0.mtp")

    lmp_text = inject_sus2_pair_style(load_lammps_template(run_dir, ensemble, temp), "current_0.mtp")
    (work_dir / "lmp.in").write_text(lmp_text, encoding="utf-8")

    run_mode = str(getattr(args, "lammps_run_mode", DEFAULT_LAMMPS_RUN_MODE) or DEFAULT_LAMMPS_RUN_MODE).lower()
    if run_mode == "local":
        _run_lammps_query_local(work_dir, scheduler, args)
    else:
        _run_lammps_query_scheduler(work_dir, scheduler, args)

    dump_path = work_dir / "force.0.dump"
    if not dump_path.exists():
        raise FileNotFoundError(f"LAMMPS finished but did not write {dump_path}")
    return read_lammps_custom_dump(dump_path, specorder)


def run_lammps_query_generation(run_dir: Path, args: argparse.Namespace, output_xyz: Path) -> Path:
    from ase.io import write
    from ase.data import atomic_numbers, chemical_symbols

    if yaml is None:
        raise RuntimeError("PyYAML is required for automatic LAMMPS query generation; pass --query to use an existing xyz.")

    runtime_config = read_json_if_exists(run_dir / "dcbf.runtime.json")
    scheduler = dict(runtime_config.get("scheduler") or {})
    parameter = read_parameter_yaml(run_dir / "init" / "parameter.yaml")
    elements = list(parameter.get("ele") or args.elements)
    sort_ele = bool(parameter.get("sort_ele", True))
    size = parse_repeat_size(parameter.get("size", "(1, 1, 1)"))
    ensemble = "npt" if runtime_config.get("workflow", {}).get("main_loop_npt") is not None else "nvt"
    temp = choose_lammps_temperature(runtime_config)

    query_root = output_xyz.parent
    query_root.mkdir(parents=True, exist_ok=True)
    specorder = [chemical_symbols[number] for number in sorted(atomic_numbers[item] for item in elements)] if sort_ele else elements

    mtp_path = find_default_mtp(run_dir)
    if mtp_path is None:
        raise FileNotFoundError(f"Could not find current.mtp/current_0.mtp under {run_dir}; pass --mtp or create the model first.")

    all_sources = find_structure_sources(run_dir)
    selected_sources = select_query_structure_sources(all_sources, getattr(args, "query_structures", "all"))
    print(
        "Selected LAMMPS query structures: "
        + ", ".join(f"{item.index}:{item.label}({item.path.name}:frame{item.frame_index})" for item in selected_sources)
    )

    all_frames = []
    manifest = []
    for source in selected_sources:
        work_dir = query_root if len(selected_sources) == 1 else query_root / "runs" / source.label
        frames = _run_lammps_query_for_structure(
            run_dir,
            args,
            work_dir,
            source.atoms,
            size,
            specorder,
            scheduler,
            ensemble,
            temp,
            mtp_path,
        )
        for atoms in frames:
            atoms.info["coverage_query_structure"] = source.label
            atoms.info["coverage_query_source"] = str(source.path)
            atoms.info["coverage_query_source_frame"] = int(source.frame_index)
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
        raise ValueError("LAMMPS query generation produced no frames.")
    if output_xyz.exists():
        output_xyz.unlink()
    write(str(output_xyz), all_frames, format="extxyz")
    (query_root / "query_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return output_xyz


def ensure_lammps_query(run_dir: Path, args: argparse.Namespace) -> Path:
    output_xyz = run_dir / "coverage_query_lammps" / "query.xyz"
    if output_xyz.exists() and not args.force_query:
        print(f"Using existing coverage LAMMPS query xyz: {output_xyz}")
        return output_xyz

    for candidate_name in ("npt.xyz", "query.xyz", DEFAULT_QUERY):
        candidate = run_dir / candidate_name
        if candidate.exists() and not args.force_query:
            print(f"Using existing query xyz from run directory: {candidate}")
            return candidate

    print(f"No existing query xyz found under {run_dir}; generating LAMMPS query at {output_xyz}")
    return run_lammps_query_generation(run_dir, args, output_xyz)


def mtp_many_body_list(mtp_type: str) -> Tuple[List[int], List[int], List[int]]:
    if mtp_type == "l2k2.mtp":
        mapping = {
            "<0>": [0, 10],
            "<11>": [27, 28, 29],
            "<22>": [30, 31, 32],
            "<211>": [72, 73, 80, 81],
            "<222>": [91, 92, 99, 100],
        }
    elif mtp_type == "l2k3.mtp":
        mapping = {
            "<0>": [0, 10, 20],
            "<11>": [46, 47, 48, 49, 50, 51],
            "<22>": [52, 53, 54, 55, 56, 57],
            "<211>": [178, 179, 180, 187, 188, 189, 196, 197, 198],
            "<222>": [211, 212, 213, 220, 221, 222, 229, 230, 231],
        }
    else:
        raise ValueError(f"Unsupported mtp_type: {mtp_type}")

    two_body: List[int] = []
    three_body: List[int] = []
    four_body: List[int] = []
    for key, value in mapping.items():
        body_order = len(key) - 1
        if body_order == 2:
            two_body += value
        elif body_order == 3:
            three_body += value
        elif body_order == 4:
            four_body += value
    return two_body, three_body, four_body


def alpha_moment_mapping(mtp_path: Path) -> List[int]:
    with mtp_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "alpha_moment_mapping" not in line:
                continue
            start = line.index("{") + 1
            end = line.index("}")
            return [int(num.strip()) for num in line[start:end].split(",") if num.strip()]
    raise ValueError(f"No alpha_moment_mapping found in {mtp_path}")


def extract_mtp_many_body_index(mtp_type: str, mtp_path: Path) -> Tuple[List[int], List[int], List[int]]:
    two_body, three_body, four_body = mtp_many_body_list(mtp_type)
    alpha_map = alpha_moment_mapping(mtp_path)
    return (
        [alpha_map.index(item) for item in two_body],
        [alpha_map.index(item) for item in three_body],
        [alpha_map.index(item) for item in four_body],
    )


def normalize_virial(value) -> np.ndarray:
    if value is None:
        return np.zeros((3, 3), dtype=float)
    arr = np.asarray(value, dtype=float)
    if arr.shape == (3, 3):
        return arr
    if arr.size == 9:
        return arr.reshape(3, 3)
    if arr.size == 6:
        xx, yy, zz, yz, xz, xy = arr.reshape(-1)
        return np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]], dtype=float)
    return np.zeros((3, 3), dtype=float)


def atoms_energy_forces_virial(atoms) -> Tuple[float, np.ndarray, np.ndarray]:
    try:
        energy = float(atoms.get_potential_energy())
    except Exception:
        energy = 0.0

    try:
        forces = np.asarray(atoms.get_forces(), dtype=float)
    except Exception:
        forces = np.zeros((len(atoms), 3), dtype=float)

    virial = atoms.info.get("virial")
    if virial is None and "stress" in atoms.info:
        virial = atoms.info.get("stress")
    return energy, forces, normalize_virial(virial)


def xyz2cfg_worker(args) -> None:
    elements, element_model, atoms_batch, out_path = args

    from ase.data import atomic_numbers, chemical_symbols

    if element_model == 1:
        sorted_numbers = sorted(atomic_numbers[item] for item in elements)
        type_elements = [chemical_symbols[item] for item in sorted_numbers]
    else:
        type_elements = list(elements)
    type_map = {element: index for index, element in enumerate(type_elements)}

    with Path(out_path).open("w", encoding="utf-8") as ff:
        for atoms in atoms_batch:
            atom_elements = atoms.get_chemical_symbols()
            nat = len(atom_elements)
            cell = atoms.get_cell()
            pos = atoms.get_positions()
            energy, forces, virial = atoms_energy_forces_virial(atoms)

            ff.write("BEGIN_CFG\n")
            ff.write(" Size\n")
            ff.write(f"  {nat:6}\n")
            ff.write(" Supercell \n")
            ff.write(f"{cell[0, 0]:15.10f} {cell[0, 1]:15.10f} {cell[0, 2]:15.10f}\n")
            ff.write(f"{cell[1, 0]:15.10f} {cell[1, 1]:15.10f} {cell[1, 2]:15.10f}\n")
            ff.write(f"{cell[2, 0]:15.10f} {cell[2, 1]:15.10f} {cell[2, 2]:15.10f}\n")
            ff.write("AtomData:  id type       cartes_x      cartes_y      cartes_z     fx          fy          fz\n")
            for atom_index in range(nat):
                element = atom_elements[atom_index]
                if element not in type_map:
                    raise ValueError(f"Element {element} is not in --elements")
                ff.write(
                    f" {atom_index + 1:6} {type_map[element]:6} "
                    f"{pos[atom_index, 0]:12.6f} {pos[atom_index, 1]:12.6f} {pos[atom_index, 2]:12.6f} "
                    f"{forces[atom_index, 0]:12.6f} {forces[atom_index, 1]:12.6f} {forces[atom_index, 2]:12.6f}\n"
                )
            ff.write("Energy \n")
            ff.write(f"\t{energy}\n")
            ff.write("PlusStress:  xx          yy          zz          yz          xz          xy \n")
            ff.write(
                f"\t{virial[0, 0]}  \t{virial[1, 1]}  \t{virial[2, 2]}  "
                f"\t{virial[1, 2]}  \t{virial[0, 2]}  \t{virial[0, 1]}\n"
            )
            ff.write("END_CFG \n")


def encode_worker(args) -> Path:
    mlp_exe, mtp_path, cfg_path = args
    cfg_path = Path(cfg_path)
    out_path = cfg_path.with_suffix(".out")
    subprocess.run(
        [str(mlp_exe), "calc-descriptors", str(mtp_path), str(cfg_path), str(out_path)],
        check=True,
    )
    return out_path


def split_nonempty(data: Sequence, n_parts: int) -> List[Sequence]:
    n_parts = max(1, min(n_parts, len(data)))
    indices = np.array_split(np.arange(len(data)), n_parts)
    return [data[int(chunk[0]): int(chunk[-1]) + 1] for chunk in indices if len(chunk) > 0]


def merge_out_files(out_files: Sequence[Path], merged_out: Path) -> None:
    with merged_out.open("w", encoding="utf-8") as outfile:
        for out_file in sorted(out_files, key=lambda item: item.name):
            with out_file.open("r", encoding="utf-8", errors="ignore") as infile:
                outfile.write(infile.read())
                outfile.write("\n")


def descriptor_out_to_pickles(
    des_out_path: Path,
    prefix: str,
    num_ele: int,
    mtp_type: str,
    mtp_path: Path,
    body_name_list: Sequence[str],
    out_path: Path,
) -> None:
    type_list = list(range(num_ele))
    total_list = [[] for _ in type_list]
    two_body_list = [[] for _ in type_list]
    three_body_list = [[] for _ in type_list]
    four_body_list = [[] for _ in type_list]

    two_body, three_body, four_body = extract_mtp_many_body_index(mtp_type, mtp_path)

    atom_index = 0
    with des_out_path.open("r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    for line_index, line in enumerate(lines):
        if "#start" not in line:
            continue
        atom_num = int(line.split()[1])
        atom_lines = lines[line_index + 1: line_index + atom_num + 1]
        for atom_line in atom_lines:
            parts = atom_line.split()
            if len(parts) < 2:
                continue
            ele_type = int(parts[0])
            values = [float(item) for item in parts[1:]]
            values.append(atom_index)
            if ele_type not in type_list:
                continue
            total_list[ele_type].append(values)
            two_values = [values[index] for index in two_body] + [values[-1]]
            three_values = [values[index] for index in three_body] + [values[-1]]
            four_values = [values[index] for index in four_body] + [values[-1]]
            two_body_list[ele_type].append(two_values)
            three_body_list[ele_type].append(three_values)
            four_body_list[ele_type].append(four_values)
        atom_index += 1

    body_lists = [two_body_list, three_body_list, four_body_list]
    body_names = [f"{prefix}_two_body_", f"{prefix}_three_body_", f"{prefix}_four_body_"]
    body_index = {"two": 0, "three": 1, "four": 2}
    selected = {body_index[item] for item in body_name_list}

    out_path.mkdir(parents=True, exist_ok=True)
    for index, (body, name) in enumerate(zip(body_lists, body_names)):
        if index in selected:
            encode_zlib_pickle(body, out_path / f"{name}coding_zlib.pkl")


def convert_xyz_to_descriptor_dir(
    input_path: Path,
    label: str,
    descriptor_dir: Path,
    elements: Sequence[str],
    element_model: int,
    workers: int,
    mlp_exe: Path,
    mtp_path: Path,
    mtp_type: str,
    data_modes: Sequence[str],
    keep_out: bool,
) -> None:
    from ase.io import iread

    start = time.time()
    atoms_list = list(iread(str(input_path)))
    if not atoms_list:
        raise ValueError(f"No structures found in {input_path}")

    workers = max(1, min(workers, len(atoms_list)))
    descriptor_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=f"{label}_", dir=str(descriptor_dir.parent)) as tmp_text:
        tmp_dir = Path(tmp_text)
        chunks = split_nonempty(atoms_list, workers)
        cfg_paths = [tmp_dir / f"data_{index}.cfg" for index in range(len(chunks))]

        with ProcessPoolExecutor(max_workers=len(chunks)) as executor:
            futures = [
                executor.submit(xyz2cfg_worker, (list(elements), element_model, chunk, str(cfg_path)))
                for chunk, cfg_path in zip(chunks, cfg_paths)
            ]
            for future in as_completed(futures):
                future.result()

        out_files: List[Path] = []
        with ProcessPoolExecutor(max_workers=len(cfg_paths)) as executor:
            futures = [executor.submit(encode_worker, (str(mlp_exe), str(mtp_path), str(cfg_path))) for cfg_path in cfg_paths]
            for future in as_completed(futures):
                out_files.append(future.result())

        merged_out = descriptor_dir / f"{label}.out"
        merge_out_files(out_files, merged_out)
        descriptor_out_to_pickles(
            merged_out,
            label,
            len(elements),
            mtp_type,
            mtp_path,
            data_modes,
            descriptor_dir,
        )
        if not keep_out:
            merged_out.unlink(missing_ok=True)

    print(f"{label}: descriptor conversion finished in {(time.time() - start) / 60:.2f} min")


def prepare_dataset(
    label: str,
    source: Path,
    output_descriptor_root: Path,
    script_dir: Path,
    elements: Sequence[str],
    element_model: int,
    workers: int,
    mlp_exe: Path,
    mtp_path: Path,
    mtp_type: str,
    data_modes: Sequence[str],
    keep_out: bool,
    force: bool,
) -> DatasetSpec:
    source = resolve_dataset_path(source, script_dir)

    if has_descriptor_pickles(source, data_modes) and not force:
        return DatasetSpec(label=label, source=source, descriptor_dir=source, converted=False)

    descriptor_dir = output_descriptor_root / label
    if has_descriptor_pickles(descriptor_dir, data_modes) and not force:
        return DatasetSpec(label=label, source=source, descriptor_dir=descriptor_dir, converted=False)

    if not source.exists():
        raise FileNotFoundError(f"Dataset source not found: {source}")
    if not source.is_file():
        raise ValueError(f"{source} is not a descriptor directory and not an xyz/traj file")

    convert_xyz_to_descriptor_dir(
        source,
        label,
        descriptor_dir,
        elements,
        element_model,
        workers,
        mlp_exe,
        mtp_path,
        mtp_type,
        data_modes,
        keep_out,
    )
    return DatasetSpec(label=label, source=source, descriptor_dir=descriptor_dir, converted=True)


def load_all_data(input_path: Path, data_modes: Iterable[str]) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], int]:
    all_aee_data: Dict[int, List[np.ndarray]] = {}
    all_stru_data: Dict[int, List[np.ndarray]] = {}

    for body in data_modes:
        files = sorted(glob.glob(str(input_path / f"*{body}*coding_zlib.pkl")))
        if not files:
            raise FileNotFoundError(f"No descriptor pickle matched mode '{body}' in {input_path}")
        loaded_list = decode_zlib_pickle(Path(files[0]))
        for ele_type, type_data in enumerate(loaded_list):
            type_data = np.asarray(type_data)
            if type_data.size == 0:
                all_aee_data.setdefault(ele_type, []).append(np.empty((0, 0)))
                all_stru_data.setdefault(ele_type, []).append(np.empty((0,)))
                continue
            if type_data.ndim != 2 or type_data.shape[1] < 2:
                raise ValueError(f"Bad descriptor shape in {files[0]}, ele_type={ele_type}: {type_data.shape}")
            all_aee_data.setdefault(ele_type, []).append(type_data[:, :-1])
            all_stru_data.setdefault(ele_type, []).append(type_data[:, -1])

    combined_aee = {}
    for ele_type, chunks in all_aee_data.items():
        nonempty = [chunk for chunk in chunks if chunk.size > 0]
        if not nonempty:
            combined_aee[ele_type] = np.empty((0, 0))
        else:
            combined_aee[ele_type] = np.hstack(nonempty)

    combined_stru = {ele_type: chunks[0] for ele_type, chunks in all_stru_data.items()}
    return combined_aee, combined_stru, len(combined_stru)


def load_aee_data_by_mode(input_path: Path, data_modes: Iterable[str]) -> Tuple[Dict[str, Dict[int, np.ndarray]], int]:
    mode_aee: Dict[str, Dict[int, np.ndarray]] = {}
    ele_num = 0

    for body in data_modes:
        files = sorted(glob.glob(str(input_path / f"*{body}*coding_zlib.pkl")))
        if not files:
            raise FileNotFoundError(f"No descriptor pickle matched mode '{body}' in {input_path}")
        loaded_list = decode_zlib_pickle(Path(files[0]))
        ele_num = max(ele_num, len(loaded_list))
        body_aee: Dict[int, np.ndarray] = {}
        for ele_type, type_data in enumerate(loaded_list):
            type_data = np.asarray(type_data)
            if type_data.size == 0:
                body_aee[ele_type] = np.empty((0, 0))
                continue
            if type_data.ndim != 2 or type_data.shape[1] < 2:
                raise ValueError(f"Bad descriptor shape in {files[0]}, ele_type={ele_type}: {type_data.shape}")
            body_aee[ele_type] = type_data[:, :-1]
        mode_aee[body] = body_aee

    return mode_aee, ele_num


def choose_fit_data(ref_data: np.ndarray, query_data: np.ndarray, source: str) -> np.ndarray:
    if source == "input":
        return ref_data
    if source == "query":
        return query_data
    if source == "combined":
        return np.vstack([ref_data, query_data])
    raise ValueError(f"Unsupported fit source: {source}")


def choose_element_fit_data(ref_arrays: Sequence[np.ndarray], query_data: np.ndarray, source: str) -> np.ndarray:
    if source == "query":
        return query_data
    if source == "input":
        return np.vstack(ref_arrays)
    if source == "combined":
        return np.vstack([*ref_arrays, query_data])
    raise ValueError(f"Unsupported fit source: {source}")


def fit_pca_model(fit_data: np.ndarray, n_components: int) -> PcaModel:
    fit_data = np.asarray(fit_data, dtype=float)
    if fit_data.shape[0] == 0:
        raise ValueError("Cannot fit PCA with empty data")

    mean = np.mean(fit_data, axis=0)
    std = np.std(fit_data, axis=0)
    std[std == 0] = 1.0

    fit_scaled = (fit_data - mean) / std
    _, singular_values, vh = np.linalg.svd(fit_scaled, full_matrices=False)
    available_components = min(n_components, vh.shape[0])
    components = np.zeros((n_components, fit_data.shape[1]), dtype=float)
    if available_components > 0:
        components[:available_components] = vh[:available_components]

    variances = singular_values ** 2
    total_variance = float(np.sum(variances))
    explained = (
        float(np.sum(variances[:available_components]) / total_variance)
        if total_variance > 0 and available_components > 0
        else 0.0
    )
    return PcaModel(mean=mean, std=std, components=components, explained=explained)


def transform_pca(data: np.ndarray, model: PcaModel) -> np.ndarray:
    data = np.asarray(data, dtype=float)
    return ((data - model.mean) / model.std) @ model.components.T


def fit_transform_pca(
    ref_data: np.ndarray,
    query_data: np.ndarray,
    n_components: int,
    fit_source: str,
) -> Tuple[np.ndarray, np.ndarray, float]:
    fit_data = np.asarray(choose_fit_data(ref_data, query_data, fit_source), dtype=float)
    ref_data = np.asarray(ref_data, dtype=float)
    query_data = np.asarray(query_data, dtype=float)

    model = fit_pca_model(fit_data, n_components)
    return transform_pca(ref_data, model), transform_pca(query_data, model), model.explained


def fd_bins(data: np.ndarray, width_factor: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = np.asarray(data, dtype=float)
    mins = np.min(data, axis=0)
    maxs = np.max(data, axis=0)
    ranges = maxs - mins
    widths = np.zeros(data.shape[1], dtype=float)
    bins = np.ones(data.shape[1], dtype=int)

    for dim in range(data.shape[1]):
        if ranges[dim] <= 0:
            widths[dim] = 0.0
            bins[dim] = 1
            continue
        iqr = np.percentile(data[:, dim], 75) - np.percentile(data[:, dim], 25)
        width = 2.0 * iqr / (len(data) ** (1.0 / 3.0))
        width *= width_factor
        if not np.isfinite(width) or width <= 0:
            width = ranges[dim]
        bins[dim] = max(1, int(ranges[dim] / width))
        widths[dim] = ranges[dim] / bins[dim]
    return mins, maxs, bins, widths


def assign_fd_bins(data: np.ndarray, mins: np.ndarray, maxs: np.ndarray, bins: np.ndarray, widths: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    data = np.asarray(data, dtype=float)
    bin_ids = np.full(data.shape, -1, dtype=int)
    valid = np.all(np.isfinite(data), axis=1)

    for dim in range(data.shape[1]):
        if widths[dim] == 0:
            dim_valid = np.isclose(data[:, dim], mins[dim])
            bin_ids[dim_valid, dim] = 0
        else:
            dim_valid = (data[:, dim] >= mins[dim]) & (data[:, dim] < maxs[dim])
            ids = np.floor((data[dim_valid, dim] - mins[dim]) / widths[dim]).astype(int)
            bin_ids[dim_valid, dim] = np.clip(ids, 0, bins[dim] - 1)
        valid &= dim_valid
    return bin_ids, valid


def grid_data_for(ref_data: np.ndarray, query_data: np.ndarray, source: str) -> np.ndarray:
    if source == "ref":
        return ref_data
    if source == "query":
        return query_data
    if source == "combined":
        return np.vstack([ref_data, query_data])
    raise ValueError(f"Unsupported grid source: {source}")


def resolve_grid_sources(args: argparse.Namespace) -> None:
    if args.coverage_grid in {"input", "current-input"}:
        args.coverage_grid = "current-loop"
    standard_to_source = {
        "query": "query",
        "current-loop": "ref",
        "last-loop": "last-ref",
    }
    base_source = standard_to_source[args.coverage_grid]
    args.grid_source_2d = base_source
    args.grid_source_1d = base_source


def normalize_coverage_grid_name(value: str) -> str:
    normalized = str(value).strip()
    if normalized in {"input", "current-input"}:
        return "current-loop"
    return normalized


def resolve_width_factors(args: argparse.Namespace) -> None:
    shared_width_factor = getattr(args, "width_factor", None)
    if shared_width_factor is not None:
        width_factor_1d = shared_width_factor
        width_factor_2d = shared_width_factor
    else:
        width_factor_1d = DEFAULT_WIDTH_FACTOR_1D
        width_factor_2d = DEFAULT_WIDTH_FACTOR_2D

    args.width_factor_1d = float(width_factor_1d)
    args.width_factor_2d = float(width_factor_2d)


def public_grid_source_name(source: str) -> str:
    return {
        "query": "query",
        "ref": "current-loop",
        "last-ref": "last-loop",
        "combined": "combined",
    }.get(source, source)


def describe_grid_source(source: str, query_label: str, final_dataset_label: str) -> str:
    if source == "query":
        return f"query ({query_label})"
    if source == "last-ref":
        return f"last-loop/final input ({final_dataset_label})"
    if source == "ref":
        return "current loop dataset"
    if source == "combined":
        return "current input + query"
    return source


def fd_grid_coverage(
    ref_data: np.ndarray,
    query_data: np.ndarray,
    width_factor: float,
    grid_source: str,
    fixed_grid_data: Optional[np.ndarray] = None,
) -> Tuple[float, np.ndarray, int]:
    if len(ref_data) == 0 or len(query_data) == 0:
        return 0.0, np.zeros(len(query_data), dtype=bool), 0

    grid_data = (
        np.asarray(fixed_grid_data, dtype=float)
        if fixed_grid_data is not None
        else grid_data_for(ref_data, query_data, grid_source)
    )
    mins, maxs, bins, widths = fd_bins(grid_data, width_factor)
    ref_bins, ref_valid = assign_fd_bins(ref_data, mins, maxs, bins, widths)
    query_bins, query_valid = assign_fd_bins(query_data, mins, maxs, bins, widths)

    occupied = {tuple(item) for item in ref_bins[ref_valid]}
    covered = query_valid & np.array([tuple(item) in occupied for item in query_bins], dtype=bool)
    covered_count = int(np.sum(covered))
    coverage = float(covered_count / len(query_data)) if len(query_data) else 0.0
    return coverage, covered, covered_count


def mean_1d_coverage(
    ref_data: np.ndarray,
    query_data: np.ndarray,
    width_factor: float,
    grid_source: str,
    fixed_grid_data: Optional[np.ndarray] = None,
) -> Tuple[float, float, np.ndarray]:
    coverages = []
    covered_counts = []
    covered_masks = []
    for dim in range(ref_data.shape[1]):
        fixed_dim_grid = fixed_grid_data[:, [dim]] if fixed_grid_data is not None else None
        coverage, covered_mask, covered_count = fd_grid_coverage(
            ref_data[:, [dim]],
            query_data[:, [dim]],
            width_factor,
            grid_source,
            fixed_dim_grid,
        )
        coverages.append(coverage)
        covered_counts.append(covered_count)
        covered_masks.append(covered_mask.astype(float))
    if not coverages:
        return 0.0, 0.0, np.zeros(len(query_data), dtype=float)
    mean_coverage = float(np.mean(coverages))
    point_scores = np.mean(np.vstack(covered_masks), axis=0)
    return mean_coverage, float(np.mean(covered_counts)), point_scores


def monotonic_topk_labels(
    results_by_key: Dict[Tuple[str, str], CoverageResult],
    dataset_labels: Sequence[str],
    elements: Sequence[str],
) -> None:
    for element in elements:
        previous: Optional[np.ndarray] = None
        previous_count = 0
        for label in dataset_labels:
            item = results_by_key.get((label, element))
            if item is None:
                continue
            if previous is None or len(previous) != item.query_count:
                previous = np.zeros(item.query_count, dtype=bool)
                previous_count = 0

            target_count = int(round(item.coverage_1d * item.query_count))
            target_count = max(previous_count, min(target_count, item.query_count))

            covered = previous.copy()
            need_add = target_count - int(np.sum(covered))
            if need_add > 0:
                candidates = np.flatnonzero(~covered)
                scores = item.covered_1d_scores[candidates]
                order = candidates[np.argsort(-scores, kind="mergesort")]
                covered[order[:need_add]] = True

            item.covered_1d_display_bool = covered
            previous = covered
            previous_count = int(np.sum(covered))


def display_covered_bool(item: CoverageResult, coverage_mode: str) -> np.ndarray:
    if coverage_mode == "1d":
        if item.covered_1d_display_bool is not None:
            return item.covered_1d_display_bool
        return item.covered_1d_scores > 0
    return item.covered_2d_bool


def save_pca_point_files(
    out_dir: Path,
    element: str,
    ref_pca: np.ndarray,
    query_pca: np.ndarray,
    covered_bool: np.ndarray,
    coverage: float,
    coverage_mode: str,
    display_label_source: str,
    width_factor: float,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ref_save = np.column_stack([ref_pca[:, 0], ref_pca[:, 1], np.ones(len(ref_pca), dtype=int)])
    query_save = np.column_stack([query_pca[:, 0], query_pca[:, 1], covered_bool.astype(int)])

    np.savetxt(
        out_dir / f"{element}_pca_A.txt",
        ref_save,
        fmt="%.8f %.8f %d",
        header=f"PC1 PC2 Is_Covered - Dataset A (Input)\nTotal points: {len(ref_pca)}, All covered",
        comments="",
    )
    np.savetxt(
        out_dir / f"{element}_pca_B.txt",
        query_save,
        fmt="%.8f %.8f %d",
        header=(
            f"PC1 PC2 Is_Covered - Dataset B (Query, primary coverage mode: {coverage_mode.upper()}; "
            f"Is_Covered labels: {display_label_source})\n"
            f"Total points: {len(query_pca)}, Primary coverage: {coverage:.6f}, "
            f"Width factor ({coverage_mode.upper()}): {width_factor:g}"
        ),
        comments="",
    )


def compute_coverage_for_pair(args) -> CoverageResult:
    (
        dataset_label,
        element,
        ele_type,
        ref_data,
        query_data,
        ref_pca,
        query_pca,
        explained,
        width_factor_2d,
        width_factor_1d,
        grid_source_2d,
        fixed_grid_data_2d,
        grid_source_1d,
        fixed_grid_data_1d,
        ref_data_by_mode,
        query_data_by_mode,
        fixed_grid_data_1d_by_mode,
        _coverage_mode,
        _pca_txt_dir,
    ) = args

    if ref_data.shape[0] == 0 or query_data.shape[0] == 0:
        raise ValueError(f"Empty descriptor data for {dataset_label}/{element}")
    if ref_data.shape[1] != query_data.shape[1]:
        raise ValueError(
            f"Feature mismatch for {dataset_label}/{element}: input={ref_data.shape}, query={query_data.shape}"
        )
    if ref_data.shape[1] < 2:
        raise ValueError(f"Need at least two descriptor features for 2D PCA: {dataset_label}/{element}")

    coverage_2d, covered_2d_bool, covered_2d_count = fd_grid_coverage(
        ref_pca,
        query_pca,
        width_factor_2d,
        grid_source_2d,
        fixed_grid_data_2d,
    )
    coverage_1d, covered_1d_mean_count, covered_1d_scores = mean_1d_coverage(
        ref_data,
        query_data,
        width_factor_1d,
        grid_source_1d,
        fixed_grid_data_1d,
    )

    coverage_1d_by_mode: Dict[str, float] = {}
    covered_1d_mean_count_by_mode: Dict[str, float] = {}
    for body, body_ref_data in ref_data_by_mode.items():
        body_query_data = query_data_by_mode.get(body)
        if body_query_data is None:
            continue
        if body_ref_data.shape[0] == 0 or body_query_data.shape[0] == 0:
            continue
        if body_ref_data.shape[1] != body_query_data.shape[1]:
            raise ValueError(
                f"Feature mismatch for {dataset_label}/{element}/{body}: "
                f"input={body_ref_data.shape}, query={body_query_data.shape}"
            )
        body_fixed_grid = fixed_grid_data_1d_by_mode.get(body) if fixed_grid_data_1d_by_mode else None
        body_coverage, body_covered_count, _ = mean_1d_coverage(
            body_ref_data,
            body_query_data,
            width_factor_1d,
            grid_source_1d,
            body_fixed_grid,
        )
        coverage_1d_by_mode[body] = body_coverage
        covered_1d_mean_count_by_mode[body] = body_covered_count

    return CoverageResult(
        dataset_label=dataset_label,
        element=element,
        ele_type=ele_type,
        coverage_2d=coverage_2d,
        coverage_1d=coverage_1d,
        coverage_1d_by_mode=coverage_1d_by_mode,
        covered_2d_count=covered_2d_count,
        covered_1d_mean_count=covered_1d_mean_count,
        covered_1d_mean_count_by_mode=covered_1d_mean_count_by_mode,
        query_count=len(query_data),
        input_count=len(ref_data),
        explained_variance_2d=explained,
        ref_pca=ref_pca,
        query_pca=query_pca,
        covered_2d_bool=covered_2d_bool,
        covered_1d_scores=covered_1d_scores,
    )


def write_summary_csv(
    csv_path: Path,
    results: Sequence[CoverageResult],
    coverage_grid: str,
    grid_source_2d: str,
    grid_source_1d: str,
    width_factor_2d: float,
    width_factor_1d: float,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    mode_names: List[str] = []
    for item in results:
        for body in item.coverage_1d_by_mode:
            if body not in mode_names:
                mode_names.append(body)

    fieldnames = [
        "dataset",
        "element",
        "ele_type",
        "coverage_grid",
        "grid_source_2d",
        "grid_source_1d",
        "width_factor_2d",
        "width_factor_1d",
        "coverage_2d",
        "coverage_2d_percent",
        "coverage_1d_mean",
        "coverage_1d_mean_percent",
        "covered_2d_count",
        "covered_1d_mean_count",
    ]
    for body in mode_names:
        fieldnames.extend([
            f"coverage_1d_{body}",
            f"coverage_1d_{body}_percent",
            f"covered_1d_{body}_mean_count",
        ])
    fieldnames.extend([
        "query_count",
        "input_count",
        "explained_variance_2d",
    ])

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        public_grid_source_2d = public_grid_source_name(grid_source_2d)
        public_grid_source_1d = public_grid_source_name(grid_source_1d)
        for item in results:
            row = {
                "dataset": item.dataset_label,
                "element": item.element,
                "ele_type": item.ele_type,
                "coverage_grid": coverage_grid,
                "grid_source_2d": public_grid_source_2d,
                "grid_source_1d": public_grid_source_1d,
                "width_factor_2d": f"{width_factor_2d:.10f}",
                "width_factor_1d": f"{width_factor_1d:.10f}",
                "coverage_2d": f"{item.coverage_2d:.10f}",
                "coverage_2d_percent": f"{item.coverage_2d * 100:.6f}",
                "coverage_1d_mean": f"{item.coverage_1d:.10f}",
                "coverage_1d_mean_percent": f"{item.coverage_1d * 100:.6f}",
                "covered_2d_count": item.covered_2d_count,
                "covered_1d_mean_count": f"{item.covered_1d_mean_count:.6f}",
                "query_count": item.query_count,
                "input_count": item.input_count,
                "explained_variance_2d": f"{item.explained_variance_2d:.10f}",
            }
            for body in mode_names:
                if body in item.coverage_1d_by_mode:
                    row[f"coverage_1d_{body}"] = f"{item.coverage_1d_by_mode[body]:.10f}"
                    row[f"coverage_1d_{body}_percent"] = f"{item.coverage_1d_by_mode[body] * 100:.6f}"
                    row[f"covered_1d_{body}_mean_count"] = f"{item.covered_1d_mean_count_by_mode.get(body, 0.0):.6f}"
                else:
                    row[f"coverage_1d_{body}"] = ""
                    row[f"coverage_1d_{body}_percent"] = ""
                    row[f"covered_1d_{body}_mean_count"] = ""
            writer.writerow(row)


def write_coverage_remark(out_dir: Path) -> Path:
    remark_path = out_dir / "coverage_remark.txt"
    remark_text = """中文说明

在 DCBF 覆盖率图中，默认使用 2D 覆盖率作为显示标准。选择 coverage-mode=2d 时，图中 query/B 点的 covered 和 uncovered 颜色由严格的二维 PCA 网格决定：先把 input/A 和 query/B 都投影到 PC1-PC2 平面，再划分二维网格；如果某个 query 点落入 input/A 已经占据的二维网格，就标记为 covered，否则标记为 uncovered。因此 2D 颜色标记直接对应 PCA 二维空间中的网格覆盖关系。

选择 coverage-mode=1d 时，颜色标记不是直接由 PC1-PC2 二维网格决定。程序先在一维特征方向上计算每个 query 点的 1D 覆盖分数，并得到整体 1D 覆盖率；然后按照这些 1D 覆盖分数排序，根据整体 1D 覆盖率截断出相应数量的 query 点标记为 covered，其余标记为 uncovered。对于多个 loop，程序会保持单调 top-k 标记，也就是后续 loop 不会把前面已经标为 covered 的点再取消。因此 1D 图上的颜色是基于一维覆盖分数排序后的统计标签，不一定表示该点在 PC1-PC2 二维网格中被覆盖。

English Remark

In the DCBF coverage plot, the default display standard is 2D coverage. When coverage-mode=2d is selected, the covered/uncovered colors of query/B points are assigned by a strict 2D PCA grid criterion. Both input/A and query/B are projected onto the PC1-PC2 plane, and a query point is marked as covered only if it falls into a 2D grid cell already occupied by input/A points; otherwise, it is marked as uncovered. Therefore, the 2D color labels directly represent grid coverage in the PCA two-dimensional space.

When coverage-mode=1d is selected, the colors are not assigned directly from the PC1-PC2 grid. The program first calculates a 1D coverage score for each query point along one-dimensional feature directions and obtains the overall 1D coverage rate. It then ranks query points by these 1D coverage scores and marks a corresponding number of top-ranked query points as covered according to the overall 1D coverage rate, while the rest are marked as uncovered. Across multiple loops, the program uses monotonic top-k labeling, meaning that points already marked as covered in earlier loops are not removed in later loops. Thus, 1D colors are statistical labels based on ranked one-dimensional coverage scores, and they do not necessarily mean that the point is covered in the PC1-PC2 grid.
"""
    out_dir.mkdir(parents=True, exist_ok=True)
    remark_path.write_text(remark_text.strip() + "\n", encoding="utf-8")
    return remark_path


def write_pca_point_files_for_results(
    results: Sequence[CoverageResult],
    coverage_mode: str,
    pca_txt_root: Path,
    width_factor_2d: float,
    width_factor_1d: float,
) -> None:
    for item in results:
        if coverage_mode == "1d":
            covered_bool = display_covered_bool(item, coverage_mode)
            coverage = item.coverage_1d
            display_label_source = "monotonic top-k labels derived from 1D descriptor coverage scores"
            width_factor = width_factor_1d
        else:
            covered_bool = item.covered_2d_bool
            coverage = item.coverage_2d
            display_label_source = "2D PCA grid labels"
            width_factor = width_factor_2d

        save_pca_point_files(
            pca_txt_root / item.dataset_label,
            item.element,
            item.ref_pca,
            item.query_pca,
            covered_bool,
            coverage,
            coverage_mode,
            display_label_source,
            width_factor,
        )


def downsample_indices(length: int, max_points: int, seed: int) -> np.ndarray:
    if max_points <= 0 or length <= max_points:
        return np.arange(length)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(length, size=max_points, replace=False))


def finite_xy_limits(
    arrays: Sequence[np.ndarray],
    padding_fraction: float,
) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
    valid_arrays = [np.asarray(array, dtype=float)[:, :2] for array in arrays if len(array) > 0]
    if not valid_arrays:
        return None

    data = np.vstack(valid_arrays)
    finite = data[np.all(np.isfinite(data), axis=1)]
    if len(finite) == 0:
        return None

    mins = np.min(finite, axis=0)
    maxs = np.max(finite, axis=0)
    ranges = maxs - mins
    padding_fraction = max(0.0, float(padding_fraction))
    pads = np.where(ranges > 0, ranges * padding_fraction, 1.0)
    return (
        (float(mins[0] - pads[0]), float(maxs[0] + pads[0])),
        (float(mins[1] - pads[1]), float(maxs[1] + pads[1])),
    )


def coverage_text(item: CoverageResult, mode: str, display_label: Optional[str] = None) -> str:
    label = display_label or item.dataset_label
    if mode == "2d":
        return f"{label} ({item.coverage_2d * 100:.2f}%)"
    if mode == "1d":
        return f"{label} ({item.coverage_1d * 100:.2f}%)"
    raise ValueError(f"Unsupported coverage text mode: {mode}")


def plot_combined_coverage(
    results_by_key: Dict[Tuple[str, str], CoverageResult],
    dataset_labels: Sequence[str],
    display_label_map: Dict[str, str],
    plot_elements: Sequence[str],
    out_path: Path,
    coverage_mode: str,
    show_input: bool,
    max_plot_points: int,
    dpi: int,
    show_ticks: bool,
    axis_padding: float,
    element_label_x: float,
    element_label_y: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.lines import Line2D

    n_rows = len(plot_elements)
    n_cols = len(dataset_labels)
    fig_width = max(4.8 * n_cols, 6)
    fig_height = max(4.2 * n_rows, 4)
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

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), squeeze=False, constrained_layout=True)
    legend_marker_size = 9
    legend_marker_width = 1.3
    legend_items = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markersize=legend_marker_size,
            markerfacecolor="white",
            markeredgecolor="#2E7D32",
            markeredgewidth=legend_marker_width,
            label="Current loop",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markersize=legend_marker_size,
            markerfacecolor="white",
            markeredgecolor="#2B6CB0",
            markeredgewidth=legend_marker_width,
            label="Query covered",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markersize=legend_marker_size,
            markerfacecolor="white",
            markeredgecolor="#C53030",
            markeredgewidth=legend_marker_width,
            label="Query uncovered",
        ),
    ]

    query_indices_by_element: Dict[str, np.ndarray] = {}
    xy_limits_by_element: Dict[str, Tuple[Tuple[float, float], Tuple[float, float]]] = {}
    for row, element in enumerate(plot_elements):
        row_arrays = []
        for label in dataset_labels:
            item = results_by_key.get((label, element))
            if item is None:
                continue
            if element not in query_indices_by_element:
                query_indices_by_element[element] = downsample_indices(
                    len(item.query_pca),
                    max_plot_points,
                    seed=2000 + row * 17,
                )
            row_arrays.append(item.query_pca)
            if show_input:
                row_arrays.append(item.ref_pca)
        limits = finite_xy_limits(row_arrays, axis_padding)
        if limits is not None:
            xy_limits_by_element[element] = limits

    for row, element in enumerate(plot_elements):
        for col, label in enumerate(dataset_labels):
            ax = axes[row, col]
            item = results_by_key.get((label, element))
            display_label = display_label_map.get(label, label)
            if item is None:
                ax.axis("off")
                continue

            ref_idx = downsample_indices(len(item.ref_pca), max_plot_points, seed=1000 + row * 17 + col)
            query_idx = query_indices_by_element.get(element)
            if query_idx is None or (len(query_idx) > 0 and int(query_idx[-1]) >= len(item.query_pca)):
                query_idx = downsample_indices(len(item.query_pca), max_plot_points, seed=2000 + row * 17)

            if show_input:
                ref_xy = item.ref_pca[ref_idx]
                ax.scatter(
                    ref_xy[:, 0],
                    ref_xy[:, 1],
                    s=24,
                    facecolors="white",
                    edgecolors="#2E7D32",
                    linewidths=0.9,
                    alpha=0.8,
                    zorder=2,
                    rasterized=True,
                )

            query_xy = item.query_pca[query_idx]
            query_cov = display_covered_bool(item, coverage_mode)[query_idx]
            if np.any(~query_cov):
                xy = query_xy[~query_cov]
                ax.scatter(
                    xy[:, 0],
                    xy[:, 1],
                    s=24,
                    facecolors="white",
                    edgecolors="#C53030",
                    linewidths=0.9,
                    alpha=0.8,
                    zorder=1,
                    rasterized=True,
                )
            if np.any(query_cov):
                xy = query_xy[query_cov]
                ax.scatter(
                    xy[:, 0],
                    xy[:, 1],
                    s=24,
                    facecolors="white",
                    edgecolors="#2B6CB0",
                    linewidths=0.9,
                    alpha=0.8,
                    zorder=3,
                    rasterized=True,
                )

            if col == 0:
                ax.set_ylabel("")
                ax.text(
                    element_label_x,
                    element_label_y,
                    element,
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=16,
                    fontweight="bold",
                    bbox={
                        "boxstyle": "round,pad=0.18",
                        "facecolor": "none",
                        "edgecolor": "none",
                        "linewidth": 0.0,
                        "alpha": 0.0,
                    },
                )
            else:
                ax.set_ylabel("")
            ax.set_xlabel("")

            limits = xy_limits_by_element.get(element)
            if limits is not None:
                ax.set_xlim(*limits[0])
                ax.set_ylim(*limits[1])

            ax.text(
                0.04,
                0.06,
                coverage_text(item, coverage_mode, display_label),
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=14,
                fontweight="bold",
                bbox={
                    "boxstyle": "round,pad=0.25",
                    "facecolor": "none",
                    "edgecolor": "black",
                    "linewidth": 1.0,
                    "alpha": 1.0,
                },
            )

            if not show_ticks:
                ax.tick_params(
                    axis="both",
                    which="both",
                    bottom=False,
                    left=False,
                    labelbottom=False,
                    labelleft=False,
                )

    legend = axes[0, 0].legend(
        handles=legend_items if show_input else legend_items[1:],
        loc="upper right",
        fontsize=11,
        frameon=True,
    )
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
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def body_mode_label(body: str) -> str:
    return {
        "two": "two-body",
        "three": "three-body",
        "four": "four-body",
    }.get(body, body)


def format_1d_mode_parts(item: CoverageResult) -> str:
    return ", ".join(
        f"{body_mode_label(body)}={coverage * 100:.2f}%"
        for body, coverage in item.coverage_1d_by_mode.items()
    )


def print_coverage_lists(
    results_by_key: Dict[Tuple[str, str], CoverageResult],
    dataset_labels: Sequence[str],
    display_label_map: Dict[str, str],
    plot_elements: Sequence[str],
    coverage_mode: str,
) -> None:
    print("")
    attr = "coverage_2d" if coverage_mode == "2d" else "coverage_1d"
    print(f"Primary coverage mode: {coverage_mode.upper()}")
    for element in plot_elements:
        values = []
        for label in dataset_labels:
            item = results_by_key.get((label, element))
            display_label = display_label_map.get(label, label)
            if item is None:
                values.append(f"{display_label} (nan%)")
            else:
                values.append(f"{display_label} ({getattr(item, attr) * 100:.2f}%)")
        print(f"{element} --> {values}")

    print("2D coverage:")
    for element in plot_elements:
        values = []
        for label in dataset_labels:
            item = results_by_key.get((label, element))
            display_label = display_label_map.get(label, label)
            values.append(f"{display_label} ({item.coverage_2d * 100:.2f}%)" if item is not None else f"{display_label} (nan%)")
        print(f"{element} --> {values}")

    print("1D coverage:")
    for element in plot_elements:
        values = []
        for label in dataset_labels:
            item = results_by_key.get((label, element))
            display_label = display_label_map.get(label, label)
            values.append(f"{display_label} ({item.coverage_1d * 100:.2f}%)" if item is not None else f"{display_label} (nan%)")
        print(f"{element} --> {values}")

    if coverage_mode == "1d":
        mode_names: List[str] = []
        for item in results_by_key.values():
            for body in item.coverage_1d_by_mode:
                if body not in mode_names:
                    mode_names.append(body)
        for body in mode_names:
            print(f"{body_mode_label(body)} 1D coverage:")
            for element in plot_elements:
                values = []
                for label in dataset_labels:
                    item = results_by_key.get((label, element))
                    display_label = display_label_map.get(label, label)
                    if item is None or body not in item.coverage_1d_by_mode:
                        values.append(f"{display_label} (nan%)")
                    else:
                        values.append(f"{display_label} ({item.coverage_1d_by_mode[body] * 100:.2f}%)")
                print(f"{element} --> {values}")


def add_coverage_pca_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--input",
        default=None,
        help=(
            "Single xyz/traj input with frame info main labels. "
            "main=-1 becomes init; main=0 becomes loop-1; main=1 becomes loop-2, cumulatively."
        ),
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help=(
            "DCBF workspace directory. Relative --input/--query/--out-dir paths are resolved from this directory; "
            "when --input is omitted, workflow output all_sample_data.xyz is used."
        ),
    )
    parser.add_argument(
        "--loop-select",
        default="middle-half",
        help=(
            "Cumulative main values to plot. Use all, middle-half, uniform-half, "
            "or a manual main-value list such as [-1,0,2,8] or -1,0,2,8. "
            "main=-1 displays as init; main=0 displays as loop-1."
        ),
    )
    parser.add_argument(
        "--main-key",
        default="main",
        help="Frame info key used to split one xyz/traj into cumulative loop datasets.",
    )
    parser.add_argument(
        "--query",
        default=DEFAULT_QUERY,
        help=(
            "Query/B/AIMD dataset as xyz/traj or descriptor directory. "
            "With --run-dir, this may be omitted when run_dir/npt.xyz exists or a LAMMPS query can be generated."
        ),
    )
    parser.add_argument(
        "--query-structures",
        nargs="+",
        default=["all"],
        help=(
            "Structures under run_dir/stru used for automatic LAMMPS query generation. "
            "Use all, first, index:N, exact file/stem/label, globs such as '*.vasp', or comma-separated values."
        ),
    )
    parser.add_argument("--out-dir", default="xyz_pca_coverage_results", help="Output directory.")
    parser.add_argument(
        "--elements",
        nargs="+",
        default=None,
        help="Element order for descriptor types. If omitted, elements are detected from --input/--query and sorted by atomic number.",
    )
    parser.add_argument(
        "--plot-elements",
        nargs="+",
        default=None,
        help="Elements to plot and print. If omitted, all resolved --elements are used.",
    )
    parser.add_argument(
        "--body-list",
        dest="data_modes",
        nargs="+",
        metavar="BODY_LIST",
        default=DEFAULT_DATA_MODES,
        help="Descriptor body modes to combine.",
    )
    parser.add_argument(
        "--mlp-exe",
        default=argparse.SUPPRESS,
        help="Path to mlp-sus2 executable. Default: current deployment runtime/bin/mlp-sus2, then PATH.",
    )
    parser.add_argument("--mtp", default=DEFAULT_MTP, help="Path to current.mtp.")
    parser.add_argument("--mtp-type", default="l2k2.mtp", choices=["l2k2.mtp", "l2k3.mtp"], help="MTP descriptor type.")
    parser.add_argument("--element-model", type=int, default=1, choices=[1, 2], help="Element mapping mode from old xyz2out script.")
    parser.add_argument("--descriptor-workers", type=int, default=DEFAULT_DESCRIPTOR_WORKERS, help="Workers for each xyz descriptor conversion.")
    parser.add_argument("--coverage-workers", type=int, default=DEFAULT_COVERAGE_WORKERS, help="Workers for coverage calculation.")
    parser.add_argument(
        "--width-factor",
        type=float,
        default=argparse.SUPPRESS,
        help=(
            "Override both 1D and 2D FD bin width multipliers. "
            f"If omitted, 1D defaults to {DEFAULT_WIDTH_FACTOR_1D:g} and 2D defaults to {DEFAULT_WIDTH_FACTOR_2D:g}; "
            "if provided, this value is used for both 1D and 2D coverage."
        ),
    )
    parser.add_argument(
        "--coverage-grid",
        type=normalize_coverage_grid_name,
        choices=["query", "current-loop", "last-loop"],
        default=DEFAULT_COVERAGE_GRID,
        help=(
            "Default FD grid standard used by both 2D and 1D coverage unless overridden. "
            "query fixes the grid to the query/B dataset; "
            "current-loop uses each loop dataset as its own grid; "
            "last-loop fixes it to the final input dataset."
        ),
    )
    parser.add_argument(
        "--pca-fit-source",
        choices=["query", "input", "combined"],
        default="query",
        help=(
            "Data used to fit one shared PCA model per element. "
            "input uses all input datasets together; combined uses all input datasets plus query."
        ),
    )
    parser.add_argument(
        "--coverage-mode",
        choices=["1d", "2d"],
        default="2d",
        help=(
            "Primary coverage mode shown in the plot label and summary. "
            "When set to 1d, query covered/uncovered point tags in the PC1/PC2 plot and pca_B txt use "
            "monotonic top-k labels derived from 1D descriptor coverage scores. "
            "When set to 2d, point tags use strict 2D PCA grid labels."
        ),
    )
    parser.add_argument("--no-plot", action="store_true", help="Only compute coverage and write CSV/txt files; skip the combined PCA figure.")
    parser.add_argument("--show-input", dest="show_input", action="store_true", help="Show input dataset points in the PCA plots.")
    parser.add_argument("--max-plot-points", type=int, default=2000000, help="Max points per group per subplot for plotting; 0 means no downsample.")
    parser.add_argument("--dpi", type=int, default=300, help="Figure DPI.")
    parser.add_argument("--show-ticks", action="store_true", help="Show tick marks and tick labels.")
    parser.add_argument("--axis-padding", type=float, default=0.1, help="Fractional padding added to each side of subplot x/y limits.")
    parser.add_argument("--element-label-x", type=float, default=0.12, help="Element label x position in the first subplot of each row, in axes fraction.")
    parser.add_argument("--element-label-y", type=float, default=0.96, help="Element label y position in the first subplot of each row, in axes fraction.")
    parser.add_argument(
        "--lammps-run-mode",
        choices=["scheduler", "local"],
        default=DEFAULT_LAMMPS_RUN_MODE,
        help=(
            "How automatic --run-dir query generation runs LAMMPS. "
            "scheduler writes bsub.lsf and submits it with the same scheduler backend as sampling; "
            "local runs directly in the current shell and is intended only for explicit debugging."
        ),
    )
    parser.add_argument(
        "--lammps-timeout-hours",
        type=float,
        default=DEFAULT_LAMMPS_TIMEOUT_HOURS,
        help="Hours to wait for scheduler LAMMPS query generation.",
    )
    parser.add_argument("--lammps-timeout", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--lammps-exe", default=None, help="LAMMPS command for automatic --run-dir query generation.")
    parser.add_argument("--lammps-env", default=None, help="Shell setup command used before automatic LAMMPS query generation.")
    parser.add_argument("--lammps-cores", type=int, default=None, help="LAMMPS cores for automatic query generation; defaults to scheduler.lmp_cores.")
    parser.add_argument("--force-query", action="store_true", help="Regenerate coverage_query_lammps/query.xyz instead of reusing an existing query xyz.")
    parser.add_argument("--keep-out", action="store_true", help="Keep merged descriptor .out files after pkl generation.")
    parser.add_argument("--force-recompute", action="store_true", help="Recompute descriptors even if pkl files already exist.")
    return parser


def build_coverage_pca_parser(prog: Optional[str] = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description=(
            "Compute descriptors, shared PCA, 1D/2D FD-grid coverage, and PCA figures for DCBF loop datasets. "
            "Typical: dcbf coverage-pca --input all_sample_data.xyz --query query.xyz --mtp current.mtp"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    return add_coverage_pca_arguments(parser)


def add_coverage_pca_subparser(subparsers) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "coverage-pca",
        help="plot PCA coverage for all_sample_data.xyz against a query/B dataset",
        description=(
            "Compute descriptors, shared PCA, 1D/2D FD-grid coverage, and PCA figures for DCBF loop datasets. "
            "Typical: dcbf coverage-pca --input all_sample_data.xyz --query query.xyz --mtp current.mtp"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    return add_coverage_pca_arguments(parser)


def normalize_coverage_pca_argv(argv: Sequence[str]) -> List[str]:
    normalized = list(argv)
    for index, token in enumerate(normalized[:-1]):
        if token == "--loop-select":
            value = normalized[index + 1]
            if re.fullmatch(r"-?\d+(?:\s*,\s*-?\d+)*", value.strip()):
                normalized[index] = f"--loop-select={value}"
                del normalized[index + 1]
                break
    return normalized


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    return build_coverage_pca_parser().parse_args(normalize_coverage_pca_argv(raw_argv))


def relative_to_run_dir(path_text: str, run_dir: Path) -> str:
    label = None
    raw = path_text
    if "=" in path_text:
        label, raw = path_text.split("=", 1)
    path = Path(raw)
    if not path.is_absolute():
        candidate = run_dir / path
        if candidate.exists() or path.suffix or raw:
            path = candidate
    return f"{label}={path}" if label is not None else str(path)


def configure_run_dir_defaults(args: argparse.Namespace, script_dir: Path) -> Path:
    run_dir = Path(args.run_dir).resolve()
    runtime_config = read_json_if_exists(run_dir / "dcbf.runtime.json")
    workflow = dict(runtime_config.get("workflow") or {})
    scheduler = dict(runtime_config.get("scheduler") or {})

    if not args.input:
        args.input = str(run_dir / workflow.get("output_xyz_name", "all_sample_data.xyz"))
    elif args.input:
        args.input = relative_to_run_dir(args.input, run_dir)

    out_path = Path(args.out_dir)
    if not out_path.is_absolute():
        args.out_dir = str((run_dir / out_path).resolve())

    if getattr(args, "mlp_exe", None) is None and scheduler.get("sus2_mlp_exe"):
        args.mlp_exe = scheduler["sus2_mlp_exe"]

    mtp_path = Path(args.mtp)
    if args.mtp == DEFAULT_MTP or not resolve_path(mtp_path, script_dir).exists():
        default_mtp = find_default_mtp(run_dir)
        if default_mtp is not None:
            args.mtp = str(default_mtp)

    query_label, query_path = parse_label_path(args.query if "=" in args.query else f"query={args.query}")
    resolved_query = query_path if query_path.is_absolute() else run_dir / query_path
    if resolved_query.exists():
        args.query = f"{query_label}={resolved_query}"
    elif args.query == DEFAULT_QUERY or args.query.endswith(f"={DEFAULT_QUERY}"):
        args.query = f"{query_label}={ensure_lammps_query(run_dir, args)}"
    else:
        args.query = f"{query_label}={resolved_query}"
    return run_dir


def resolve_dataset_items_from_args(
    args: argparse.Namespace,
    out_dir: Path,
    script_dir: Path,
) -> List[Tuple[str, Path]]:
    if not args.input:
        raise ValueError("No input dataset was provided. Use --input all_sample_data.xyz.")

    input_label, split_source = parse_label_path(args.input)
    if input_label and "=" in args.input:
        print(f"Ignoring --input label {input_label!r}; loop labels come from {args.main_key}.")
    label_values = discover_main_label_values(split_source, args.main_key, script_dir)
    selected_labels = select_loop_labels(label_values, args.loop_select)
    print(f"Selected loop labels: {selected_labels}")
    return split_xyz_by_info_label(
        split_source,
        args.main_key,
        label_values,
        out_dir,
        script_dir,
        selected_labels=selected_labels,
    )


def resolve_elements_from_sources(
    args: argparse.Namespace,
    dataset_items: Sequence[Tuple[str, Path]],
    query_source: Path,
    script_dir: Path,
) -> None:
    if args.elements:
        args.elements = list(args.elements)
    else:
        detected = set()
        for _, source in dataset_items:
            detected.update(collect_elements_from_xyz(source, script_dir))
        detected.update(collect_elements_from_xyz(query_source, script_dir))
        if not detected:
            raise ValueError(
                "Could not auto-detect elements from --input/--query xyz files. "
                "Pass --elements explicitly when using descriptor directories or unreadable inputs."
            )
        args.elements = sort_elements_by_atomic_number(detected)
        print(f"Auto-detected elements: {args.elements}")

    if args.plot_elements:
        args.plot_elements = list(args.plot_elements)
    else:
        args.plot_elements = list(args.elements)


def main_from_args(args: argparse.Namespace) -> int:
    resolve_width_factors(args)
    resolve_grid_sources(args)
    script_dir = Path(__file__).resolve().parent
    if args.run_dir:
        configure_run_dir_defaults(args, script_dir)
    out_dir = Path(args.out_dir).resolve()
    descriptor_root = out_dir / "descriptors"
    pca_txt_root = out_dir / "pca_txt"

    configured_mlp_exe = getattr(args, "mlp_exe", None)
    if configured_mlp_exe is None or str(configured_mlp_exe).strip().lower() in {"", "auto", "default"}:
        mlp_exe = resolve_default_mlp_exe(script_dir)
        args.mlp_exe = str(mlp_exe)
    else:
        mlp_exe = resolve_path(Path(configured_mlp_exe), script_dir)
    mtp_path = resolve_path(Path(args.mtp), script_dir)

    if not mtp_path.exists():
        raise FileNotFoundError(f"MTP file not found: {mtp_path}")
    if not mlp_exe.exists():
        print(f"Warning: mlp executable not found now: {mlp_exe}")
        print("Descriptor conversion will fail unless this path exists in the runtime environment.")

    dataset_items = resolve_dataset_items_from_args(args, out_dir, script_dir)

    dataset_labels = [label for label, _ in dataset_items]
    if len(set(dataset_labels)) != len(dataset_labels):
        raise ValueError(f"Dataset labels must be unique: {dataset_labels}")
    query_label, query_source = parse_label_path(args.query if "=" in args.query else f"query={args.query}")
    resolve_elements_from_sources(args, dataset_items, query_source, script_dir)
    display_label_map = build_display_label_map(dataset_labels)

    print("Preparing query dataset...")
    query_spec = prepare_dataset(
        query_label,
        query_source,
        descriptor_root,
        script_dir,
        args.elements,
        args.element_model,
        args.descriptor_workers,
        mlp_exe,
        mtp_path,
        args.mtp_type,
        args.data_modes,
        args.keep_out,
        args.force_recompute,
    )

    dataset_specs: List[DatasetSpec] = []
    print("Preparing input datasets...")
    for label, source in dataset_items:
        spec = prepare_dataset(
            label,
            source,
            descriptor_root,
            script_dir,
            args.elements,
            args.element_model,
            args.descriptor_workers,
            mlp_exe,
            mtp_path,
            args.mtp_type,
            args.data_modes,
            args.keep_out,
            args.force_recompute,
        )
        dataset_specs.append(spec)

    print("Loading descriptors...")
    query_aee, _, query_ele_num = load_all_data(query_spec.descriptor_dir, args.data_modes)
    query_aee_by_mode, query_mode_ele_num = load_aee_data_by_mode(query_spec.descriptor_dir, args.data_modes)
    ele_num = len(args.elements)
    if query_ele_num != ele_num:
        raise ValueError(f"Query element count mismatch: expected {ele_num}, got {query_ele_num}")
    if query_mode_ele_num != ele_num:
        raise ValueError(f"Query body-mode element count mismatch: expected {ele_num}, got {query_mode_ele_num}")

    ref_aee_by_label: Dict[str, Dict[int, np.ndarray]] = {}
    ref_aee_by_label_by_mode: Dict[str, Dict[str, Dict[int, np.ndarray]]] = {}
    for spec in dataset_specs:
        ref_aee, _, ref_ele_num = load_all_data(spec.descriptor_dir, args.data_modes)
        ref_aee_by_mode, ref_mode_ele_num = load_aee_data_by_mode(spec.descriptor_dir, args.data_modes)
        if ref_ele_num != ele_num:
            raise ValueError(f"{spec.label} element count mismatch: expected {ele_num}, got {ref_ele_num}")
        if ref_mode_ele_num != ele_num:
            raise ValueError(f"{spec.label} body-mode element count mismatch: expected {ele_num}, got {ref_mode_ele_num}")
        ref_aee_by_label[spec.label] = ref_aee
        ref_aee_by_label_by_mode[spec.label] = ref_aee_by_mode

    element_to_type = {element: index for index, element in enumerate(args.elements)}
    for element in args.plot_elements:
        if element not in element_to_type:
            raise ValueError(f"Plot element {element} is not present in --elements")

    pca_model_by_type: Dict[int, PcaModel] = {}
    for element in args.elements:
        ele_type = element_to_type[element]
        ref_arrays = [ref_aee_by_label[spec.label][ele_type] for spec in dataset_specs]
        fit_data = choose_element_fit_data(ref_arrays, query_aee[ele_type], args.pca_fit_source)
        pca_model_by_type[ele_type] = fit_pca_model(fit_data, 2)

    fixed_grid_2d_by_type: Dict[int, Optional[np.ndarray]] = {}
    fixed_grid_1d_by_type: Dict[int, Optional[np.ndarray]] = {}
    fixed_grid_1d_by_mode_by_type: Dict[int, Dict[str, Optional[np.ndarray]]] = {}
    final_dataset_label = dataset_specs[-1].label
    print(f"FD width factors: 1D={args.width_factor_1d:g}, 2D={args.width_factor_2d:g}")
    if getattr(args, "width_factor", None) is not None:
        print(f"FD width factor override: shared --width-factor={args.width_factor:g}")
    print(f"Coverage grid standard: {args.coverage_grid}")
    print(f"Coverage grid source: {describe_grid_source(args.grid_source_2d, query_label, final_dataset_label)}")
    print(f"Primary coverage mode: {args.coverage_mode.upper()}")
    uses_last_ref_grid = args.grid_source_2d == "last-ref" or args.grid_source_1d == "last-ref"
    if uses_last_ref_grid:
        print(f"Using fixed FD grid from final input dataset: {final_dataset_label}")

    for element in args.elements:
        ele_type = element_to_type[element]
        if args.grid_source_2d == "last-ref":
            pca_model = pca_model_by_type[ele_type]
            final_ref_data = ref_aee_by_label[final_dataset_label][ele_type]
            fixed_grid_2d_by_type[ele_type] = transform_pca(final_ref_data, pca_model)
        else:
            fixed_grid_2d_by_type[ele_type] = None

        fixed_grid_1d_by_mode_by_type[ele_type] = {}
        if args.grid_source_1d == "last-ref":
            fixed_grid_1d_by_type[ele_type] = ref_aee_by_label[final_dataset_label][ele_type]
            for body in args.data_modes:
                fixed_grid_1d_by_mode_by_type[ele_type][body] = ref_aee_by_label_by_mode[final_dataset_label][body][ele_type]
        else:
            fixed_grid_1d_by_type[ele_type] = None
            for body in args.data_modes:
                fixed_grid_1d_by_mode_by_type[ele_type][body] = None

    tasks = []
    for spec in dataset_specs:
        txt_dir = pca_txt_root / spec.label
        for element in args.elements:
            ele_type = element_to_type[element]
            ref_data = ref_aee_by_label[spec.label][ele_type]
            query_data = query_aee[ele_type]
            pca_model = pca_model_by_type[ele_type]
            tasks.append((
                spec.label,
                element,
                ele_type,
                ref_data,
                query_data,
                transform_pca(ref_data, pca_model),
                transform_pca(query_data, pca_model),
                pca_model.explained,
                args.width_factor_2d,
                args.width_factor_1d,
                args.grid_source_2d,
                fixed_grid_2d_by_type[ele_type],
                args.grid_source_1d,
                fixed_grid_1d_by_type[ele_type],
                {body: ref_aee_by_label_by_mode[spec.label][body][ele_type] for body in args.data_modes},
                {body: query_aee_by_mode[body][ele_type] for body in args.data_modes},
                fixed_grid_1d_by_mode_by_type[ele_type],
                args.coverage_mode,
                txt_dir,
            ))

    print(f"Computing coverage for {len(tasks)} dataset-element pairs...")
    results: List[CoverageResult] = []
    with ThreadPoolExecutor(max_workers=max(1, args.coverage_workers)) as executor:
        future_to_name = {
            executor.submit(compute_coverage_for_pair, task): (task[0], task[1])
            for task in tasks
        }
        for future in as_completed(future_to_name):
            label, element = future_to_name[future]
            try:
                result = future.result()
                results.append(result)
                if element in args.plot_elements:
                    display_label = display_label_map.get(label, label)
                    mode_parts = format_1d_mode_parts(result)
                    mode_detail = f", {mode_parts}" if mode_parts else ""
                    print(
                        f"{display_label}/{element}: 1D={result.coverage_1d * 100:.2f}%, "
                        f"2D={result.coverage_2d * 100:.2f}%{mode_detail}, "
                        f"width(1D/2D)={args.width_factor_1d:g}/{args.width_factor_2d:g}, "
                        f"primary={args.coverage_mode.upper()}, "
                        f"var2={result.explained_variance_2d * 100:.2f}%"
                    )
            except Exception as exc:
                print(f"{label}/{element}: failed: {exc}")

    results.sort(key=lambda item: (dataset_labels.index(item.dataset_label), item.ele_type))
    results_by_key = {(item.dataset_label, item.element): item for item in results}
    if args.coverage_mode == "1d":
        monotonic_topk_labels(results_by_key, dataset_labels, args.elements)
    write_pca_point_files_for_results(
        results,
        args.coverage_mode,
        pca_txt_root,
        args.width_factor_2d,
        args.width_factor_1d,
    )

    summary_csv = out_dir / "coverage_summary.csv"
    write_summary_csv(
        summary_csv,
        results,
        args.coverage_grid,
        args.grid_source_2d,
        args.grid_source_1d,
        args.width_factor_2d,
        args.width_factor_1d,
    )
    remark_path = write_coverage_remark(out_dir)

    figure_path = None
    if not args.no_plot:
        figure_name = "combined_pca_coverage_" + "_".join(args.plot_elements) + ".jpg"
        figure_path = out_dir / figure_name
        try:
            plot_combined_coverage(
                results_by_key,
                dataset_labels,
                display_label_map,
                args.plot_elements,
                figure_path,
                args.coverage_mode,
                args.show_input,
                args.max_plot_points,
                args.dpi,
                args.show_ticks,
                args.axis_padding,
                args.element_label_x,
                args.element_label_y,
            )
        except Exception as exc:
            print(f"Plotting failed after coverage calculation: {exc}")
            figure_path = None

    print_coverage_lists(results_by_key, dataset_labels, display_label_map, args.plot_elements, args.coverage_mode)
    print("")
    if figure_path is not None:
        print(f"Figure saved to: {figure_path}")
    elif args.no_plot:
        print("Figure skipped (--no-plot).")
    else:
        print("Figure was not saved because plotting failed.")
    print(f"Summary CSV saved to: {summary_csv}")
    print(f"Coverage remark saved to: {remark_path}")
    print(f"PCA txt files saved under: {pca_txt_root}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    return main_from_args(parse_args(argv))


def handle_coverage_pca_command(args: argparse.Namespace) -> int:
    return main_from_args(args)


def run_coverage_pca_from_config(config: Dict, run_dir: Path) -> int:
    coverage_cfg = dict(config.get("coverage_plot") or {})
    if not coverage_cfg.get("enabled", False):
        return 0

    workflow = dict(config.get("workflow") or {})
    scheduler = dict(config.get("scheduler") or {})
    run_dir = Path(run_dir).resolve()

    unsupported_keys = [
        "coverage_label",
        "data_modes",
        "loops",
        "label_vars",
        "n1_label",
        "n2_label",
        "main_labels",
    ]
    present_unsupported = [key for key in unsupported_keys if key in coverage_cfg and coverage_cfg[key] is not None]
    if present_unsupported:
        raise ValueError(
            "Unsupported coverage_plot field(s): "
            + ", ".join(present_unsupported)
            + ". Use coverage_mode, loop_select, and body_list with the simplified coverage-pca interface."
        )

    argv: List[str] = [
        "--run-dir",
        str(run_dir),
        "--input",
        str(run_dir / workflow.get("output_xyz_name", "all_sample_data.xyz")),
        "--out-dir",
        str(coverage_cfg.get("output_dir", "xyz_pca_coverage_results")),
        "--loop-select",
        str(coverage_cfg.get("loop_select", "middle-half")),
        "--coverage-mode",
        str(coverage_cfg.get("coverage_mode", "2d")),
        "--coverage-grid",
        str(coverage_cfg.get("coverage_grid", DEFAULT_COVERAGE_GRID)),
        "--pca-fit-source",
        str(coverage_cfg.get("pca_fit_source", "query")),
    ]

    if coverage_cfg.get("query"):
        argv.extend(["--query", str(coverage_cfg["query"])])
    elif coverage_cfg.get("query_xyz"):
        argv.extend(["--query", str(coverage_cfg["query_xyz"])])

    for key, option in (
        ("width_factor", "--width-factor"),
        ("axis_padding", "--axis-padding"),
        ("max_plot_points", "--max-plot-points"),
        ("dpi", "--dpi"),
        ("descriptor_workers", "--descriptor-workers"),
        ("coverage_workers", "--coverage-workers"),
        ("lammps_run_mode", "--lammps-run-mode"),
        ("lammps_timeout_hours", "--lammps-timeout-hours"),
        ("lammps_cores", "--lammps-cores"),
        ("mtp_type", "--mtp-type"),
        ("element_model", "--element-model"),
    ):
        if key in coverage_cfg and coverage_cfg[key] is not None:
            argv.extend([option, str(coverage_cfg[key])])

    if coverage_cfg.get("lammps_timeout_hours") is None and coverage_cfg.get("lammps_timeout") is not None:
        argv.extend(["--lammps-timeout-hours", str(float(coverage_cfg["lammps_timeout"]) / 3600)])

    if coverage_cfg.get("query_structures") is not None:
        query_structures = coverage_cfg["query_structures"]
        argv.append("--query-structures")
        if isinstance(query_structures, (list, tuple)):
            argv.extend(str(item) for item in query_structures)
        else:
            argv.append(str(query_structures))

    if coverage_cfg.get("mlp_exe"):
        argv.extend(["--mlp-exe", str(coverage_cfg["mlp_exe"])])
    elif scheduler.get("sus2_mlp_exe"):
        argv.extend(["--mlp-exe", str(scheduler["sus2_mlp_exe"])])

    if coverage_cfg.get("mtp"):
        argv.extend(["--mtp", str(coverage_cfg["mtp"])])
    if coverage_cfg.get("lammps_exe"):
        argv.extend(["--lammps-exe", str(coverage_cfg["lammps_exe"])])
    if coverage_cfg.get("lammps_env"):
        argv.extend(["--lammps-env", str(coverage_cfg["lammps_env"])])

    elements = coverage_cfg.get("elements")
    if elements:
        argv.append("--elements")
        argv.extend(str(item) for item in elements)

    plot_elements = coverage_cfg.get("plot_elements")
    if plot_elements:
        argv.append("--plot-elements")
        argv.extend(str(item) for item in plot_elements)

    data_modes = coverage_cfg.get("body_list")
    if data_modes:
        argv.append("--body-list")
        argv.extend(str(item) for item in data_modes)

    if coverage_cfg.get("show_input", True):
        argv.append("--show-input")
    if coverage_cfg.get("no_plot", False):
        argv.append("--no-plot")
    if coverage_cfg.get("show_ticks", False):
        argv.append("--show-ticks")
    if coverage_cfg.get("force_query", False):
        argv.append("--force-query")
    if coverage_cfg.get("force_recompute", False):
        argv.append("--force-recompute")
    if coverage_cfg.get("keep_out", False):
        argv.append("--keep-out")

    print("[coverage_pca] " + " ".join(argv))
    return main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
