from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import tempfile
import time
import warnings

from ase.data import atomic_numbers, chemical_symbols
from ase.io import iread, write
import numpy as np

from .das.file_conversion import xyz2cfg
from .encode.data_distri import Freedman_Diaconis_bins, scott, data_base_distribution
from .encode.find_min_cover_set import find_min_cover_set
from .encode.dimension_min_cover import (
    build_dimension_tasks,
    normalize_dimension_min_cover_workers,
    solve_dimension_tasks,
)
from .encode.compact_indices import find_multi_cover_set_compact, group_compact_indices
from .encode.descriptor_store import (
    DescriptorRows,
    build_descriptor_store,
    column,
    concatenate_rows,
    descriptor_store_signature,
    file_fingerprint,
    load_descriptor_store,
    values_and_indices,
)
from .encode.mlp_encode_sample_flow import md_extract
from .encode.mlp_encoding_extract import decode, des_out2pkl
from .fast_extxyz import (
    MalformedXYZ,
    UnsupportedFastXYZ,
    complement_indices,
    index_xyz_frames,
    supports_raw_xyz,
    write_indexed_frames,
    write_position_cfg_part,
    write_position_cfg_parts,
)
from .mtp import normalize_mtp_type
from .memory_guard import MIB, atomic_json, descriptor_guard, require_memory, stage_progress, tree_memory
from .runtime_config import load_json_config
from .selection.core import group_structure_indices_by_interval


DEFAULT_DIRECT_ELEMENTS = [
    "H", "Li", "Be", "B", "C", "N", "O", "F", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Rb",
    "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I",
    "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Ac", "Th", "Pa",
    "U", "Np", "Pu",
]

DEFAULT_UNIVERSAL_MTP_NAME = "MP_UIP_l2k3.mtp"
DEFAULT_UNIVERSAL_MTP_TYPE = "l2k3"
XYZ_IO_MODES = {"fast_extxyz", "auto", "ase"}

# Scratch directories are tagged with host + PID so a shared filesystem can tell
# an abandoned run from a live one on another node.
_SCRATCH_HOST = re.sub(r"[^A-Za-z0-9_.-]", "-", socket.gethostname()) or "unknown"


def default_reduce_sus2_mlp_exe() -> str:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "runtime" / "bin" / "mlp-sus2"
        if candidate.exists():
            return str(candidate)
    return "mlp-sus2"


def _resolve_path(base_dir, raw_path):
    if raw_path is None:
        return None
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _write_xyz(path, atoms):
    path = Path(path)
    if atoms:
        write(path, atoms, format="extxyz")
    else:
        path.write_text("", encoding="utf-8")


def _count_xyz_structures(path):
    if path is None:
        return 0
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return 0
    return sum(1 for _ in iread(str(path)))


def _coerce_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "on"}:
            return True
        if lowered in {"false", "0", "no", "off"}:
            return False
    return bool(value)


class _ProgressTracker:
    def __init__(self, label, total):
        self.label = label
        self.total = max(1, int(total))
        self.start = time.perf_counter()

    def update(self, completed, detail=""):
        completed = min(max(int(completed), 0), self.total)
        fraction = completed / self.total
        elapsed = time.perf_counter() - self.start
        eta = (elapsed / fraction - elapsed) if fraction > 0 and completed < self.total else 0.0
        width = 24
        filled = min(width, int(round(width * fraction)))
        bar = "[" + "#" * filled + "-" * (width - filled) + "]"
        message = (
            f"\r[reduce] {self.label} {bar} {completed}/{self.total} "
            f"{fraction * 100:6.2f}% elapsed={elapsed / 3600.0:.4f}h eta={eta / 3600.0:.4f}h"
        )
        if detail:
            message += f" {detail}"
        print(message, end="", flush=True)
        if completed >= self.total:
            print(flush=True)


def _infer_elements_from_xyz(paths):
    element_set = set()
    for path in paths:
        if path is None:
            continue
        for atoms in iread(str(path)):
            element_set.update(atoms.get_chemical_symbols())
    ordered_atomic_numbers = sorted(atomic_numbers[element] for element in element_set)
    return [chemical_symbols[number] for number in ordered_atomic_numbers]


def _read_mtp_species_count(mtp_path):
    try:
        text = Path(mtp_path).read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    match = re.search(r"species_count\s*=\s*(\d+)", text)
    if match is None:
        return None
    return int(match.group(1))


class _SelectedRowStore:
    """Frame-ordered rows of already-selected frames, held on disk.

    Chunked reduce encodes one chunk at a time so the descriptor cache never
    covers the whole input. Later chunks still need every earlier selection as
    reference data, so those rows are appended here and mapped back read-only
    instead of keeping the chunk caches alive.
    """

    def __init__(self, directory, bodies, element_count, dimensions=None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.bodies = list(dict.fromkeys(bodies))
        self.element_count = int(element_count)
        self._dimensions = dict(dimensions or {})
        self._counts = {}
        self._manifest_path = self.directory / "manifest.json"
        self._load_manifest()

    def _manifest(self):
        return {
            "version": 1,
            "bodies": list(self.bodies),
            "element_count": self.element_count,
            "dimensions": {str(key): int(value) for key, value in self._dimensions.items()},
            "counts": {
                f"{body}:{element}": int(count)
                for (body, element), count in self._counts.items()
            },
        }

    def _save_manifest(self):
        atomic_json(self._manifest_path, self._manifest(), durable=False)

    def _load_manifest(self):
        if not self._manifest_path.exists():
            if any(self.directory.glob("*.bin")):
                raise RuntimeError(
                    f"Selected-row store has data but no manifest: {self.directory}"
                )
            return
        manifest = json.loads(self._manifest_path.read_text(encoding="utf-8"))
        if manifest.get("version") != 1:
            raise RuntimeError(f"Unsupported selected-row store: {self.directory}")
        if manifest.get("bodies") != self.bodies or int(manifest.get("element_count", -1)) != self.element_count:
            raise RuntimeError(f"Selected-row store configuration mismatch: {self.directory}")
        self._dimensions.update({str(key): int(value) for key, value in manifest.get("dimensions", {}).items()})
        self._counts = {}
        for key, value in manifest.get("counts", {}).items():
            body, element = key.rsplit(":", 1)
            self._counts[(body, int(element))] = int(value)
        self._repair_files()

    def _repair_files(self):
        expected_names = set()
        for body in self.bodies:
            dimensions = int(self._dimensions.get(body, 0))
            for element in range(self.element_count):
                count = self._count(body, element)
                value_name = f"{body}_{element}.values.bin"
                index_name = f"{body}_{element}.indices.bin"
                expected_names.update((value_name, index_name))
                expected_values = count * dimensions * np.dtype(np.float64).itemsize
                expected_indices = count * np.dtype(np.int64).itemsize
                for name, expected in ((value_name, expected_values), (index_name, expected_indices)):
                    path = self.directory / name
                    actual = path.stat().st_size if path.exists() else 0
                    if actual < expected:
                        raise RuntimeError(f"Truncated selected-row store file: {path}")
                    if actual > expected:
                        with path.open("r+b") as handle:
                            handle.truncate(expected)
        for path in self.directory.glob("*.bin"):
            if path.name not in expected_names:
                path.unlink()

    def restore_manifest(self, manifest):
        if manifest is None:
            return
        if manifest.get("version") != 1:
            raise RuntimeError("Unsupported selected-row checkpoint manifest")
        expected = dict(manifest)
        current = self._manifest()
        if expected.get("bodies") != current.get("bodies") or int(expected.get("element_count", -1)) != current.get("element_count"):
            raise RuntimeError("Selected-row checkpoint configuration mismatch")
        self._dimensions = {str(key): int(value) for key, value in expected.get("dimensions", {}).items()}
        self._counts = {}
        for key, value in expected.get("counts", {}).items():
            body, element = key.rsplit(":", 1)
            self._counts[(body, int(element))] = int(value)
        self._repair_files()
        self._save_manifest()

    def reset(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        for path in self.directory.glob("*.bin"):
            path.unlink()
        self._counts = {}
        self._save_manifest()

    def _count(self, body, element):
        return self._counts.get((body, element), 0)

    def _append(self, name, payload):
        # Opened per write on purpose: one handle per (body, element) would be
        # hundreds of descriptors held for the whole run, and the process also
        # needs descriptors for every memmapped store it reads.
        with (self.directory / f"{name}.bin").open("ab") as handle:
            handle.write(payload)

    def append_selection(self, chunk_store, body, selected_indices):
        selected = np.asarray(sorted({int(index) for index in selected_indices}), dtype=np.int64)
        if not len(selected):
            return
        dimensions = len(chunk_store.manifest["body_columns"][body])
        self._dimensions[body] = dimensions
        for element, rows in enumerate(chunk_store.body(body)):
            picked = rows.select_frames(selected)
            if not len(picked):
                continue
            values = np.ascontiguousarray(
                np.concatenate([part[0] for part in picked.parts]) if len(picked.parts) > 1
                else picked.parts[0][0]
            )
            indices = np.ascontiguousarray(
                np.concatenate([part[1] for part in picked.parts]) if len(picked.parts) > 1
                else picked.parts[0][1]
            )
            self._append(f"{body}_{element}.values", memoryview(values))
            self._append(f"{body}_{element}.indices", memoryview(indices))
            self._counts[(body, element)] = self._count(body, element) + len(indices)
        self._save_manifest()

    def freeze(self):
        """Expose the accumulated rows read-only, mapped from the appended files."""
        frozen = {}
        for body in self.bodies:
            dimensions = self._dimensions.get(body, 0)
            rows = []
            for element in range(self.element_count):
                count = self._count(body, element)
                values_path = self.directory / f"{body}_{element}.values.bin"
                indices_path = self.directory / f"{body}_{element}.indices.bin"
                if count and dimensions:
                    values = np.memmap(
                        values_path, dtype=np.float64, mode="r", shape=(count, dimensions)
                    )
                    indices = np.memmap(
                        indices_path, dtype=np.int64, mode="r", shape=(count,)
                    )
                else:
                    values = np.empty((0, dimensions), dtype=np.float64)
                    indices = np.empty(0, dtype=np.int64)
                rows.append(DescriptorRows([(values, indices)], dimensions))
            frozen[body] = rows
        return frozen


def _raise_file_limit():
    """Lift the soft descriptor limit towards the hard limit.

    A descriptor store keeps one open mapping per (body, element) file, and the
    selection path reads several stores at once, so a run over a wide element
    table needs far more descriptors than the usual soft limit of 1024.
    """
    try:
        import resource
    except ImportError:
        return None
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        if hard != resource.RLIM_INFINITY and soft < hard:
            resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
            return int(hard)
        return int(soft) if soft != resource.RLIM_INFINITY else None
    except (ValueError, OSError):
        return None


def _process_alive(pid):
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except (PermissionError, OSError):
        return True
    return True


class DCBFReducer:
    def __init__(self, config_path):
        self.config_path = Path(config_path).resolve()
        self.config_dir = self.config_path.parent
        self.config = load_json_config(self.config_path)
        self.input_config_snapshot = json.loads(json.dumps(self.config))

        reduce_cfg = self.config.get("reduce", {})
        parameter = self.config.get("parameter", {})
        scheduler = self.config.get("scheduler", {})
        legacy_reduce_keys = [
            key for key in (
                "train_xyz",
                "bw_ref_xyz",
                "iw_ref_xyz",
                "new_database",
                "new_xyz",
                "current_database",
                "coverage_count_threshold",
                "report_coverage_count_threshold_zero_baseline",
                "mean_descriptor_coverage_count_threshold",
                "bw_method",
                "bw",
                "bw_coff",
                "iw_method",
                "iw",
                "iw_scale",
                "dynamic_iw",
            )
            if key in reduce_cfg
        ]
        legacy_parameter_keys = [
            key for key in (
                "ele_model",
                "bw_method",
                "bw",
                "bw_coff",
                "coverage_count_threshold",
                "report_coverage_count_threshold_zero_baseline",
                "mean_descriptor_coverage_count_threshold",
                "iw_method",
                "iw",
                "iw_scale",
                "dynamic_iw",
            )
            if key in parameter
        ]
        if legacy_reduce_keys or legacy_parameter_keys:
            raise ValueError(
                "Legacy keys are no longer supported. "
                f"reduce: {legacy_reduce_keys}, parameter: {legacy_parameter_keys}. "
                "Use sort_ele / dq_width_method / dq_width / dq_width_factor / dynamic_dq_width / "
                "interval_ref_xyz / candidate_only / reference_guided / state_population."
            )
        self.default_universal_mtp_path = (
            Path(__file__).resolve().parent / "default_reduce_assets" / DEFAULT_UNIVERSAL_MTP_NAME
        )

        raw_mode = str(reduce_cfg.get("mode", "candidate_only")).strip().lower().replace("-", "_")
        if raw_mode == "candidate_only":
            self.mode = "candidate_only"
            self._mode_impl = "single"
        elif raw_mode == "reference_guided":
            self.mode = "reference_guided"
            self._mode_impl = "chunked"
        else:
            raise ValueError("reduce.mode must be one of: candidate_only, reference_guided")

        self.input_xyz = _resolve_path(
            self.config_dir,
            reduce_cfg.get("input_xyz"),
        )
        if self.input_xyz is None:
            raise ValueError("reduce.input_xyz is required")

        self.current_xyz = _resolve_path(
            self.config_dir,
            reduce_cfg.get("current_xyz"),
        )
        if self._mode_impl == "chunked" and self.current_xyz is None:
            raise ValueError("reduce.current_xyz is required when reduce.mode is reference_guided")
        interval_ref_raw = reduce_cfg.get("interval_ref_xyz")
        if interval_ref_raw is None and self._mode_impl == "chunked":
            self.interval_ref_xyz = self.current_xyz
        else:
            self.interval_ref_xyz = _resolve_path(
                self.config_dir,
                interval_ref_raw,
            )

        self.output_xyz = _resolve_path(
            self.config_dir,
            reduce_cfg.get("output_xyz", "dcbf_reduce_sample.xyz"),
        )
        self.remain_xyz = _resolve_path(
            self.config_dir,
            reduce_cfg.get("remain_xyz", "dcbf_reduce_remain.xyz"),
        )
        self.report_json = _resolve_path(
            self.config_dir,
            reduce_cfg.get("report_json", "dcbf_reduce_report.json"),
        )
        self.work_dir = _resolve_path(
            self.config_dir,
            reduce_cfg.get("work_dir", ".dcbf_reduce_work"),
        )
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.encoding_seconds = 0.0
        self.descriptor_cache_dir = self.work_dir / "descriptor_cache"
        self.descriptor_cache_dir.mkdir(parents=True, exist_ok=True)
        self._descriptor_sources = {}
        self._descriptor_cache_records = []
        self._peak_memory_bytes = tree_memory(os.getpid())
        self._output_temporary_paths = []
        self._fast_xyz_indexes = {}
        self._fast_xyz_fallbacks = []
        self._fast_xyz_fallback_keys = set()
        self._used_fast_xyz_io = False
        self._used_ase_xyz_io = False
        self.xyz_io_mode = str(
            reduce_cfg.get("xyz_io_mode", "fast_extxyz")
        ).strip().lower()
        if self.xyz_io_mode not in XYZ_IO_MODES:
            raise ValueError(
                "reduce.xyz_io_mode must be one of: fast_extxyz, auto, ase"
            )

        self.chunk_size = int(reduce_cfg.get("chunk_size", 1000000))
        if self.chunk_size <= 0:
            raise ValueError("reduce.chunk_size must be > 0")
        self.keep_intermediate = bool(reduce_cfg.get("keep_intermediate", False))
        self.append_current = bool(reduce_cfg.get("append_current", True))
        self.dynamic_dq_width = _coerce_bool(
            reduce_cfg.get("dynamic_dq_width", parameter.get("dynamic_dq_width")),
            default=False,
        )
        self.state_population = max(0, int(reduce_cfg.get("state_population", parameter.get("state_population", 0))))
        self.encoding_cores = int(reduce_cfg.get("encoding_cores", 5))
        if self.encoding_cores <= 0:
            raise ValueError("encoding_cores must be > 0")
        self.dimension_min_cover_workers = normalize_dimension_min_cover_workers(
            reduce_cfg.get("dimension_min_cover_workers", -1)
        )
        self.dimension_min_cover_records = []
        self._dimension_scheduler_warning_printed = False
        self.interval_width_history = []
        self.fixed_interval_widths = None
        self.fixed_interval_width_source = None

        raw_sus2_mlp_exe = scheduler.get("sus2_mlp_exe")
        self.sus2_mlp_exe = raw_sus2_mlp_exe or default_reduce_sus2_mlp_exe()
        if not self.sus2_mlp_exe:
            raise ValueError("scheduler.sus2_mlp_exe is required for reduce mode")
        if not raw_sus2_mlp_exe:
            print(f"[reduce] scheduler.sus2_mlp_exe not set; using default: {self.sus2_mlp_exe}")

        mtp_path_raw = reduce_cfg.get("mtp_path", parameter.get("mtp_path"))
        explicit_elements = reduce_cfg.get("ele", parameter.get("ele"))
        self.use_universal_potential = _coerce_bool(
            reduce_cfg.get("use_universal_potential", parameter.get("use_universal_potential")),
            default=False,
        )
        self.using_default_universal_assets = self.use_universal_potential or mtp_path_raw is None or explicit_elements is None
        if self.using_default_universal_assets:
            self.mtp_path = self.default_universal_mtp_path
        else:
            self.mtp_path = _resolve_path(self.config_dir, mtp_path_raw)
        if not self.mtp_path.exists():
            raise ValueError(f"Reduce potential does not exist: {self.mtp_path}")
        self.mtp_species_count = _read_mtp_species_count(self.mtp_path)

        raw_sort_ele = parameter.get("sort_ele", reduce_cfg.get("sort_ele", True))
        self.sort_elements_by_atomic_number = _coerce_bool(raw_sort_ele, default=True)
        self.dq_width_method = parameter.get(
            "dq_width_method",
            reduce_cfg.get(
                "dq_width_method",
                "Freedman_Diaconis",
            ),
        )
        self.dq_width = float(
            parameter.get(
                "dq_width",
                reduce_cfg.get(
                    "dq_width",
                    0.01,
                ),
            )
        )
        self.dq_width_factor = float(
            parameter.get(
                "dq_width_factor",
                reduce_cfg.get(
                    "dq_width_factor",
                    1.0,
                ),
            )
        )
        self.body_list = list(parameter.get("body_list", reduce_cfg.get("body_list", ["two", "three"])))
        requested_mtp_type = parameter.get("mtp_type", reduce_cfg.get("mtp_type"))
        if self.using_default_universal_assets:
            self.mtp_type = DEFAULT_UNIVERSAL_MTP_TYPE
            if requested_mtp_type and normalize_mtp_type(requested_mtp_type) != DEFAULT_UNIVERSAL_MTP_TYPE:
                print(
                    f"[reduce] Universal potential forces mtp_type={DEFAULT_UNIVERSAL_MTP_TYPE}. "
                    f"Ignoring requested mtp_type={requested_mtp_type}."
                )
        else:
            self.mtp_type = requested_mtp_type
            if not self.mtp_type:
                raise ValueError("parameter.mtp_type (or reduce.mtp_type) is required for reduce mode")
            self.mtp_type = normalize_mtp_type(self.mtp_type)

        inferred_elements = []
        if not self.using_default_universal_assets and not explicit_elements:
            inferred_elements = self._infer_elements(
                [self.input_xyz, self.current_xyz, self.interval_ref_xyz]
            )
        self.direct_self_dedup = self._mode_impl == "single" and self.current_xyz is None and self.interval_ref_xyz is None

        if self.using_default_universal_assets:
            self.elements = list(DEFAULT_DIRECT_ELEMENTS)
        elif explicit_elements:
            self.elements = list(explicit_elements)
        elif (
            self.direct_self_dedup
            and self.mtp_species_count is not None
            and self.mtp_species_count == len(DEFAULT_DIRECT_ELEMENTS)
            and len(inferred_elements) != self.mtp_species_count
        ):
            # Preserve the universal type mapping used by the legacy direct reduce script.
            self.elements = list(DEFAULT_DIRECT_ELEMENTS)
        else:
            self.elements = inferred_elements

        if not self.elements:
            raise ValueError("No elements could be inferred. Please provide parameter.ele or reduce.ele")
        if self.sort_elements_by_atomic_number:
            self.elements = [symbol for _, symbol in sorted((atomic_numbers[item], item) for item in self.elements)]

        if self.mtp_species_count is not None and explicit_elements and len(self.elements) != self.mtp_species_count:
            raise ValueError(
                f"Provided element list length ({len(self.elements)}) does not match mtp species_count "
                f"({self.mtp_species_count})."
            )
        if (
            self.direct_self_dedup
            and self.mtp_species_count is not None
            and len(self.elements) != self.mtp_species_count
        ):
            raise ValueError(
                "candidate_only reduce needs the full potential element mapping. "
                "Please set parameter.ele or reduce.ele to match the mtp species order."
            )
        if self.using_default_universal_assets:
            print(f"[reduce] Using default universal potential: {self.mtp_path}")
            print(f"[reduce] Using default element mapping ({len(self.elements)}): {self.elements}")
        print(
            "[reduce] Element ordering by atomic number: "
            f"{self.sort_elements_by_atomic_number}"
        )
        print(f"[reduce] Encoding cores: {self.encoding_cores}")
        print(f"[reduce] XYZ I/O mode: {self.xyz_io_mode}")
        print(f"[reduce] Dimension min-cover workers: {self.dimension_min_cover_workers}")
        print(f"[reduce] State population threshold: {self.state_population}")
        print(f"[reduce] Dynamic dq width update: {self.dynamic_dq_width}")

    def _build_effective_config(self):
        return {
            "mode": self.mode,
            "use_universal_potential": self.use_universal_potential,
            "using_default_universal_assets": self.using_default_universal_assets,
            "sort_ele": self.sort_elements_by_atomic_number,
            "encoding_cores": self.encoding_cores,
            "xyz_io_mode": self.xyz_io_mode,
            "dimension_min_cover_workers": self.dimension_min_cover_workers,
            "mtp_type": self.mtp_type,
            "body_list": list(self.body_list),
            "elements": list(self.elements),
            "dq_width_method": self.dq_width_method,
            "dq_width": self.dq_width,
            "dq_width_factor": self.dq_width_factor,
            "dynamic_dq_width": self.dynamic_dq_width,
            "state_population": self.state_population,
            "chunk_size": self.chunk_size,
            "append_current": self.append_current,
            "keep_intermediate": self.keep_intermediate,
            "paths": {
                "input_xyz": str(self.input_xyz),
                "current_xyz": str(self.current_xyz) if self.current_xyz is not None else None,
                "interval_ref_xyz": str(self.interval_ref_xyz) if self.interval_ref_xyz is not None else None,
                "mtp_path": str(self.mtp_path),
                "output_xyz": str(self.output_xyz),
                "remain_xyz": str(self.remain_xyz),
                "report_json": str(self.report_json),
                "work_dir": str(self.work_dir),
            },
        }

    def _resume_signature(self):
        """Return the immutable inputs that determine a chunked reduce result."""
        effective = self._build_effective_config()
        effective.pop("keep_intermediate", None)
        paths = effective.get("paths", {})
        for key in ("output_xyz", "remain_xyz", "report_json"):
            paths.pop(key, None)
        sources = {}
        for name, path in (
            ("input_xyz", self.input_xyz),
            ("current_xyz", self.current_xyz),
            ("interval_ref_xyz", self.interval_ref_xyz),
        ):
            sources[name] = file_fingerprint(path) if path is not None else None
        return {
            "format": 1,
            "effective_config": effective,
            "source_fingerprints": sources,
            "model_sha256": hashlib.sha256(self.mtp_path.read_bytes()).hexdigest(),
        }

    @property
    def _resume_checkpoint_path(self):
        return self.work_dir / "reduce_checkpoint.json"

    @property
    def _resume_indices_path(self):
        return self.work_dir / "selected_indices.bin"

    def _write_resume_indices(self, selected_indices):
        temporary = self._resume_indices_path.with_name(
            f".{self._resume_indices_path.name}.{os.getpid()}.partial"
        )
        np.asarray(sorted(int(index) for index in selected_indices), dtype=np.int64).tofile(temporary)
        os.replace(temporary, self._resume_indices_path)

    def _read_resume_indices(self):
        if not self._resume_indices_path.exists():
            return set()
        size = self._resume_indices_path.stat().st_size
        if size % np.dtype(np.int64).itemsize:
            raise RuntimeError(f"Invalid selected index checkpoint: {self._resume_indices_path}")
        return set(np.fromfile(self._resume_indices_path, dtype=np.int64).astype(np.int64).tolist())

    def _remove_resume_artifacts(self):
        for path in (self._resume_checkpoint_path, self._resume_indices_path):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        shutil.rmtree(self.work_dir / "selected_rows", ignore_errors=True)

    def _load_resume_state(self, total_chunks):
        path = self._resume_checkpoint_path
        if not path.exists():
            return None
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
            if state.get("format") != 1 or state.get("signature") != self._resume_signature():
                print("[reduce] Ignoring incompatible Reduce checkpoint and starting from chunk 0.")
                self._remove_resume_artifacts()
                return None
            if state.get("status") == "complete":
                print("[reduce] Existing completed checkpoint will not be reused; starting a new Reduce run.")
                self._remove_resume_artifacts()
                return None
            owner_host = state.get("host")
            owner_pid = int(state.get("pid", 0) or 0)
            if owner_host == _SCRATCH_HOST and owner_pid and owner_pid != os.getpid() and _process_alive(owner_pid):
                raise RuntimeError(
                    f"Reduce checkpoint belongs to a live process: host={owner_host} pid={owner_pid}"
                )
            next_chunk = int(state.get("next_chunk", 0))
            if not 0 <= next_chunk <= total_chunks:
                raise RuntimeError("Reduce checkpoint has an invalid next chunk")
            selected = self._read_resume_indices()
            if int(state.get("selected_count", len(selected))) != len(selected):
                raise RuntimeError("Reduce checkpoint selected-index count does not match")
            return state, selected
        except (OSError, ValueError, TypeError, RuntimeError) as exc:
            print(f"[reduce] Invalid Reduce checkpoint ({exc}); starting from chunk 0.")
            self._remove_resume_artifacts()
            return None

    def _save_resume_state(
        self,
        total_chunks,
        next_chunk,
        selected_indices,
        selected_rows=None,
        status="running",
        error=None,
    ):
        self._write_resume_indices(selected_indices)
        state = {
            "format": 1,
            "status": status,
            "host": _SCRATCH_HOST,
            "pid": os.getpid(),
            "updated_at": time.time(),
            "signature": self._resume_signature(),
            "total_chunks": int(total_chunks),
            "next_chunk": int(next_chunk),
            "selected_count": len(selected_indices),
            "selected_rows": selected_rows,
        }
        if error:
            state["error"] = str(error)
        atomic_json(self._resume_checkpoint_path, state, durable=False)
        self._resume_last_state = state

    def _prepare_resume(self, total_chunks, selected_rows=None):
        loaded = self._load_resume_state(total_chunks)
        if loaded is None:
            selected = set()
            next_chunk = 0
            if selected_rows is not None:
                selected_rows.reset()
            self._save_resume_state(total_chunks, next_chunk, selected, selected_rows)
            return next_chunk, selected
        state, selected = loaded
        if selected_rows is not None:
            selected_rows.restore_manifest(state.get("selected_rows"))
        print(
            f"[reduce] Resuming reference-guided reduce from chunk "
            f"{int(state['next_chunk']):05d}/{total_chunks}; selected={len(selected)}"
        )
        self._resume_last_state = state
        return int(state["next_chunk"]), selected

    def _mark_resume_interrupted(self, error):
        state = getattr(self, "_resume_last_state", None)
        if not state or self.mode != "reference_guided":
            return
        try:
            state = dict(state)
            state["status"] = "interrupted"
            state["updated_at"] = time.time()
            state["error"] = str(error)
            atomic_json(self._resume_checkpoint_path, state, durable=False)
            self._resume_last_state = state
        except OSError as exc:
            warnings.warn(f"Could not update Reduce checkpoint after failure: {exc}", RuntimeWarning)

    def _record_fast_xyz_fallback(self, path, stage, reason):
        if self.xyz_io_mode == "fast_extxyz":
            raise ValueError(
                f"reduce.xyz_io_mode=fast_extxyz cannot process {Path(path).resolve()} "
                f"during {stage}: {reason}. Use xyz_io_mode=auto to allow an ASE "
                "fallback, or xyz_io_mode=ase to force the legacy ASE path."
            )
        item = {
            "path": str(Path(path).resolve()),
            "stage": str(stage),
            "reason": str(reason),
        }
        key = (item["path"], item["stage"], item["reason"])
        if key in self._fast_xyz_fallback_keys:
            return
        self._fast_xyz_fallback_keys.add(key)
        self._fast_xyz_fallbacks.append(item)
        self._used_ase_xyz_io = True
        warnings.warn(
            f"Fast EXTXYZ {stage} is unavailable for {item['path']}; using ASE: {item['reason']}",
            RuntimeWarning,
        )

    def _get_fast_xyz_index(self, path):
        if path is None:
            return None
        path = Path(path).resolve()
        key = str(path)
        if key in self._fast_xyz_indexes:
            return self._fast_xyz_indexes[key]
        if self.xyz_io_mode == "ase":
            self._used_ase_xyz_io = True
            self._fast_xyz_indexes[key] = None
            return None
        if not supports_raw_xyz(path):
            self._fast_xyz_indexes[key] = None
            self._record_fast_xyz_fallback(path, "frame indexing", "unsupported file suffix")
            return None
        try:
            frame_index = index_xyz_frames(path, collect_elements=True)
        except UnsupportedFastXYZ as exc:
            self._fast_xyz_indexes[key] = None
            self._record_fast_xyz_fallback(path, "frame indexing", exc)
            return None
        except MalformedXYZ as exc:
            raise ValueError(f"Malformed XYZ input: {exc}") from exc
        self._fast_xyz_indexes[key] = frame_index
        self._used_fast_xyz_io = True
        return frame_index

    def _drop_fast_xyz_index(self, path):
        if path is not None:
            self._fast_xyz_indexes.pop(str(Path(path).resolve()), None)

    def _make_scratch_dir(self):
        """Scratch space tagged with host and owner PID so a crashed run can be reclaimed."""
        return tempfile.TemporaryDirectory(
            dir=str(self.work_dir),
            prefix=f"tmp{_SCRATCH_HOST}_{os.getpid()}_",
        )

    def _sweep_stale_scratch(self):
        """Remove scratch directories left behind by runs that are no longer alive.

        A killed reduce leaves its intermediate descriptors on disk forever
        otherwise, because TemporaryDirectory cleanup never runs. Only
        directories owned by this host are considered, so a shared filesystem
        cannot make another node's live run look abandoned.
        """
        pattern = re.compile(rf"^tmp{re.escape(_SCRATCH_HOST)}_(\d+)_")
        for entry in self.work_dir.iterdir():
            match = pattern.match(entry.name)
            if match is None or not entry.is_dir():
                continue
            owner = int(match.group(1))
            if owner == os.getpid() or _process_alive(owner):
                continue
            print(f"[reduce] Reclaiming scratch directory from dead run {owner}: {entry}")
            shutil.rmtree(entry, ignore_errors=True)

    def _discard_intermediate(self, paths):
        """Drop descriptor shards once their consumer has finished reading them."""
        if self.keep_intermediate:
            return
        for path in paths:
            if path is None:
                continue
            try:
                Path(path).unlink()
            except FileNotFoundError:
                pass
            except OSError as exc:
                warnings.warn(f"Could not remove intermediate {path}: {exc}", RuntimeWarning)

    def _infer_elements(self, paths):
        element_set = set()
        for path in paths:
            if path is None:
                continue
            frame_index = self._get_fast_xyz_index(path)
            if frame_index is not None and frame_index.elements:
                element_set.update(frame_index.elements)
                continue
            if frame_index is not None:
                self._record_fast_xyz_fallback(
                    path,
                    "element inference",
                    frame_index.descriptor_reason or "element columns are unavailable",
                )
            for atoms in iread(str(path)):
                element_set.update(atoms.get_chemical_symbols())
        ordered_atomic_numbers = sorted(atomic_numbers[element] for element in element_set)
        return [chemical_symbols[number] for number in ordered_atomic_numbers]

    def _structure_count(self, path):
        if path is None:
            return 0
        path = Path(path)
        if not path.exists() or path.stat().st_size == 0:
            return 0
        frame_index = self._get_fast_xyz_index(path)
        if frame_index is not None:
            return len(frame_index)
        return _count_xyz_structures(path)

    def _raw_block_indexes(self, paths):
        indexes = []
        for path in paths:
            if path is None:
                indexes.append(None)
                continue
            frame_index = self._get_fast_xyz_index(path)
            if frame_index is None:
                return None
            indexes.append(frame_index)
        return indexes

    def _extract_width_lists(self, large_max_min, large_bins):
        width_lists = []
        for type_max_min, type_bins in zip(large_max_min, large_bins):
            type_widths = []
            for max_min, bin_count in zip(type_max_min, type_bins):
                if not bin_count:
                    type_widths.append(0.0)
                    continue
                span = abs(float(max_min[0]) - float(max_min[1]))
                type_widths.append(span / float(bin_count) if span > 0 else 0.0)
            width_lists.append(type_widths)
        return width_lists

    def _distribution_with_widths(self, array_data, width_list):
        values, _ = values_and_indices(array_data)
        D = values.dimensions if isinstance(values, DescriptorRows) else values.shape[1]
        zero_freq_intervals_list = []
        max_min = []
        bins = []

        for dim in range(D):
            new_data = column(values, dim)
            data_max = float(np.max(new_data))
            data_min = float(np.min(new_data))
            max_min.append([data_max, data_min])
            width = width_list[dim] if dim < len(width_list) else 0.0
            if width <= 0 or data_max == data_min:
                bin_count = 1
            else:
                bin_count = max(1, int(np.ceil((data_max - data_min) / width)))
            bins.append(bin_count)
            frequencies, bin_edges = np.histogram(new_data, bins=bin_count)
            zero_freq_intervals = [
                [float(bin_edges[index]), float(bin_edges[index + 1])]
                for index in range(len(bin_edges) - 1)
                if frequencies[index] <= self.state_population
            ]
            zero_freq_intervals_list.append(zero_freq_intervals)
        return zero_freq_intervals_list, max_min, bins

    def _data_base_distribution_with_widths(self, data_base_data, width_lists_by_type):
        train_data = decode(data_base_data)
        large_zero_freq_intervals_list = []
        large_max_min = []
        large_bins = []

        for type_index, type_atoms in enumerate(train_data):
            if len(type_atoms):
                widths = width_lists_by_type[type_index] if type_index < len(width_lists_by_type) else []
                zero_freq_intervals_list, max_min, bins = self._distribution_with_widths(type_atoms, widths)
            else:
                zero_freq_intervals_list, max_min, bins = [], [], []
            large_zero_freq_intervals_list.append(zero_freq_intervals_list)
            large_max_min.append(max_min)
            large_bins.append(bins)
        return large_zero_freq_intervals_list, large_max_min, large_bins

    def _prepare_fixed_interval_widths(self):
        if self.fixed_interval_widths is not None:
            return

        reference_xyz = self.interval_ref_xyz or self.current_xyz or self.input_xyz
        if reference_xyz is None:
            self.fixed_interval_widths = {}
            return

        temp_dir_obj = self._make_scratch_dir() if not self.keep_intermediate else None
        out_dir = Path(temp_dir_obj.name) if temp_dir_obj is not None else (self.work_dir / "iw_reference")
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            source = self._descriptor_source(reference_xyz, "iw_ref", out_dir)
            self.fixed_interval_widths = {}
            body_widths = {}
            for body in self.body_list:
                data_base_data = self._descriptor_body(source, body)
                if data_base_data is None:
                    continue
                _, large_max_min, large_bins = data_base_distribution(
                    data_base_data,
                    self.dq_width,
                    self.dq_width_method,
                    body,
                    plot_model=False,
                    dq_width_factor=self.dq_width_factor,
                )
                width_lists = self._extract_width_lists(large_max_min, large_bins)
                self.fixed_interval_widths[body] = width_lists
                body_widths[body] = width_lists
            self.fixed_interval_width_source = str(reference_xyz)
            self.interval_width_history.append(
                {
                    "stage": "reference",
                    "source_xyz": self.fixed_interval_width_source,
                    "dynamic_dq_width": False,
                    "body_widths": body_widths,
                }
            )
        finally:
            if temp_dir_obj is not None:
                temp_dir_obj.cleanup()
            elif not self.keep_intermediate and out_dir.exists():
                shutil.rmtree(out_dir, ignore_errors=True)

    def _run_calc_descriptors(self, cfg_path, out_path):
        command = [self.sus2_mlp_exe, "calc-descriptors", str(self.mtp_path), str(cfg_path), str(out_path)]
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
        if completed.returncode != 0:
            raise RuntimeError(
                f"calc-descriptors failed (exit={completed.returncode}): {' '.join(command)}\n"
                f"{completed.stderr[-2000:]}"
            )

    def _partition_atoms(self, atoms, parts):
        parts = max(1, min(int(parts), len(atoms)))
        base, remainder = divmod(len(atoms), parts)
        groups = []
        start = 0
        for index in range(parts):
            stop = start + base + (1 if index < remainder else 0)
            groups.append(atoms[start:stop])
            start = stop
        return [group for group in groups if group]

    def _preflight_ase_load(self, paths, stage):
        estimated = 256 * MIB
        for path in paths:
            if path is not None and Path(path).exists():
                estimated += Path(path).stat().st_size * 3
        stage_progress(stage, input_path=next((path for path in paths if path is not None), None))
        require_memory(estimated)

    def _build_chunk_ranges(self, total_count):
        if total_count <= 0:
            return []
        chunk_count = total_count // self.chunk_size
        if chunk_count == 0:
            chunk_count = 1
        base, remainder = divmod(total_count, chunk_count)
        ranges = []
        start = 0
        for index in range(chunk_count):
            stop = start + base + (1 if index < remainder else 0)
            ranges.append((start, stop))
            start = stop
        return ranges

    def _encode_xyz_to_pickles_ase(self, xyz_path, prefix, out_dir):
        self._used_ase_xyz_io = True
        self._preflight_ase_load([xyz_path], "reduce_ase_descriptor_load")
        atoms = list(iread(str(xyz_path)))
        cfg_path = out_dir / f"{prefix}.cfg"
        out_path = out_dir / f"{prefix}.out"
        worker_count = min(self.encoding_cores, len(atoms)) if atoms else 1

        if worker_count <= 1:
            xyz2cfg(
                self.elements,
                self.sort_elements_by_atomic_number,
                str(xyz_path),
                str(cfg_path),
                allow_missing_labels=True,
            )
            self._run_calc_descriptors(str(cfg_path), str(out_path))
            # calc-descriptors has already consumed the geometry.
            self._discard_intermediate([cfg_path])
            return [out_path]

        part_xyz_paths = []
        part_cfg_paths = []
        part_out_paths = []
        for index, chunk_atoms in enumerate(self._partition_atoms(atoms, worker_count)):
            part_xyz = out_dir / f"{prefix}.part_{index:04d}.xyz"
            part_cfg = out_dir / f"{prefix}.part_{index:04d}.cfg"
            part_out = out_dir / f"{prefix}.part_{index:04d}.out"
            _write_xyz(part_xyz, chunk_atoms)
            xyz2cfg(
                self.elements,
                self.sort_elements_by_atomic_number,
                str(part_xyz),
                str(part_cfg),
                allow_missing_labels=True,
            )
            part_xyz_paths.append(part_xyz)
            part_cfg_paths.append(part_cfg)
            part_out_paths.append(part_out)

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(self._run_calc_descriptors, part_cfg, part_out)
                for part_cfg, part_out in zip(part_cfg_paths, part_out_paths)
            ]
            for future in futures:
                future.result()

        self._discard_intermediate(part_xyz_paths + part_cfg_paths)
        return part_out_paths

    def _encode_xyz_to_pickles_fast(self, xyz_path, prefix, out_dir, frame_index, frame_range=None):
        out_path = out_dir / f"{prefix}.out"
        if frame_range is None:
            first_frame, last_frame = 0, len(frame_index)
        else:
            first_frame, last_frame = int(frame_range[0]), int(frame_range[1])
        frame_count = max(0, last_frame - first_frame)
        if frame_count == 0:
            out_path.write_text("", encoding="utf-8")
            return [out_path]

        worker_count = min(self.encoding_cores, frame_count)
        ranges = self._partition_atoms(range(first_frame, last_frame), worker_count)
        ranges = [(group[0], group[-1] + 1) for group in ranges]
        if worker_count == 1:
            part_cfg_paths = [out_dir / f"{prefix}.cfg"]
            part_out_paths = [out_path]
        else:
            part_cfg_paths = [
                out_dir / f"{prefix}.part_{index:04d}.cfg"
                for index in range(worker_count)
            ]
            part_out_paths = [path.with_suffix(".out") for path in part_cfg_paths]

        if worker_count == 1:
            write_position_cfg_parts(
                frame_index,
                part_cfg_paths,
                ranges,
                self.elements,
                self.sort_elements_by_atomic_number,
            )
        else:
            with ProcessPoolExecutor(max_workers=worker_count) as executor:
                futures = [
                    executor.submit(
                        write_position_cfg_part,
                        frame_index,
                        part_cfg,
                        frame_range,
                        self.elements,
                        self.sort_elements_by_atomic_number,
                    )
                    for part_cfg, frame_range in zip(part_cfg_paths, ranges)
                ]
                for future in futures:
                    future.result()

        if worker_count == 1:
            self._run_calc_descriptors(part_cfg_paths[0], part_out_paths[0])
        else:
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = [
                    executor.submit(self._run_calc_descriptors, part_cfg, part_out)
                    for part_cfg, part_out in zip(part_cfg_paths, part_out_paths)
                ]
                for future in futures:
                    future.result()

        # The shards stay in frame order, so consumers can read them in sequence
        # instead of materialising one merged copy of every descriptor.
        self._discard_intermediate(part_cfg_paths)
        return part_out_paths

    def _encode_xyz_to_pickles(self, xyz_path, prefix, out_dir):
        start = time.perf_counter()
        try:
            out_paths = self._descriptor_shards(xyz_path, prefix, out_dir)
            try:
                des_out2pkl(
                    [str(path) for path in out_paths],
                    prefix,
                    len(self.elements),
                    self.mtp_type,
                    str(self.mtp_path),
                    self.body_list,
                    str(out_dir),
                    column_subset=True,
                )
            finally:
                self._discard_intermediate(out_paths)
        finally:
            self.encoding_seconds += time.perf_counter() - start

    def _descriptor_shards(self, xyz_path, prefix, out_dir):
        try:
            frame_index = self._get_fast_xyz_index(xyz_path)
        except UnsupportedFastXYZ as exc:
            self._record_fast_xyz_fallback(xyz_path, "descriptor conversion", exc)
            return self._encode_xyz_to_pickles_ase(xyz_path, prefix, out_dir)
        if frame_index is not None and frame_index.descriptor_compatible:
            try:
                return self._encode_xyz_to_pickles_fast(xyz_path, prefix, out_dir, frame_index)
            except UnsupportedFastXYZ as exc:
                self._record_fast_xyz_fallback(xyz_path, "descriptor conversion", exc)
                return self._encode_xyz_to_pickles_ase(xyz_path, prefix, out_dir)
        if frame_index is not None:
            self._record_fast_xyz_fallback(
                xyz_path,
                "descriptor conversion",
                frame_index.descriptor_reason or "unsupported EXTXYZ descriptor fields",
            )
        return self._encode_xyz_to_pickles_ase(xyz_path, prefix, out_dir)

    def _descriptor_cache_key(self, xyz_path):
        resolved = str(Path(xyz_path).resolve())
        return hashlib.sha256(resolved.encode("utf-8")).hexdigest()[:20]

    def _descriptor_signature(self, xyz_path):
        return descriptor_store_signature(
            [],
            self.elements,
            self.mtp_type,
            self.mtp_path,
            self.body_list,
            mean_enabled=False,
            source_fingerprint=file_fingerprint(xyz_path),
        )

    def _encode_xyz_to_store(self, xyz_path):
        xyz_path = Path(xyz_path).resolve()
        key = str(xyz_path)
        cached = self._descriptor_sources.get(key)
        if cached is not None:
            return cached

        frame_index = self._get_fast_xyz_index(xyz_path)
        if frame_index is None:
            return None
        if not frame_index.descriptor_compatible:
            self._record_fast_xyz_fallback(
                xyz_path,
                "descriptor conversion",
                frame_index.descriptor_reason or "unsupported EXTXYZ descriptor fields",
            )
            return None

        cache_dir = self.descriptor_cache_dir / self._descriptor_cache_key(xyz_path)
        cache_dir.mkdir(parents=True, exist_ok=True)
        source_fingerprint = file_fingerprint(xyz_path)
        signature = self._descriptor_signature(xyz_path)
        store = load_descriptor_store(cache_dir, "data", signature)
        cache_reused = store is not None
        started = time.perf_counter()
        if store is None:
            out_paths = self._encode_xyz_to_pickles_fast(
                xyz_path,
                "data",
                cache_dir,
                frame_index,
            )
            try:
                store = build_descriptor_store(
                    [str(path) for path in out_paths],
                    "data",
                    self.elements,
                    self.mtp_type,
                    self.mtp_path,
                    self.body_list,
                    cache_dir,
                    mean_enabled=False,
                    source_fingerprint=source_fingerprint,
                )
            finally:
                self._discard_intermediate(out_paths)
            if file_fingerprint(xyz_path) != source_fingerprint:
                raise RuntimeError(f"Reduce input changed during descriptor encoding: {xyz_path}")
            self.encoding_seconds += time.perf_counter() - started
        cache_bytes = sum(
            path.stat().st_size
            for path in store.path.rglob("*")
            if path.is_file()
        )
        self._descriptor_cache_records.append(
            {
                "input": key,
                "store": str(store.path),
                "cache_reused": bool(cache_reused),
                "cache_bytes": int(cache_bytes),
                "frame_count": int(store.manifest.get("frame_count", 0)),
            }
        )
        self._descriptor_sources[key] = store
        stage_progress("reduce_descriptor_ready", store.manifest.get("frame_count", 0), xyz_path)
        return store

    def _encode_frame_range_to_store(self, xyz_path, frame_index, start, stop, tag):
        """Encode a single frame range into a throwaway descriptor store.

        Building the store here, rather than encoding the whole file up front,
        is what bounds peak disk: one chunk of descriptors exists at a time and
        is released before the next chunk starts.
        """
        cache_dir = self.descriptor_cache_dir / f"{self._descriptor_cache_key(xyz_path)}.chunk_{tag}"
        shutil.rmtree(cache_dir, ignore_errors=True)
        cache_dir.mkdir(parents=True, exist_ok=True)
        shards = self._encode_xyz_to_pickles_fast(
            xyz_path,
            "chunk",
            cache_dir,
            frame_index,
            frame_range=(start, stop),
        )
        try:
            store = build_descriptor_store(
                [str(path) for path in shards],
                "chunk",
                self.elements,
                self.mtp_type,
                self.mtp_path,
                self.body_list,
                cache_dir,
                mean_enabled=False,
                source_fingerprint=file_fingerprint(xyz_path),
                index_offset=int(start),
            )
        finally:
            self._discard_intermediate(shards)
        cache_bytes = sum(
            path.stat().st_size for path in store.path.rglob("*") if path.is_file()
        )
        self._descriptor_cache_records.append(
            {
                "input": str(Path(xyz_path).resolve()),
                "store": str(store.path),
                "cache_reused": False,
                "cache_bytes": int(cache_bytes),
                "frame_count": int(store.manifest.get("frame_count", 0)),
            }
        )
        return store

    def _release_chunk_store(self, store):
        if self.keep_intermediate:
            return
        shutil.rmtree(Path(store.path).parent, ignore_errors=True)

    def _descriptor_source(self, xyz_path, prefix, out_dir):
        store = self._encode_xyz_to_store(xyz_path)
        if store is not None:
            return store
        self._encode_xyz_to_pickles(xyz_path, prefix, out_dir)
        return {
            body: out_dir / f"{prefix}_{body}_body_coding_zlib.pkl"
            for body in self.body_list
        }

    @staticmethod
    def _descriptor_body(source, body):
        if hasattr(source, "body"):
            return source.body(body)
        path = source.get(body)
        if path is None:
            return None
        if isinstance(path, list):
            return path
        if not path.exists():
            return None
        return path

    @staticmethod
    def _combine_body_rows(sources, body):
        rows_by_element = []
        body_sources = [source.body(body) for source in sources if source is not None]
        if not body_sources:
            return rows_by_element
        for element_index in range(len(body_sources[0])):
            rows_by_element.append(
                concatenate_rows([source[element_index] for source in body_sources])
            )
        return rows_by_element

    def _update_peak_memory(self):
        self._peak_memory_bytes = max(self._peak_memory_bytes, tree_memory(os.getpid()))
        return self._peak_memory_bytes

    def _resolve_histogram_bins(self, values):
        if self.dq_width_method == "Freedman_Diaconis":
            return Freedman_Diaconis_bins(values, dq_width_factor=self.dq_width_factor)
        if self.dq_width_method == "self_input":
            if self.dq_width <= 0:
                raise ValueError("dq_width must be > 0 when dq_width_method is self_input")
            return max(1, int(np.ceil((float(np.max(values)) - float(np.min(values))) / self.dq_width)))
        if self.dq_width_method == "scott":
            return scott(values, dq_width_factor=self.dq_width_factor)
        if self.dq_width_method == "std":
            std_dev = float(np.std(values))
            if std_dev == 0:
                return 1
            return max(1, int(np.ceil((float(np.max(values)) - float(np.min(values))) / (std_dev / 10))))
        raise ValueError(f"Unsupported dq_width_method for candidate_only reduce: {self.dq_width_method}")

    def _build_occupied_intervals(self, values):
        values = np.asarray(values, dtype=float)
        if values.size == 0:
            return []
        bin_count = self._resolve_histogram_bins(values)
        frequencies, bin_edges = np.histogram(values, bins=bin_count)
        return [
            [float(bin_edges[index]), float(bin_edges[index + 1])]
            for index in range(len(bin_edges) - 1)
            if frequencies[index] > 0
        ]

    def _build_occupied_intervals_with_targets(self, values):
        values = np.asarray(values, dtype=float)
        if values.size == 0:
            return [], []
        bin_count = self._resolve_histogram_bins(values)
        frequencies, bin_edges = np.histogram(values, bins=bin_count)
        intervals = []
        targets = []
        for index, frequency in enumerate(frequencies):
            if frequency <= 0:
                continue
            intervals.append([float(bin_edges[index]), float(bin_edges[index + 1])])
            if self.state_population <= 1:
                targets.append(1)
            else:
                targets.append(int(min(int(frequency), self.state_population)))
        return intervals, targets

    def _find_multi_cover_set(self, classes, targets):
        return find_multi_cover_set_compact(classes, targets)

    def _run_dimension_min_cover(self, tasks, label):
        selected, stats, _ = solve_dimension_tasks(
            tasks,
            self.dimension_min_cover_workers,
        )
        record = dict(stats)
        record["label"] = str(label)
        self.dimension_min_cover_records.append(record)
        if (
            stats.get("task_count", 0) > 0
            and self.dimension_min_cover_workers == -1
            and stats.get("scheduler") is None
            and not self._dimension_scheduler_warning_printed
        ):
            print(
                "[reduce] No scheduler allocation was detected; "
                "dimension_min_cover_workers=-1 uses affinity-visible CPUs."
            )
            self._dimension_scheduler_warning_printed = True
        return selected

    def _select_direct_indices(self, candidate_xyz_path):
        if self._structure_count(candidate_xyz_path) == 0:
            return []
        progress = _ProgressTracker("candidate-only-reduce", 3)

        temp_dir_obj = self._make_scratch_dir() if not self.keep_intermediate else None
        out_dir = Path(temp_dir_obj.name) if temp_dir_obj is not None else (self.work_dir / "direct_intermediate")
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            source = self._descriptor_source(candidate_xyz_path, "database", out_dir)
            progress.update(1, "encoding-complete")

            classes = []
            class_targets = []
            dimension_tasks = []
            for body in self.body_list:
                data_base_data = self._descriptor_body(source, body)
                if data_base_data is None:
                    continue
                decoded = decode(data_base_data)
                for element_index, type_atoms in enumerate(decoded):
                    if not len(type_atoms):
                        continue
                    array_data, stru_indexs = values_and_indices(type_atoms)
                    dimensions = (
                        array_data.dimensions
                        if isinstance(array_data, DescriptorRows)
                        else array_data.shape[1]
                    )
                    if len(type_atoms) == 0 or dimensions == 0:
                        continue
                    for dim in range(dimensions):
                        dimension_values = column(array_data, dim)
                        intervals, targets = self._build_occupied_intervals_with_targets(dimension_values)
                        if not intervals:
                            continue
                        grouped = group_compact_indices(
                            stru_indexs,
                            dimension_values,
                            intervals,
                        )
                        dimension_classes = [] if self.dimension_min_cover_workers != 0 else None
                        dimension_targets = [] if self.dimension_min_cover_workers != 0 else None
                        for bucket, target in zip(grouped, targets):
                            if bucket:
                                classes.append(bucket)
                                class_targets.append(target)
                                if dimension_classes is not None:
                                    dimension_classes.append(bucket)
                                    dimension_targets.append(target)
                        if dimension_classes:
                            element = (
                                self.elements[element_index]
                                if element_index < len(self.elements)
                                else f"type-{element_index}"
                            )
                            dimension_tasks.append(
                                {
                                    "key": (str(body), str(element), int(dim)),
                                    "classes": dimension_classes,
                                    "targets": (
                                        dimension_targets
                                        if self.state_population > 1
                                        else None
                                    ),
                                }
                            )
            progress.update(2, f"class_count={len(classes)}")
            self._update_peak_memory()

            if not classes:
                progress.update(3, "no-classes")
                return []
            if self.dimension_min_cover_workers != 0:
                selected = self._run_dimension_min_cover(
                    dimension_tasks,
                    "candidate_only",
                )
            elif self.state_population <= 1:
                selected = sorted(set(int(index) for index in find_min_cover_set(classes)))
            else:
                selected = sorted(set(int(index) for index in self._find_multi_cover_set(classes, class_targets)))
            progress.update(3, f"selected={len(selected)}")
            return selected
        finally:
            if temp_dir_obj is not None:
                temp_dir_obj.cleanup()
            elif not self.keep_intermediate and out_dir.exists():
                shutil.rmtree(out_dir, ignore_errors=True)

    def _select_descriptor_sources(
        self,
        train_source,
        candidate_source,
        candidate_count,
        stage_label=None,
        source_label=None,
    ):
        if candidate_count == 0:
            return []

        classes = []
        dimension_tasks = []
        stage_body_widths = {}
        for body in self.body_list:
            data_base_data = self._descriptor_body(train_source, body)
            md_data = self._descriptor_body(candidate_source, body)
            if data_base_data is None or md_data is None:
                continue
            if self.dynamic_dq_width:
                large_zero_freq_intervals_list, large_max_min, large_bins = data_base_distribution(
                    data_base_data,
                    self.dq_width,
                    self.dq_width_method,
                    body,
                    plot_model=False,
                    dq_width_factor=self.dq_width_factor,
                    state_population=self.state_population,
                )
                stage_body_widths[body] = self._extract_width_lists(large_max_min, large_bins)
            else:
                self._prepare_fixed_interval_widths()
                width_lists = self.fixed_interval_widths.get(body, [])
                large_zero_freq_intervals_list, large_max_min, large_bins = self._data_base_distribution_with_widths(
                    data_base_data,
                    width_lists,
                )
            _, dimension_classes, _, no_set_need_index_list = md_extract(
                md_data,
                large_zero_freq_intervals_list,
                large_max_min,
                large_bins,
            )
            classes.extend(no_set_need_index_list)
            if self.dimension_min_cover_workers != 0:
                dimension_tasks.extend(
                    build_dimension_tasks(
                        dimension_classes,
                        body,
                        self.elements,
                    )
                )

        if self.dynamic_dq_width and stage_label is not None and stage_body_widths:
            self.interval_width_history.append(
                {
                    "stage": stage_label,
                    "source_xyz": str(source_label) if source_label is not None else None,
                    "dynamic_dq_width": True,
                    "body_widths": stage_body_widths,
                }
            )
        self._update_peak_memory()
        if not classes:
            return []
        if self.dimension_min_cover_workers != 0:
            return self._run_dimension_min_cover(
                dimension_tasks,
                stage_label or "reference_guided",
            )
        return sorted(set(int(index) for index in find_min_cover_set(classes)))

    def _select_indices(self, train_xyz_path, candidate_xyz_path, stage_label=None):
        candidate_count = self._structure_count(candidate_xyz_path)
        if candidate_count == 0:
            return []
        if self._structure_count(train_xyz_path) == 0:
            return list(range(candidate_count))

        temp_dir_obj = self._make_scratch_dir() if not self.keep_intermediate else None
        out_dir = Path(temp_dir_obj.name) if temp_dir_obj is not None else (self.work_dir / "intermediate")
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            train_source = self._descriptor_source(train_xyz_path, "database", out_dir)
            candidate_source = self._descriptor_source(candidate_xyz_path, "md", out_dir)
            return self._select_descriptor_sources(
                train_source,
                candidate_source,
                candidate_count,
                stage_label=stage_label,
                source_label=train_xyz_path,
            )
        finally:
            if temp_dir_obj is not None:
                temp_dir_obj.cleanup()
            elif not self.keep_intermediate and out_dir.exists():
                shutil.rmtree(out_dir, ignore_errors=True)

    def _run_single_fast(self, input_index, current_index):
        input_count = len(input_index)
        current_count = len(current_index) if current_index is not None else 0

        if self.direct_self_dedup:
            selected_indices = self._select_direct_indices(self.input_xyz)
            selection_basis = "candidate_only_self_dedup"
        else:
            if self.interval_ref_xyz is not None:
                train_source_path = self.interval_ref_xyz
            elif self.current_xyz is not None:
                train_source_path = self.current_xyz
            else:
                train_source_path = self.input_xyz
            selected_indices = self._select_indices(
                train_source_path,
                self.input_xyz,
                stage_label="single",
            )
            selection_basis = "against_existing_reference"

        selected_index_set = {
            int(index) for index in selected_indices if 0 <= int(index) < input_count
        }
        selected_order = sorted(selected_index_set)
        output_selections = []
        if self.append_current and current_index is not None and current_count:
            output_selections.append((current_index, current_index.all_indices()))
        output_selections.append((input_index, selected_order))

        output_count = write_indexed_frames(self.output_xyz, output_selections)
        remain_count = write_indexed_frames(
            self.remain_xyz,
            [(input_index, complement_indices(input_count, selected_index_set))],
        )
        return {
            "mode": "candidate_only",
            "selection_basis": selection_basis,
            "using_default_universal_assets": self.using_default_universal_assets,
            "mtp_path": str(self.mtp_path),
            "mtp_species_count": self.mtp_species_count,
            "element_count": len(self.elements),
            "sort_elements_by_atomic_number": self.sort_elements_by_atomic_number,
            "interval_ref_xyz": str(self.interval_ref_xyz) if self.interval_ref_xyz is not None else None,
            "dq_width_method": self.dq_width_method,
            "dq_width": self.dq_width,
            "dq_width_factor": self.dq_width_factor,
            "state_population": self.state_population,
            "input_count": input_count,
            "current_count": current_count,
            "selected_from_input": len(selected_order),
            "remain_from_input": remain_count,
            "output_count": output_count,
            "output_xyz": str(self.output_xyz),
            "remain_xyz": str(self.remain_xyz),
            "file_counts": {
                "input_xyz": input_count,
                "current_xyz": current_count,
                "interval_ref_xyz": self._structure_count(self.interval_ref_xyz),
                "output_xyz": output_count,
                "remain_xyz": remain_count,
            },
        }

    def _run_single_ase(self):
        self._used_ase_xyz_io = True
        self._preflight_ase_load([self.input_xyz, self.current_xyz], "reduce_ase_input_load")
        input_atoms = list(iread(str(self.input_xyz)))
        current_atoms = list(iread(str(self.current_xyz))) if self.current_xyz else []

        if self.direct_self_dedup:
            selected_indices = self._select_direct_indices(self.input_xyz)
            selection_basis = "candidate_only_self_dedup"
        else:
            if self.interval_ref_xyz is not None:
                train_source_path = self.interval_ref_xyz
            elif self.current_xyz is not None:
                train_source_path = self.current_xyz
            else:
                train_source_path = self.input_xyz
            selected_indices = self._select_indices(train_source_path, self.input_xyz, stage_label="single")
            selection_basis = "against_existing_reference"

        selected_index_set = set(selected_indices)
        selected_atoms = [atoms for idx, atoms in enumerate(input_atoms) if idx in selected_index_set]
        remain_atoms = [atoms for idx, atoms in enumerate(input_atoms) if idx not in selected_index_set]

        if self.append_current and current_atoms:
            output_atoms = current_atoms + selected_atoms
        else:
            output_atoms = selected_atoms

        _write_xyz(self.output_xyz, output_atoms)
        _write_xyz(self.remain_xyz, remain_atoms)
        return {
            "mode": "candidate_only",
            "selection_basis": selection_basis,
            "using_default_universal_assets": self.using_default_universal_assets,
            "mtp_path": str(self.mtp_path),
            "mtp_species_count": self.mtp_species_count,
            "element_count": len(self.elements),
            "sort_elements_by_atomic_number": self.sort_elements_by_atomic_number,
            "interval_ref_xyz": str(self.interval_ref_xyz) if self.interval_ref_xyz is not None else None,
            "dq_width_method": self.dq_width_method,
            "dq_width": self.dq_width,
            "dq_width_factor": self.dq_width_factor,
            "state_population": self.state_population,
            "input_count": len(input_atoms),
            "current_count": len(current_atoms),
            "selected_from_input": len(selected_atoms),
            "remain_from_input": len(remain_atoms),
            "output_count": len(output_atoms),
            "output_xyz": str(self.output_xyz),
            "remain_xyz": str(self.remain_xyz),
            "file_counts": {
                "input_xyz": len(input_atoms),
                "current_xyz": len(current_atoms),
                "interval_ref_xyz": _count_xyz_structures(self.interval_ref_xyz),
                "output_xyz": _count_xyz_structures(self.output_xyz),
                "remain_xyz": _count_xyz_structures(self.remain_xyz),
            },
        }

    def _run_single(self):
        paths = [self.input_xyz]
        if self.current_xyz is not None:
            paths.append(self.current_xyz)
        indexes = self._raw_block_indexes(paths)
        if indexes is None:
            return self._run_single_ase()
        input_index = indexes[0]
        current_index = indexes[1] if len(indexes) > 1 else None
        return self._run_single_fast(input_index, current_index)

    def _run_chunked_fast_legacy(self, input_index, current_index):
        input_count = len(input_index)
        current_count = len(current_index)
        chunk_ranges = self._build_chunk_ranges(input_count)
        total_chunks = len(chunk_ranges)
        start_chunk, selected_global_indices = self._prepare_resume(total_chunks)
        progress = _ProgressTracker("reference-guided-reduce", total_chunks)

        for chunk_id in range(start_chunk, total_chunks):
            start, end = chunk_ranges[chunk_id]
            if start >= end:
                continue
            chunk_dir = self.work_dir / f"chunk_{chunk_id:05d}"
            chunk_dir.mkdir(parents=True, exist_ok=True)
            train_xyz_path = chunk_dir / "train.xyz"
            candidate_xyz_path = chunk_dir / "candidate.xyz"

            train_selections = []
            if current_count:
                train_selections.append((current_index, current_index.all_indices()))
            if selected_global_indices:
                train_selections.append((input_index, sorted(selected_global_indices)))
            if not train_selections:
                train_selections.append((input_index, range(start, end)))
            write_indexed_frames(train_xyz_path, train_selections)
            write_indexed_frames(
                candidate_xyz_path,
                [(input_index, range(start, end))],
            )

            try:
                local_indices = self._select_indices(
                    train_xyz_path,
                    candidate_xyz_path,
                    stage_label=f"chunk_{chunk_id:05d}",
                )
                for local_index in local_indices:
                    local_index = int(local_index)
                    if 0 <= local_index < end - start:
                        selected_global_indices.add(start + local_index)

                progress.update(
                    chunk_id + 1,
                    f"selected={len(selected_global_indices)} "
                    f"remain={input_count - len(selected_global_indices)}",
                )
                self._save_resume_state(
                    total_chunks,
                    chunk_id + 1,
                    selected_global_indices,
                )
            finally:
                self._drop_fast_xyz_index(train_xyz_path)
                self._drop_fast_xyz_index(candidate_xyz_path)
                if not self.keep_intermediate:
                    shutil.rmtree(chunk_dir, ignore_errors=True)

        selected_order = sorted(selected_global_indices)
        output_selections = []
        if self.append_current and current_count:
            output_selections.append((current_index, current_index.all_indices()))
        output_selections.append((input_index, selected_order))
        output_count = write_indexed_frames(self.output_xyz, output_selections)
        remain_count = write_indexed_frames(
            self.remain_xyz,
            [(input_index, complement_indices(input_count, selected_global_indices))],
        )
        return {
            "mode": "reference_guided",
            "chunk_size": self.chunk_size,
            "using_default_universal_assets": self.using_default_universal_assets,
            "mtp_path": str(self.mtp_path),
            "mtp_species_count": self.mtp_species_count,
            "element_count": len(self.elements),
            "sort_elements_by_atomic_number": self.sort_elements_by_atomic_number,
            "interval_ref_xyz": str(self.interval_ref_xyz) if self.interval_ref_xyz is not None else None,
            "dq_width_method": self.dq_width_method,
            "dq_width": self.dq_width,
            "dq_width_factor": self.dq_width_factor,
            "state_population": self.state_population,
            "input_count": input_count,
            "current_count": current_count,
            "selected_from_input": len(selected_order),
            "remain_from_input": remain_count,
            "output_count": output_count,
            "output_xyz": str(self.output_xyz),
            "remain_xyz": str(self.remain_xyz),
            "file_counts": {
                "input_xyz": input_count,
                "current_xyz": current_count,
                "interval_ref_xyz": self._structure_count(self.interval_ref_xyz),
                "output_xyz": output_count,
                "remain_xyz": remain_count,
            },
        }

    def _run_chunked_fast(self, input_index, current_index):
        current_store = self._encode_xyz_to_store(self.current_xyz)
        if current_store is None:
            return self._run_chunked_fast_legacy(input_index, current_index)

        input_count = len(input_index)
        current_count = len(current_index)
        chunk_ranges = self._build_chunk_ranges(input_count)
        progress = _ProgressTracker("reference-guided-reduce", len(chunk_ranges))
        selected_row_kwargs = {
            "directory": self.work_dir / "selected_rows",
            "bodies": self.body_list,
            "element_count": len(self.elements),
            "dimensions": {
                body: len(current_store.manifest["body_columns"][body])
                for body in self.body_list
            },
        }
        try:
            selected_rows = _SelectedRowStore(**selected_row_kwargs)
        except (OSError, ValueError, RuntimeError) as exc:
            print(f"[reduce] Resetting invalid selected-row checkpoint: {exc}")
            self._remove_resume_artifacts()
            selected_rows = _SelectedRowStore(**selected_row_kwargs)
        start_chunk, selected_global_indices = self._prepare_resume(
            len(chunk_ranges), selected_rows
        )

        for chunk_id in range(start_chunk, len(chunk_ranges)):
            start, end = chunk_ranges[chunk_id]
            if start >= end:
                continue
            chunk_store = self._encode_frame_range_to_store(
                self.input_xyz,
                input_index,
                start,
                end,
                f"{chunk_id:05d}",
            )
            try:
                candidate_source = {}
                train_source = {}
                selected_order = sorted(selected_global_indices)
                accumulated = selected_rows.freeze() if selected_order else None
                for body in self.body_list:
                    chunk_rows = chunk_store.body(body)
                    current_rows = current_store.body(body)
                    candidate_source[body] = chunk_rows
                    train_rows = []
                    for element_index in range(len(chunk_rows)):
                        parts = []
                        if current_count:
                            parts.append(current_rows[element_index])
                        if accumulated is not None:
                            parts.append(accumulated[body][element_index])
                        if not parts:
                            parts.append(candidate_source[body][element_index])
                        train_rows.append(concatenate_rows(parts))
                    train_source[body] = train_rows

                selected_chunk_indices = self._select_descriptor_sources(
                    train_source,
                    candidate_source,
                    end - start,
                    stage_label=f"chunk_{chunk_id:05d}",
                    source_label=f"{self.current_xyz} + selected input frames",
                )
                fresh_selection = set()
                for global_index in selected_chunk_indices:
                    global_index = int(global_index)
                    if start <= global_index < end and global_index not in selected_global_indices:
                        selected_global_indices.add(global_index)
                        fresh_selection.add(global_index)
                # Carry this chunk's picks forward before its cache is released.
                for body in self.body_list:
                    selected_rows.append_selection(chunk_store, body, fresh_selection)
                self._save_resume_state(
                    len(chunk_ranges),
                    chunk_id + 1,
                    selected_global_indices,
                    selected_rows._manifest(),
                )
            finally:
                self._release_chunk_store(chunk_store)

            progress.update(
                chunk_id + 1,
                f"selected={len(selected_global_indices)} "
                f"remain={input_count - len(selected_global_indices)}",
            )
            stage_progress(
                "reduce_reference_chunk",
                chunk_id + 1,
                self.input_xyz,
                workers=self.dimension_min_cover_workers,
            )
            self._update_peak_memory()

        selected_order = sorted(selected_global_indices)
        output_selections = []
        if self.append_current and current_count:
            output_selections.append((current_index, current_index.all_indices()))
        output_selections.append((input_index, selected_order))
        output_count = write_indexed_frames(self.output_xyz, output_selections)
        remain_count = write_indexed_frames(
            self.remain_xyz,
            [(input_index, complement_indices(input_count, selected_global_indices))],
        )
        return {
            "mode": "reference_guided",
            "chunk_size": self.chunk_size,
            "using_default_universal_assets": self.using_default_universal_assets,
            "mtp_path": str(self.mtp_path),
            "mtp_species_count": self.mtp_species_count,
            "element_count": len(self.elements),
            "sort_elements_by_atomic_number": self.sort_elements_by_atomic_number,
            "interval_ref_xyz": str(self.interval_ref_xyz) if self.interval_ref_xyz is not None else None,
            "dq_width_method": self.dq_width_method,
            "dq_width": self.dq_width,
            "dq_width_factor": self.dq_width_factor,
            "state_population": self.state_population,
            "input_count": input_count,
            "current_count": current_count,
            "selected_from_input": len(selected_order),
            "remain_from_input": remain_count,
            "output_count": output_count,
            "output_xyz": str(self.output_xyz),
            "remain_xyz": str(self.remain_xyz),
            "file_counts": {
                "input_xyz": input_count,
                "current_xyz": current_count,
                "interval_ref_xyz": self._structure_count(self.interval_ref_xyz),
                "output_xyz": output_count,
                "remain_xyz": remain_count,
            },
        }

    def _run_chunked_ase(self):
        self._used_ase_xyz_io = True
        self._preflight_ase_load([self.input_xyz, self.current_xyz], "reduce_ase_input_load")
        input_atoms = list(iread(str(self.input_xyz)))
        current_atoms = list(iread(str(self.current_xyz))) if self.current_xyz else []

        chunk_ranges = self._build_chunk_ranges(len(input_atoms))
        total_chunks = len(chunk_ranges)
        start_chunk, selected_global_indices = self._prepare_resume(total_chunks)
        selected_atoms = [input_atoms[index] for index in sorted(selected_global_indices)]
        progress = _ProgressTracker("reference-guided-reduce", total_chunks)

        for chunk_id in range(start_chunk, total_chunks):
            start, end = chunk_ranges[chunk_id]
            chunk_atoms = input_atoms[start:end]
            if not chunk_atoms:
                continue

            train_atoms = current_atoms + selected_atoms
            if not train_atoms:
                train_atoms = chunk_atoms

            chunk_dir = self.work_dir / f"chunk_{chunk_id:05d}"
            chunk_dir.mkdir(parents=True, exist_ok=True)
            train_xyz_path = chunk_dir / "train.xyz"
            candidate_xyz_path = chunk_dir / "candidate.xyz"
            _write_xyz(train_xyz_path, train_atoms)
            _write_xyz(candidate_xyz_path, chunk_atoms)

            local_indices = self._select_indices(
                train_xyz_path,
                candidate_xyz_path,
                stage_label=f"chunk_{chunk_id:05d}",
            )
            for local_index in local_indices:
                if 0 <= local_index < len(chunk_atoms):
                    global_index = start + local_index
                    if global_index not in selected_global_indices:
                        selected_global_indices.add(global_index)
                        selected_atoms.append(chunk_atoms[local_index])

            progress.update(
                chunk_id + 1,
                f"selected={len(selected_global_indices)} remain={len(input_atoms) - len(selected_global_indices)}",
            )
            self._save_resume_state(
                total_chunks,
                chunk_id + 1,
                selected_global_indices,
            )

            if not self.keep_intermediate:
                shutil.rmtree(chunk_dir, ignore_errors=True)

        remain_atoms = [atoms for idx, atoms in enumerate(input_atoms) if idx not in selected_global_indices]
        if self.append_current and current_atoms:
            output_atoms = current_atoms + selected_atoms
        else:
            output_atoms = selected_atoms

        _write_xyz(self.output_xyz, output_atoms)
        _write_xyz(self.remain_xyz, remain_atoms)
        return {
            "mode": "reference_guided",
            "chunk_size": self.chunk_size,
            "using_default_universal_assets": self.using_default_universal_assets,
            "mtp_path": str(self.mtp_path),
            "mtp_species_count": self.mtp_species_count,
            "element_count": len(self.elements),
            "sort_elements_by_atomic_number": self.sort_elements_by_atomic_number,
            "interval_ref_xyz": str(self.interval_ref_xyz) if self.interval_ref_xyz is not None else None,
            "dq_width_method": self.dq_width_method,
            "dq_width": self.dq_width,
            "dq_width_factor": self.dq_width_factor,
            "state_population": self.state_population,
            "input_count": len(input_atoms),
            "current_count": len(current_atoms),
            "selected_from_input": len(selected_atoms),
            "remain_from_input": len(remain_atoms),
            "output_count": len(output_atoms),
            "output_xyz": str(self.output_xyz),
            "remain_xyz": str(self.remain_xyz),
            "file_counts": {
                "input_xyz": len(input_atoms),
                "current_xyz": len(current_atoms),
                "interval_ref_xyz": _count_xyz_structures(self.interval_ref_xyz),
                "output_xyz": _count_xyz_structures(self.output_xyz),
                "remain_xyz": _count_xyz_structures(self.remain_xyz),
            },
        }

    def _run_chunked(self):
        indexes = self._raw_block_indexes([self.input_xyz, self.current_xyz])
        if indexes is None:
            return self._run_chunked_ase()
        return self._run_chunked_fast(indexes[0], indexes[1])

    def _run_impl(self):
        total_start = time.perf_counter()
        if self._mode_impl == "single":
            report = self._run_single()
        elif self._mode_impl == "chunked":
            report = self._run_chunked()
        else:
            raise ValueError(f"Unsupported reduce mode: {self.mode}")

        total_seconds = time.perf_counter() - total_start
        processing_seconds = max(0.0, total_seconds - self.encoding_seconds)
        report["encoding_hours"] = self.encoding_seconds / 3600.0
        report["processing_hours"] = processing_seconds / 3600.0
        report["total_hours"] = total_seconds / 3600.0
        report["input_config"] = self.input_config_snapshot
        if self._used_fast_xyz_io and self._used_ase_xyz_io:
            effective_xyz_io_mode = "mixed"
        elif self._used_fast_xyz_io:
            effective_xyz_io_mode = "fast_extxyz"
        elif self._used_ase_xyz_io:
            effective_xyz_io_mode = "ase"
        else:
            effective_xyz_io_mode = "none"
        effective_config = self._build_effective_config()
        effective_config["xyz_io_backend"] = effective_xyz_io_mode
        effective_config["xyz_io_fallbacks"] = list(self._fast_xyz_fallbacks)
        report["effective_config"] = effective_config
        report["fast_xyz_io"] = {
            "requested_mode": self.xyz_io_mode,
            "effective_mode": effective_xyz_io_mode,
            "indexed_file_count": sum(
                1 for frame_index in self._fast_xyz_indexes.values() if frame_index is not None
            ),
            "fallback_count": len(self._fast_xyz_fallbacks),
            "fallbacks": list(self._fast_xyz_fallbacks),
        }
        if self.dimension_min_cover_workers == 0:
            report["dimension_min_cover"] = {
                "mode": "joint",
                "requested_workers": 0,
                "call_count": 0,
                "task_count": 0,
                "elapsed_seconds": 0.0,
                "calls": [],
            }
        else:
            report["dimension_min_cover"] = {
                "mode": "per_dimension",
                "requested_workers": self.dimension_min_cover_workers,
                "call_count": len(self.dimension_min_cover_records),
                "task_count": sum(
                    item["task_count"] for item in self.dimension_min_cover_records
                ),
                "effective_workers": max(
                    (item["effective_workers"] for item in self.dimension_min_cover_records),
                    default=0,
                ),
                "sum_selected": sum(
                    item["sum_selected"] for item in self.dimension_min_cover_records
                ),
                "union_selected": sum(
                    item["union_selected"] for item in self.dimension_min_cover_records
                ),
                "overlap_removed": sum(
                    item["overlap_removed"] for item in self.dimension_min_cover_records
                ),
                "global_prune_removed": sum(
                    item.get("global_prune", {}).get("removed", 0)
                    for item in self.dimension_min_cover_records
                ),
                "global_prune_seconds": sum(
                    item.get("global_prune", {}).get("elapsed_seconds", 0.0)
                    for item in self.dimension_min_cover_records
                ),
                "dimension_solve_seconds": sum(
                    item.get("dimension_solve_seconds", 0.0)
                    for item in self.dimension_min_cover_records
                ),
                "elapsed_seconds": sum(
                    item["elapsed_seconds"] for item in self.dimension_min_cover_records
                ),
                "calls": self.dimension_min_cover_records,
            }
        if self.interval_width_history:
            report["interval_width_history"] = self.interval_width_history
        self._update_peak_memory()
        report["descriptor_storage"] = {
            "backend": "descriptor_store" if self._descriptor_cache_records else "legacy_pickle",
            "source_count": len(self._descriptor_cache_records),
            "cache_reused_count": sum(
                int(item["cache_reused"]) for item in self._descriptor_cache_records
            ),
            "cache_bytes": sum(
                int(item["cache_bytes"]) for item in self._descriptor_cache_records
            ),
            "sources": list(self._descriptor_cache_records),
        }
        report["memory"] = {
            "peak_process_tree_bytes": int(self._peak_memory_bytes),
            "file_descriptor_limit": getattr(self, "file_descriptor_limit", None),
        }
        return report

    def run(self):
        self.file_descriptor_limit = _raise_file_limit()
        if not self.keep_intermediate:
            self._sweep_stale_scratch()
        final_output = self.output_xyz
        final_remain = self.remain_xyz
        final_report = self.report_json
        token = f"{os.getpid()}-{time.time_ns()}"
        temporary_output = final_output.with_name(
            f".{final_output.stem}.{token}.partial{final_output.suffix}"
        )
        temporary_remain = final_remain.with_name(
            f".{final_remain.stem}.{token}.partial{final_remain.suffix}"
        )
        temporary_report = final_report.with_name(
            f".{final_report.stem}.{token}.partial{final_report.suffix}"
        )
        for path in (temporary_output, temporary_remain, temporary_report):
            path.parent.mkdir(parents=True, exist_ok=True)
        self.output_xyz = temporary_output
        self.remain_xyz = temporary_remain
        self.report_json = temporary_report
        succeeded = False
        try:
            with descriptor_guard(self.work_dir, task_type="reduce"):
                report = self._run_impl()
            report["output_xyz"] = str(final_output)
            report["remain_xyz"] = str(final_remain)
            report["report_json"] = str(final_report)
            atomic_json(temporary_report, report)
            os.replace(temporary_output, final_output)
            os.replace(temporary_remain, final_remain)
            os.replace(temporary_report, final_report)
            succeeded = True
            print(
                "[reduce] timing (hours): "
                f"encoding={report['encoding_hours']:.6f}, "
                f"processing={report['processing_hours']:.6f}, "
                f"total={report['total_hours']:.6f}"
            )
            return report
        except BaseException as exc:
            self._mark_resume_interrupted(exc)
            raise
        finally:
            self.output_xyz = final_output
            self.remain_xyz = final_remain
            self.report_json = final_report
            for path in (temporary_output, temporary_remain, temporary_report):
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
            if succeeded and not self.keep_intermediate:
                shutil.rmtree(self.descriptor_cache_dir, ignore_errors=True)
                self._remove_resume_artifacts()
            elif succeeded and self.mode == "reference_guided" and self._resume_checkpoint_path.exists():
                try:
                    state = json.loads(self._resume_checkpoint_path.read_text(encoding="utf-8"))
                    state["status"] = "complete"
                    state["updated_at"] = time.time()
                    atomic_json(self._resume_checkpoint_path, state, durable=False)
                except (OSError, ValueError) as exc:
                    warnings.warn(f"Could not mark Reduce checkpoint complete: {exc}", RuntimeWarning)
