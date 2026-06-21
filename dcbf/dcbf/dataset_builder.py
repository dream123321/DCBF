from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Iterable, Sequence

from ase import units
from ase.build import make_supercell
from ase.calculators.calculator import Calculator
from ase.data import atomic_numbers
from ase.io import iread, write
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
import numpy as np

from .das.main_calc import main_calc
from .das.work_dir import check_finish, check_scf, scf_dir
from .path_names import DFT_WORK_DIR
from .phonon_displacement import generate_phonopy_like_supercells
from .runtime_config import build_scheduler_spec, normalize_scheduler_config

try:
    from pymlip.core import MTPCalactor, PyConfiguration

    PYMLIP_AVAILABLE = True
except ImportError:
    PYMLIP_AVAILABLE = False


STRUCTURE_SUFFIXES = {".vasp", ".cif", ".xyz", ".extxyz"}
STRUCTURE_PREFIXES = ("POSCAR", "CONTCAR")
DEFAULT_UNIVERSAL_MATTERSIM_NAME = "mattersim-v1.0.0-1M.pth"
DEFAULT_UNIVERSAL_NEP_NAME = "nep.txt"
DEFAULT_UNIVERSAL_NEP_FALLBACK_PATHS = [
    Path("/share/home/xill/hj/NEP/gpumd_dynamic/nep.txt"),
]
DEFAULT_UNIVERSAL_SUS2_MTP_NAME = "MP_UIP.mtp"
DEFAULT_UNIVERSAL_SUS2_ELEMENTS = [
    "H", "Li", "Be", "B", "C", "N", "O", "F", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Rb",
    "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I",
    "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Ac", "Th", "Pa",
    "U", "Np", "Pu",
]


def _run_single_md_in_process(payload):
    atoms, md_cfg, structure_index, structure_name = payload
    return InitialDatasetBuilder._run_single_md_core(
        atoms,
        md_cfg,
        structure_index,
        structure_name,
        logger=None,
    )


class SUS2Calculator(Calculator):
    implemented_properties = ["energy", "forces", "stress"]

    def __init__(self, potential="p.sus2", ele_list=None, compute_stress=True, stress_weight=1.0, **kwargs):
        if not PYMLIP_AVAILABLE:
            raise ImportError("pymlip is required for the sus2 calculator")
        super().__init__(**kwargs)
        self.potential = potential
        self.compute_stress = compute_stress
        self.stress_weight = stress_weight
        self.mtpcalc = MTPCalactor(self.potential)
        if ele_list is None:
            raise ValueError("sus2 requires ele_list")
        self.unique_numbers = [atomic_numbers[element] for element in ele_list]

    def calculate(self, atoms=None, properties=None, system_changes=None):
        properties = properties or ["energy"]
        system_changes = system_changes or self.all_changes
        super().calculate(atoms=atoms, properties=properties, system_changes=system_changes)
        cfg = PyConfiguration.from_ase_atoms(atoms, unique_numbers=self.unique_numbers)
        volume = atoms.cell.volume if atoms.cell.volume > 0 else 1.0
        self.mtpcalc.calc(cfg)
        self.results["energy"] = np.array(cfg.energy)
        self.results["forces"] = cfg.force
        if self.compute_stress and hasattr(cfg, "stresses") and cfg.stresses is not None:
            stresses = cfg.stresses
            self.results["stress"] = -np.array(
                [
                    stresses[0, 0],
                    stresses[1, 1],
                    stresses[2, 2],
                    stresses[1, 2],
                    stresses[0, 2],
                    stresses[0, 1],
                ]
            ) * self.stress_weight / volume


@dataclass
class DatasetBuildResult:
    output_xyz: Path
    report_path: Path
    built_now: bool
    reused_existing: bool
    should_pause: bool


class InitialDatasetBuilder:
    DEFAULTS = {
        "enabled": False,
        "include_source_structures": False,
        "generation_mode": "configured",
        "output_xyz": "init_dataset/init_dataset.xyz",
        "reuse_if_exists": True,
        "post_build_action": "continue",
        "report_path": "init_dataset/build_report.json",
        "random_displacement": {
            "enabled": False,
            "supercell": [1, 1, 1],
            "strain": [1.0],
            "rattle_count": 0,
            "rattle_step": 0.005,
            "seed": 42,
            "phonon_displacement": {
                "enabled": False,
                "supercell": [1, 1, 1],
                "distance": 0.01,
                "diag": True,
                "plusminus": "auto",
                "trigonal": False,
                "symprec": 1e-5,
                "include_in_initial_train_set": True,
            },
        },
        "md": {
            "enabled": False,
            "supercell": [1, 1, 1],
            "parallel_workers": 1,
            "calc_type": "nep",
            "model": None,
            "ele_list": None,
            "device": "cpu",
            "temperature": 300.0,
            "pressure": 1.01325,
            "timestep": 1.0,
            "npt_steps": 0,
            "nvt_steps": 0,
            "npt_type": "berendsen",
            "ttime": None,
            "ttime_factor": 100.0,
            "nhc_length": 3,
            "log_interval": 100,
            "traj_interval": 100,
            "seed": 42,
        },
        "scf": {
            "calc_dir_num": 5,
            "force_threshold": 20,
            "pending_warning_hours": None,
        },
    }

    def __init__(self, run_dir: Path, config: dict, config_dir: Path):
        self.run_dir = Path(run_dir).resolve()
        self.config = config
        self.config_dir = Path(config_dir).resolve()
        self.dataset = dict(config.get("dataset", {}))
        raw_builder = dict(self.dataset.get("builder") or {})
        self.builder = self._normalize_builder_config(self.dataset.get("builder"))
        raw_builder_scf = dict(raw_builder.get("scf") or {})
        parameter_dft = dict((config.get("parameter") or {}).get("dft") or {})
        for key in ("calc_dir_num", "force_threshold", "pending_warning_hours"):
            if key not in raw_builder_scf and key in parameter_dft:
                self.builder["scf"][key] = parameter_dft[key]
        self.generation_mode = str(self.builder["generation_mode"]).strip().lower()
        if self.generation_mode not in {"configured", "random_displacement", "md", "both"}:
            raise ValueError(
                "dataset.builder.generation_mode must be one of "
                "'configured', 'random_displacement', 'md', or 'both'"
            )
        self.post_build_action = str(self.builder["post_build_action"]).strip().lower()
        if self.post_build_action not in {"continue", "pause"}:
            raise ValueError("dataset.builder.post_build_action must be 'continue' or 'pause'")
        self.scheduler = build_scheduler_spec(normalize_scheduler_config(dict(config["scheduler"])))
        self.output_xyz = self._resolve_output_path(self.builder["output_xyz"])
        self.report_path = self._resolve_output_path(self.builder["report_path"])
        self.build_root = self.run_dir / "init_dataset_build"
        self.build_workspace = self.build_root / "gen_0"
        self.logger = self._create_logger()

    @classmethod
    def _normalize_builder_config(cls, raw_builder):
        builder = dict(raw_builder or {})
        construction_methods = builder.pop("construction_methods", None)
        if construction_methods is not None:
            methods = dict(construction_methods or {})
            if "random_displacement" in methods:
                builder["random_displacement"] = dict(methods.get("random_displacement") or {})
            if "phonon_displacement" in methods:
                random_cfg = dict(builder.get("random_displacement") or {})
                random_cfg["phonon_displacement"] = dict(methods.get("phonon_displacement") or {})
                builder["random_displacement"] = random_cfg
            if "md" in methods:
                builder["md"] = dict(methods.get("md") or {})
        if "dft" in builder:
            dft_cfg = dict(builder.pop("dft") or {})
            scf_cfg = dict(builder.get("scf") or {})
            scf_cfg.update(dft_cfg)
            builder["scf"] = scf_cfg
        builder["post_build_action"] = "continue"
        builder["include_source_structures"] = False
        normalized = {}
        for key, default_value in cls.DEFAULTS.items():
            if isinstance(default_value, dict):
                merged = dict(default_value)
                merged.update(builder.get(key) or {})
                normalized[key] = merged
            elif key in builder:
                normalized[key] = builder[key]
            else:
                normalized[key] = default_value
        return normalized

    def is_enabled(self):
        return bool(self.builder.get("enabled", False))

    def ensure_dataset(self):
        if not self.is_enabled():
            raise ValueError("initial dataset builder is disabled")

        if self._can_reuse_existing_dataset():
            return DatasetBuildResult(
                output_xyz=self.output_xyz,
                report_path=self.report_path,
                built_now=False,
                reused_existing=True,
                should_pause=False,
            )

        resume_state = self._detect_resumable_build_state()
        if resume_state is not None:
            return self._resume_existing_build(resume_state)

        self._prepare_build_root()
        base_structures = self._load_source_structures()

        counts = {
            "source_structures": len(base_structures),
            "source_candidates": 0,
            "random_displacement_candidates": 0,
            "phonon_displacement_candidates": 0,
            "phonon_displacement_in_training_set": 0,
            "md_candidates": 0,
            "unique_candidates": 0,
            "labeled_structures": 0,
            "scf_completed": 0,
            "scf_collected": 0,
            "scf_force_threshold_count": 0,
        }

        try:
            candidates = []
            if self._coerce_bool(self.builder.get("include_source_structures", True), default=True):
                source_candidates = [self._clone_candidate(atoms, "source", atoms.info.get("builder_structure_name", "structure"), 0) for atoms in base_structures]
                counts["source_candidates"] = len(source_candidates)
                candidates.extend(source_candidates)

            random_candidates = []
            random_cfg = self.builder["random_displacement"]
            phonon_cfg = self._normalize_phonon_config(random_cfg.get("phonon_displacement"))
            md_cfg = self.builder["md"]
            use_random_as_output, use_md_as_output = self._resolve_generation_outputs()

            if use_random_as_output:
                random_candidates = self._generate_random_displacement_candidates(base_structures, random_cfg)
                counts["random_displacement_candidates"] = len(random_candidates)
                self._write_xyz(self.build_root / "random_displacement.xyz", random_candidates)
                if use_random_as_output:
                    candidates.extend(random_candidates)

            phonon_candidates = []
            if self._coerce_bool(phonon_cfg.get("enabled", False), default=False):
                phonon_candidates = self._generate_phonon_displacement_candidates(base_structures, phonon_cfg)
                counts["phonon_displacement_candidates"] = len(phonon_candidates)
                self._write_xyz(self.build_root / "phonon_displacement.xyz", phonon_candidates)
                if self._coerce_bool(phonon_cfg.get("include_in_initial_train_set", True), default=True):
                    counts["phonon_displacement_in_training_set"] = len(phonon_candidates)
                    candidates.extend(phonon_candidates)

            if use_md_as_output:
                md_inputs = list(base_structures)
                md_candidates = self._generate_md_candidates(md_inputs, md_cfg)
                counts["md_candidates"] = len(md_candidates)
                self._write_xyz(self.build_root / "md_candidates.xyz", md_candidates)
                candidates.extend(md_candidates)

            if not candidates:
                raise ValueError("initial dataset builder produced no candidate structures")

            unique_candidates = self._deduplicate_structures(candidates)
            counts["unique_candidates"] = len(unique_candidates)
            candidate_xyz = self.build_root / "candidate_pool.xyz"
            self._write_xyz(candidate_xyz, unique_candidates)
            self._log_candidate_structure_summary(base_structures, unique_candidates, counts)

            scf_stats = self._run_scf_labelling(unique_candidates)
            counts.update(scf_stats)

            report = {
                "status": "completed",
                "built_at": datetime.now().isoformat(timespec="seconds"),
                "output_xyz": str(self.output_xyz),
                "report_path": str(self.report_path),
                "generation_mode": self.generation_mode,
                "post_build_action": self.post_build_action,
                "counts": counts,
            }
            self._write_report(report)
        except Exception as exc:
            report = {
                "status": "failed",
                "built_at": datetime.now().isoformat(timespec="seconds"),
                "output_xyz": str(self.output_xyz),
                "report_path": str(self.report_path),
                "error": str(exc),
                "generation_mode": self.generation_mode,
                "post_build_action": self.post_build_action,
                "counts": counts,
            }
            self._write_report(report)
            raise

        should_pause = self.post_build_action == "pause"
        return DatasetBuildResult(
            output_xyz=self.output_xyz,
            report_path=self.report_path,
            built_now=True,
            reused_existing=False,
            should_pause=should_pause,
        )

    def _detect_resumable_build_state(self):
        candidate_pool = self.build_root / "candidate_pool.xyz"
        scf_root = self.build_workspace / DFT_WORK_DIR / "scf"
        scf_filter_root = scf_root / "filter"
        if not candidate_pool.exists() or candidate_pool.stat().st_size == 0:
            return None
        if not scf_filter_root.exists():
            return None
        try:
            task_dirs = scf_dir(str(self.build_workspace))
        except Exception:
            return None
        if not task_dirs:
            return None
        return {
            "candidate_pool": candidate_pool,
            "scf_root": scf_root,
            "task_dirs": task_dirs,
        }

    def _resume_existing_build(self, resume_state):
        base_structures = self._load_source_structures()
        random_cfg = self.builder["random_displacement"]
        phonon_cfg = self._normalize_phonon_config(random_cfg.get("phonon_displacement"))
        counts = {
            "source_structures": len(base_structures),
            "source_candidates": len(base_structures) if self._coerce_bool(self.builder.get("include_source_structures", True), default=True) else 0,
            "random_displacement_candidates": self._count_xyz_frames(self.build_root / "random_displacement.xyz"),
            "phonon_displacement_candidates": self._count_xyz_frames(self.build_root / "phonon_displacement.xyz"),
            "phonon_displacement_in_training_set": 0,
            "md_candidates": self._count_xyz_frames(self.build_root / "md_candidates.xyz"),
            "unique_candidates": self._count_xyz_frames(resume_state["candidate_pool"]),
            "labeled_structures": 0,
            "scf_completed": 0,
            "scf_collected": 0,
            "scf_force_threshold_count": 0,
        }
        if self._coerce_bool(phonon_cfg.get("include_in_initial_train_set", True), default=True):
            counts["phonon_displacement_in_training_set"] = counts["phonon_displacement_candidates"]

        candidate_atoms = list(iread(str(resume_state["candidate_pool"]), index=":"))
        self.logger.info("=" * 72)
        self.logger.info("[builder.resume] Existing candidate structures and SCF tasks detected. Waiting for SCF completion without regenerating inputs.")
        self.logger.info("[builder.resume] candidate_pool=%s", resume_state["candidate_pool"])
        self.logger.info("[builder.resume] scf_root=%s", resume_state["scf_root"])
        self.logger.info("[builder.resume] existing_task_dirs=%s", len(resume_state["task_dirs"]))
        self.logger.info("=" * 72)
        self._log_candidate_structure_summary(base_structures, candidate_atoms, counts)

        scf_stats = self._collect_existing_scf_results(candidate_atoms, wait_for_completion=True)
        counts.update(scf_stats)
        report = {
            "status": "completed",
            "built_at": datetime.now().isoformat(timespec="seconds"),
            "output_xyz": str(self.output_xyz),
            "report_path": str(self.report_path),
            "generation_mode": self.generation_mode,
            "post_build_action": self.post_build_action,
            "counts": counts,
            "resumed_from_existing_build": True,
        }
        self._write_report(report)

        should_pause = self.post_build_action == "pause"
        return DatasetBuildResult(
            output_xyz=self.output_xyz,
            report_path=self.report_path,
            built_now=True,
            reused_existing=False,
            should_pause=should_pause,
        )

    def _create_logger(self):
        self.build_root.mkdir(parents=True, exist_ok=True)
        logger = logging.getLogger(f"dcbf.dataset_builder.{id(self)}")
        logger.setLevel(logging.INFO)
        logger.propagate = False
        logger.handlers.clear()

        formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
        file_handler = logging.FileHandler(self.build_root / "app.log", encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        return logger

    def _resolve_output_path(self, raw_path):
        path = Path(raw_path)
        if path.is_absolute():
            return path
        return (self.run_dir / path).resolve()

    def _resolve_config_path(self, raw_path):
        path = Path(raw_path)
        if path.is_absolute():
            return path
        return (self.config_dir / path).resolve()

    @staticmethod
    def _default_universal_sus2_model_path():
        return Path(__file__).resolve().parent / "default_reduce_assets" / DEFAULT_UNIVERSAL_SUS2_MTP_NAME

    @staticmethod
    def _default_universal_nep_model_path():
        candidates = [Path(__file__).resolve().parent / "default_reduce_assets" / DEFAULT_UNIVERSAL_NEP_NAME]
        candidates.extend(DEFAULT_UNIVERSAL_NEP_FALLBACK_PATHS)
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            "Could not find a default NEP model. Checked: "
            + ", ".join(str(path) for path in candidates)
        )

    @staticmethod
    def _default_universal_mattersim_model_path():
        candidate = Path(__file__).resolve().parent / "default_reduce_assets" / DEFAULT_UNIVERSAL_MATTERSIM_NAME
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Could not find the default MatterSim model: {candidate}")

    @staticmethod
    def _coerce_bool(value, default=True):
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

    def _resolve_generation_outputs(self):
        random_enabled = self._coerce_bool(self.builder["random_displacement"].get("enabled", False), default=False)
        md_enabled = self._coerce_bool(self.builder["md"].get("enabled", False), default=False)
        if self.generation_mode == "configured":
            return random_enabled, md_enabled
        if self.generation_mode == "random_displacement":
            return True, False
        if self.generation_mode == "md":
            return False, True
        return True, True

    def _load_report(self):
        if not self.report_path.exists():
            return None
        try:
            return json.loads(self.report_path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _can_reuse_existing_dataset(self):
        if not self._coerce_bool(self.builder.get("reuse_if_exists", True), default=True):
            return False
        if not self.output_xyz.exists() or self.output_xyz.stat().st_size == 0:
            return False
        report = self._load_report()
        return bool(report and report.get("status") == "completed")

    def _prepare_build_root(self):
        self.build_root.mkdir(parents=True, exist_ok=True)
        for directory in (
            self.build_root / "candidates",
            self.build_workspace,
            self.build_workspace / DFT_WORK_DIR,
        ):
            if directory.exists():
                shutil.rmtree(directory)
        for path in (
            self.build_root / "random_displacement.xyz",
            self.build_root / "phonon_displacement.xyz",
            self.build_root / "md_candidates.xyz",
            self.build_root / "candidate_pool.xyz",
            self.output_xyz,
            self.report_path,
        ):
            if path.exists():
                path.unlink()
        self.output_xyz.parent.mkdir(parents=True, exist_ok=True)
        self.report_path.parent.mkdir(parents=True, exist_ok=True)

    def _write_report(self, report):
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    @staticmethod
    def _count_xyz_frames(path):
        path = Path(path)
        if not path.exists() or path.stat().st_size == 0:
            return 0
        return sum(1 for _ in iread(str(path), index=":"))

    def _iter_structure_files(self):
        structure_dir = self.run_dir / "stru"
        if not structure_dir.exists():
            raise FileNotFoundError(f"structure source directory does not exist: {structure_dir}")
        for path in sorted(structure_dir.iterdir()):
            if not path.is_file():
                continue
            if path.suffix.lower() in STRUCTURE_SUFFIXES or path.name.startswith(STRUCTURE_PREFIXES):
                yield path

    def _load_source_structures(self):
        atoms_list = []
        for path in self._iter_structure_files():
            for index, atoms in enumerate(iread(str(path), index=":")):
                atoms = atoms.copy()
                atoms.calc = None
                atoms.pbc = [True, True, True]
                atoms.info = dict(atoms.info)
                atoms.info["builder_structure_file"] = str(path)
                atoms.info["builder_structure_name"] = f"{path.stem}_{index}"
                atoms_list.append(atoms)
        if not atoms_list:
            raise ValueError(f"no structures were found in {self.run_dir / 'stru'}")
        return atoms_list

    def _generate_random_displacement_candidates(self, base_structures, random_cfg):
        supercell = self._parse_supercell(random_cfg.get("supercell", [1, 1, 1]))
        strain_values = list(random_cfg.get("strain") or [1.0])
        rattle_count = int(random_cfg.get("rattle_count", 0))
        rattle_step = float(random_cfg.get("rattle_step", 0.005))
        seed = int(random_cfg.get("seed", 42))

        generated = []
        candidate_index = 0
        for structure_index, atoms in enumerate(base_structures):
            supercell_atoms = make_supercell(atoms, supercell)
            scaled_positions = supercell_atoms.get_scaled_positions()
            cell = supercell_atoms.get_cell()
            structure_name = atoms.info.get("builder_structure_name", f"structure_{structure_index}")
            for strain_factor in strain_values:
                strained = supercell_atoms.copy()
                strained.set_cell(cell * float(strain_factor))
                strained.set_scaled_positions(scaled_positions)
                generated.append(self._clone_candidate(strained, "random_displacement", structure_name, candidate_index))
                candidate_index += 1
                for rattle_index in range(rattle_count):
                    rattled = strained.copy()
                    local_seed = seed + structure_index * 100000 + int(round(float(strain_factor) * 1000)) * 100 + rattle_index
                    rattled.rattle((rattle_index + 1) * rattle_step, rng=np.random.RandomState(local_seed))
                    generated.append(self._clone_candidate(rattled, "random_displacement", structure_name, candidate_index))
                    candidate_index += 1
        return generated

    def _generate_phonon_displacement_candidates(self, base_structures, phonon_cfg):
        generated = []
        candidate_index = 0
        distance = float(phonon_cfg.get("distance", 0.01))
        plusminus = phonon_cfg.get("plusminus", "auto")
        if isinstance(plusminus, str):
            lowered = plusminus.strip().lower()
            if lowered == "auto":
                plusminus = "auto"
            elif lowered in {"true", "yes", "1", "on"}:
                plusminus = True
            elif lowered in {"false", "no", "0", "off"}:
                plusminus = False
            else:
                raise ValueError("builder.random_displacement.phonon_displacement.plusminus must be 'auto', true, or false")
        is_diagonal = self._coerce_bool(phonon_cfg.get("diag", True), default=True)
        is_trigonal = self._coerce_bool(phonon_cfg.get("trigonal", False), default=False)
        symprec = float(phonon_cfg.get("symprec", 1e-5))
        supercell = self._parse_supercell(phonon_cfg.get("supercell", [1, 1, 1]))

        for structure_index, atoms in enumerate(base_structures):
            structure_name = atoms.info.get("builder_structure_name", f"structure_{structure_index}")
            displaced_supercells, specs, _ = generate_phonopy_like_supercells(
                atoms,
                supercell,
                distance=distance,
                plusminus=plusminus,
                is_diagonal=is_diagonal,
                is_trigonal=is_trigonal,
                symprec=symprec,
            )
            if len(displaced_supercells) != len(specs):
                raise ValueError("phonon displaced supercells and specs have inconsistent lengths")
            for displaced, spec in zip(displaced_supercells, specs):
                displaced.info["builder_source_kind"] = "phonon_displacement"
                displaced.info["builder_parent_structure"] = structure_name
                displaced.info["builder_candidate_index"] = int(candidate_index)
                displaced.info["phonon_supercell"] = np.asarray(supercell, dtype=int).tolist()
                generated.append(self._clone_candidate(displaced, "phonon_displacement", structure_name, candidate_index))
                candidate_index += 1
        return generated

    @staticmethod
    def _normalize_phonon_config(raw_phonon):
        phonon_default = {
            "enabled": False,
            "supercell": [1, 1, 1],
            "distance": 0.01,
            "diag": True,
            "plusminus": "auto",
            "trigonal": False,
            "symprec": 1e-5,
            "include_in_initial_train_set": True,
        }
        phonon_default.update(raw_phonon or {})
        return phonon_default

    def _generate_md_candidates(self, md_inputs, md_cfg):
        if not md_inputs:
            raise ValueError("builder.md is enabled but no input structures are available")

        generated = []
        md_run_root = self.build_root / "md_runs"
        md_run_root.mkdir(parents=True, exist_ok=True)
        candidate_index = 0
        parallel_workers = max(1, int(md_cfg.get("parallel_workers", 1)))

        tasks = []
        for structure_index, atoms in enumerate(md_inputs):
            structure_name = atoms.info.get("builder_structure_name", f"structure_{structure_index}")
            workdir = md_run_root / structure_name
            workdir.mkdir(parents=True, exist_ok=True)
            tasks.append((structure_index, atoms, structure_name, workdir))

        self.logger.info(
            "[builder.md] structures=%s parallel_workers=%s",
            len(tasks),
            min(parallel_workers, len(tasks)) if tasks else parallel_workers,
        )

        results = []
        if parallel_workers == 1 or len(tasks) <= 1:
            for structure_index, atoms, structure_name, workdir in tasks:
                md_samples = self._run_single_md(atoms, md_cfg, structure_index, structure_name)
                results.append((structure_index, structure_name, workdir, md_samples))
        else:
            max_workers = min(parallel_workers, len(tasks))
            process_md_cfg = dict(md_cfg)
            if process_md_cfg.get("model") is not None:
                process_md_cfg["model"] = str(self._resolve_config_path(process_md_cfg["model"]))
            self.logger.info("[builder.md] using process pool workers=%s", max_workers)
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        _run_single_md_in_process,
                        (atoms, process_md_cfg, structure_index, structure_name),
                    ): (
                        structure_index,
                        structure_name,
                        workdir,
                    )
                    for structure_index, atoms, structure_name, workdir in tasks
                }
                for future in as_completed(futures):
                    structure_index, structure_name, workdir = futures[future]
                    md_samples = future.result()
                    self.logger.info(
                        "[builder.md] finished structure=%s samples=%s",
                        structure_name,
                        len(md_samples),
                    )
                    results.append((structure_index, structure_name, workdir, md_samples))

        results.sort(key=lambda item: item[0])
        for _, structure_name, workdir, md_samples in results:
            self._write_xyz(workdir / "trajectory_samples.xyz", md_samples)
            for sample in md_samples:
                generated.append(self._clone_candidate(sample, "md", structure_name, candidate_index))
                candidate_index += 1
        return generated

    def _run_single_md(self, atoms, md_cfg, structure_index, structure_name):
        md_cfg_for_run = dict(md_cfg)
        model_path = md_cfg_for_run.get("model")
        if model_path is not None:
            md_cfg_for_run["model"] = str(self._resolve_config_path(model_path))
        return self._run_single_md_core(
            atoms,
            md_cfg_for_run,
            structure_index,
            structure_name,
            logger=self.logger,
        )

    @staticmethod
    def _run_single_md_core(atoms, md_cfg, structure_index, structure_name, logger=None):
        calc_type = str(md_cfg.get("calc_type", "nep")).strip().lower()
        model_path = md_cfg.get("model")
        if model_path is not None:
            model_path = str(model_path)
        ele_list = md_cfg.get("ele_list")
        device = str(md_cfg.get("device", "cpu")).strip().lower()
        temperature = float(md_cfg.get("temperature", 300.0))
        pressure = float(md_cfg.get("pressure", 1.01325))
        dt_fs = float(md_cfg.get("timestep", 1.0))
        npt_steps = int(md_cfg.get("npt_steps", 0))
        nvt_steps = int(md_cfg.get("nvt_steps", 0))
        npt_type = str(md_cfg.get("npt_type", "berendsen")).strip().lower()
        ttime_fs = md_cfg.get("ttime")
        if ttime_fs is not None:
            ttime_fs = float(ttime_fs)
        else:
            ttime_fs = dt_fs * float(md_cfg.get("ttime_factor", 100.0))
        nhc_length = int(md_cfg.get("nhc_length", 3))
        log_interval = int(md_cfg.get("log_interval", 100))
        traj_interval = int(md_cfg.get("traj_interval", 100))
        seed = int(md_cfg.get("seed", 42)) + structure_index
        if log_interval <= 0:
            raise ValueError("builder.md.log_interval must be > 0")
        if traj_interval <= 0:
            raise ValueError("builder.md.traj_interval must be > 0")

        calculator = InitialDatasetBuilder._setup_calculator(
            calc_type,
            model_path=model_path,
            device=device,
            ele_list=ele_list,
        )
        md_supercell = InitialDatasetBuilder._parse_supercell(md_cfg.get("supercell", [1, 1, 1]))
        md_atoms = make_supercell(atoms, md_supercell)
        md_atoms.calc = calculator
        md_atoms.pbc = [True, True, True]

        if logger is not None:
            logger.info(
                "[builder.md] start structure=%s calc_type=%s atoms=%s npt_steps=%s nvt_steps=%s dt_fs=%s",
                structure_name,
                calc_type,
                len(md_atoms),
                npt_steps,
                nvt_steps,
                dt_fs,
            )

        rng = np.random.RandomState(seed)
        MaxwellBoltzmannDistribution(md_atoms, temperature_K=temperature, rng=rng)
        Stationary(md_atoms)
        ZeroRotation(md_atoms)

        samples = []
        if npt_steps > 0:
            npt_dyn = InitialDatasetBuilder._get_npt_integrator(
                md_atoms,
                temperature_K=temperature,
                pressure_bar=pressure,
                timestep_fs=dt_fs,
                ttime=ttime_fs,
                npt_type=npt_type,
            )
            for step in range(1, npt_steps + 1):
                npt_dyn.run(1)
                InitialDatasetBuilder._log_md_progress(
                    md_atoms,
                    structure_name,
                    "NPT",
                    step,
                    npt_steps,
                    log_interval,
                    logger=logger,
                )
                if step % traj_interval == 0:
                    samples.append(InitialDatasetBuilder._snapshot_atoms(md_atoms, "md_npt", structure_name, step))

        if nvt_steps > 0:
            from ase.md.nose_hoover_chain import NoseHooverChainNVT

            nvt_dyn = NoseHooverChainNVT(
                md_atoms,
                timestep=dt_fs * units.fs,
                temperature_K=temperature,
                tchain=nhc_length,
                tdamp=ttime_fs * units.fs,
            )
            for step in range(1, nvt_steps + 1):
                nvt_dyn.run(1)
                InitialDatasetBuilder._log_md_progress(
                    md_atoms,
                    structure_name,
                    "NVT",
                    step,
                    nvt_steps,
                    log_interval,
                    logger=logger,
                )
                if step % traj_interval == 0:
                    samples.append(InitialDatasetBuilder._snapshot_atoms(md_atoms, "md_nvt", structure_name, step))

        if not samples and (npt_steps > 0 or nvt_steps > 0):
            final_step = nvt_steps if nvt_steps > 0 else npt_steps
            final_stage = "md_nvt" if nvt_steps > 0 else "md_npt"
            samples.append(InitialDatasetBuilder._snapshot_atoms(md_atoms, final_stage, structure_name, final_step))
        return samples

    @staticmethod
    def _log_md_progress(atoms, structure_name, stage, step, total_steps, log_interval, logger=None):
        if step != 1 and step != total_steps and step % log_interval != 0:
            return
        if logger is None:
            return
        temperature = atoms.get_kinetic_energy() / (1.5 * units.kB * len(atoms))
        volume = atoms.get_volume()
        logger.info(
            "[builder.md] structure=%s stage=%s step=%s/%s temperature=%.2fK volume=%.3f",
            structure_name,
            stage,
            step,
            total_steps,
            temperature,
            volume,
        )

    @staticmethod
    def _parse_supercell(raw_supercell):
        matrix = np.array(raw_supercell, dtype=int)
        if matrix.shape == (3,):
            return np.diag(matrix)
        if matrix.shape == (3, 3):
            return matrix
        raise ValueError("builder.random_displacement.supercell must be [a, b, c] or a 3x3 matrix")

    def _clone_candidate(self, atoms, source_kind, structure_name, candidate_index):
        cloned = atoms.copy()
        cloned.calc = None
        cloned.pbc = [True, True, True]
        cloned.info = dict(cloned.info)
        cloned.info["builder_source_kind"] = source_kind
        cloned.info["builder_parent_structure"] = structure_name
        cloned.info["builder_candidate_index"] = int(candidate_index)
        if "momenta" in cloned.arrays:
            del cloned.arrays["momenta"]
        return cloned

    @staticmethod
    def _snapshot_atoms(atoms, stage, structure_name, step):
        snapshot = atoms.copy()
        snapshot.calc = None
        snapshot.pbc = [True, True, True]
        snapshot.info = dict(snapshot.info)
        snapshot.info["builder_source_kind"] = stage
        snapshot.info["builder_parent_structure"] = structure_name
        snapshot.info["builder_step"] = int(step)
        if "momenta" in snapshot.arrays:
            del snapshot.arrays["momenta"]
        return snapshot

    def _deduplicate_structures(self, structures: Iterable):
        unique = []
        seen = set()
        for atoms in structures:
            key = self._fingerprint_atoms(atoms)
            if key in seen:
                continue
            seen.add(key)
            unique.append(atoms)
        return unique

    @staticmethod
    def _fingerprint_atoms(atoms):
        symbols = tuple(atoms.get_chemical_symbols())
        cell = tuple(np.round(np.asarray(atoms.get_cell()), 8).reshape(-1))
        scaled = tuple(np.round(atoms.get_scaled_positions(wrap=True), 6).reshape(-1))
        return symbols, cell, scaled

    @staticmethod
    def _write_xyz(path: Path, structures: Sequence):
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            path.unlink()
        for atoms in structures:
            write(str(path), atoms, format="extxyz", append=True)

    @staticmethod
    def _summarize_atoms_collection(structures):
        summary = {
            "count": len(structures),
            "kind_counts": {},
            "parent_counts": {},
            "formula_counts": {},
            "atom_count_min": 0,
            "atom_count_max": 0,
            "atom_count_mean": 0.0,
        }
        if not structures:
            return summary

        kind_counts = {}
        parent_counts = {}
        formula_counts = {}
        atom_counts = []
        for atoms in structures:
            kind = atoms.info.get("builder_source_kind", "unknown")
            parent = atoms.info.get("builder_parent_structure", "unknown")
            formula = atoms.get_chemical_formula()
            kind_counts[kind] = kind_counts.get(kind, 0) + 1
            parent_counts[parent] = parent_counts.get(parent, 0) + 1
            formula_counts[formula] = formula_counts.get(formula, 0) + 1
            atom_counts.append(len(atoms))

        summary["kind_counts"] = dict(sorted(kind_counts.items()))
        summary["parent_counts"] = dict(sorted(parent_counts.items()))
        summary["formula_counts"] = dict(sorted(formula_counts.items()))
        summary["atom_count_min"] = int(min(atom_counts))
        summary["atom_count_max"] = int(max(atom_counts))
        summary["atom_count_mean"] = float(sum(atom_counts) / len(atom_counts))
        return summary

    def _log_candidate_structure_summary(self, base_structures, unique_candidates, counts):
        source_summary = self._summarize_atoms_collection(base_structures)
        candidate_summary = self._summarize_atoms_collection(unique_candidates)
        self.logger.info("=" * 72)
        self.logger.info("[builder.summary] Candidate structure summary")
        self.logger.info(
            "[builder.summary] source_structures=%s source_candidates=%s random_displacement=%s phonon=%s phonon_in_training=%s md=%s unique_candidates=%s",
            counts["source_structures"],
            counts["source_candidates"],
            counts["random_displacement_candidates"],
            counts["phonon_displacement_candidates"],
            counts["phonon_displacement_in_training_set"],
            counts["md_candidates"],
            counts["unique_candidates"],
        )
        self.logger.info(
            "[builder.summary] source_formula_counts=%s source_atom_count[min,max,mean]=[%s,%s,%.2f]",
            source_summary["formula_counts"],
            source_summary["atom_count_min"],
            source_summary["atom_count_max"],
            source_summary["atom_count_mean"],
        )
        self.logger.info(
            "[builder.summary] candidate_kind_counts=%s candidate_formula_counts=%s",
            candidate_summary["kind_counts"],
            candidate_summary["formula_counts"],
        )
        self.logger.info(
            "[builder.summary] candidate_parent_counts=%s candidate_atom_count[min,max,mean]=[%s,%s,%.2f]",
            candidate_summary["parent_counts"],
            candidate_summary["atom_count_min"],
            candidate_summary["atom_count_max"],
            candidate_summary["atom_count_mean"],
        )
        self.logger.info("[builder.summary] candidate_pool=%s", self.build_root / "candidate_pool.xyz")
        self.logger.info("=" * 72)

    def _run_scf_labelling(self, structures):
        scf_cfg = self.builder["scf"]
        calc_dir_num = int(scf_cfg.get("calc_dir_num", 5))
        force_threshold = float(scf_cfg.get("force_threshold", 20))
        effective_calc_dir_num = len(structures) if calc_dir_num <= 0 else min(calc_dir_num, len(structures))
        if effective_calc_dir_num == 0:
            raise ValueError("initial dataset builder has no structures for SCF labelling")

        workspace = self.build_workspace
        scf_path = workspace / DFT_WORK_DIR / "scf"
        scf_path.mkdir(parents=True, exist_ok=True)
        cwd = Path.cwd()
        try:
            os.chdir(scf_path)
            count = main_calc(list(structures), effective_calc_dir_num, str(workspace), self.scheduler)
            if count == 0:
                raise ValueError("failed to create SCF tasks for the initial dataset")
            subprocess.run([sys.executable, "start_calc.py"], check=True)
        finally:
            os.chdir(cwd)

        return self._collect_existing_scf_results(
            input_structures=structures,
            wait_for_completion=True,
            effective_calc_dir_num=effective_calc_dir_num,
        )

    def _collect_existing_scf_results(self, input_structures, wait_for_completion=True, effective_calc_dir_num=None):
        scf_cfg = self.builder["scf"]
        force_threshold = float(scf_cfg.get("force_threshold", 20))
        workspace = self.build_workspace
        if effective_calc_dir_num is None:
            effective_calc_dir_num = int(scf_cfg.get("calc_dir_num", 5))

        if wait_for_completion and not check_scf(str(workspace)):
            self.logger.info("[builder.resume] SCF completion pending")
        self.logger.info("Waiting for initial SCF tasks to finish")
        check_finish(
            scf_dir(str(workspace)),
            self.logger,
            "Initial SCF calculations have been completed",
            pending_warning_hours=scf_cfg.get("pending_warning_hours"),
        )

        collector = self._resolve_scf_handler()
        current = workspace / DFT_WORK_DIR / "scf" / "filter"
        original_output = self.output_xyz.parent / "ori_init_dataset.xyz"
        if original_output.exists():
            original_output.unlink()
        ok_count, len_count, no_success_paths, force_count, _ = collector(
            str(current),
            str(self.output_xyz),
            str(original_output),
            force_threshold,
        )
        if not self.output_xyz.exists() or self.output_xyz.stat().st_size == 0:
            raise RuntimeError("SCF finished but no labeled initial dataset was collected")
        labeled_atoms = list(iread(str(self.output_xyz), index=":"))
        self.logger.info(
            "Initial dataset collected: completed=%s, collected=%s, below_force_threshold=%s",
            ok_count,
            len_count,
            force_count,
        )
        if no_success_paths:
            self.logger.warning("Some SCF paths did not collect successfully: %s", len(no_success_paths))
        self._log_scf_summary(
            input_structures=input_structures,
            labeled_structures=labeled_atoms,
            effective_calc_dir_num=effective_calc_dir_num,
            ok_count=ok_count,
            len_count=len_count,
            no_success_paths=no_success_paths,
            force_count=force_count,
        )
        return {
            "labeled_structures": len(labeled_atoms),
            "scf_completed": ok_count,
            "scf_collected": len_count,
            "scf_force_threshold_count": force_count,
        }

    def _log_scf_summary(self, input_structures, labeled_structures, effective_calc_dir_num, ok_count, len_count, no_success_paths, force_count):
        input_summary = self._summarize_atoms_collection(input_structures)
        labeled_summary = self._summarize_atoms_collection(labeled_structures)
        self.logger.info("=" * 72)
        self.logger.info("[builder.summary] SCF result summary")
        self.logger.info(
            "[builder.summary] scf_input_structures=%s calc_dir_num=%s completed_tasks=%s collected_structures=%s accepted_by_force_threshold=%s failed_paths=%s",
            len(input_structures),
            effective_calc_dir_num,
            ok_count,
            len_count,
            force_count,
            len(no_success_paths),
        )
        self.logger.info(
            "[builder.summary] scf_input_kind_counts=%s scf_input_formula_counts=%s",
            input_summary["kind_counts"],
            input_summary["formula_counts"],
        )
        self.logger.info(
            "[builder.summary] labeled_formula_counts=%s labeled_atom_count[min,max,mean]=[%s,%s,%.2f]",
            labeled_summary["formula_counts"],
            labeled_summary["atom_count_min"],
            labeled_summary["atom_count_max"],
            labeled_summary["atom_count_mean"],
        )
        self.logger.info(
            "[builder.summary] labeled_output=%s original_scf_output=%s",
            self.output_xyz,
            self.output_xyz.parent / "ori_init_dataset.xyz",
        )
        if no_success_paths:
            self.logger.warning("[builder.summary] scf_failed_paths=%s", no_success_paths)
        self.logger.info("=" * 72)

    def _resolve_scf_handler(self):
        scf_cal_engine = self.scheduler.scf_cal_engine
        if scf_cal_engine == "abacus":
            from .das.abacus_main_xyz import abacus_main_xyz as handler
        elif scf_cal_engine == "cp2k":
            from .das.cp2k_main_xyz import cp2k_main_xyz as handler
        elif scf_cal_engine == "qe":
            from .das.qe_main_xyz import qe_main_xyz as handler
        elif scf_cal_engine == "vasp":
            from .das.vasp_main_xyz import vasp_main_xyz as handler
        else:
            raise ValueError(f"{scf_cal_engine} does not exist")
        return handler

    @staticmethod
    def _get_npt_integrator(atoms, temperature_K, pressure_bar, timestep_fs, ttime=None, pfactor=None, npt_type="berendsen"):
        dt = timestep_fs * units.fs
        if ttime is None:
            ttime = timestep_fs * 100.0

        if npt_type == "berendsen":
            from ase.md.nptberendsen import NPTBerendsen

            return NPTBerendsen(
                atoms,
                timestep=dt,
                temperature_K=temperature_K,
                pressure_au=pressure_bar * 1e-5 / units.Pascal,
                taut=ttime * units.fs,
                taup=ttime * 10 * units.fs,
                compressibility_au=4.57e-5 / units.bar,
            )

        if npt_type == "npt_ase":
            from ase.md.npt import NPT as ASE_NPT

            pressure_gpa = pressure_bar / 10000.0
            if pfactor is None:
                pfactor = (75.0 * units.fs) ** 2 * 100.0 * units.GPa
            return ASE_NPT(
                atoms,
                timestep=dt,
                temperature_K=temperature_K,
                externalstress=pressure_gpa * units.GPa,
                ttime=ttime * units.fs,
                pfactor=pfactor,
            )

        raise ValueError(f"unsupported builder.md.npt_type: {npt_type}")

    @staticmethod
    def _setup_calculator(calc_type, model_path=None, device="cpu", ele_list=None):
        if calc_type == "nep":
            from pynep.calculate import NEP

            if not model_path:
                model_path = str(InitialDatasetBuilder._default_universal_nep_model_path())
            return NEP(model_path)

        if calc_type == "mace":
            from mace.calculators import MACECalculator

            if not model_path:
                raise ValueError("mace requires builder.md.model")
            return MACECalculator(model_paths=model_path, device=device)

        if calc_type == "dp":
            from deepmd.calculator import DP

            if not model_path:
                raise ValueError("dp requires builder.md.model")
            return DP(model=model_path, device=device)

        if calc_type == "chgnet":
            from chgnet.model import CHGNet
            from chgnet.model.dynamics import CHGNetCalculator

            model = CHGNet.from_file(model_path) if model_path else CHGNet.load()
            return CHGNetCalculator(model, use_device=device)

        if calc_type == "m3gnet":
            import matgl
            from matgl.ext.ase import PESCalculator

            matgl.set_backend("DGL")
            model = matgl.load_model(model_path) if model_path else matgl.load_model("M3GNet-MP-2021.2.8-PES")
            return PESCalculator(model)

        if calc_type == "mattersim":
            from mattersim.forcefield import MatterSimCalculator

            if not model_path:
                model_path = str(InitialDatasetBuilder._default_universal_mattersim_model_path())
            return MatterSimCalculator(load_path=model_path, device=device)

        if calc_type == "sus2":
            using_default_universal_model = not model_path
            if using_default_universal_model:
                model_path = str(InitialDatasetBuilder._default_universal_sus2_model_path())
            if ele_list is None and using_default_universal_model:
                ele_list = list(DEFAULT_UNIVERSAL_SUS2_ELEMENTS)
            if ele_list is None:
                raise ValueError("sus2 requires builder.md.ele_list when builder.md.model is explicitly provided")
            return SUS2Calculator(potential=model_path, ele_list=ele_list)

        raise ValueError(f"unsupported builder.md.calc_type: {calc_type}")
