from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import warnings

from ase.data import atomic_numbers, chemical_symbols
from ase.io import iread, write
import yaml

from .dataset_builder import InitialDatasetBuilder
from .builtin_models import BUILTIN_SUS2_87, LEGACY_BUILTIN_SUS2_87, resolve_model_path
from .encode.coverage_policy import (
    normalize_selection_budget_scope,
    validate_selection_schedules,
)
from .npt_volume_filter import normalize_npt_max_cell_volume_filter_factor
from .path_names import DFT_WORK_DIR
from .runtime_config import build_md_loop_schedule, load_json_config, normalize_scheduler_config, save_runtime_config
from .sus2_training import (
    DEFAULT_FINAL_ITERATIONS,
    DEFAULT_INITIAL_ITERATIONS,
    DEFAULT_SUBSEQUENT_ITERATIONS,
)


def _write_parameter_if_changed(init_dir, parameter):
    """Write init/parameter.yaml only when the config-derived values changed.

    A rerun calls ``prepare_workspace`` again.  If the JSON config is
    unchanged, an existing ``init/parameter.yaml`` may contain deliberate user
    edits, so it must be preserved.  If the config changed, the new
    config-derived document wins.  The snapshot stores the hash of the last
    config-derived document, not of the user-edited file.
    """
    init_dir = Path(init_dir)
    text = yaml.safe_dump(parameter, default_flow_style=False, sort_keys=False)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    parameter_path = init_dir / "parameter.yaml"
    snapshot_path = init_dir / ".dcbf_parameter_source.json"
    previous = None
    if snapshot_path.is_file():
        try:
            previous = json.loads(snapshot_path.read_text(encoding="utf-8")).get("sha256")
        except (OSError, ValueError):
            previous = None
    if parameter_path.is_file() and previous == digest:
        return False
    parameter_path.write_text(text, encoding="utf-8")
    snapshot_path.write_text(
        json.dumps({"sha256": digest, "source": "config"}, indent=2) + "\n",
        encoding="utf-8",
    )
    return True


class WorkspaceBootstrapper:
    LEGACY_INIT_FILES = ("sub_loop.py", "old_parameter.yaml", "bsub_script.py", "l2k2.mtp", "l2k3.mtp", "original.cfg", "original.mtp")
    STRUCTURE_FILE_SUFFIXES = {".vasp", ".cif", ".xyz", ".extxyz"}
    STRUCTURE_FILE_NAMES = {"POSCAR", "CONTCAR"}
    SELECTION_MODE_KEYS = ("mlp_encode_model",)
    PUBLIC_PARAMETER_RENAMES = {
        "ambiguity_histogram_max": "end",
        "ambiguity_histogram_bins": "num_elements",
    }
    MLP_ENCODE_MODEL_ONLY_KEYS = (
        "selection_budget_schedule",
        "coverage_threshold_schedule",
        "coverage_rate_method",
        "selection_budget_scope",
        "report_per_configuration_details",
        "candidate_trigger",
        "plateau_generations",
        "min_coverage_delta",
        "dimension_min_cover_workers",
    )
    REMOVED_PARAMETER_RENAMES = {
        "coverage_count_threshold": "state_population",
        "report_coverage_count_threshold_zero_baseline": "report_state_population_zero_baseline",
        "mean_descriptor_coverage_count_threshold": "mean_descriptor_state_population",
        "iw_method": "dq_width_method",
        "iw": "dq_width",
        "iw_scale": "dq_width_factor",
        "dynamic_iw": "dynamic_dq_width",
        "bw_method": "dq_width_method",
        "bw": "dq_width",
        "bw_coff": "dq_width_factor",
    }
    SUBMISSION_BACKENDS = {
        "bsub": {
            "task_submission_method": "bsub<bsub.lsf",
            "start_calc_command": "bsub<",
        },
        "sbatch": {
            "task_submission_method": "sbatch bsub.lsf",
            "start_calc_command": "sbatch",
        },
    }
    DEFAULT_PARAMETER_VALUES = {
        "mlp_nums": 3,
        "size": "(1, 1, 1)",
        "sort_ele": True,
        "nvt_lattice_scaling_factor": [1],
        "npt_max_cell_volume_filter_factor": 1.5,
        "mlp_encode_model": True,
        "encoding_cores": 2,
        "dimension_min_cover_workers": 4,
        "dq_width_method": "Freedman_Diaconis",
        "dq_width": 0.01,
        "dq_width_factor": 1.0,
        "body_list": ["two", "three"],
         "selection_budget_schedule": [12, 8, 5],
         "coverage_threshold_schedule": [99.5, 99.9, 99.95],
         "coverage_rate_method": "mean",
         "selection_budget_scope": "per_configuration",
         "candidate_trigger": 10,
         "state_population": 0,
         "report_state_population_zero_baseline": False,
         "plateau_generations": None,
         "min_coverage_delta": None,
        "report_per_configuration_details": True,
        "dft": {
            "calc_dir_num": 5,
            "force_threshold": 20,
            "pending_warning_hours": None,
        },
    }

    def __init__(self, config_path):
        self.config_path = Path(config_path).resolve()
        self.config_dir = self.config_path.parent
        self.config = self.normalize_config_layout(load_json_config(self.config_path))
        removed_top_level = sorted(
            key for key in (
                "coverage_pca",
                "das",
                "das_adaptive",
                "das_fixed",
                "fixed_divergence_threshold",
                "divergence_threshold",
            )
            if key in self.config
        )
        if removed_top_level:
            raise ValueError(
                "Removed sampling features are not supported in this SUS2-only "
                "release: " + ", ".join(removed_top_level)
            )
        training = dict(self.config.get("training") or {})
        forbidden_training = sorted(
            key for key in (
                "do-lin", "do-lin-rescale", "std_weight", "stdd_weight",
                "version_sus2", "mtp_type",
            ) if key in training
        )
        if forbidden_training:
            raise ValueError(
                "Legacy SUS2/MTP training flags are not supported: "
                + ", ".join(forbidden_training)
            )
        self.config["potential"] = self.normalize_potential_config(self.config.get("potential"))
        if self.config["potential"].get("model_path") in (BUILTIN_SUS2_87, LEGACY_BUILTIN_SUS2_87):
            self.config["potential"]["model_path"] = str(resolve_model_path(BUILTIN_SUS2_87))

    @classmethod
    def normalize_potential_config(cls, raw_potential):
        potential = dict(raw_potential or {})
        forbidden = sorted(
            key for key in ("family", "format", "version_sus2", "mtp_type")
            if key in potential
        )
        if forbidden:
            raise ValueError(
                "This release supports SUS2-SH only; remove legacy potential keys: "
                + ", ".join(forbidden)
            )
        model_yaml = potential.get("model_yaml")
        if not model_yaml:
            raise ValueError("potential.model_yaml is required for SUS2-SH")
        backend = str(potential.get("md_backend", "cpu")).strip().lower()
        if backend != "cpu":
            raise ValueError(
                "SUS2-SH CPU-only mode is active; potential.md_backend must be 'cpu'"
            )
        potential["md_backend"] = "cpu"
        potential["model_yaml"] = str(model_yaml)
        if potential.get("model_path") is not None:
            model_path = str(potential["model_path"])
            if model_path not in (BUILTIN_SUS2_87, LEGACY_BUILTIN_SUS2_87) and Path(model_path).suffix.lower() != ".sus2":
                raise ValueError(
                    "potential.model_path must point to a .sus2 model or builtin:sus2_87"
                )
            potential["model_path"] = model_path
        return potential

    @staticmethod
    def normalize_config_layout(config):
        normalized = dict(config)
        if isinstance(normalized.get("workflow"), dict):
            top_level_workflow = dict(normalized["workflow"])
            top_level_workflow.pop("mode", None)
            normalized["workflow"] = top_level_workflow
        init_dataset = normalized.pop("init_dataset", None)
        sampling = normalized.pop("sampling", None)

        if "parameter" in normalized:
            raise ValueError(
                "Top-level parameter is no longer supported in public configs; "
                "use sampling.structure_selection instead."
            )

        if init_dataset is None and sampling is None:
            return normalized

        if "training" in normalized:
            raise ValueError(
                "Top-level training is no longer supported in public sample configs; "
                "move it to sampling.scheduler.train."
            )
        if "potential" in normalized:
            raise ValueError(
                "Top-level potential is no longer supported in public sample configs; "
                "put model_yaml under sampling.scheduler.train."
            )

        dataset = dict(normalized.get("dataset", {}))
        workflow = dict(normalized.get("workflow", {}))
        parameter = {}
        scheduler = dict(normalized.get("scheduler", {}))
        training = dict(normalized.get("training", {}))
        dft = dict(normalized.get("dft", {}))

        if init_dataset is not None:
            dataset.update(dict(init_dataset))

        if sampling is not None:
            sampling = dict(sampling)
            if "parameter" in sampling:
                raise ValueError(
                    "sampling.parameter is no longer supported; "
                    "use sampling.structure_selection instead."
                )
            workflow.update(dict(sampling.get("workflow", {})))
            workflow.pop("mode", None)
            structure_selection = sampling.get("structure_selection")
            if structure_selection is None:
                raise ValueError("sampling.structure_selection is required.")
            grouped_scheduler = dict(sampling.get("scheduler") or {})
            scheduler, training, scheduler_dft = WorkspaceBootstrapper._normalize_grouped_scheduler(
                grouped_scheduler
            )
            dft.update(scheduler_dft)
            selection_for_normalization = copy.deepcopy(structure_selection)
            selection_dft = dict(selection_for_normalization.get("dft") or {})
            for key in ("calc_dir_num", "force_threshold", "pending_warning_hours"):
                if key in scheduler_dft:
                    selection_dft.setdefault(key, scheduler_dft[key])
            selection_for_normalization["dft"] = selection_dft
            parameter.update(WorkspaceBootstrapper._normalize_structure_selection(selection_for_normalization))
            if sampling.get("coverage_plot") is not None:
                merged_coverage_plot = dict(normalized.get("coverage_plot") or {})
                merged_coverage_plot.update(dict(sampling.get("coverage_plot") or {}))
                normalized["coverage_plot"] = merged_coverage_plot

        workflow["output_xyz"] = True
        workflow.pop("mode", None)
        dataset = WorkspaceBootstrapper._normalize_dataset_config(dataset)

        normalized["dataset"] = dataset
        normalized["workflow"] = workflow
        normalized["parameter"] = parameter
        normalized["scheduler"] = scheduler
        normalized["training"] = training
        normalized["potential"] = {
            "model_yaml": training["model_yaml"],
            "model_path": BUILTIN_SUS2_87,
            "md_backend": "cpu",
        }
        if "coverage_plot" not in normalized:
            normalized["coverage_plot"] = {
                "enabled": True,
                "body_list": ["two", "three"],
                "output_dir": "xyz_pca_coverage_results",
                "query_structures": ["all"],
                "loop_select": "middle-half",
                "pca_fit_source": "query",
                "coverage_mode": "2d",
                "coverage_grid": "last-loop",
                "descriptor_workers": 8,
                "coverage_workers": 8,
                "lammps_run_mode": "scheduler",
                "lammps_timeout_hours": 24,
                "show_input": False,
                "max_plot_points": 2000000,
                "force_query": False,
            }
        normalized["dft"] = dft
        return normalized

    @staticmethod
    def _normalize_grouped_scheduler(grouped_scheduler):
        grouped = dict(grouped_scheduler or {})
        train = dict(grouped.get("train") or {})
        lmp = dict(grouped.get("lmp") or {})
        scf = dict(grouped.get("scf") or {})
        missing = [name for name, value in (("train", train), ("lmp", lmp), ("scf", scf)) if not value]
        if missing:
            raise ValueError(
                "sampling.scheduler must contain train, lmp, and scf groups; "
                f"missing: {', '.join(missing)}"
            )
        model_yaml = train.get("model_yaml")
        if not model_yaml:
            raise ValueError("sampling.scheduler.train.model_yaml is required")

        training = {
            key: copy.deepcopy(value)
            for key, value in train.items()
            if key not in {
                "sus2_mlp_exe", "train_env", "original_command", "subsequent_command",
                "queue", "cores", "ptile", "model_yaml",
            }
        }
        training["model_yaml"] = str(model_yaml)
        training.setdefault("sampling", {
            "initial_iterations": DEFAULT_INITIAL_ITERATIONS,
            "subsequent_iterations": DEFAULT_SUBSEQUENT_ITERATIONS,
        })
        training.setdefault("final", {
            "enabled": True,
            "iterations": DEFAULT_FINAL_ITERATIONS,
            "model_name": "final.sus2",
        })

        scheduler = {
            "submission_backend": grouped.get("submission_backend", "bsub"),
            "sus2_mlp_exe": train.get("sus2_mlp_exe"),
            "train_env": train.get("train_env", ""),
            "original_command": train.get("original_command", ""),
            "subsequent_command": train.get("subsequent_command", ""),
            "train_sus_queue": train.get("queue", 33),
            "train_sus_cores": train.get("cores", 40),
            "train_sus_ptile": train.get("ptile", 40),
            "lmp_exe": lmp.get("lmp_exe"),
            "lmp_env": lmp.get("lmp_env", ""),
            "lmp_queue": lmp.get("queue", 33),
            "lmp_cores": lmp.get("cores", 40),
            "lmp_ptile": lmp.get("ptile", 40),
            "scf_queue": scf.get("queue", 33),
            "scf_cores": scf.get("cores", 40),
            "scf_ptile": scf.get("ptile", 40),
            "scf_cal_engine": scf.get("cal_engine", "vasp"),
            "dft_clean_dcbf_environment": scf.get(
                "clean_dcbf_environment", scf.get("dft_clean_dcbf_environment", True)
            ),
            "dft_env": scf.get("dft_env", ""),
            "dft_command": scf.get("dft_command", ""),
        }
        dft = {
            key: copy.deepcopy(scf[key])
            for key in ("template_dir", "calc_dir_num", "force_threshold", "pending_warning_hours")
            if key in scf
        }
        dft.setdefault("template_dir", "./init")
        dft.setdefault("calc_dir_num", 5)
        dft.setdefault("force_threshold", 20)
        dft.setdefault("pending_warning_hours", None)
        return scheduler, training, dft

    @classmethod
    def _reject_removed_parameter_keys(cls, label, mapping):
        removed = [key for key in cls.REMOVED_PARAMETER_RENAMES if key in (mapping or {})]
        if removed:
            replacements = ", ".join(
                f"{key} -> {cls.REMOVED_PARAMETER_RENAMES[key]}" for key in removed
            )
            raise ValueError(f"Removed configuration keys in {label}: {replacements}. Use the new field names only.")

    @classmethod
    def _normalize_structure_selection(cls, raw_selection):
        selection = dict(raw_selection or {})
        common = dict(selection.get("common") or {})
        legacy_coverage_mode = "coverage_calculation_mode" in common
        common.pop("coverage_calculation_mode", None)
        common.pop("dimension_min_cover_global_prune", None)
        dft = dict(selection.get("dft") or {})
        modes = {}
        for mode_name, raw_mode_config in dict(selection.get("modes") or {}).items():
            mode_config = dict(raw_mode_config or {})
            if "coverage_calculation_mode" in mode_config:
                legacy_coverage_mode = True
                mode_config.pop("coverage_calculation_mode", None)
            modes[mode_name] = mode_config
        if legacy_coverage_mode:
            warnings.warn(
                "coverage_calculation_mode is no longer used; coverage is always "
                "calculated per configuration.",
                RuntimeWarning,
            )
        cls._reject_removed_parameter_keys("sampling.structure_selection.common", common)
        misplaced_common_keys = [key for key in cls.MLP_ENCODE_MODEL_ONLY_KEYS if key in common]
        if misplaced_common_keys:
            raise ValueError(
                "These keys belong under sampling.structure_selection.modes.mlp_encode_model, "
                f"not sampling.structure_selection.common: {misplaced_common_keys}"
            )
        for mode_name, mode_cfg in modes.items():
            cls._reject_removed_parameter_keys(
                f"sampling.structure_selection.modes.{mode_name}",
                mode_cfg or {},
            )
        if not modes:
            raise ValueError("sampling.structure_selection.modes is required.")
        removed_modes = {
            name for name in ("das_adaptive", "das_fixed")
            if cls._coerce_bool((modes.get(name) or {}).get("enabled", False), default=False)
        }
        if removed_modes:
            raise ValueError("DAS selection modes were removed; enable mlp_encode_model only.")

        active_modes = []
        for mode_key in cls.SELECTION_MODE_KEYS:
            mode_cfg = modes.get(mode_key) or {}
            if cls._coerce_bool(mode_cfg.get("enabled", False), default=False):
                active_modes.append(mode_key)

        if len(active_modes) != 1:
            warnings.warn(
                "Exactly one structure_selection mode should be enabled. "
                "Defaulting to mlp_encode_model.",
                RuntimeWarning,
            )
            active_mode = "mlp_encode_model"
        else:
            active_mode = active_modes[0]

        mode_config = dict(modes.get(active_mode) or {})
        mode_config.pop("enabled", None)
        mode_config.pop("dimension_min_cover_global_prune", None)
        parameter = {}
        parameter.update(common)
        parameter.update(mode_config)
        parameter["dft"] = dft
        # Sampling keeps the established low-cost selector default: two and
        # three body outer descriptors.  Reduce keeps its explicit body_list
        # semantics, so users can still request four (or another metadata
        # body) there when needed.
        parameter.setdefault("body_list", ["two", "three"])

        parameter["mlp_encode_model"] = True

        for public_key, internal_key in cls.PUBLIC_PARAMETER_RENAMES.items():
            if public_key in parameter:
                parameter[internal_key] = parameter.pop(public_key)
        if "report_zero_count_baseline" in parameter:
            raise ValueError(
                "report_zero_count_baseline has been renamed to "
                "report_state_population_zero_baseline."
            )

        parameter["sort_ele"] = True
        return parameter

    @classmethod
    def _normalize_dataset_config(cls, dataset):
        normalized = dict(dataset or {})
        builder = dict(normalized.get("builder") or {})
        dataset_mode = normalized.get("dataset_mode")
        if dataset_mode is not None:
            builder["dataset_mode"] = dataset_mode
        elif "dataset_mode" in builder:
            normalized["dataset_mode"] = builder["dataset_mode"]
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
        normalized["builder"] = builder
        return normalized

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

    @staticmethod
    def resolve_path(base_dir, raw_path):
        path = Path(raw_path)
        if path.is_absolute():
            return path
        return (base_dir / path).resolve()

    @staticmethod
    def portable_path(path, base_dir):
        path = Path(path).resolve()
        base_dir = Path(base_dir).resolve()
        try:
            path.relative_to(base_dir.parent)
        except ValueError:
            return str(path)
        try:
            return os.path.relpath(path, base_dir)
        except ValueError:
            return str(path)

    @staticmethod
    def portable_init_model_path(path, init_dir):
        path = Path(path).resolve()
        init_dir = Path(init_dir).resolve()
        bundled_copy = init_dir / path.name
        if bundled_copy.is_file():
            try:
                if bundled_copy.read_bytes() == path.read_bytes():
                    return os.path.join("..", init_dir.name, path.name)
            except OSError:
                pass
        return WorkspaceBootstrapper.portable_path(path, init_dir)

    def prepare_workspace(self):
        workflow = self.config["workflow"]
        build_md_loop_schedule(workflow.get("main_loop_npt"), workflow.get("main_loop_nvt"))
        dataset = self.config["dataset"]
        run_dir = self.resolve_path(self.config_dir, self.config.get("run_dir", "."))
        init_source_dir = self.resolve_path(self.config_dir, dataset.get("init_source_dir", "init"))
        init_dir = run_dir / "init"
        init_dir.mkdir(parents=True, exist_ok=True)
        if init_source_dir.resolve() != init_dir.resolve():
            shutil.copytree(init_source_dir, init_dir, dirs_exist_ok=True)

        for legacy_name in self.LEGACY_INIT_FILES:
            legacy_path = init_dir / legacy_name
            if legacy_path.exists():
                legacy_path.unlink()

        self._copy_structure_source(run_dir, dataset)

        should_pause_after_build = False
        builder = InitialDatasetBuilder(run_dir, self.config, self.config_dir)
        if builder.is_enabled():
            build_result = builder.ensure_dataset()
            xyz_input = build_result.output_xyz
            if build_result.built_now and build_result.should_pause:
                should_pause_after_build = True
        else:
            xyz_input_raw = dataset.get("xyz_input")
            if not xyz_input_raw:
                raise ValueError("dataset.xyz_input is required when dataset.builder.enabled is false")
            xyz_input = self.resolve_path(self.config_dir, xyz_input_raw)

        elements = self.infer_elements_from_xyz(xyz_input)
        self.validate_initial_dataset_elements(dataset, xyz_input, elements)
        self.refresh_runtime_config(run_dir, elements, xyz_input)
        return run_dir, xyz_input, elements, "sus2", should_pause_after_build

    def refresh_runtime_config(self, run_dir, elements, xyz_input):
        """Rewrite init/parameter.yaml and dcbf.runtime.json from this config.

        Used for hot reload while a workflow is running.  Dataset construction
        and structure copying are intentionally not repeated here; the caller
        must keep the same run_dir/xyz_input.
        """
        run_dir = Path(run_dir).resolve()
        init_dir = run_dir / "init"
        init_dir.mkdir(parents=True, exist_ok=True)
        self.write_parameter_yaml(init_dir, elements, xyz_input)
        runtime_config = dict(self.config)
        runtime_dataset = dict(runtime_config.get("dataset", {}))
        runtime_dataset["xyz_input"] = self.portable_path(xyz_input, run_dir)
        runtime_config["dataset"] = runtime_dataset
        runtime_parameter = dict(runtime_config.get("parameter", {}))
        runtime_parameter["ele"] = list(elements)
        model_yaml_path = self.resolve_path(self.config_dir, self.config["potential"]["model_yaml"])
        runtime_parameter["model_yaml"] = self.portable_path(model_yaml_path, run_dir)
        runtime_parameter["model_path"] = (
            self.portable_path(self.resolve_path(self.config_dir, self.config["potential"]["model_path"]), run_dir)
            if self.config["potential"].get("model_path") is not None else None
        )
        runtime_parameter["md_backend"] = "cpu"
        runtime_config["parameter"] = runtime_parameter
        runtime_config["scheduler"] = self.normalize_scheduler_keys(dict(self.config["scheduler"]))
        save_runtime_config(run_dir, runtime_config)
        return runtime_config

    def _copy_structure_source(self, run_dir, dataset):
        structure_source_raw = dataset.get("structure_source_dir", "stru")
        structure_source_dir = self.resolve_path(self.config_dir, structure_source_raw)
        structure_target_dir = run_dir / "stru"
        if structure_source_dir.exists():
            structure_target_dir.mkdir(parents=True, exist_ok=True)
            if structure_source_dir.resolve() != structure_target_dir.resolve():
                shutil.copytree(structure_source_dir, structure_target_dir, dirs_exist_ok=True)
            return
        if not structure_target_dir.exists():
            raise FileNotFoundError(
                f"Could not find dataset.structure_source_dir={structure_source_dir} and run_dir/stru does not exist"
            )

    @staticmethod
    def infer_elements_from_xyz(xyz_input):
        element_set = set()
        for atoms in iread(str(xyz_input)):
            element_set.update(atoms.get_chemical_symbols())
        ordered_atomic_numbers = sorted(atomic_numbers[element] for element in element_set)
        return [chemical_symbols[number] for number in ordered_atomic_numbers]

    def validate_initial_dataset_elements(self, dataset, xyz_input, dataset_elements):
        structure_elements_by_file = self._collect_sampling_structure_elements(dataset)
        if not structure_elements_by_file:
            return

        sampling_elements = set()
        for elements in structure_elements_by_file.values():
            sampling_elements.update(elements)

        dataset_element_set = set(dataset_elements)
        missing = sorted(sampling_elements - dataset_element_set, key=lambda item: atomic_numbers[item])
        if not missing:
            return

        missing_set = set(missing)
        offending = [
            str(path)
            for path, elements in sorted(structure_elements_by_file.items(), key=lambda item: str(item[0]))
            if elements & missing_set
        ]
        sampling_sorted = sorted(sampling_elements, key=lambda item: atomic_numbers[item])
        raise ValueError(
            "Initial dataset element mismatch:\n"
            f"dataset xyz = {xyz_input}\n"
            f"dataset elements = {dataset_elements}\n"
            f"sampling structure elements = {sampling_sorted}\n"
            f"missing in initial dataset = {missing}\n"
            f"offending structures = {offending}"
        )

    def _collect_sampling_structure_elements(self, dataset):
        structure_source_raw = dataset.get("structure_source_dir", "stru")
        structure_source_dir = self.resolve_path(self.config_dir, structure_source_raw)
        if not structure_source_dir.exists():
            run_dir = self.resolve_path(self.config_dir, self.config.get("run_dir", "."))
            structure_source_dir = run_dir / "stru"
        if not structure_source_dir.exists():
            return {}

        elements_by_file = {}
        for path in sorted(structure_source_dir.rglob("*")):
            if not path.is_file():
                continue
            if path.name not in self.STRUCTURE_FILE_NAMES and path.suffix.lower() not in self.STRUCTURE_FILE_SUFFIXES:
                continue
            read_kwargs = {"index": ":"}
            if path.name in self.STRUCTURE_FILE_NAMES:
                read_kwargs["format"] = "vasp"
            try:
                frames = list(iread(str(path), **read_kwargs))
            except Exception as exc:
                raise ValueError(f"Could not read sampling structure file for element check: {path}") from exc
            file_elements = set()
            for atoms in frames:
                file_elements.update(atoms.get_chemical_symbols())
            if file_elements:
                elements_by_file[path] = file_elements
        return elements_by_file

    def write_parameter_yaml(self, init_dir, elements, xyz_input):
        raw_parameter = dict(self.config["parameter"])
        legacy_keys = [
            key
            for key in (
                "ele_model",
                "bw_method",
                "bw",
                "bw_coff",
                "iw_method",
                "iw",
                "iw_scale",
                "dynamic_iw",
                "stru_num",
                "coverage_rate_threshold",
            )
            if key in raw_parameter
        ]
        if legacy_keys:
            raise ValueError(
                "Legacy parameter keys are no longer supported: "
                f"{legacy_keys}. Use sort_ele / selection_budget_schedule / coverage_threshold_schedule / "
                "dq_width_method / dq_width / dq_width_factor instead."
            )
        parameter = self.apply_parameter_defaults(raw_parameter)
        parameter.pop("dimension_min_cover_global_prune", None)
        parameter.pop("init_threshold", None)
        parameter.pop("threshold_coff", None)
        parameter["ele"] = elements
        parameter["sort_ele"] = True
        parameter["encoding_cores"] = int(parameter.get("encoding_cores", 2))
        raw_min_cover_workers = parameter.get("dimension_min_cover_workers", 4)
        if isinstance(raw_min_cover_workers, bool):
            raise ValueError("dimension_min_cover_workers must be an integer")
        try:
            parameter["dimension_min_cover_workers"] = int(raw_min_cover_workers)
        except (TypeError, ValueError) as exc:
            raise ValueError("dimension_min_cover_workers must be an integer") from exc
        if (
            isinstance(raw_min_cover_workers, float)
            and not raw_min_cover_workers.is_integer()
        ):
            raise ValueError("dimension_min_cover_workers must be an integer")
        if parameter["dimension_min_cover_workers"] < -1:
            raise ValueError(
                "dimension_min_cover_workers must be -1, 0, or a positive integer"
            )
        parameter["selection_budget_scope"] = normalize_selection_budget_scope(
            parameter.get("selection_budget_scope")
        )
        parameter["selection_budget_schedule"], _ = validate_selection_schedules(
            parameter["selection_budget_schedule"],
            parameter["coverage_threshold_schedule"],
            len(elements),
        )
        parameter["dq_width"] = float(parameter.get("dq_width", 0.01))
        parameter["dq_width_factor"] = float(parameter.get("dq_width_factor", 1.0))
        run_dir = Path(init_dir).resolve().parent
        parameter["dataset_xyz_input"] = self.portable_path(xyz_input, run_dir)
        potential = self.config["potential"]
        from .sus2_training import load_model_yaml
        model_yaml_path = self.resolve_path(self.config_dir, potential["model_yaml"])
        load_model_yaml(model_yaml_path)
        parameter["model_yaml"] = str(model_yaml_path)
        parameter["model_path"] = (
            str(self.resolve_path(self.config_dir, potential["model_path"]))
            if potential.get("model_path") is not None else None
        )
        parameter["md_backend"] = "cpu"
        training = dict(self.config.get("training") or {})
        sampling = dict(training.get("sampling") or {})
        final = dict(training.get("final") or {})
        initial_iterations = int(sampling.get("initial_iterations", DEFAULT_INITIAL_ITERATIONS))
        subsequent_iterations = int(sampling.get("subsequent_iterations", DEFAULT_SUBSEQUENT_ITERATIONS))
        final_iterations = int(final.get("iterations", DEFAULT_FINAL_ITERATIONS))
        if min(initial_iterations, subsequent_iterations, final_iterations) <= 0:
            raise ValueError("SUS2 training iterations must be positive")
        final_name = str(final.get("model_name", "final.sus2"))
        if Path(final_name).suffix.lower() != ".sus2":
            raise ValueError("training.final.model_name must end with .sus2")
        final_model_yaml = final.get("model_yaml") or training["model_yaml"]
        final_model_yaml_path = self.resolve_path(self.config_dir, final_model_yaml)
        load_model_yaml(final_model_yaml_path)
        if "mode" in final:
            raise ValueError(
                "training.final.mode is no longer supported; "
                "the final model_yaml determines continuation or fresh initialization"
            )
        parameter["sus2_training"] = {
            "initial_iterations": initial_iterations,
            "subsequent_iterations": subsequent_iterations,
            "final": {
                "enabled": bool(final.get("enabled", True)),
                "iterations": final_iterations,
                "model_name": final_name,
                "model_yaml": str(final_model_yaml_path),
            },
        }
        _write_parameter_if_changed(init_dir, parameter)
        return "sus2"

    @classmethod
    def normalize_parameter_keys(cls, parameter):
        normalized = dict(parameter)
        if "coverage_calculation_mode" in normalized:
            warnings.warn(
                "coverage_calculation_mode is no longer used; coverage is always "
                "calculated per configuration.",
                RuntimeWarning,
            )
            normalized.pop("coverage_calculation_mode", None)
        if "ele_model" in normalized and "sort_ele" not in normalized:
            normalized["sort_ele"] = int(normalized["ele_model"]) == 1
        if "stru_num" in normalized and "selection_budget_schedule" not in normalized:
            normalized["selection_budget_schedule"] = normalized["stru_num"]
        if "coverage_rate_threshold" in normalized and "coverage_threshold_schedule" not in normalized:
            normalized["coverage_threshold_schedule"] = normalized["coverage_rate_threshold"]
        return normalized

    @classmethod
    def apply_parameter_defaults(cls, parameter):
        normalized = cls.normalize_parameter_keys(dict(parameter))
        cls._reject_removed_parameter_keys("parameter", normalized)
        forbidden = sorted(
            key for key in (
                "version_sus2", "mtp_type", "mean_descriptor_enabled",
                "mean_descriptor_state_population",
                    ) if key in normalized
        )
        if forbidden:
            raise ValueError(
                "Legacy MTP/mean-descriptor configuration is not supported: "
                + ", ".join(forbidden)
            )
        for key, default_value in cls.DEFAULT_PARAMETER_VALUES.items():
            if isinstance(default_value, dict):
                merged = copy.deepcopy(default_value)
                merged.update(normalized.get(key) or {})
                normalized[key] = merged
            elif key not in normalized:
                normalized[key] = copy.deepcopy(default_value)
        normalized["npt_max_cell_volume_filter_factor"] = (
            normalize_npt_max_cell_volume_filter_factor(
                normalized.get("npt_max_cell_volume_filter_factor")
            )
        )
        return normalized

    @classmethod
    def normalize_scheduler_keys(cls, scheduler):
        return normalize_scheduler_config(scheduler)

    @staticmethod
    def _count_xyz_frames(path):
        path = Path(path)
        if not path.exists() or path.stat().st_size == 0:
            return 0
        return sum(1 for _ in iread(str(path), index=":"))

    @staticmethod
    def _numeric_child_dirs(parent, prefix):
        children = []
        parent = Path(parent)
        if not parent.exists():
            return children
        for path in parent.iterdir():
            if not path.is_dir() or not path.name.startswith(prefix):
                continue
            try:
                index = int(path.name.replace(prefix, "", 1))
            except ValueError:
                continue
            children.append((index, path))
        return sorted(children, key=lambda item: item[0])

    @staticmethod
    def _resolve_dataset_xyz_input(run_dir, yaml_data):
        raw_path = yaml_data.get("dataset_xyz_input")
        if not raw_path:
            return None
        path = Path(raw_path)
        if path.is_absolute():
            return path
        return Path(run_dir) / path

    @staticmethod
    def _warn_final_xyz_annotation(run_dir, message):
        line = f"{time.strftime('%Y-%m-%d %H:%M')} WARNING: [workflow.final_xyz] {message}"
        print(line)
        app_log = Path(run_dir) / "app.log"
        try:
            app_log.parent.mkdir(parents=True, exist_ok=True)
            with open(app_log, "a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        except OSError as exc:
            warnings.warn(f"Could not append final xyz warning to {app_log}: {exc}")

    @staticmethod
    def _final_xyz_main_segments(run_dir, yaml_data):
        run_dir = Path(run_dir)
        segments = []

        initial_dataset = WorkspaceBootstrapper._resolve_dataset_xyz_input(run_dir, yaml_data)
        if initial_dataset is not None:
            initial_count = WorkspaceBootstrapper._count_xyz_frames(initial_dataset)
            if initial_count > 0:
                segments.append((-1, initial_count, initial_dataset))
            else:
                WorkspaceBootstrapper._warn_final_xyz_annotation(
                    run_dir,
                    f"Initial dataset has no readable frames, so main=-1 was not assigned from {initial_dataset}",
                )
        else:
            WorkspaceBootstrapper._warn_final_xyz_annotation(
                run_dir,
                "parameter.yaml does not contain dataset_xyz_input; initial frames cannot be assigned main=-1",
            )

        for main_index, main_dir in WorkspaceBootstrapper._numeric_child_dirs(run_dir, "main_"):
            main_count = 0
            for _, gen_dir in WorkspaceBootstrapper._numeric_child_dirs(main_dir, "gen_"):
                main_count += WorkspaceBootstrapper._count_xyz_frames(gen_dir / DFT_WORK_DIR / "scf_filter.xyz")
            if main_count > 0:
                segments.append((main_index, main_count, main_dir))
        return segments

    @staticmethod
    def _annotate_final_xyz_main_labels(run_dir, output_path, config, yaml_data):
        output_path = Path(output_path)
        atoms = list(iread(str(output_path), index=":"))
        dataset_config = dict(config.get("dataset") or {})
        label = dataset_config.get("all_label")

        segments = WorkspaceBootstrapper._final_xyz_main_segments(run_dir, yaml_data)
        expected_total = sum(count for _, count, _ in segments)
        cursor = 0
        assigned = 0
        actual_total = len(atoms)
        if actual_total == 0:
            WorkspaceBootstrapper._warn_final_xyz_annotation(run_dir, f"Final xyz has no readable frames: {output_path}")
            return

        for main_value, frame_count, source_path in segments:
            if cursor >= actual_total:
                WorkspaceBootstrapper._warn_final_xyz_annotation(
                    run_dir,
                    f"No final frames remain for main={main_value}; expected {frame_count} frames from {source_path}",
                )
                continue
            end = min(cursor + frame_count, actual_total)
            for atom in atoms[cursor:end]:
                if main_value == -1 and not label:
                    atom.info.pop("label", None)
                atom.info["main"] = main_value
            assigned += end - cursor
            if end - cursor != frame_count:
                WorkspaceBootstrapper._warn_final_xyz_annotation(
                    run_dir,
                    f"Only assigned {end - cursor}/{frame_count} frames for main={main_value} from {source_path}",
                )
            cursor = end

        if label:
            for atom in atoms:
                atom.info["label"] = label

        if expected_total != actual_total:
            unlabeled = sum(1 for atom in atoms if "main" not in atom.info)
            WorkspaceBootstrapper._warn_final_xyz_annotation(
                run_dir,
                "Frame count mismatch while assigning main labels: "
                f"expected={expected_total}, actual={actual_total}, assigned={assigned}, unlabeled={unlabeled}. "
                "Matched frames were written; unmatched frames were left unchanged.",
            )

        write(str(output_path), atoms, format="extxyz")

    @staticmethod
    def export_final_xyz(run_dir, config):
        workflow = config["workflow"]
        if not workflow.get("output_xyz", True):
            return

        output_name = workflow.get("output_xyz_name", "sampled_training_dataset.xyz")
        output_path = Path(run_dir) / output_name
        from .training_dataset import TrainingDatasetStore

        TrainingDatasetStore(run_dir, config).export_final_xyz(output_path)
