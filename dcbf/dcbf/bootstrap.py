from __future__ import annotations

import copy
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
from .das.file_conversion import cfg2xyz
from .mtp import normalize_mtp_type
from .path_names import DFT_WORK_DIR
from .runtime_config import load_json_config, normalize_scheduler_config, save_runtime_config


class WorkspaceBootstrapper:
    LEGACY_INIT_FILES = ("sub_loop.py", "old_parameter.yaml", "bsub_script.py", "l2k2.mtp", "l2k3.mtp", "original.cfg", "original.mtp")
    STRUCTURE_FILE_SUFFIXES = {".vasp", ".cif", ".xyz", ".extxyz"}
    STRUCTURE_FILE_NAMES = {"POSCAR", "CONTCAR"}
    SELECTION_MODE_KEYS = ("mlp_encode_model", "das_adaptive", "das_fixed")
    PUBLIC_PARAMETER_RENAMES = {
        "ambiguity_histogram_max": "end",
        "ambiguity_histogram_bins": "num_elements",
    }
    MLP_ENCODE_MODEL_ONLY_KEYS = (
        "selection_budget_schedule",
        "coverage_threshold_schedule",
        "coverage_rate_method",
        "coverage_calculation_mode",
        "report_per_configuration_details",
        "candidate_trigger",
        "plateau_generations",
        "min_coverage_delta",
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
        "das_ambiguity": True,
        "af_default": 0.01,
        "af_limit": 0.2,
        "af_failed": 0.5,
        "over_fitting_factor": 1.1,
        "af_adaptive": None,
        "threshold_low": 0.08,
        "threshold_high": 0.3,
        "select_stru_num": None,
        "end": 1,
        "num_elements": 6,
        "sample": {
            "n": 5,
            "cluster_threshold_init": 0.5,
            "k": 2,
            "clustering_by_ambiguity": True,
        },
        "mlp_encode_model": True,
        "encoding_cores": 2,
        "dq_width_method": "Freedman_Diaconis",
        "dq_width": 0.01,
        "dq_width_factor": 1.0,
        "body_list": ["two", "three"],
        "mtp_type": "l2k2",
         "selection_budget_schedule": [20, 15, 10],
         "coverage_threshold_schedule": [99.5, 99.9, 99.95],
         "coverage_rate_method": "mean",
         "candidate_trigger": 10,
         "state_population": 0,
         "report_state_population_zero_baseline": False,
         "mean_descriptor_enabled": False,
         "mean_descriptor_state_population": 0,
         "plateau_generations": None,
         "min_coverage_delta": None,
         "coverage_calculation_mode": "per_configuration",
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

    @staticmethod
    def normalize_config_layout(config):
        normalized = dict(config)
        init_dataset = normalized.pop("init_dataset", None)
        sampling = normalized.pop("sampling", None)

        if "parameter" in normalized:
            raise ValueError(
                "Top-level parameter is no longer supported in public configs; "
                "use sampling.structure_selection instead."
            )

        if init_dataset is None and sampling is None:
            return normalized

        dataset = dict(normalized.get("dataset", {}))
        workflow = dict(normalized.get("workflow", {}))
        parameter = {}
        scheduler = dict(normalized.get("scheduler", {}))

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
            structure_selection = sampling.get("structure_selection")
            if structure_selection is None:
                raise ValueError("sampling.structure_selection is required.")
            parameter.update(WorkspaceBootstrapper._normalize_structure_selection(structure_selection))
            scheduler.update(dict(sampling.get("scheduler", {})))
            coverage_plot = sampling.get("coverage_plot")
            if coverage_plot is not None:
                merged_coverage_plot = dict(normalized.get("coverage_plot") or {})
                merged_coverage_plot.update(dict(coverage_plot))
                normalized["coverage_plot"] = merged_coverage_plot

        workflow["output_xyz"] = True
        dataset = WorkspaceBootstrapper._normalize_dataset_config(dataset)

        normalized["dataset"] = dataset
        normalized["workflow"] = workflow
        normalized["parameter"] = parameter
        normalized["scheduler"] = scheduler
        return normalized

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
        dft = dict(selection.get("dft") or {})
        modes = dict(selection.get("modes") or {})
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
        parameter = {}
        parameter.update(common)
        parameter.update(mode_config)
        parameter["dft"] = dft

        if active_mode == "mlp_encode_model":
            parameter["mlp_encode_model"] = True
            parameter["das_ambiguity"] = False
        elif active_mode == "das_adaptive":
            parameter["mlp_encode_model"] = False
            parameter["das_ambiguity"] = True
        else:
            parameter["mlp_encode_model"] = False
            parameter["das_ambiguity"] = False

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

    def prepare_workspace(self):
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
        mtp_type = self.write_parameter_yaml(init_dir, elements, xyz_input)
        runtime_config = dict(self.config)
        runtime_dataset = dict(runtime_config.get("dataset", {}))
        runtime_dataset["xyz_input"] = str(xyz_input)
        runtime_config["dataset"] = runtime_dataset
        runtime_config["scheduler"] = self.normalize_scheduler_keys(dict(self.config["scheduler"]))
        save_runtime_config(run_dir, runtime_config)
        return run_dir, xyz_input, elements, mtp_type, should_pause_after_build

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
        parameter.pop("init_threshold", None)
        parameter.pop("threshold_coff", None)
        parameter["ele"] = elements
        parameter["sort_ele"] = True
        parameter["encoding_cores"] = int(parameter.get("encoding_cores", 2))
        parameter["dq_width"] = float(parameter.get("dq_width", 0.01))
        parameter["dq_width_factor"] = float(parameter.get("dq_width_factor", 1.0))
        parameter["dataset_xyz_input"] = str(xyz_input)
        parameter["mtp_type"] = normalize_mtp_type(parameter.get("mtp_type", "l2k2"))
        mtp_type = parameter["mtp_type"]
        with open(init_dir / "parameter.yaml", "w", encoding="utf-8") as handle:
            yaml.safe_dump(parameter, handle, default_flow_style=False, sort_keys=False)
        return mtp_type

    @classmethod
    def normalize_parameter_keys(cls, parameter):
        normalized = dict(parameter)
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
        for key, default_value in cls.DEFAULT_PARAMETER_VALUES.items():
            if isinstance(default_value, dict):
                merged = copy.deepcopy(default_value)
                merged.update(normalized.get(key) or {})
                normalized[key] = merged
            elif key not in normalized:
                normalized[key] = copy.deepcopy(default_value)
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
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} WARNING: [workflow.final_xyz] {message}"
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

        main_list = [item for item in os.listdir(run_dir) if item.startswith("main_")]
        if not main_list:
            return
        main_list = sorted(main_list, key=lambda item: int(item.replace("main_", "")))
        last_main = main_list[-1]

        gen_dir = Path(run_dir) / last_main
        gen_list = [item for item in os.listdir(gen_dir) if item.startswith("gen_")]
        gen_list = sorted(gen_list, key=lambda item: int(item.replace("gen_", "")))
        last_gen = gen_list[-1]

        output_name = workflow.get("output_xyz_name", "all_sample_data.xyz")
        output_path = Path(run_dir) / output_name
        if output_path.exists():
            output_path.unlink()

        parameter_path = Path(run_dir) / "init" / "parameter.yaml"
        with open(parameter_path, "r", encoding="utf-8") as handle:
            yaml_data = yaml.safe_load(handle)
        ele = yaml_data["ele"]
        sort_ele = WorkspaceBootstrapper._coerce_bool(yaml_data.get("sort_ele", True), default=True)

        train_cfg = Path(run_dir) / last_main / last_gen / "train_mlp" / "train.cfg"
        cfg2xyz(ele, sort_ele, str(train_cfg), str(output_path))
        WorkspaceBootstrapper._annotate_final_xyz_main_labels(run_dir, output_path, config, yaml_data)
