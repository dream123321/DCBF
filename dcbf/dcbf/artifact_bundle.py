from __future__ import annotations

from collections import Counter
import json
import shutil
from pathlib import Path

from ase.io import iread, write

from .core_hours import write_core_hours_report
from .path_names import SUS2_MODEL_DIR


class ArtifactBundler:
    DEFAULTS = {
        "enabled": False,
        "output_dir": "summary_bundle",
    }
    DATASET_EXPORTS = {
        "initial_dataset": {
            "filename": "initial_dataset.xyz",
            "description": "Force-threshold-filtered initial dataset used as the workflow input dataset.",
        },
        "initial_dataset_raw": {
            "filename": "initial_dataset_raw.xyz",
            "description": "Original SCF-collected initial dataset before force-threshold filtering.",
        },
        "all_dataset": {
            "filename": "all.xyz",
            "description": "Final exported dataset containing the full workflow result.",
        },
        "dcbf_sampling": {
            "filename": "dcbf_sampling.xyz",
            "description": "Sampling-only dataset derived as all.xyz minus exact matches from initial_dataset.xyz.",
        },
    }

    def __init__(self, config: dict, run_dir: Path, config_path: Path):
        self.config = dict(config)
        self.run_dir = Path(run_dir).resolve()
        self.config_path = Path(config_path).resolve()
        summary = dict(self.config.get("summary") or {})
        merged = dict(self.DEFAULTS)
        merged.update(summary)
        self.summary = merged
        self.bundle_dir = self._resolve_output_dir(self.summary["output_dir"])
        self.manifest = {
            "bundle_dir": str(self.bundle_dir),
            "copied": {},
            "missing": {},
            "datasets": {},
        }

    def is_enabled(self):
        return bool(self.summary.get("enabled", False))

    def collect(self):
        if not self.is_enabled():
            return None

        self.bundle_dir.mkdir(parents=True, exist_ok=True)
        self._collect_datasets()
        self._collect_reports_and_logs()
        self._collect_source_directories()
        self._collect_models_and_analysis()
        manifest_path = self.bundle_dir / "manifest.json"
        manifest_path.write_text(json.dumps(self.manifest, indent=2), encoding="utf-8")
        return manifest_path

    def _resolve_output_dir(self, raw_path):
        path = Path(raw_path)
        if path.is_absolute():
            return path
        return (self.run_dir / path).resolve()

    def _copy_file(self, source: Path, destination: Path, manifest_key: str):
        source = Path(source)
        if not source.exists():
            self.manifest["missing"][manifest_key] = str(source)
            return
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        self.manifest["copied"][manifest_key] = str(destination)

    def _copy_xyz_with_main_label(
        self,
        source: Path,
        destination: Path,
        manifest_key: str,
        main_value: int,
        *,
        replace_label: bool = False,
    ):
        source = Path(source)
        if not source.exists():
            self.manifest["missing"][manifest_key] = str(source)
            return
        atoms_list = list(iread(str(source), index=":"))
        for atoms in atoms_list:
            if replace_label:
                atoms.info.pop("label", None)
            atoms.info["main"] = main_value
        self._write_xyz(destination, atoms_list)
        self.manifest["copied"][manifest_key] = str(destination)

    def _copy_tree(self, source: Path, destination: Path, manifest_key: str):
        source = Path(source)
        if not source.exists():
            self.manifest["missing"][manifest_key] = str(source)
            return
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, destination, dirs_exist_ok=True)
        self.manifest["copied"][manifest_key] = str(destination)

    def _dataset_config(self):
        return dict(self.config.get("dataset", {}))

    def _workflow_config(self):
        return dict(self.config.get("workflow", {}))

    def _training_config(self):
        return dict(self.config.get("training", {}))

    def _builder_output_path(self):
        builder = dict(self._dataset_config().get("builder") or {})
        raw = builder.get("output_xyz")
        if not raw:
            return None
        path = Path(raw)
        if path.is_absolute():
            return path
        return (self.run_dir / path).resolve()

    def _builder_original_output_path(self):
        output_path = self._builder_output_path()
        if output_path is None:
            return None
        return output_path.parent / "ori_init_dataset.xyz"

    def _sampling_output_path(self):
        workflow = self._workflow_config()
        output_name = workflow.get("output_xyz_name", "all_sample_data.xyz")
        return self.run_dir / output_name

    def _prune_dataset_dir(self, datasets_dir: Path):
        datasets_dir.mkdir(parents=True, exist_ok=True)
        allowed_names = {spec["filename"] for spec in self.DATASET_EXPORTS.values()}
        for path in datasets_dir.glob("*.xyz"):
            if path.name not in allowed_names:
                path.unlink()

    def _record_dataset(
        self,
        dataset_key: str,
        dataset_path: Path,
        *,
        match_mode: str | None = None,
        derived_from=None,
        main_labels=None,
    ):
        spec = self.DATASET_EXPORTS[dataset_key]
        record = {
            "path": str(dataset_path),
            "description": spec["description"],
        }
        if match_mode is not None:
            record["match_mode"] = match_mode
        if derived_from is not None:
            record["derived_from"] = list(derived_from)
        if main_labels is not None:
            record["main_labels"] = main_labels
        self.manifest["datasets"][dataset_key] = record

    def _collect_datasets(self):
        datasets_dir = self.bundle_dir / "datasets"
        self._prune_dataset_dir(datasets_dir)
        initial_dataset = self._builder_output_path()
        if initial_dataset is None:
            xyz_input = self._dataset_config().get("xyz_input")
            if xyz_input:
                initial_dataset = (self.run_dir / xyz_input).resolve() if not Path(xyz_input).is_absolute() else Path(xyz_input)
        if initial_dataset is not None:
            initial_dataset_bundle_path = datasets_dir / self.DATASET_EXPORTS["initial_dataset"]["filename"]
            self._copy_xyz_with_main_label(
                initial_dataset,
                initial_dataset_bundle_path,
                "datasets.initial_dataset",
                -1,
                replace_label=True,
            )
            if Path(initial_dataset).exists():
                self._record_dataset("initial_dataset", initial_dataset_bundle_path, main_labels=[-1])

        original_initial = self._builder_original_output_path()
        if original_initial is not None:
            original_initial_bundle_path = datasets_dir / self.DATASET_EXPORTS["initial_dataset_raw"]["filename"]
            self._copy_xyz_with_main_label(
                original_initial,
                original_initial_bundle_path,
                "datasets.initial_dataset_raw",
                -1,
                replace_label=True,
            )
            if Path(original_initial).exists():
                self._record_dataset("initial_dataset_raw", original_initial_bundle_path, main_labels=[-1])

        all_dataset = self._sampling_output_path()
        all_dataset_bundle_path = datasets_dir / self.DATASET_EXPORTS["all_dataset"]["filename"]
        self._copy_file(all_dataset, all_dataset_bundle_path, "datasets.all_dataset")
        if all_dataset.exists():
            self._record_dataset("all_dataset", all_dataset_bundle_path, main_labels="preserved_from_sampling_output")

        if initial_dataset is not None and Path(initial_dataset).exists() and all_dataset.exists():
            sampling_only_atoms = self._subtract_initial_dataset_strict(all_dataset, initial_dataset)
            sampling_only_path = datasets_dir / self.DATASET_EXPORTS["dcbf_sampling"]["filename"]
            self._write_xyz(sampling_only_path, sampling_only_atoms)
            self.manifest["copied"]["datasets.dcbf_sampling"] = str(sampling_only_path)
            self._record_dataset(
                "dcbf_sampling",
                sampling_only_path,
                match_mode="strict_exact_geometry",
                derived_from=[
                    self.DATASET_EXPORTS["all_dataset"]["filename"],
                    self.DATASET_EXPORTS["initial_dataset"]["filename"],
                ],
                main_labels="preserved_from_all_dataset",
            )

    @staticmethod
    def _strict_fingerprint_atoms(atoms):
        return (
            tuple(atoms.get_chemical_symbols()),
            tuple(float(value) for value in atoms.get_cell().array.reshape(-1)),
            tuple(float(value) for value in atoms.get_positions().reshape(-1)),
        )

    def _subtract_initial_dataset_strict(self, all_dataset_path: Path, initial_dataset_path: Path):
        all_atoms = list(iread(str(all_dataset_path), index=":"))
        initial_atoms = list(iread(str(initial_dataset_path), index=":"))
        initial_counter = Counter(self._strict_fingerprint_atoms(atoms) for atoms in initial_atoms)
        sampling_only_atoms = []
        for atoms in all_atoms:
            fingerprint = self._strict_fingerprint_atoms(atoms)
            if initial_counter[fingerprint] > 0:
                initial_counter[fingerprint] -= 1
            else:
                sampling_only_atoms.append(atoms)
        return sampling_only_atoms

    @staticmethod
    def _write_xyz(path: Path, atoms_list):
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            path.unlink()
        for atoms in atoms_list:
            write(str(path), atoms, format="extxyz", append=True)

    def _collect_reports_and_logs(self):
        reports_dir = self.bundle_dir / "reports"
        logs_dir = self.bundle_dir / "logs"
        self._copy_file(self.run_dir / "dcbf.runtime.json", reports_dir / "dcbf.runtime.json", "reports.runtime")
        self._copy_file(self.run_dir / "app.log", logs_dir / "app.log", "logs.app")

        builder = dict(self._dataset_config().get("builder") or {})
        report_path = builder.get("report_path")
        if report_path:
            report_source = (self.run_dir / report_path).resolve() if not Path(report_path).is_absolute() else Path(report_path)
            self._copy_file(report_source, reports_dir / "init_dataset_build_report.json", "reports.init_dataset_build")

        init_build_log = self.run_dir / "init_dataset_build" / "app.log"
        self._copy_file(init_build_log, logs_dir / "init_dataset_build.log", "logs.init_dataset_build")

        training_cfg = self._training_config()
        training_root = self.run_dir / training_cfg.get("work_dir", "high_precision_training")
        self._copy_file(training_root / "training_report.json", reports_dir / "training_report.json", "reports.training")
        self._copy_file(training_root / "training.log", logs_dir / "training.log", "logs.training")
        core_hours_path = write_core_hours_report(self.run_dir, self.config, reports_dir / "core_hours.txt")
        self.manifest["copied"]["reports.core_hours"] = str(core_hours_path)

    def _collect_source_directories(self):
        sources_dir = self.bundle_dir / "sources"
        self._copy_tree(self.run_dir / "init", sources_dir / "init", "sources.init")
        self._copy_tree(self.run_dir / "stru", sources_dir / "stru", "sources.stru")

    def _find_last_generation_dir(self):
        main_dirs = [path for path in self.run_dir.iterdir() if path.is_dir() and path.name.startswith("main_")]
        if not main_dirs:
            return None
        main_dirs.sort(key=lambda path: int(path.name.replace("main_", "")))
        last_main = main_dirs[-1]
        gen_dirs = [path for path in last_main.iterdir() if path.is_dir() and path.name.startswith("gen_")]
        if not gen_dirs:
            return None
        gen_dirs.sort(key=lambda path: int(path.name.replace("gen_", "")))
        return gen_dirs[-1]

    def _collect_models_and_analysis(self):
        models_dir = self.bundle_dir / "models"
        analysis_dir = self.bundle_dir / "analysis"

        last_gen = self._find_last_generation_dir()
        if last_gen is not None:
            sampling_mtp_dir = last_gen / SUS2_MODEL_DIR
            if sampling_mtp_dir.exists():
                self._copy_tree(sampling_mtp_dir, models_dir / "sampling_last_potential", "models.sampling_last_potential")

        training_cfg = self._training_config()
        training_root = self.run_dir / training_cfg.get("work_dir", "high_precision_training")
        model_name = training_cfg.get("model_name", "trained.mtp")
        self._copy_file(training_root / model_name, models_dir / "final_training_potential" / model_name, "models.final_training_potential")

        plot_cfg = dict(training_cfg.get("plot") or {})
        plot_output = plot_cfg.get("output", "sus2_errors.jpg")
        self._copy_file(training_root / plot_output, analysis_dir / Path(plot_output).name, "analysis.error_plot")

        prediction_cfg = dict(training_cfg.get("predict") or {})
        prediction_dir = training_root / prediction_cfg.get("output_dir", "prediction")
        if prediction_dir.exists():
            self._copy_tree(prediction_dir, analysis_dir / "prediction", "analysis.prediction")

        coverage_cfg = dict(self.config.get("coverage_plot") or {})
        if coverage_cfg.get("enabled"):
            coverage_dir = self.run_dir / coverage_cfg.get("output_dir", "xyz_pca_coverage_results")
            self._copy_tree(coverage_dir, analysis_dir / "coverage", "analysis.coverage")
            coverage_query_dir = self.run_dir / "coverage_query_lammps"
            self._copy_tree(
                coverage_query_dir,
                analysis_dir / "coverage_query_lammps",
                "analysis.coverage_query_lammps",
            )
