from __future__ import annotations

from collections import Counter
import gzip
import json
import os
import shutil
from pathlib import Path

import numpy as np
from ase.io import iread, write

from .core_hours import write_core_hours_report
from .path_names import SUS2_MODEL_DIR


class ArtifactBundler:
    DEFAULTS = {
        "enabled": False,
        "output_dir": "summary_bundle",
    }
    DATASET_EXPORTS = {
        "existing_dataset": {
            "filename": "existing_dataset.xyz",
            "description": "User-provided existing dataset before optional builder augmentation.",
        },
        "builder_dataset": {
            "filename": "builder_dataset.xyz",
            "description": "Unique builder structures actually added to the sampling base dataset.",
        },
        "all_dataset": {
            "filename": "all.xyz",
            "description": "Final dataset with existing, builder, and DCBF sampling origins annotated.",
        },
        "dcbf_sampling": {
            "filename": "dcbf_sampling.xyz",
            "description": "Structures added by the DCBF active-learning sampling loops.",
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
            "status": "complete",
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
        temporary = manifest_path.with_name(manifest_path.name + ".tmp")
        temporary.write_text(json.dumps(self.manifest, indent=2), encoding="utf-8")
        os.replace(temporary, manifest_path)
        self._cleanup_coverage_intermediates()
        return manifest_path

    def _resolve_output_dir(self, raw_path):
        path = Path(raw_path)
        if path.is_absolute():
            return path
        return (self.run_dir.parent / path).resolve()

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

    def _copy_gzip(self, source: Path, destination: Path, manifest_key: str):
        source = Path(source)
        if not source.exists():
            self.manifest["missing"][manifest_key] = str(source)
            return
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".tmp")
        with open(source, "rb") as reader, gzip.open(temporary, "wb", compresslevel=9) as writer:
            shutil.copyfileobj(reader, writer, length=1024 * 1024)
        with gzip.open(temporary, "rb") as reader:
            while reader.read(1024 * 1024):
                pass
        os.replace(temporary, destination)
        self.manifest["copied"][manifest_key] = str(destination)

    def _dataset_config(self):
        return dict(self.config.get("dataset", {}))

    def _workflow_config(self):
        return dict(self.config.get("workflow", {}))

    def _training_config(self):
        return dict(self.config.get("training", {}))

    def _builder_config(self):
        return dict(self._dataset_config().get("builder") or {})

    def _builder_enabled(self):
        return bool(self._builder_config().get("enabled", False))

    def _dataset_mode(self):
        dataset = self._dataset_config()
        builder = self._builder_config()
        return str(dataset.get("dataset_mode", builder.get("dataset_mode", "generated_only"))).strip().lower()

    def _resolve_config_path(self, raw_path):
        path = Path(raw_path)
        if path.is_absolute():
            return path
        return (self.config_path.parent / path).resolve()

    def _existing_dataset_path(self):
        dataset = self._dataset_config()
        if self._builder_enabled() and self._dataset_mode() != "augment_existing":
            return None
        raw = dataset.get("xyz_input")
        if not raw:
            return None
        path = self._resolve_config_path(raw)
        if self._builder_enabled() and path == self._builder_output_path():
            report_path = self._builder_config().get("report_path")
            if report_path:
                report_path = Path(report_path)
                if not report_path.is_absolute():
                    report_path = (self.run_dir / report_path).resolve()
                if report_path.exists():
                    report = json.loads(report_path.read_text(encoding="utf-8"))
                    existing_path = report.get("existing_dataset_xyz")
                    if existing_path:
                        path = Path(existing_path)
        return path

    def _builder_output_path(self):
        if not self._builder_enabled():
            return None
        builder = self._builder_config()
        raw = builder.get("output_xyz")
        if not raw:
            return None
        path = Path(raw)
        if path.is_absolute():
            return path
        return (self.run_dir / path).resolve()

    def _sampling_base_path(self):
        builder_output = self._builder_output_path()
        if builder_output is not None:
            return builder_output
        return self._existing_dataset_path()

    def _sampling_output_path(self):
        workflow = self._workflow_config()
        output_name = workflow.get("output_xyz_name", "sampled_training_dataset.xyz")
        return self.run_dir / output_name

    def _prune_dataset_dir(self, datasets_dir: Path):
        datasets_dir.mkdir(parents=True, exist_ok=True)
        for path in datasets_dir.glob("*.xyz"):
            path.unlink()

    def _record_dataset(
        self,
        dataset_key: str,
        dataset_path: Path,
        *,
        match_mode: str | None = None,
        derived_from=None,
        main_labels=None,
        structure_count: int | None = None,
        data_origin: str | None = None,
    ):
        spec = self.DATASET_EXPORTS[dataset_key]
        record = {
            "path": str(dataset_path),
            "description": spec["description"],
        }
        if match_mode is not None:
            record["match_mode"] = match_mode
        if derived_from is not None:
            record["derived_from"] = [
                item for item in derived_from if item is not None
            ]
        if main_labels is not None:
            record["main_labels"] = main_labels
        if structure_count is not None:
            record["structure_count"] = int(structure_count)
        if data_origin is not None:
            record["data_origin"] = data_origin
        self.manifest["datasets"][dataset_key] = record

    def _collect_datasets(self):
        datasets_dir = self.bundle_dir / "datasets"
        self._prune_dataset_dir(datasets_dir)
        existing_path = self._existing_dataset_path()
        sampling_base_path = self._sampling_base_path()
        all_dataset = self._sampling_output_path()
        required_paths = {
            "sampling_base": sampling_base_path,
            "all_dataset": all_dataset,
        }
        if existing_path is not None:
            required_paths["existing_dataset"] = existing_path
        missing_required = {
            key: str(path)
            for key, path in required_paths.items()
            if path is None or not Path(path).exists()
        }
        if missing_required:
            self.manifest["missing"].update(
                {f"datasets.{key}": path for key, path in missing_required.items()}
            )
            return

        existing_atoms = (
            list(iread(str(existing_path), index=":"))
            if existing_path is not None
            else []
        )
        sampling_base_atoms = list(iread(str(sampling_base_path), index=":"))
        all_atoms = list(iread(str(all_dataset), index=":"))

        existing_retained, builder_added = self._split_sampling_base(
            sampling_base_atoms,
            existing_atoms,
        )
        annotated_all, sampling_only_atoms, unmatched_base_count = self._annotate_all_origins(
            all_atoms,
            existing_retained,
            builder_added,
        )
        if unmatched_base_count:
            raise RuntimeError(
                "summary dataset split failed: "
                f"{unmatched_base_count} sampling-base structures are absent from {all_dataset}"
            )

        if existing_path is not None:
            existing_export = [self._annotate(atoms, "existing", main=-1) for atoms in existing_atoms]
            existing_output = datasets_dir / self.DATASET_EXPORTS["existing_dataset"]["filename"]
            self._write_xyz(existing_output, existing_export)
            self.manifest["copied"]["datasets.existing_dataset"] = str(existing_output)
            self._record_dataset(
                "existing_dataset",
                existing_output,
                main_labels=[-1],
                structure_count=len(existing_export),
                data_origin="existing",
            )

        if self._builder_enabled() and builder_added:
            builder_output = datasets_dir / self.DATASET_EXPORTS["builder_dataset"]["filename"]
            builder_export = [self._annotate(atoms, "builder", main=-1) for atoms in builder_added]
            self._write_xyz(builder_output, builder_export)
            self.manifest["copied"]["datasets.builder_dataset"] = str(builder_output)
            self._record_dataset(
                "builder_dataset",
                builder_output,
                match_mode="builder_geometry_fingerprint",
                derived_from=[Path(sampling_base_path).name],
                main_labels=[-1],
                structure_count=len(builder_export),
                data_origin="builder",
            )

        sampling_output = datasets_dir / self.DATASET_EXPORTS["dcbf_sampling"]["filename"]
        self._write_xyz(sampling_output, sampling_only_atoms)
        self.manifest["copied"]["datasets.dcbf_sampling"] = str(sampling_output)
        self._record_dataset(
            "dcbf_sampling",
            sampling_output,
            match_mode="builder_geometry_fingerprint",
            derived_from=[Path(all_dataset).name, Path(sampling_base_path).name],
            main_labels="preserved_from_all_dataset",
            structure_count=len(sampling_only_atoms),
            data_origin="dcbf_sampling",
        )

        all_output = datasets_dir / self.DATASET_EXPORTS["all_dataset"]["filename"]
        self._write_xyz(all_output, annotated_all)
        self.manifest["copied"]["datasets.all_dataset"] = str(all_output)
        self._record_dataset(
            "all_dataset",
            all_output,
            match_mode="builder_geometry_fingerprint",
            derived_from=[
                self.DATASET_EXPORTS["existing_dataset"]["filename"]
                if existing_path is not None
                else None,
                self.DATASET_EXPORTS["builder_dataset"]["filename"]
                if self._builder_enabled()
                else None,
                self.DATASET_EXPORTS["dcbf_sampling"]["filename"],
            ],
            main_labels="preserved_from_sampling_output",
            structure_count=len(annotated_all),
            data_origin="mixed",
        )

        self.manifest["dataset_counts"] = {
            "existing_source": len(existing_atoms),
            "existing_retained_in_sampling_base": len(existing_retained),
            "existing_duplicates_removed_by_builder": max(
                0, len(existing_atoms) - len(existing_retained)
            ),
            "builder_added": len(builder_added),
            "sampling_base": len(sampling_base_atoms),
            "dcbf_sampling": len(sampling_only_atoms),
            "all": len(annotated_all),
        }

    @staticmethod
    def _fingerprint_atoms(atoms):
        symbols = tuple(atoms.get_chemical_symbols())
        cell = tuple(np.round(np.asarray(atoms.get_cell()), 8).reshape(-1))
        scaled = tuple(np.round(atoms.get_scaled_positions(wrap=True), 6).reshape(-1))
        return symbols, cell, scaled

    @staticmethod
    def _annotate(atoms, origin, main=None):
        annotated = atoms.copy()
        annotated.calc = atoms.calc
        annotated.info = dict(atoms.info)
        annotated.info["data_origin"] = origin
        if main is not None:
            annotated.info["main"] = main
        return annotated

    def _split_sampling_base(self, sampling_base_atoms, existing_atoms):
        existing_counter = Counter(self._fingerprint_atoms(atoms) for atoms in existing_atoms)
        existing_retained = []
        builder_added = []
        default_origin = "builder" if self._builder_enabled() else "existing"
        for atoms in sampling_base_atoms:
            fingerprint = self._fingerprint_atoms(atoms)
            if existing_counter[fingerprint] > 0:
                existing_counter[fingerprint] -= 1
                existing_retained.append(atoms)
            elif default_origin == "existing":
                existing_retained.append(atoms)
            else:
                builder_added.append(atoms)
        return existing_retained, builder_added

    def _annotate_all_origins(self, all_atoms, existing_retained, builder_added):
        origin_counters = {
            "existing": Counter(
                self._fingerprint_atoms(atoms) for atoms in existing_retained
            ),
            "builder": Counter(
                self._fingerprint_atoms(atoms) for atoms in builder_added
            ),
        }
        annotated_all = []
        sampling_only = []
        for atoms in all_atoms:
            fingerprint = self._fingerprint_atoms(atoms)
            if origin_counters["existing"][fingerprint] > 0:
                origin_counters["existing"][fingerprint] -= 1
                origin = "existing"
            elif origin_counters["builder"][fingerprint] > 0:
                origin_counters["builder"][fingerprint] -= 1
                origin = "builder"
            else:
                origin = "dcbf_sampling"
            annotated = self._annotate(atoms, origin)
            annotated_all.append(annotated)
            if origin == "dcbf_sampling":
                sampling_only.append(annotated)
        unmatched = sum(
            sum(counter.values()) for counter in origin_counters.values()
        )
        return annotated_all, sampling_only, unmatched

    @staticmethod
    def _write_xyz(path: Path, atoms_list):
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            path.unlink()
        path.touch()
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
            sampling_sus2_dir = last_gen / SUS2_MODEL_DIR
            if sampling_sus2_dir.exists():
                self._copy_tree(sampling_sus2_dir, models_dir / "sampling_last_potential", "models.sampling_last_potential")

        training_cfg = self._training_config()
        training_root = self.run_dir / training_cfg.get("work_dir", "high_precision_training")
        final_cfg = dict(training_cfg.get("final") or {})
        model_name = training_cfg.get("model_name") or final_cfg.get(
            "model_name", "final.sus2"
        )
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
            coverage_dir = self.run_dir / coverage_cfg.get(
                "output_dir", "xyz_pca_coverage_results"
            )
            self._copy_tree(coverage_dir, analysis_dir / "coverage", "analysis.coverage")
            coverage_query_dir = self.run_dir / "coverage_query_lammps"
            self._copy_tree(
                coverage_query_dir,
                analysis_dir / "coverage_query_lammps",
                "analysis.coverage_query_lammps",
            )

    def _cleanup_coverage_intermediates(self):
        # Standalone SUS2 PCA/coverage artifacts are user analysis outputs.  Do
        # not remove them from the workspace or silently fold them into the
        # sampling selector; the selector and the analysis command remain
        # separate by design.
        return

