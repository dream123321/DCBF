from __future__ import annotations

import importlib
import json
import logging
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Sequence

from ase.data import atomic_numbers, chemical_symbols
from ase.io import iread

from .das.file_conversion import xyz2cfg
from .runtime_config import build_scheduler_spec, normalize_scheduler_config


TRAINING_TEMPLATE_CHOICES = {
    "l2k2": "l2k2.mtp",
    "l2k3": "l2k3.mtp",
    "l3k3": "l3k3.mtp",
    "l4k3": "l4k3.mtp",
    "l4k4": "l4k4.mtp",
    "l4k6": "l4k6.mtp",
}


class HighPrecisionTrainer:
    DEFAULTS = {
        "enabled": False,
        "input_xyz": None,
        "work_dir": "high_precision_training",
        "template_type": "l2k3",
        "model_name": "trained.mtp",
        "elements": None,
        "sort_ele": True,
        "r_max": None,
        "submit": True,
        "wait": True,
        "command_prefix": None,
        "max_iter": 6000,
        "predict": {
            "enabled": True,
            "input_xyz": None,
            "calc_type": "sus2",
            "output_dir": "prediction",
            "output_format": "extxyz",
            "suffix": "pred",
            "device": "cpu",
            "num_workers": 1,
            "chunksize": 1,
            "log_level": "INFO",
        },
        "plot": {
            "enabled": True,
            "output": "sus2_errors.jpg",
            "mlip_name": "SUS2",
            "force_mode": "components",
            "num_processes": 8,
            "keep_temp": False,
            "show_r2": True,
            "save_data": False,
            "stress_unit": "eV",
            "cmap": "Spectral_r",
            "skip_structure_indices": [],
            "axis_padding": 0.04,
            "signed_axis_symmetric": False,
            "fontsize": 30,
            "tick_labelsize": 30,
            "legend_fontsize": 20,
            "title_fontsize": 32,
            "annotation_fontsize": 20,
            "cbar_fontsize": 28,
            "cbar_tick_size": 22,
            "linewidth": 4,
            "scatter_size": 10,
            "bins": 120,
            "dpi": 300,
            "figsize": [30, 10],
        },
    }

    def __init__(self, config: dict, config_path: Path, run_dir: Path):
        self.config = dict(config)
        self.config_path = Path(config_path).resolve()
        self.config_dir = self.config_path.parent
        self.run_dir = Path(run_dir).resolve()
        self.training = self._normalize_training_config(self.config.get("training"))
        self.scheduler = build_scheduler_spec(normalize_scheduler_config(dict(self._scheduler_config())))
        self.training_root = self._resolve_path(self.training["work_dir"], base_dir=self.run_dir)
        self.training_root.mkdir(parents=True, exist_ok=True)
        self.logger = self._create_logger()

    @classmethod
    def _normalize_training_config(cls, raw_training):
        training = dict(raw_training or {})
        normalized = {}
        for key, default_value in cls.DEFAULTS.items():
            if isinstance(default_value, dict):
                merged = dict(default_value)
                merged.update(training.get(key) or {})
                normalized[key] = merged
            elif key in training:
                normalized[key] = training[key]
            else:
                normalized[key] = default_value
        return normalized

    def is_enabled(self):
        return bool(self.training.get("enabled", False))

    def run(self):
        if not self.is_enabled():
            return None

        input_xyz = self._resolve_input_xyz()
        elements = self._resolve_elements(input_xyz)
        train_cfg = self.training_root / "train.cfg"
        model_path = self.training_root / self.training["model_name"]
        template_path = self._training_assets_dir() / TRAINING_TEMPLATE_CHOICES[self.training["template_type"]]
        job_dir = self.training_root / "train_job"
        job_dir.mkdir(parents=True, exist_ok=True)

        xyz2cfg(elements, self._sort_ele_flag(), str(input_xyz), str(train_cfg))
        rendered_template = self._prepare_training_template(template_path, job_dir / "hyx.mtp", elements)
        self.logger.info("[training] input_xyz=%s", input_xyz)
        self.logger.info("[training] elements=%s", elements)
        self.logger.info("[training] template_type=%s template=%s", self.training["template_type"], template_path)
        self.logger.info("[training] rendered_template=%s", rendered_template)
        self.logger.info("[training] species_count=%s", len(elements))
        if self.training.get("r_max") is not None:
            self.logger.info("[training] r_max=%s", self.training.get("r_max"))
        self.logger.info("[training] training_root=%s", self.training_root)

        if self.training.get("submit", True):
            self._submit_training_job(job_dir, model_path)
            if self.training.get("wait", True):
                self._wait_for_training_completion(job_dir)
        else:
            self._run_training_locally(job_dir, model_path)

        if not model_path.exists():
            raise RuntimeError(f"Training finished but model was not created: {model_path}")

        prediction_path = None
        if self.training["predict"].get("enabled", False):
            prediction_path = self._run_prediction(input_xyz, elements, model_path)

        plot_path = None
        plot_error = None
        if self.training["plot"].get("enabled", False) and prediction_path is not None:
            try:
                plot_path = self._run_plot(input_xyz, prediction_path, elements)
            except Exception as exc:
                plot_error = str(exc)
                self.logger.exception("[training] plot failed; continuing without plot: %s", exc)

        metrics = self._extract_training_metrics(job_dir / "logout")
        report = {
            "input_xyz": str(input_xyz),
            "elements": elements,
            "template_type": self.training["template_type"],
            "template_path": str(template_path),
            "rendered_template": str(rendered_template),
            "species_count": len(elements),
            "r_max": self.training.get("r_max"),
            "train_cfg": str(train_cfg),
            "model_path": str(model_path),
            "prediction_path": str(prediction_path) if prediction_path is not None else None,
            "plot_path": str(plot_path) if plot_path is not None else None,
            "plot_error": plot_error,
            "metrics": metrics,
        }
        report_path = self.training_root / "training_report.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        self.logger.info("[training] report=%s", report_path)
        return report

    def _scheduler_config(self):
        sampling = self.config.get("sampling")
        if isinstance(sampling, dict):
            return sampling.get("scheduler", {})
        return self.config.get("scheduler", {})

    def _workflow_config(self):
        sampling = self.config.get("sampling")
        if isinstance(sampling, dict):
            return sampling.get("workflow", {})
        return self.config.get("workflow", {})

    def _resolve_input_xyz(self):
        raw_input = self.training.get("input_xyz")
        if raw_input:
            return self._resolve_path(raw_input)

        workflow = self._workflow_config()
        output_name = workflow.get("output_xyz_name", "all_sample_data.xyz")
        candidate = self.run_dir / output_name
        if candidate.exists():
            return candidate

        raise FileNotFoundError(
            "training.input_xyz is not set and no sampling output xyz was found at "
            f"{candidate}"
        )

    def _resolve_elements(self, xyz_path: Path):
        explicit = self.training.get("elements")
        if explicit:
            return list(explicit)

        element_set = set()
        for atoms in iread(str(xyz_path)):
            element_set.update(atoms.get_chemical_symbols())
        ordered_atomic_numbers = sorted(atomic_numbers[element] for element in element_set)
        return [chemical_symbols[number] for number in ordered_atomic_numbers]

    def _sort_ele_flag(self):
        value = self.training.get("sort_ele", True)
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "1", "yes", "on"}:
                return True
            if lowered in {"false", "0", "no", "off"}:
                return False
        return bool(value)

    def _build_command_prefix(self):
        command_prefix = self.training.get("command_prefix")
        if command_prefix:
            return str(command_prefix)
        sus2_mlp_exe = self._scheduler_config().get("sus2_mlp_exe", self.scheduler.sus2_mlp_exe)
        max_iter = int(self.training.get("max_iter", 6000))
        return (
            f"mpirun {sus2_mlp_exe} train hyx.mtp ../train.cfg "
            f"--stdd-weight=0.0 --std-weight=0.0000 --max-iter={max_iter} --do-lin --curr-pot-name="
        )

    def _prepare_training_template(self, template_path: Path, output_path: Path, elements):
        text = template_path.read_text(encoding="utf-8")
        species_count = len(elements)
        text, species_replacements = re.subn(
            r"(^\s*species_count\s*=\s*)(\d+)",
            rf"\g<1>{species_count}",
            text,
            count=1,
            flags=re.MULTILINE,
        )
        if species_replacements == 0:
            raise ValueError(f"Could not find species_count in template: {template_path}")

        r_max = self.training.get("r_max")
        if r_max is not None:
            r_max = float(r_max)
            if r_max <= 0:
                raise ValueError("training.r_max must be > 0 when provided")
            text, max_dist_replacements = re.subn(
                r"(^\s*max_dist\s*=\s*)([-+0-9.eE]+)",
                rf"\g<1>{r_max}",
                text,
                count=1,
                flags=re.MULTILINE,
            )
            if max_dist_replacements == 0:
                raise ValueError(f"Could not find max_dist in template: {template_path}")

        output_path.write_text(text, encoding="utf-8")
        return output_path

    def _submit_training_job(self, job_dir: Path, model_path: Path):
        command_prefix = self._build_command_prefix()
        train_env = str(self._scheduler_config().get("train_env", "") or "").strip()
        train_env_text = f"{train_env}\n" if train_env else ""
        script_text = (
            "#!/bin/bash\n"
            f"{self.scheduler.bsub_script_train_sus_job_name}high_precision_train\n"
            f"{self.scheduler.bsub_script_train_sus}"
            f"{train_env_text}"
            "\nstart_time=$(date +%s.%N)\n"
            "touch __start__\n"
            f"COMMAND_std=\"{command_prefix}{model_path.as_posix()}\"\n"
            "$COMMAND_std > logout 2>&1\n"
            "touch __ok__\n"
            "end_time=$(date +%s.%N)\n"
            "runtime=$(echo \"$end_time - $start_time\" | bc)\n"
            "echo \"total_runtime:$runtime s\" >> time.txt\n"
        )
        script_path = job_dir / "bsub.lsf"
        script_path.write_text(script_text, encoding="utf-8")
        self.logger.info("[training] submitting job from %s", job_dir)
        subprocess.run(self.scheduler.task_submission_method, cwd=job_dir, shell=True, check=True)

    def _wait_for_training_completion(self, job_dir: Path):
        self.logger.info("[training] waiting for training completion")
        while True:
            if (job_dir / "__ok__").exists():
                self.logger.info("[training] training completed")
                return
            if (job_dir / "__error__").exists():
                raise RuntimeError(f"Training failed: {job_dir}")
            if (job_dir / "logout").exists():
                text = (job_dir / "logout").read_text(encoding="utf-8", errors="ignore")
                if "Killed" in text or "Traceback" in text:
                    raise RuntimeError(f"Training failed, see logout: {job_dir / 'logout'}")
            import time

            time.sleep(10)

    def _run_training_locally(self, job_dir: Path, model_path: Path):
        command_prefix = self._build_command_prefix()
        command = command_prefix + model_path.as_posix()
        self.logger.info("[training] running locally: %s", command)
        subprocess.run(command, cwd=job_dir, shell=True, check=True)

    def _run_prediction(self, input_xyz: Path, elements, model_path: Path):
        prediction_cfg = self.training["predict"]
        output_dir = self.training_root / prediction_cfg.get("output_dir", "prediction")
        output_dir.mkdir(parents=True, exist_ok=True)
        calc_type = prediction_cfg.get("calc_type", "sus2")
        suffix = prediction_cfg.get("suffix", "pred")
        output_format = prediction_cfg.get("output_format", "extxyz")
        args = [
            sys.executable,
            "-m",
            "dcbf.high_precision_tools",
            "predict",
            str(input_xyz),
            "--calc_type",
            str(calc_type),
            "--model",
            str(model_path),
            "--device",
            str(prediction_cfg.get("device", "cpu")),
            "--output",
            str(output_dir),
            "--format",
            str(output_format),
            "--suffix",
            str(suffix),
            "--num-workers",
            str(int(prediction_cfg.get("num_workers", 1))),
            "--log-level",
            str(prediction_cfg.get("log_level", "INFO")),
        ]
        if self._predict_cli_supports("--chunksize"):
            args.extend(["--chunksize", str(int(prediction_cfg.get("chunksize", 1)))])
        if calc_type == "sus2":
            args.extend(["--ele_list", *elements])
        self.logger.info("[training] prediction command=%s", args)
        subprocess.run(args, check=True)

        stem = input_xyz.stem
        if output_format == input_xyz.suffix.lstrip("."):
            predicted_name = f"{calc_type}_{stem}_{suffix}{input_xyz.suffix}" if suffix else f"{calc_type}_{stem}{input_xyz.suffix}"
        elif output_format == "extxyz":
            predicted_name = f"{calc_type}_{stem}_{suffix}.xyz" if suffix else f"{calc_type}_{stem}.xyz"
        else:
            predicted_name = f"{calc_type}_{stem}_{suffix}.{output_format}" if suffix else f"{calc_type}_{stem}.{output_format}"
        predicted_path = output_dir / predicted_name
        if not predicted_path.exists():
            raise FileNotFoundError(f"Predicted xyz was not created: {predicted_path}")
        self.logger.info("[training] prediction_xyz=%s", predicted_path)
        return predicted_path

    @staticmethod
    def _predict_cli_supports(option: str) -> bool:
        tools = importlib.import_module("dcbf.high_precision_tools")
        parser_factory = getattr(tools, "build_parser", None)
        if parser_factory is None:
            return False
        help_text = parser_factory().format_help()
        return option in help_text

    def _run_plot(self, dft_xyz: Path, predicted_xyz: Path, elements):
        plot_cfg = self.training["plot"]
        output_path = self.training_root / plot_cfg.get("output", "sus2_errors.jpg")
        plot_script = Path(__file__).resolve().parent / "sus2_plot_errors_v3.py"
        args = [
            sys.executable,
            str(plot_script),
            str(dft_xyz),
            str(predicted_xyz),
            "--mlip-name",
            str(plot_cfg.get("mlip_name", "SUS2")),
            "--force-mode",
            str(plot_cfg.get("force_mode", "components")),
            "--num-processes",
            str(int(plot_cfg.get("num_processes", 8))),
            "--output",
            str(output_path),
            "--stress-unit",
            str(plot_cfg.get("stress_unit", "eV")),
            "--axis-padding",
            str(float(plot_cfg.get("axis_padding", 0.04))),
            "--signed-axis-symmetric",
            str(bool(plot_cfg.get("signed_axis_symmetric", False))),
            "--cmap",
            str(plot_cfg.get("cmap", "Spectral_r")),
            "--keep-temp",
            str(bool(plot_cfg.get("keep_temp", False))),
            "--show-r2",
            str(bool(plot_cfg.get("show_r2", True))),
            "--save-data",
            str(bool(plot_cfg.get("save_data", False))),
            "--fontsize",
            str(int(plot_cfg.get("fontsize", 30))),
            "--tick-labelsize",
            str(int(plot_cfg.get("tick_labelsize", 30))),
            "--legend-fontsize",
            str(int(plot_cfg.get("legend_fontsize", 20))),
            "--title-fontsize",
            str(int(plot_cfg.get("title_fontsize", 32))),
            "--annotation-fontsize",
            str(int(plot_cfg.get("annotation_fontsize", 20))),
            "--cbar-fontsize",
            str(int(plot_cfg.get("cbar_fontsize", 28))),
            "--cbar-tick-size",
            str(int(plot_cfg.get("cbar_tick_size", 22))),
            "--linewidth",
            str(float(plot_cfg.get("linewidth", 4))),
            "--scatter-size",
            str(int(plot_cfg.get("scatter_size", 10))),
            "--bins",
            str(int(plot_cfg.get("bins", 120))),
            "--dpi",
            str(int(plot_cfg.get("dpi", 300))),
            "--figsize",
            str(float(plot_cfg.get("figsize", [30, 10])[0])),
            str(float(plot_cfg.get("figsize", [30, 10])[1])),
            "--elements",
            *(plot_cfg.get("elements") or elements),
        ]
        skip_structure_indices = plot_cfg.get("skip_structure_indices") or []
        if skip_structure_indices:
            args.extend(["--skip-structure-indices", *[str(index) for index in skip_structure_indices]])
        self.logger.info("[training] plot command=%s", args)
        env = os.environ.copy()
        env["MPLBACKEND"] = str(plot_cfg.get("matplotlib_backend", "Agg"))
        subprocess.run(args, check=True, env=env)
        self.logger.info("[training] plot_output=%s", output_path)
        return output_path

    @staticmethod
    def _extract_training_metrics(logout_path: Path):
        metrics = {
            "energy_mae_mev_per_atom": None,
            "force_mae_mev_per_a": None,
            "stress_mae_ev": None,
        }
        if not logout_path.exists():
            return metrics

        patterns = {
            "energy_mae_mev_per_atom": re.compile(r"Energy MAE \(meV/atom\):\s*([0-9.+\-eE]+)"),
            "force_mae_mev_per_a": re.compile(r"Force MAE\s+\(meV/A\)\s*:\s*([0-9.+\-eE]+)"),
            "stress_mae_ev": re.compile(r"Stress MAE\s+\(eV\)\s*:\s*([0-9.+\-eE]+)"),
        }
        text = logout_path.read_text(encoding="utf-8", errors="ignore")
        for key, pattern in patterns.items():
            match = pattern.search(text)
            if match is not None:
                metrics[key] = float(match.group(1))
        return metrics

    def _resolve_path(self, raw_path, base_dir: Path | None = None):
        path = Path(raw_path)
        if path.is_absolute():
            return path
        if base_dir is None:
            base_dir = self.config_dir
        return (base_dir / path).resolve()

    @staticmethod
    def _training_assets_dir():
        return Path(__file__).resolve().parent / "training_assets"

    def _create_logger(self):
        logger = logging.getLogger(f"dcbf.high_precision_training.{id(self)}")
        logger.setLevel(logging.INFO)
        logger.propagate = False
        logger.handlers.clear()
        formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
        file_handler = logging.FileHandler(self.training_root / "training.log", encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        return logger
