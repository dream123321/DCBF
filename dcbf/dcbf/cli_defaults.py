from __future__ import annotations

import json
import os
from pathlib import Path


DEFAULTS_PATH = Path(os.environ.get("DCBF_CLI_DEFAULTS", Path.home() / ".dcbf" / "sus2sh_cli_defaults.json"))

NONE_TOKENS = {"", "none", "null", "auto", "default"}
TRUE_TOKENS = {"1", "true", "yes", "y", "on"}
FALSE_TOKENS = {"0", "false", "no", "n", "off"}


CLI_DEFAULTS = {
    "train": {
        "template": "sus2_sh",
        "model_yaml": None,
        "min_dist": None,
        "max_dist": None,
        "radial_basis_size": None,
        "backend": "bsub",
        "queue": "33",
        "cores": None,
        "ptile": None,
        "max_iter": 6000,
        "sus2_exe": None,
        "train_env": "",
        "work_dir": None,
        "submit": False,
        "elements": None,
        "keep_order": False,
    },
    "mp_search": {
        "api_key": None,
        "output_dir": "mp-stru",
        "csv_name": "summary.csv",
    },
    "efs_distri": {
        "force_threshold": None,
        "bins": 120,
        "energy_bins": None,
        "force_bins": None,
        "stress_bins": None,
        "density": False,
        "fit": False,
        "figsize": [30.0, 10.0],
        "dpi": 300,
        "log_y": False,
        "output": None,
    },
    "predict_xyz": {
        "calc_type": "sus2",
        "model": None,
        "elements": None,
        "device": "cpu",
        "output": "out_files",
        "format": "extxyz",
        "append": False,
        "log_level": "INFO",
        "suffix": "",
        "num_workers": 1,
        "split_workers": 1,
    },
    "plot_errors": {
        "mlip_name": "SUS²",
        "elements": None,
        "keep_temp": False,
        "force_mode": "components",
        "num_processes": 24,
        "output": "efs.jpg",
        "stress_unit": "eV",
        "skip_structure_indices": [],
        "axis_padding": 0.04,
        "signed_axis_symmetric": False,
        "figsize": [30.0, 10.0],
        "dpi": 300,
        "cmap": "Spectral_r",
        "scatter_size": 10,
        "bins": 120,
        "fontsize": 30,
        "tick_labelsize": 30,
        "legend_fontsize": 20,
        "title_fontsize": 32,
        "annotation_fontsize": 20,
        "cbar_fontsize": 28,
        "cbar_tick_size": 22,
        "linewidth": 4.0,
        "show_r2": True,
        "save_data": False,
    },
}


def _parse_bool(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in TRUE_TOKENS:
        return True
    if normalized in FALSE_TOKENS:
        return False
    raise ValueError(f"Expected a boolean value, got {value!r}")


def _parse_optional_str(value: str):
    if str(value).strip().lower() in NONE_TOKENS:
        return None
    return str(value)


def _parse_optional_int(value: str):
    if str(value).strip().lower() in NONE_TOKENS:
        return None
    return int(value)


def _parse_optional_float(value: str):
    if str(value).strip().lower() in NONE_TOKENS:
        return None
    return float(value)


def _parse_str_list(value: str):
    normalized = str(value).strip()
    if normalized.lower() in NONE_TOKENS:
        return None
    items = [item.strip() for item in normalized.split(",") if item.strip()]
    return items or None


def _parse_int_list(value: str):
    normalized = str(value).strip()
    if normalized.lower() in NONE_TOKENS:
        return []
    return [int(item.strip()) for item in normalized.split(",") if item.strip()]


def _parse_float_pair(value: str):
    normalized = str(value).strip()
    if normalized.lower() in NONE_TOKENS:
        return None
    items = [item.strip() for item in normalized.split(",") if item.strip()]
    if len(items) != 2:
        raise ValueError(f"Expected two comma-separated numbers, got {value!r}")
    return [float(items[0]), float(items[1])]


SECTION_SCHEMAS = {
    "train": {
        "template": str,
        "model_yaml": _parse_optional_str,
        "min_dist": _parse_optional_float,
        "max_dist": _parse_optional_float,
        "radial_basis_size": _parse_optional_int,
        "backend": str,
        "queue": str,
        "cores": _parse_optional_int,
        "ptile": _parse_optional_int,
        "max_iter": int,
        "sus2_exe": str,
        "train_env": _parse_optional_str,
        "work_dir": _parse_optional_str,
        "submit": _parse_bool,
        "elements": _parse_str_list,
        "keep_order": _parse_bool,
    },
    "mp_search": {
        "api_key": _parse_optional_str,
        "output_dir": str,
        "csv_name": str,
    },
    "efs_distri": {
        "force_threshold": _parse_optional_float,
        "bins": int,
        "energy_bins": _parse_optional_int,
        "force_bins": _parse_optional_int,
        "stress_bins": _parse_optional_int,
        "density": _parse_bool,
        "fit": _parse_bool,
        "figsize": _parse_float_pair,
        "dpi": int,
        "log_y": _parse_bool,
        "output": _parse_optional_str,
    },
    "predict_xyz": {
        "calc_type": str,
        "model": _parse_optional_str,
        "elements": _parse_str_list,
        "device": str,
        "output": str,
        "format": str,
        "append": _parse_bool,
        "log_level": str,
        "suffix": str,
        "num_workers": int,
        "split_workers": int,
    },
    "plot_errors": {
        "mlip_name": str,
        "elements": _parse_str_list,
        "keep_temp": _parse_bool,
        "force_mode": str,
        "num_processes": int,
        "output": str,
        "stress_unit": str,
        "skip_structure_indices": _parse_int_list,
        "axis_padding": float,
        "signed_axis_symmetric": _parse_bool,
        "figsize": _parse_float_pair,
        "dpi": int,
        "cmap": str,
        "scatter_size": int,
        "bins": int,
        "fontsize": int,
        "tick_labelsize": int,
        "legend_fontsize": int,
        "title_fontsize": int,
        "annotation_fontsize": int,
        "cbar_fontsize": int,
        "cbar_tick_size": int,
        "linewidth": float,
        "show_r2": _parse_bool,
        "save_data": _parse_bool,
    },
}

SECTION_ALIASES = {
    "train": {
        "ele_list": "elements",
        "keep_order": "keep_order",
        "max_dist": "max_dist",
        "min_dist": "min_dist",
        "radial_basis_size": "radial_basis_size",
    },
    "mp_search": {
        "api-key": "api_key",
        "api_key": "api_key",
        "output-dir": "output_dir",
        "output_dir": "output_dir",
        "csv-name": "csv_name",
        "csv_name": "csv_name",
    },
    "efs_distri": {
        "log_y": "log_y",
    },
    "predict_xyz": {
        "ele_list": "elements",
        "num_workers": "num_workers",
        "split_workers": "split_workers",
        "log_level": "log_level",
        "calc_type": "calc_type",
    },
    "plot_errors": {
        "mlip-name": "mlip_name",
        "mlip_name": "mlip_name",
        "keep-temp": "keep_temp",
        "keep_temp": "keep_temp",
        "force-mode": "force_mode",
        "force_mode": "force_mode",
        "num-processes": "num_processes",
        "num_processes": "num_processes",
        "stress-unit": "stress_unit",
        "stress_unit": "stress_unit",
        "skip-structure-indices": "skip_structure_indices",
        "skip_structure_indices": "skip_structure_indices",
        "axis-padding": "axis_padding",
        "axis_padding": "axis_padding",
        "signed-axis-symmetric": "signed_axis_symmetric",
        "signed_axis_symmetric": "signed_axis_symmetric",
        "scatter-size": "scatter_size",
        "scatter_size": "scatter_size",
        "tick-labelsize": "tick_labelsize",
        "tick_labelsize": "tick_labelsize",
        "legend-fontsize": "legend_fontsize",
        "legend_fontsize": "legend_fontsize",
        "title-fontsize": "title_fontsize",
        "title_fontsize": "title_fontsize",
        "annotation-fontsize": "annotation_fontsize",
        "annotation_fontsize": "annotation_fontsize",
        "cbar-fontsize": "cbar_fontsize",
        "cbar_fontsize": "cbar_fontsize",
        "cbar-tick-size": "cbar_tick_size",
        "cbar_tick_size": "cbar_tick_size",
        "show-r2": "show_r2",
        "show_r2": "show_r2",
        "save-data": "save_data",
        "save_data": "save_data",
    },
}


def _normalize_assignment_key(section: str, key: str) -> str:
    normalized = key.strip().lower().replace("-", "_")
    return SECTION_ALIASES.get(section, {}).get(normalized, normalized)


def load_user_defaults(path: Path | None = None):
    defaults_path = Path(path or DEFAULTS_PATH)
    if not defaults_path.exists():
        return {}
    with open(defaults_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def save_user_defaults(payload, path: Path | None = None):
    defaults_path = Path(path or DEFAULTS_PATH)
    defaults_path.parent.mkdir(parents=True, exist_ok=True)
    with open(defaults_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return defaults_path


def get_section_defaults(section: str, path: Path | None = None):
    if section not in CLI_DEFAULTS:
        raise KeyError(f"Unknown CLI defaults section: {section}")
    defaults = dict(CLI_DEFAULTS[section])
    payload = load_user_defaults(path)
    overrides = payload.get(section, {})
    if isinstance(overrides, dict):
        for key, value in overrides.items():
            if key in defaults:
                defaults[key] = value
    return defaults


def set_section_defaults(section: str, assignments, path: Path | None = None):
    if section not in CLI_DEFAULTS:
        raise KeyError(f"Unknown CLI defaults section: {section}")
    schema = SECTION_SCHEMAS[section]
    payload = load_user_defaults(path)
    current = dict(payload.get(section, {}))

    for assignment in assignments:
        if "=" not in assignment:
            raise ValueError(f"Expected key=value, got {assignment!r}")
        raw_key, raw_value = assignment.split("=", 1)
        key = _normalize_assignment_key(section, raw_key)
        if key not in schema:
            valid_keys = ", ".join(sorted(schema))
            raise ValueError(f"Unknown key {raw_key!r} for {section}; valid keys: {valid_keys}")
        current[key] = schema[key](raw_value)

    payload[section] = current
    save_user_defaults(payload, path)
    return get_section_defaults(section, path)


def _format_value(value):
    if value is None:
        return "<auto>"
    if isinstance(value, list):
        return ",".join(str(item) for item in value)
    return str(value)


def format_section_defaults(section: str, path: Path | None = None):
    defaults = get_section_defaults(section, path)
    lines = [f"{key}={_format_value(defaults[key])}" for key in sorted(defaults)]
    return "\n".join(lines)


def defaults_epilog(section: str, path: Path | None = None):
    return (
        "Current saved defaults:\n"
        f"{format_section_defaults(section, path)}\n\n"
        f"Use `dcbf {section.replace('_', '-')} set key=value ...` to update them."
    )
