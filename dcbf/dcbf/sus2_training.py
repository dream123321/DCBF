"""SUS2-SH grouped YAML training helpers.

The public DCBF configuration deliberately contains only the path to the
architecture YAML. This module turns that YAML into the fresh-model or
continuation YAML consumed by mlp-sus2 train.
"""
from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
import subprocess

import yaml


SUPPORTED_KINDS = {"static-radial", "coordination-effect"}
PROFILE_BY_KIND = {
    "static-radial": "native-static-v1",
    "coordination-effect": "native-effect-v1",
}

# Sampling defaults.  These values are also used when a public JSON omits the
# corresponding training section; explicit user values remain authoritative.
DEFAULT_INITIAL_ITERATIONS = 600
DEFAULT_SUBSEQUENT_ITERATIONS = 300
DEFAULT_FINAL_ITERATIONS = 3000


class Sus2TrainingConfigError(ValueError):
    pass


def _as_mapping(value, label):
    if not isinstance(value, dict):
        raise Sus2TrainingConfigError(f"{label} must be a YAML mapping")
    return value


def load_model_yaml(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"SUS2 model_yaml does not exist: {path}")
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise Sus2TrainingConfigError(f"Invalid SUS2 model YAML: {path}") from exc
    document = _as_mapping(document, "model_yaml")
    model = _as_mapping(document.get("model"), "model_yaml.model")
    kind = str(model.get("kind", "")).strip().lower()
    if kind == "coordination-adaptive":
        raise Sus2TrainingConfigError(
            "coordination-adaptive is not supported by this DCBF build; "
            "use static-radial or coordination-effect"
        )
    if kind not in SUPPORTED_KINDS:
        raise Sus2TrainingConfigError(
            "model.kind must be one of: coordination-effect, static-radial"
        )
    training = _as_mapping(document.get("training", {}), "model_yaml.training")
    device = str(training.get("device", "cpu")).strip().lower()
    if device != "cpu":
        raise Sus2TrainingConfigError(
            "SUS2-SH CPU-only mode is active; model_yaml.training.device must be cpu"
        )
    dtype = str(training.get("dtype", "float64")).strip().lower()
    if dtype not in {"float64", "double"}:
        raise Sus2TrainingConfigError(
            "SUS2-SH training requires model_yaml.training.dtype=float64"
        )
    profile = str(training.get("profile", PROFILE_BY_KIND[kind])).strip()
    expected = PROFILE_BY_KIND[kind]
    if profile != expected:
        raise Sus2TrainingConfigError(
            f"model_yaml.training.profile must be {expected!r} for model.kind={kind!r}"
        )
    radial = training.get('radial_parameterization')
    if radial is not None and radial not in {'physical', 'weight-norm'}:
        raise Sus2TrainingConfigError('training.radial_parameterization must be physical or weight-norm')
    return document, kind


def continuation_document(document, checkpoint):
    """Preserve the seed's optimizer chart without repairing physical coefficients."""
    from .builtin_models import check_continuation_allowed, continuation_training_settings
    check_continuation_allowed(checkpoint)
    source = deepcopy(document)
    training = source.setdefault('training', {})
    for key, value in continuation_training_settings(checkpoint).items():
        if key in training and training[key] != value:
            raise Sus2TrainingConfigError(
                f'This packaged checkpoint requires training.{key}={value!r}; '
                'refusing a parameterization conversion or sign repair')
        training[key] = value
    return source


def _validate_complete_checkpoint(checkpoint):
    checkpoint = Path(checkpoint).resolve()
    if checkpoint.suffix.lower() != ".sus2":
        raise Sus2TrainingConfigError(
            f"SUS2 continuation requires a .sus2 checkpoint: {checkpoint}"
        )
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"SUS2 continuation checkpoint does not exist: {checkpoint}"
        )
    from .encode.model_metadata import read_sus2_descriptor_metadata
    read_sus2_descriptor_metadata(checkpoint)
    text = checkpoint.read_text(encoding="utf-8", errors="ignore")
    required = ("species_coeffs", "moment_coeffs")
    missing = [key for key in required if not f"{key} =" in text]
    if not any(f"{key} =" in text for key in ("shift_coeffs", "scal_coeffs")):
        missing.append("shift_coeffs/scal_coeffs")
    if missing:
        raise Sus2TrainingConfigError(
            "Continuation requires a complete trained SUS2 checkpoint; "
            f"{checkpoint} is missing parameter blocks: {', '.join(missing)}"
        )
    return checkpoint


def _training_block(document, kind, iterations):
    iterations = int(iterations)
    if iterations < 0:
        raise Sus2TrainingConfigError("training iterations must be non-negative")
    training = deepcopy(document.get("training") or {})
    training["engine"] = "native"
    training["device"] = "cpu"
    training["dtype"] = "float64"
    training["profile"] = PROFILE_BY_KIND[kind]
    training["optimizer"] = "bfgs"
    training["iterations"] = iterations
    return training


def _adapt_species_count(document, elements):
    """Adapt only the generated YAML copy to the actual training elements.

    The public architecture YAML is a reusable template and must remain
    untouched.  SUS2's grouped YAML needs the number of species used by the
    generated CFG, which is the sorted element list written to parameter.yaml.
    """
    if elements is None:
        return document
    ordered = [str(element) for element in elements]
    if not ordered or len(set(ordered)) != len(ordered):
        raise Sus2TrainingConfigError("Training elements must be a non-empty unique sequence")
    source = deepcopy(document)
    model = _as_mapping(source.get("model"), "model_yaml.model")
    model["species_count"] = len(ordered)
    source["model"] = model
    return source


def _validate_checkpoint_species_count(checkpoint, elements):
    if elements is None:
        return
    from .encode.model_metadata import read_sus2_descriptor_metadata

    expected = len(elements)
    actual = int(read_sus2_descriptor_metadata(checkpoint).species_count)
    if actual != expected:
        raise Sus2TrainingConfigError(
            "SUS2 checkpoint species_count does not match the training dataset: "
            f"checkpoint={actual}, dataset={expected}; use the complete matching element order"
        )


def _output_reference(value, output_dir):
    path = Path(value)
    if path.is_absolute():
        return os.path.relpath(path.resolve(), Path(output_dir).resolve())
    return str(value)


def build_run_document(model_document, *, train_cfg, output_dir, iterations,
                       continuation_model=None, output_name="current.sus2",
                       best_name="best.sus2", final_name=None, elements=None):
    """Build a grouped SUS2 training document."""
    source = _adapt_species_count(model_document, elements)
    if continuation_model is not None:
        source = continuation_document(source, continuation_model)
    model = _as_mapping(source.get("model"), "model_yaml.model")
    kind = str(model.get("kind", "")).strip().lower()
    if kind not in SUPPORTED_KINDS:
        raise Sus2TrainingConfigError(f"unsupported SUS2 model kind: {kind!r}")
    source["schema_version"] = source.get("schema_version", 1)
    source["training"] = _training_block(source, kind, iterations)
    data = deepcopy(source.get("data") or {})
    output_dir = Path(output_dir).resolve()
    train_path = Path(train_cfg).resolve()
    data["train"] = os.path.relpath(train_path, output_dir)
    if continuation_model is not None:
        checkpoint = Path(continuation_model).resolve()
        if checkpoint.suffix.lower() != ".sus2":
            raise Sus2TrainingConfigError(
                f"SUS2 continuation requires a complete .sus2 checkpoint: {checkpoint}"
            )
        checkpoint = _validate_complete_checkpoint(checkpoint)
        _validate_checkpoint_species_count(checkpoint, elements)
        source.pop("model", None)
        data["model"] = os.path.relpath(checkpoint, output_dir)
        source["data"] = data
        source["initialization"] = {"parameters": "same"}
    else:
        data.pop("model", None)
        source["data"] = data
        initialization = deepcopy(source.get("initialization") or {})
        initialization.setdefault("parameters", "random")
        source["initialization"] = initialization

    output = deepcopy(source.get("output") or {})
    output["directory"] = "."
    output["current"] = _output_reference(output_name, output_dir)
    output["best"] = _output_reference(best_name, output_dir)
    if final_name:
        output["final"] = _output_reference(final_name, output_dir)
    source["output"] = output
    return source


def write_run_yaml(path, model_document, **kwargs):
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    document = build_run_document(model_document, **kwargs)
    path.write_text(
        yaml.safe_dump(document, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )
    return path


def grouped_train_command(executable, run_yaml):
    return [str(executable), "train", "--config", str(Path(run_yaml).resolve())]


def check_complete_checkpoint(checkpoint):
    return _validate_complete_checkpoint(checkpoint)


def write_element_sidecars(directory):
    import json
    from .encode.model_metadata import read_sus2_descriptor_metadata
    from .reduce_state import sha, save_json
    directory = Path(directory)
    elements = json.loads((directory/'model_elements.json').read_text())['elements']
    for name in ('current.sus2', 'best.sus2', 'final.sus2'):
        model = directory/name
        if model.is_file():
            if read_sus2_descriptor_metadata(model).species_count != len(elements):
                raise Sus2TrainingConfigError('Output model and complete element mapping disagree')
            save_json(Path(str(model)+'.elements.json'), {'elements': elements, 'model_sha256': sha(model)})


def write_model_element_sidecar(model, elements):
    """Persist the chemical-symbol order paired with one SUS2 checkpoint."""
    from .encode.model_metadata import read_sus2_descriptor_metadata
    from .reduce_state import sha, save_json

    model = Path(model)
    ordered = [str(element) for element in elements]
    if read_sus2_descriptor_metadata(model).species_count != len(ordered):
        raise Sus2TrainingConfigError(
            f"Output model and element mapping disagree: {model}"
        )
    save_json(
        Path(str(model) + ".elements.json"),
        {"elements": ordered, "model_sha256": sha(model)},
    )


def run_grouped_training(executable, run_yaml, *, cwd=None, log_path=None):
    command = grouped_train_command(executable, run_yaml)
    output = None
    if log_path is not None:
        output = Path(log_path).open("w", encoding="utf-8")
    try:
        return subprocess.run(
            command, cwd=cwd, stdout=output, stderr=subprocess.STDOUT, check=True
        )
    finally:
        if output is not None:
            output.close()
