"""Explicit, hash-verified packaged models and their authoritative type order."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .runtime_paths import deployment_root

BUILTIN_SUS2_87 = "builtin:sus2_87"
LEGACY_BUILTIN_SUS2_87 = "builtin:NEP_data"
BUILTIN_ALIASES = {LEGACY_BUILTIN_SUS2_87: BUILTIN_SUS2_87}


def _registry():
    root = deployment_root()
    if root is None:
        return None, {}
    path = root / "models/registry.json"
    if not path.is_file():
        return root, {}
    return root, json.loads(path.read_text(encoding="utf-8"))["models"]


def resolve_model_path(value, base_dir=None):
    if not value:
        raise ValueError("model_path is required; use builtin:sus2_87 or a complete .sus2 model")
    raw = str(value)
    canonical = BUILTIN_ALIASES.get(raw, raw)
    if canonical.startswith("builtin:"):
        root, registry = _registry()
        key = canonical.removeprefix("builtin:")
        if key not in registry:
            raise ValueError(f"Unknown or unavailable bundled SUS2 model: {canonical}")
        record = registry[key]
        path = (root / record["file"]).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("Bundled model registry contains a path outside the deployment")
        if not path.is_file():
            raise FileNotFoundError(f"Bundled model is missing: {path}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError(f"Bundled model SHA-256 mismatch: {path}")
    else:
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = Path(base_dir or Path.cwd()) / path
        path = path.resolve()
    if path.suffix.lower() != ".sus2":
        raise ValueError("model_path must be a .sus2 model or builtin:sus2_87")
    if not path.is_file():
        raise FileNotFoundError(f"SUS2 model does not exist: {path}")
    return path


def registered_elements(model_path, explicit=None):
    """Recognize model contents, never infer species IDs from a filename/count."""
    _, registry = _registry()
    if not registry:
        return None
    digest = hashlib.sha256(Path(model_path).read_bytes()).hexdigest()
    for record in registry.values():
        if digest == record["sha256"]:
            elements = list(record["elements"])
            if len(elements) != record["species_count"] or len(set(elements)) != len(elements):
                raise ValueError("Bundled model registry contains an invalid species mapping")
            if explicit is not None and list(explicit) != elements:
                raise ValueError("Element order disagrees with the bundled model; omit ele/ele_list to use its verified full mapping")
            return elements
    return None


def check_continuation_allowed(model_path):
    """Honor the packaged model's declared use without modifying its parameters."""
    _, registry = _registry()
    if not registry:
        return
    digest = hashlib.sha256(Path(model_path).read_bytes()).hexdigest()
    for record in registry.values():
        if digest == record['sha256'] and record.get('allow_continuation') is False:
            raise ValueError(
                'This packaged model registry does not allow continuation training'
            )


def continuation_training_settings(model_path):
    """Content-addressed hints; a copied/renamed packaged seed has the same policy."""
    _, registry = _registry()
    digest = hashlib.sha256(Path(model_path).read_bytes()).hexdigest()
    for record in registry.values():
        if digest == record['sha256']:
            return dict(record.get('continuation_training') or {})
    return {}
