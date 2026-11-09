"""Resolve the active deployment's CPU programs without site-specific paths."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil


def deployment_root():
    # The imported source wins over an environment left by another deployment.
    for parent in Path(__file__).resolve().parents:
        if (parent / "activate.sh").is_file() and (parent / "runtime/bin").is_dir():
            return parent
    configured = os.environ.get("DCBF_HOME")
    if configured:
        root = Path(configured).expanduser().resolve()
        if (root / "runtime/bin").is_dir():
            return root
    return None


def resolve_program(name, override=None, *, base_dir=None):
    """An explicit invalid override is an error, never a fallback request.

    ``$VAR``/``${VAR}`` references (for example ``${DCBF_HOME}/runtime/bin/x``)
    are expanded before the file check, so a configured override may point into
    the active deployment.  An override whose variables expand to a missing or
    non-executable path still raises instead of falling back.
    """
    root = deployment_root()
    if override:
        raw = os.path.expandvars(str(override))
        if "/" in raw or os.sep in raw:
            path = Path(raw).expanduser()
            if not path.is_absolute():
                path = Path(base_dir or Path.cwd()) / path
        else:
            found = shutil.which(raw)
            path = Path(found) if found else Path(raw)
    elif root is not None:
        path = root / "runtime/bin" / name
    else:
        found = shutil.which(name)
        if not found:
            raise FileNotFoundError(f"Cannot locate {name}; activate a SUS2-SH deployment")
        path = Path(found)
    path = path.absolute()
    if not path.is_file() or not os.access(path, os.X_OK):
        raise FileNotFoundError(f"Invalid {name} executable: {path}")
    return str(path)


def default_md_command():
    return "mpirun -np $NP lmp_mpi"


def validate_md_command(command):
    tokens = shlex.split(str(command))
    if not tokens:
        raise ValueError("scheduler.lmp_exe must be a non-empty launch command")
    launcher = Path(tokens[0]).name
    if launcher in {"mpirun", "mpiexec", "mpiexec.hydra", "srun"}:
        # DCBF's lmp_exe is a launch prefix; -in is appended by the workflow.
        if Path(tokens[-1]).name not in {"lmp_mpi", "lmp-cpu"}:
            resolve_program("lmp_mpi", tokens[-1])
    elif launcher not in {"lmp_mpi", "lmp-cpu"}:
        resolve_program("LAMMPS launcher", tokens[0])
    return str(command)


def runtime_identity():
    root = deployment_root()
    if root is None:
        return {"deployment": None}
    manifest = root / "runtime/COMPONENTS.json"
    result = {"deployment": str(root)}
    if manifest.is_file():
        result["manifest_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
        result["components"] = json.loads(manifest.read_text(encoding="utf-8"))
    return result
