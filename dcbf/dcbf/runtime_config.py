from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
from typing import Optional

from .runtime_paths import default_md_command, deployment_root, resolve_program, validate_md_command


ENGINE_DEFAULT_DFT = {
    "abacus": {
        "dft_env": "",
        "dft_command": "mpirun -n $NP abacus",
    },
    "cp2k": {
        "dft_env": "",
        "dft_command": "mpirun -n $NP cp2k.popt -i cp2k.inp",
    },
    "qe": {
        "dft_env": "",
        "dft_command": "mpirun -n $NP pw.x -in qe.in",
    },
    "vasp": {
        "dft_env": "",
        "dft_command": "mpirun -n $NP vasp_std",
    },
}

ENGINE_LEGACY_KEYS = {
    "abacus": ("abacus_env", "abacus_command"),
    "cp2k": ("cp2k_env", "cp2k_command"),
    "qe": ("qe_env", "qe_command"),
    "vasp": ("vasp_env", "vasp_command"),
}
RUNTIME_CONFIG_NAME = "dcbf.runtime.json"

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

SUBMISSION_BACKEND_ALIASES = {
    "lsf": "bsub",
    "slurm": "sbatch",
}


def build_md_loop_schedule(main_loop_npt, main_loop_nvt):
    loops = {
        "main_loop_npt": main_loop_npt,
        "main_loop_nvt": main_loop_nvt,
    }
    if main_loop_npt is None and main_loop_nvt is None:
        raise ValueError("workflow.main_loop_npt and workflow.main_loop_nvt cannot both be null")

    for name, values in loops.items():
        if values is not None and not isinstance(values, list):
            raise ValueError(f"workflow.{name} must be a list of temperature lists or null")

    if main_loop_npt is not None and main_loop_nvt is not None:
        if len(main_loop_npt) != len(main_loop_nvt):
            raise ValueError(
                "workflow.main_loop_npt and workflow.main_loop_nvt must have the same outer length: "
                f"{len(main_loop_npt)} != {len(main_loop_nvt)}"
            )

    loop_count = len(main_loop_npt if main_loop_npt is not None else main_loop_nvt)
    if loop_count == 0:
        raise ValueError("workflow MD loop schedule cannot be empty")

    schedule = []
    for index in range(loop_count):
        loop_values = {}
        for name, values in loops.items():
            temperatures = [] if values is None else values[index]
            if not isinstance(temperatures, list):
                raise ValueError(
                    f"workflow.{name}[{index}] must be a temperature list; use [] to skip this ensemble"
                )
            for temperature_index, temperature in enumerate(temperatures):
                if isinstance(temperature, bool):
                    raise ValueError(
                        f"workflow.{name}[{index}][{temperature_index}] must be numeric, got {temperature!r}"
                    )
                try:
                    numeric_temperature = float(temperature)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"workflow.{name}[{index}][{temperature_index}] must be numeric, got {temperature!r}"
                    ) from exc
                if not math.isfinite(numeric_temperature):
                    raise ValueError(
                        f"workflow.{name}[{index}][{temperature_index}] must be finite, got {temperature!r}"
                    )
            loop_values[name.removeprefix("main_loop_")] = list(temperatures)

        if not loop_values["npt"] and not loop_values["nvt"]:
            raise ValueError(
                f"workflow.main_loop_npt[{index}] and workflow.main_loop_nvt[{index}] cannot both be empty"
            )
        schedule.append(loop_values)
    return schedule


def _strip_json_comments(text):
    result = []
    index = 0
    length = len(text)
    in_string = False
    escape = False
    while index < length:
        char = text[index]
        nxt = text[index + 1] if index + 1 < length else ""
        if in_string:
            result.append(char)
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == "\"":
                in_string = False
            index += 1
            continue
        if char == "\"":
            in_string = True
            result.append(char)
            index += 1
            continue
        if char == "/" and nxt == "*":
            index += 2
            while index < length - 1 and not (text[index] == "*" and text[index + 1] == "/"):
                index += 1
            index += 2
            continue
        if char == "/" and nxt == "/":
            index += 2
            while index < length and text[index] != "\n":
                index += 1
            continue
        if char == "#":
            index += 1
            while index < length and text[index] != "\n":
                index += 1
            continue
        result.append(char)
        index += 1
    return "".join(result)


def load_json_config(path):
    raw = Path(path).read_text(encoding="utf-8")
    return json.loads(_strip_json_comments(raw))


def save_runtime_config(run_dir, config):
    run_dir = Path(run_dir)
    runtime_path = run_dir / RUNTIME_CONFIG_NAME
    runtime_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return runtime_path


def find_runtime_config(start_path):
    current = Path(start_path).resolve()
    for candidate in [current, *current.parents]:
        path = candidate / RUNTIME_CONFIG_NAME
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find {RUNTIME_CONFIG_NAME} from {start_path}")


def load_runtime_config(start_path):
    return load_json_config(find_runtime_config(start_path))


def normalize_package_activation(value):
    """Use the active deployment variable for a package activate script."""
    text = str(value or "").strip()
    if not text:
        return ""
    pattern = re.compile(
        r"(?P<prefix>(?:^|[;\n])\s*(?:source|\.)\s+)"
        r"(?P<quote>[\"']?)(?P<path>[^\"'\s;]+/activate\.sh)(?P=quote)"
    )

    def replace(match):
        raw_path = match.group("path").replace("\\", "/")
        if "/runtime/" not in raw_path and "DCBF" not in raw_path.upper():
            return match.group(0)
        return f'{match.group("prefix")}"${{DCBF_HOME}}/activate.sh"'

    return pattern.sub(replace, text)


def normalize_scheduler_config(scheduler):
    normalized = dict(scheduler)
    def package_command(value, name):
        if not value:
            return name
        text = str(value).strip().strip('"').strip("'")
        package_suffix = f"/runtime/bin/{name}"
        if (
            text in {name, package_suffix, f"${{DCBF_HOME}}{package_suffix}"}
            or (Path(text).name == name and "/runtime/bin/" in text.replace("\\", "/"))
        ):
            return name
        return text

    normalized["sus2_mlp_exe"] = package_command(normalized.get("sus2_mlp_exe"), "mlp-sus2")
    raw_lmp = normalized.get("lmp_exe")
    raw_lmp_text = str(raw_lmp or "").strip().strip('"').strip("'")
    if raw_lmp and (
        raw_lmp_text in {"lmp_mpi", "${DCBF_HOME}/runtime/bin/lmp_mpi"}
        or (Path(raw_lmp_text).name == "lmp_mpi" and "/runtime/bin/" in raw_lmp_text.replace("\\", "/"))
    ):
        normalized["lmp_exe"] = default_md_command()
    else:
        normalized["lmp_exe"] = (
            validate_md_command(raw_lmp) if raw_lmp else default_md_command()
        )
    for key in ("original_command", "subsequent_command"):
        command = str(normalized.get(key) or "").strip()
        if command and "runtime/bin/mlp-sus2" in command:
            command = re.sub(
                r"[\"']?[^\s\"']*/runtime/bin/mlp-sus2[\"']?",
                "mlp-sus2",
                command,
            )
        if command and ("{sus2_mlp_exe}" in command or "mlp-sus2" in command) and not any(
            launcher in command.split() for launcher in ("mpirun", "mpiexec", "srun")
        ):
            command = 'mpirun -np "${NP:-1}" ' + command
        normalized[key] = command
    normalized.setdefault("original_command", "")
    normalized.setdefault("subsequent_command", "")
    normalized["train_env"] = normalize_package_activation(normalized.get("train_env", ""))
    normalized.setdefault("lmp_env", "")
    if "clean_dft_environment" in normalized:
        raise ValueError(
            "clean_dft_environment has been renamed to "
            "dft_clean_dcbf_environment"
        )
    dft_clean_dcbf_environment = normalized.get("dft_clean_dcbf_environment", False)
    if not isinstance(dft_clean_dcbf_environment, bool):
        raise ValueError("dft_clean_dcbf_environment must be true or false")
    normalized["dft_clean_dcbf_environment"] = dft_clean_dcbf_environment
    scf_cal_engine = str(normalized.get("scf_cal_engine", "abacus")).strip().lower()
    legacy_env_key, legacy_command_key = ENGINE_LEGACY_KEYS.get(scf_cal_engine, (None, None))
    defaults = ENGINE_DEFAULT_DFT.get(scf_cal_engine, {})

    if "dft_env" not in normalized:
        if legacy_env_key and legacy_env_key in normalized:
            normalized["dft_env"] = normalized[legacy_env_key]
        elif "qe_env" in normalized:
            normalized["dft_env"] = normalized["qe_env"]
        else:
            normalized["dft_env"] = defaults.get("dft_env", "")

    if "dft_command" not in normalized:
        if legacy_command_key and legacy_command_key in normalized:
            normalized["dft_command"] = normalized[legacy_command_key]
        elif "qe_command" in normalized:
            normalized["dft_command"] = normalized["qe_command"]
        else:
            normalized["dft_command"] = defaults.get("dft_command", "")

    backend = normalized.get("submission_backend")
    if backend is not None:
        backend = str(backend).strip().lower()
        backend = SUBMISSION_BACKEND_ALIASES.get(backend, backend)
        if backend not in SUBMISSION_BACKENDS:
            raise ValueError(f"submission_backend must be one of {sorted(SUBMISSION_BACKENDS)}, got {backend!r}")
        normalized.update(SUBMISSION_BACKENDS[backend])
        normalized["submission_backend"] = backend
        return normalized

    task_submission_method = normalized.get("task_submission_method")
    start_calc_command = normalized.get("start_calc_command")
    if task_submission_method == "bsub<bsub.lsf" or start_calc_command == "bsub<":
        normalized.update(SUBMISSION_BACKENDS["bsub"])
        normalized["submission_backend"] = "bsub"
    elif task_submission_method == "sbatch bsub.lsf" or start_calc_command == "sbatch":
        normalized.update(SUBMISSION_BACKENDS["sbatch"])
        normalized["submission_backend"] = "sbatch"
    else:
        normalized.update(SUBMISSION_BACKENDS["bsub"])
        normalized["submission_backend"] = "bsub"
    return normalized


@dataclass
class SchedulerSpec:
    train_sus_queue: int
    train_sus_cores: Optional[int]
    train_sus_ptile: Optional[int]
    train_env: str
    sus2_mlp_exe: str
    original_COMMAND: str
    subsequent_COMMAND: str
    lmp_queue: int
    lmp_cores: int
    lmp_ptile: int
    lmp_env: str
    lmp_exe: str
    scf_queue: int
    scf_cores: int
    scf_ptile: int
    scf_cal_engine: str
    submission_backend: str
    start_calc_command: str
    task_submission_method: str
    dft_clean_dcbf_environment: bool
    dft_env: str
    dft_command: str
    bsub_script_train_sus_job_name: str
    bsub_script_lmp_job_name: str
    bsub_script_scf_job_name: str
    bsub_script_train_sus: str
    bsub_script_lmp: str
    bsub_script_scf: str


def _render_optional_env(config, key):
    env = str(config.get(key, "") or "").strip()
    if not env:
        return ""
    return env + "\n"


def _resolve_dft_environment_cleaner():
    candidates = []
    configured_root = os.environ.get("DCBF_HOME")
    if configured_root:
        candidates.append(Path(configured_root) / "runtime" / "clean_dft_environment.sh")

    module_path = Path(__file__).resolve()
    candidates.extend(
        parent / "runtime" / "clean_dft_environment.sh"
        for parent in module_path.parents
    )

    checked = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in checked:
            continue
        checked.append(resolved)
        if resolved.is_file():
            return resolved

    locations = ", ".join(str(path) for path in checked)
    raise FileNotFoundError(
        "Cannot locate runtime/clean_dft_environment.sh; checked: " + locations
    )


def _render_cores_directive(config, cores_key, ptile_key, backend):
    """Resource directives for one script section.

    ``None`` cores means the scheduler default should be used: the request line
    is omitted entirely instead of guessing a node size, and NP falls back to
    the allocation the scheduler actually granted.
    """
    cores = config.get(cores_key)
    lines = []
    if cores is None:
        return lines
    lines.append(f"#SBATCH -n {cores}" if backend == "sbatch" else f"#BSUB -n {cores}")
    ptile = config.get(ptile_key)
    if ptile is not None:
        if backend == "sbatch":
            lines.append(f"#SBATCH --ntasks-per-node={ptile}")
        else:
            lines.append(f'#BSUB -R "span[ptile={ptile}]"')
    return lines


def _render_np_block(config, cores_key, backend):
    """NP from the granted allocation, falling back to the explicit cores value.

    The historical bsub scripts counted ``$LSB_DJOB_HOSTFILE`` lines when cores
    were configured explicitly; that behaviour is preserved.  With cores=None
    the scheduler default applies, so NP uses the allocated slot count.
    """
    cores = config.get(cores_key)
    if backend == "bsub" and cores is not None:
        return "hostfile=`echo $LSB_DJOB_HOSTFILE`\nNP=`cat $hostfile | wc -l`\n"
    if cores is None:
        return "NP=${SLURM_NTASKS:-${LSB_DJOB_NUMPROC:-1}}\n"
    return f"NP=${{SLURM_NTASKS:-{cores}}}\n"


def _render_engine_section(config):
    clean_environment_line = ""
    if config.get("dft_clean_dcbf_environment", False):
        clean_environment = _resolve_dft_environment_cleaner()
        clean_environment_line = f'source "{clean_environment}"\n'
    backend = config["submission_backend"]
    resource_lines = "\n".join(
        _render_cores_directive(config, "scf_cores", "scf_ptile", backend)
    )
    resource_block = f"{resource_lines}\n" if resource_lines else ""
    np_block = _render_np_block(config, "scf_cores", backend)
    if backend == "sbatch":
        return f'''
#SBATCH -p {config["scf_queue"]}
{resource_block}#SBATCH --error=%j.err
#SBATCH --output=%j.out

{np_block}cd ${{SLURM_SUBMIT_DIR:-$PWD}}
{clean_environment_line}{_render_optional_env(config, "dft_env")}COMMAND_std="{config["dft_command"]}"
'''
    return f'''
#BSUB -q {config["scf_queue"]}
{resource_block}#BSUB -e %J.err
#BSUB -o %J.out

{np_block}cd $LS_SUBCWD
{clean_environment_line}{_render_optional_env(config, "dft_env")}COMMAND_std="{config["dft_command"]}"
'''


def _render_train_section(config):
    backend = config["submission_backend"]
    resource_lines = "\n".join(
        _render_cores_directive(config, "train_sus_cores", "train_sus_ptile", backend)
    )
    resource_block = f"{resource_lines}\n" if resource_lines else ""
    np_block = _render_np_block(config, "train_sus_cores", backend)
    if backend == "sbatch":
        return f'''
#SBATCH -p {config["train_sus_queue"]}
{resource_block}#SBATCH --error=%j.err
#SBATCH --output=%j.out
{np_block}cd ${{SLURM_SUBMIT_DIR:-$PWD}}
'''
    return f'''
#BSUB -q {config["train_sus_queue"]}
{resource_block}#BSUB -e %J.err
#BSUB -o %J.out
{np_block}cd $LS_SUBCWD
'''


def _render_lammps_section(config):
    if config["submission_backend"] == "sbatch":
        return f'''
#SBATCH -p {config["lmp_queue"]}
#SBATCH -n {config["lmp_cores"]}
#SBATCH --error=%j.err
#SBATCH --output=%j.out
#SBATCH --ntasks-per-node={config["lmp_ptile"]}

NP=${{SLURM_NTASKS:-{config["lmp_cores"]}}}
cd ${{SLURM_SUBMIT_DIR:-$PWD}}
COMMAND_0="{config["lmp_exe"]}"
'''
    return f'''
#BSUB -q {config["lmp_queue"]}
#BSUB -n {config["lmp_cores"]}
#BSUB -e %J.err
#BSUB -o %J.out
#BSUB -R "span[ptile={config["lmp_ptile"]}]"

hostfile=`echo $LSB_DJOB_HOSTFILE`
NP=`cat $hostfile | wc -l`
cd $LS_SUBCWD
COMMAND_0="{config["lmp_exe"]}"
'''


def build_scheduler_spec(scheduler):
    defaults = {
        "train_sus_queue": 33,
        "train_sus_cores": None,
        "train_sus_ptile": None,
        "lmp_queue": 33,
        "lmp_cores": 40,
        "lmp_ptile": 40,
        "scf_queue": 33,
        "scf_cores": 40,
        "scf_ptile": 40,
        "submission_backend": "bsub",
    }
    config = normalize_scheduler_config({**defaults, **scheduler})
    job_name_prefix = "\n#SBATCH -J " if config["submission_backend"] == "sbatch" else "\n#BSUB -J "
    return SchedulerSpec(
        train_sus_queue=config["train_sus_queue"],
        train_sus_cores=config["train_sus_cores"],
        train_sus_ptile=config["train_sus_ptile"],
        train_env=config["train_env"],
        sus2_mlp_exe=config["sus2_mlp_exe"],
        original_COMMAND=config["original_command"],
        subsequent_COMMAND=config["subsequent_command"],
        lmp_queue=config["lmp_queue"],
        lmp_cores=config["lmp_cores"],
        lmp_ptile=config["lmp_ptile"],
        lmp_env=config["lmp_env"],
        lmp_exe=config["lmp_exe"],
        scf_queue=config["scf_queue"],
        scf_cores=config["scf_cores"],
        scf_ptile=config["scf_ptile"],
        scf_cal_engine=config["scf_cal_engine"],
        submission_backend=config["submission_backend"],
        start_calc_command=config["start_calc_command"],
        task_submission_method=config["task_submission_method"],
        dft_clean_dcbf_environment=config["dft_clean_dcbf_environment"],
        dft_env=config["dft_env"],
        dft_command=config["dft_command"],
        bsub_script_train_sus_job_name=job_name_prefix,
        bsub_script_lmp_job_name=job_name_prefix,
        bsub_script_scf_job_name=job_name_prefix,
        bsub_script_train_sus=_render_train_section(config),
        bsub_script_lmp=_render_lammps_section(config),
        bsub_script_scf=_render_engine_section(config),
    )
