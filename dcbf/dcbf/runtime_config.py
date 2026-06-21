from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re


ENGINE_DEFAULT_DFT = {
    "abacus": {
        "dft_env": "module load compiler/2022.1.0 mpi/2021.6.0 mkl/2022.2.0\nexport PATH=/work/phy-huangj/apps/il/abacus/3.6.5/bin:$PATH",
        "dft_command": "mpirun -n 40 abacus",
    },
    "cp2k": {
        "dft_env": "module purge\nmodule load cp2k/2024.1_oneapi-e5",
        "dft_command": "mpirun -n 40 cp2k.popt -i cp2k.inp",
    },
    "qe": {
        "dft_env": "module load compiler/2022.1.0 mpi/2021.6.0 mkl/2022.2.0\nexport PATH=/work/phy-huangj/apps/qe/7.5/build-intel/bin:$PATH",
        "dft_command": "mpirun -n 40 pw.x -in qe.in",
    },
    "vasp": {
        "dft_env": "export PATH=/work/phy-huangj/app/vasp.5.4.4/bin:$PATH",
        "dft_command": "mpirun -n 40 vasp_std",
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


def normalize_scheduler_config(scheduler):
    normalized = dict(scheduler)
    normalized.setdefault("train_env", "")
    normalized.setdefault("lmp_env", "")
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
    train_sus_cores: int
    train_sus_ptile: int
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


def _render_engine_section(config):
    if config["submission_backend"] == "sbatch":
        return f'''
#SBATCH -p {config["scf_queue"]}
#SBATCH -n {config["scf_cores"]}
#SBATCH --error=%j.err
#SBATCH --output=%j.out
#SBATCH --ntasks-per-node={config["scf_ptile"]}

NP=${{SLURM_NTASKS:-{config["scf_cores"]}}}
cd ${{SLURM_SUBMIT_DIR:-$PWD}}
{_render_optional_env(config, "dft_env")}COMMAND_std="{config["dft_command"]}"
'''
    return f'''
#BSUB -q {config["scf_queue"]}
#BSUB -n {config["scf_cores"]}
#BSUB -e %J.err
#BSUB -o %J.out
#BSUB -R "span[ptile={config["scf_ptile"]}]"

hostfile=`echo $LSB_DJOB_HOSTFILE`
NP=`cat $hostfile | wc -l`
cd $LS_SUBCWD
{_render_optional_env(config, "dft_env")}COMMAND_std="{config["dft_command"]}"
'''


def _render_train_section(config):
    if config["submission_backend"] == "sbatch":
        return f'''
#SBATCH -p {config["train_sus_queue"]}
#SBATCH -n {config["train_sus_cores"]}
#SBATCH --ntasks-per-node={config["train_sus_ptile"]}
#SBATCH --error=%j.err
#SBATCH --output=%j.out
NP=${{SLURM_NTASKS:-{config["train_sus_cores"]}}}
cd ${{SLURM_SUBMIT_DIR:-$PWD}}
'''
    return f'''
#BSUB -q {config["train_sus_queue"]}
#BSUB -n {config["train_sus_cores"]}
#BSUB -R "span[ptile={config["train_sus_ptile"]}]"
#BSUB -e %J.err
#BSUB -o %J.out
hostfile=`echo $LSB_DJOB_HOSTFILE`
NP=`cat $hostfile | wc -l`
cd $LS_SUBCWD
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
        "train_sus_cores": 40,
        "train_sus_ptile": 40,
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
        dft_env=config["dft_env"],
        dft_command=config["dft_command"],
        bsub_script_train_sus_job_name=job_name_prefix,
        bsub_script_lmp_job_name=job_name_prefix,
        bsub_script_scf_job_name=job_name_prefix,
        bsub_script_train_sus=_render_train_section(config),
        bsub_script_lmp=_render_lammps_section(config),
        bsub_script_scf=_render_engine_section(config),
    )
