from __future__ import annotations

import re
from pathlib import Path

from .path_names import DFT_WORK_DIR, MD_WORK_DIR


RUNTIME_PATTERN = re.compile(r"total_runtime(?:[:\uFF1A])?\s*([0-9.+\-eE]+)\s*s")
LAMMPS_WALLTIME_PATTERN = re.compile(r"Total wall time:\s*(?:(\d+)-)?(\d+):(\d+):(\d+(?:\.\d+)?)")


def _parse_runtime_file(path: Path):
    path = Path(path)
    runtimes = []
    if not path.exists():
        return runtimes
    text = path.read_text(encoding="utf-8", errors="ignore")
    for match in RUNTIME_PATTERN.finditer(text):
        try:
            runtimes.append(float(match.group(1)))
        except ValueError:
            continue
    return runtimes


def _parse_lammps_walltime_file(path: Path):
    path = Path(path)
    runtimes = []
    if not path.exists():
        return runtimes
    text = path.read_text(encoding="utf-8", errors="ignore")
    for match in LAMMPS_WALLTIME_PATTERN.finditer(text):
        days = int(match.group(1) or 0)
        hours = int(match.group(2))
        minutes = int(match.group(3))
        seconds = float(match.group(4))
        runtimes.append(days * 86400.0 + hours * 3600.0 + minutes * 60.0 + seconds)
    return runtimes


def _iter_sampling_train_dirs(run_dir: Path):
    for train_root in sorted(run_dir.glob("main_*/gen_*/train_mlp")):
        if not train_root.is_dir():
            continue
        for child in sorted(train_root.iterdir()):
            if child.is_dir() and ((child / "bsub.lsf").exists() or (child / "logout").exists()):
                yield child


def _iter_dft_filter_dirs(run_dir: Path):
    for filter_dir in sorted(run_dir.glob(f"**/{DFT_WORK_DIR}/scf/filter")):
        if filter_dir.is_dir():
            yield filter_dir


def _iter_sampling_md_structure_dirs(run_dir: Path):
    for md_root in sorted(run_dir.glob(f"main_*/gen_*/{MD_WORK_DIR}")):
        if not md_root.is_dir():
            continue
        for structure_dir in sorted(md_root.iterdir()):
            if structure_dir.is_dir() and (structure_dir / "bsub.lsf").exists():
                yield structure_dir


def _iter_sampling_md_log_files(run_dir: Path):
    seen = set()
    for structure_dir in _iter_sampling_md_structure_dirs(run_dir):
        for log_path in sorted(structure_dir.glob("**/log.lammps")):
            if not log_path.is_file():
                continue
            key = str(log_path.resolve())
            if key in seen:
                continue
            seen.add(key)
            yield log_path


def _parse_bsub_cores(script_path: Path):
    if not script_path.exists():
        return None
    text = script_path.read_text(encoding="utf-8", errors="ignore")
    patterns = (
        re.compile(r"^#BSUB -n (\d+)$", re.MULTILINE),
        re.compile(r"^#SBATCH -n (\d+)$", re.MULTILINE),
    )
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            return int(match.group(1))
    return None


def _accumulate_category(category_name, task_infos, default_cores):
    total_seconds = 0.0
    counted_tasks = 0
    missing = []

    for task in task_infos:
        runtime_file = Path(task["runtime_file"])
        parser = task.get("parser", _parse_runtime_file)
        runtimes = parser(runtime_file)
        cores = task.get("cores")
        if cores is None:
            cores = default_cores
        if runtimes and cores:
            total_seconds += sum(runtimes) * float(cores)
            counted_tasks += len(runtimes)
        else:
            missing.append(str(runtime_file))

    return {
        "category": category_name,
        "task_count": len(task_infos),
        "counted_count": counted_tasks,
        "core_hours": total_seconds / 3600.0,
        "missing_runtime_records": missing,
    }


def write_core_hours_report(run_dir, config, output_path):
    run_dir = Path(run_dir).resolve()
    output_path = Path(output_path).resolve()
    scheduler = dict(config.get("scheduler", {}))
    training = dict(config.get("training", {}))

    sampling_train_tasks = [
        {
            "runtime_file": path / "time.txt",
            "cores": int(scheduler.get("train_sus_cores", 0) or 0),
        }
        for path in _iter_sampling_train_dirs(run_dir)
    ]

    final_train_root = run_dir / training.get("work_dir", "high_precision_training") / "train_job"
    final_train_cores = _parse_bsub_cores(final_train_root / "bsub.lsf")
    final_train_tasks = []
    if final_train_root.exists():
        final_train_tasks.append(
            {
                "runtime_file": final_train_root / "time.txt",
                "cores": final_train_cores or int(scheduler.get("train_sus_cores", 0) or 0),
            }
        )

    dft_tasks = [
        {
            "runtime_file": filter_dir / "time.txt",
            "cores": int(scheduler.get("scf_cores", 0) or 0),
        }
        for filter_dir in _iter_dft_filter_dirs(run_dir)
    ]
    sampling_md_tasks = [
        {
            "runtime_file": log_path,
            "cores": int(scheduler.get("lmp_cores", 0) or 0),
            "parser": _parse_lammps_walltime_file,
        }
        for log_path in _iter_sampling_md_log_files(run_dir)
    ]

    categories = [
        _accumulate_category(
            "sampling_train_core_hours",
            sampling_train_tasks,
            int(scheduler.get("train_sus_cores", 0) or 0),
        ),
        _accumulate_category(
            "sampling_md_core_hours",
            sampling_md_tasks,
            int(scheduler.get("lmp_cores", 0) or 0),
        ),
        _accumulate_category(
            "final_training_core_hours",
            final_train_tasks,
            int(scheduler.get("train_sus_cores", 0) or 0),
        ),
        _accumulate_category(
            "dft_core_hours",
            dft_tasks,
            int(scheduler.get("scf_cores", 0) or 0),
        ),
    ]

    total_core_hours = sum(item["core_hours"] for item in categories)
    missing_records = []
    for item in categories:
        for record in item["missing_runtime_records"]:
            missing_records.append(f"{item['category']}: {record}")

    lines = []
    for item in categories:
        lines.append(f"{item['category']}: {item['core_hours']:.6f}")
        lines.append(f"  task_count: {item['task_count']}")
        lines.append(f"  counted_count: {item['counted_count']}")
    lines.append(f"total_core_hours: {total_core_hours:.6f}")
    lines.append("missing_runtime_records:")
    if missing_records:
        for record in missing_records:
            lines.append(f"  - {record}")
    else:
        lines.append("  - none")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path
