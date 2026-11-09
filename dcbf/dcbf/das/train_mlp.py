"""SUS2-only generation training orchestration.

Generation zero creates a model from the architecture YAML. A candidate DFT
batch may then create one continuation checkpoint in that same generation.
Generation boundaries reuse the latest complete checkpoint directly; they do
not create another training task merely to enter the next MD generation.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

from .work_dir import check_finish
from ..path_names import DFT_WORK_DIR, SUS2_MODEL_DIR
from ..sus2_training import (
    DEFAULT_INITIAL_ITERATIONS,
    DEFAULT_SUBSEQUENT_ITERATIONS,
    check_complete_checkpoint,
    load_model_yaml,
    write_run_yaml,
)
from ..training_dataset import TrainingDatasetStore


BATCH_MERGED_MARKER = "__candidate_batch_merged__"
_CANONICAL_CURRENT_MODEL = re.compile(r"^current_\d+\.sus2$")


def training_task_dirs(pwd):
    train_root = Path(pwd) / "train_mlp"
    if not train_root.is_dir():
        return []
    return [
        str(path) for path in sorted(
            (path for path in train_root.iterdir()
             if path.is_dir() and path.name.isdigit()),
            key=lambda path: int(path.name),
        )
    ]


def mkdir(directory):
    Path(directory).mkdir(parents=True, exist_ok=True)


def check(file):
    if not os.path.exists(file):
        sys.exit(f"{file} does not exist")


def _canonical_current_models(directory):
    directory = Path(directory)
    return sorted(
        (path for path in directory.glob("current_*.sus2")
         if path.is_file() and _CANONICAL_CURRENT_MODEL.fullmatch(path.name)),
        key=lambda path: int(path.stem.split("_")[1]),
    )


def _parameter_for(pwd):
    path = Path(pwd) / "parameter.yaml"
    if not path.exists():
        path = Path(pwd).parent.parent / "init" / "parameter.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    model_yaml = data.get("model_yaml")
    if not model_yaml:
        raise ValueError("parameter.yaml is missing model_yaml")
    model_yaml = _resolve_parameter_model_yaml(path, model_yaml)
    document, kind = load_model_yaml(model_yaml)
    training = dict(data.get("sus2_training") or {})
    return data, document, kind, training


def _resolve_parameter_model_yaml(parameter_path, raw_model_yaml):
    """Resolve model YAML copied between init/ and generation directories."""
    raw_path = Path(raw_model_yaml).expanduser()
    if raw_path.is_absolute():
        return raw_path

    parameter_path = Path(parameter_path).resolve()
    roots = [parameter_path.parent]
    roots.extend(parameter_path.parent.parents)
    candidates = []
    for root in roots:
        candidates.append(root / raw_path)
        candidates.append(root / "init" / raw_path)
        if raw_path.parent == Path("."):
            candidates.append(root / "init" / raw_path.name)

    checked = []
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in checked:
            continue
        checked.append(candidate)
        if candidate.is_file():
            return candidate

    rendered = ", ".join(str(candidate) for candidate in checked)
    raise FileNotFoundError(
        f"SUS2 model_yaml does not exist: {raw_model_yaml!r}; checked {rendered}"
    )


def _scheduler_executable(scheduler):
    executable = getattr(scheduler, "sus2_mlp_exe", None)
    if not executable:
        raise ValueError("scheduler.sus2_mlp_exe is required for SUS2 training")
    return str(executable)


def _iterations(training, continuation):
    key = "subsequent_iterations" if continuation else "initial_iterations"
    default = (
        DEFAULT_SUBSEQUENT_ITERATIONS
        if continuation
        else DEFAULT_INITIAL_ITERATIONS
    )
    value = int(training.get(key, default))
    if value <= 0:
        raise ValueError(f"sus2_training.{key} must be positive")
    return value


def _write_task_yaml(task_dir, model_document, *, train_cfg, output_model,
                     continuation_model, iterations, elements):
    task_dir = Path(task_dir)
    task_dir.mkdir(parents=True, exist_ok=True)
    if continuation_model is not None:
        continuation_model = check_complete_checkpoint(continuation_model)
    return write_run_yaml(
        task_dir / "run.yaml",
        model_document,
        train_cfg=train_cfg,
        output_dir=task_dir,
        iterations=iterations,
        continuation_model=continuation_model,
        output_name=str(Path(output_model).resolve()),
        best_name="best.sus2",
        elements=elements,
    )


def _configured_training_command(scheduler, run_yaml, continuation):
    executable = _scheduler_executable(scheduler)
    key = "subsequent_COMMAND" if continuation else "original_COMMAND"
    template = str(getattr(scheduler, key, "") or "").strip()
    if not template:
        return f'mpirun -np "${{NP:-1}}" "{executable}" train --config "{Path(run_yaml).name}"'
    command = template.replace("{sus2_mlp_exe}", f'"{executable}"')
    yaml_path = f'"{Path(run_yaml).name}"'
    command = command.replace("{initial_yaml}", yaml_path)
    command = command.replace("{continuation_yaml}", yaml_path)
    command = command.replace(f" train {yaml_path}", f" train --config {yaml_path}")
    return command


def train_mlp_lsf(current_num, out, run_yaml, pwd, scheduler, continuation=False):
    """Create an LSF script for one grouped SUS2 training task."""
    run_yaml = Path(run_yaml)
    script_text = f"""#!/bin/bash
{scheduler.bsub_script_train_sus_job_name}{current_num}
{scheduler.bsub_script_train_sus}
"""
    train_env = str(getattr(scheduler, "train_env", "") or "").strip()
    if train_env:
        script_text += train_env + "\n"
    command = _configured_training_command(scheduler, run_yaml, continuation)
    script_text += f"""
start_time=$(date +%s.%N)
touch __start__
{command} > logout 2>&1
status=$?
end_time=$(date +%s.%N)
if [ "$status" -eq 0 ]; then touch __ok__; else touch __error__ __failed__; fi
runtime=$(echo "$end_time - $start_time" | bc)
echo "total_runtime: $runtime s" >> time.txt
exit "$status"
"""
    Path(out).write_text(script_text, encoding="utf-8")
    Path(out).chmod(0o755)


def _reset_training_dir_state(train_dir):
    for file_name in ("__start__", "__ok__", "__error__", "__failed__", "logout", "time.txt"):
        path = Path(train_dir) / file_name
        if path.exists():
            path.unlink()


def _reference_models(reference_path):
    # current_N are chronological checkpoints, not an ensemble.
    return _canonical_current_models(Path(reference_path) / SUS2_MODEL_DIR)[-1:]


def _reuse_reference_models(reference_path, destination_path, mlp_nums):
    """Materialize the previous generation's final model for direct MD use.

    A generation boundary is a sampling boundary, not an additional training
    event.  Candidate DFT updates already produce the newest checkpoint in the
    preceding generation; the next generation must consume that checkpoint
    directly.  Normalize its name to current_0.sus2 because the MD templates
    use that stable entry point.
    """
    sources = _reference_models(reference_path)
    if len(sources) < int(mlp_nums):
        raise RuntimeError(
            f"Previous generation has {len(sources)} SUS2 checkpoints, "
            f"expected {mlp_nums}"
        )

    destination_path = Path(destination_path)
    destination_path.mkdir(parents=True, exist_ok=True)
    targets = []
    for index, source in enumerate(sources[:int(mlp_nums)]):
        source = check_complete_checkpoint(source)
        target = destination_path / f"current_{index}.sus2"
        shutil.copy2(source, target)
        source_sidecar = Path(str(source) + ".elements.json")
        if source_sidecar.is_file():
            shutil.copy2(source_sidecar, Path(str(target) + ".elements.json"))
        targets.append(target)
    return targets


def _prepare_tasks(pwd, source_models, *, initial, ele, scheduler):
    pwd = Path(pwd).resolve()
    model_dir = pwd / SUS2_MODEL_DIR
    train_root = pwd / "train_mlp"
    model_dir.mkdir(parents=True, exist_ok=True)
    train_root.mkdir(parents=True, exist_ok=True)
    parameter, model_document, _, training = _parameter_for(pwd)
    train_cfg = TrainingDatasetStore.from_generation(pwd).global_cfg
    sources = list(source_models)
    if initial and not sources:
        sources = [None]
    for index, source in enumerate(sources):
        task_dir = train_root / str(index)
        task_dir.mkdir(parents=True, exist_ok=True)
        _reset_training_dir_state(task_dir)
        target = model_dir / f"current_{index}.sus2"
        if target.exists():
            continue
        _write_task_yaml(
            task_dir,
            model_document,
            train_cfg=train_cfg,
            output_model=target,
            continuation_model=source,
            iterations=_iterations(training, source is not None),
            elements=ele,
        )
        train_mlp_lsf(index, task_dir / "bsub.lsf", task_dir / "run.yaml",
                      pwd, scheduler, continuation=source is not None)
    return len(sources)


def pre_train_mlp(pwd, mlp_nums, ele, ele_model, logger, scheduler):
    training_store = TrainingDatasetStore.from_generation(pwd)
    training_store.ensure_generation_link(pwd)
    pwd = Path(pwd).resolve()
    model_dir = pwd / SUS2_MODEL_DIR
    model_dir.mkdir(parents=True, exist_ok=True)
    existing = _canonical_current_models(model_dir)
    if existing:
        check_complete_checkpoint(existing[-1])
        logger.info("SUS2 model already exists")
        return False

    gen_num = int(pwd.name.replace("gen_", ""))
    main_num = int(pwd.parent.name.replace("main_", ""))
    if gen_num == 0 and main_num == 0:
        _prepare_tasks(
            pwd,
            [None] * max(1, int(mlp_nums)),
            initial=True,
            ele=ele,
            scheduler=scheduler,
        )
    else:
        if gen_num == 0:
            previous_main = pwd.parent.parent / f"main_{main_num - 1}"
            previous_gens = sorted(
                (p for p in previous_main.glob("gen_*") if p.is_dir()),
                key=lambda p: int(p.name.split("_")[1]),
            )
            reference = previous_gens[-1] if previous_gens else None
        else:
            reference = pwd.parent / f"gen_{gen_num - 1}"
        if reference is None:
            raise FileNotFoundError("No previous generation checkpoint for continuation")
        targets = _reuse_reference_models(reference, model_dir, mlp_nums)
        logger.info("SUS2 generation gen_%s reuses the latest checkpoint", gen_num)
        return False
    logger.info("SUS2 generation gen_%s initial training prepared", gen_num)
    return True


def update_mlp_from_current_batch(pwd, mlp_nums, ele, ele_model, logger, scheduler):
    pwd = Path(pwd).resolve()
    train_root = pwd / "train_mlp"
    model_dir = pwd / SUS2_MODEL_DIR
    train_root.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    training_store = TrainingDatasetStore.from_generation(pwd)
    train_cfg = training_store.global_cfg
    training_store.ensure_generation_link(pwd)
    scf_filter_xyz = pwd / DFT_WORK_DIR / "scf_filter.xyz"
    merged_marker = train_root / BATCH_MERGED_MARKER
    check(train_cfg)
    check(scf_filter_xyz)
    if not merged_marker.exists():
        training_store.append_generation(pwd, scf_filter_xyz)
        merged_marker.touch()

    _, model_document, _, training = _parameter_for(pwd)
    sources = _canonical_current_models(model_dir)
    next_index = max(
        (int(path.stem.split("_")[1]) for path in sources), default=-1
    ) + 1
    if not sources:
        raise FileNotFoundError(f"No trained checkpoint available for update: {model_dir}")
    for source in sources[-1:]:
        # Grouped SUS2 refuses reusing best/final/log outputs. Keep each update
        # in a new task directory, preserving the preceding optimization run.
        task_index = next_index
        task_dir = train_root / str(task_index)
        _reset_training_dir_state(task_dir)
        target = model_dir / f"current_{next_index}.sus2"
        next_index += 1
        _write_task_yaml(
            task_dir,
            model_document,
            train_cfg=train_cfg,
            output_model=target,
            continuation_model=source,
            iterations=_iterations(training, True),
            elements=ele,
        )
        train_mlp_lsf(task_index, task_dir / "bsub.lsf", task_dir / "run.yaml",
                      pwd, scheduler, continuation=True)

    start_train(str(pwd), scheduler.task_submission_method, mlp_nums, logger)


def start_train(pwd, task_submission_method, mlp_nums, logger):
    dirs = training_task_dirs(pwd)
    pending = [directory for directory in dirs if not (Path(directory) / "__ok__").exists()]
    if pending:
        phase = "continuation" if any(Path(directory).name != "0" for directory in pending) else "initial"
        logger.info("SUS2 %s training started", phase)
    for directory in pending:
        if not (Path(directory) / "__start__").exists():
            subprocess.run(str(task_submission_method), cwd=directory, shell=True, check=True)
    if pending:
        check_finish(pending, logger, "SUS2 MLIP training completed.")
    for directory in dirs:
        directory = Path(directory)
        document = yaml.safe_load((directory / "run.yaml").read_text())
        # output.current is stored relative to the training task directory;
        # it must not be resolved against the caller's working directory.
        current = Path(document["output"]["current"])
        if not current.is_absolute():
            current = directory / current
        check_complete_checkpoint(current)
    model_dir = Path(pwd) / SUS2_MODEL_DIR
    names = _canonical_current_models(model_dir)
    if not names:
        raise SystemExit("Please check the SUS2 training process! No checkpoint was produced")
    parameter_path = Path(pwd) / "parameter.yaml"
    if not parameter_path.exists():
        parameter_path = Path(pwd).parent.parent / "init" / "parameter.yaml"
    parameter = yaml.safe_load(parameter_path.read_text(encoding="utf-8")) or {}
    elements = list(parameter.get("ele") or [])
    if elements:
        from ..sus2_training import write_model_element_sidecar
        for model in names:
            write_model_element_sidecar(model, elements)
