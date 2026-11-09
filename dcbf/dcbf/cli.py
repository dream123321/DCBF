import argparse
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time

from ase.io import iread

from .cli_defaults import defaults_epilog, get_section_defaults
from .das.gen_while_loop import check_run_position, gen_while_loop, mkdir
from .das.logger import format_log_context

from .bootstrap import WorkspaceBootstrapper
from .cli_examples import (
    advanced_command_example_epilog,
    append_command_example,
    top_level_epilog,
)
from .coverage_pca import add_coverage_pca_subparser, handle_coverage_pca_command
from .high_precision_cli import (
    add_high_precision_subparsers,
    handle_efs_distri_command,
    handle_plot_errors_command,
    handle_predict_xyz_command,
    handle_train_command,
)
from .mp_search import handle_mp_search_command
from .path_names import DFT_WORK_DIR
from .raw_dft_archive import (
    extract_raw_dft_archive,
    pack_raw_dft_directory,
    verify_raw_dft_archive,
)
from .runtime_config import build_md_loop_schedule, load_json_config

MANAGED_RUN_CHILD_FLAG = "--managed-run-child"
PID_FILE_NAME = "pid.txt"
LOG_FILE_NAME = "logout"
APP_LOG_FILE_NAME = "app.log"
ADVANCED_TOP_LEVEL_COMMANDS = {"run-generation", "benchmark-selection", "calibrate-selection"}
REDUCE_WORKER_ENV = "DCBF_REDUCE_WORKER"


def _required_epilog(required_text, defaults_section=None, command=None):
    sections = [f"Required input:\n  {required_text}"]
    if defaults_section:
        sections.append(defaults_epilog(defaults_section))
    text = "\n\n".join(sections)
    return append_command_example(text, command) if command else text


def _resolve_run_dir_from_config(config_path):
    config_path = Path(config_path).resolve()
    raw_config = load_json_config(config_path)
    normalized = WorkspaceBootstrapper.normalize_config_layout(raw_config)
    return WorkspaceBootstrapper.resolve_path(config_path.parent, normalized.get("run_dir", ".")).resolve()


def _resolve_reduce_work_dir(config_path):
    config_path = Path(config_path).resolve()
    raw_config = load_json_config(config_path)
    reduce_config = raw_config.get("reduce", {})
    raw_path = Path(reduce_config.get("work_dir", ".dcbf_reduce_work"))
    return raw_path.resolve() if raw_path.is_absolute() else (config_path.parent / raw_path).resolve()


def _resolve_kill_target_dir(target):
    if target is None:
        return Path.cwd().resolve()
    path = Path(target)
    if path.exists() and path.is_dir():
        return path.resolve()
    return _resolve_run_dir_from_config(path)


def _pid_file_path(run_dir):
    return Path(run_dir) / PID_FILE_NAME


def _log_file_path(run_dir):
    return Path(run_dir) / LOG_FILE_NAME


def _app_log_file_path(run_dir):
    return Path(run_dir) / APP_LOG_FILE_NAME


def _append_pid(pid_path, pid):
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    with open(pid_path, "a", encoding="utf-8") as handle:
        handle.write(f"{pid}\n")


def _read_last_pid(pid_path):
    if not pid_path.exists():
        return None
    for line in reversed(pid_path.read_text(encoding="utf-8", errors="ignore").splitlines()):
        text = line.strip()
        if not text:
            continue
        try:
            return int(text.split()[0])
        except ValueError:
            continue
    return None


def _is_process_running(pid):
    if pid is None or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    if os.name != "nt":
        result = subprocess.run(["ps", "-p", str(pid), "-o", "stat="], capture_output=True, text=True, check=False)
        if result.returncode == 0:
            state = result.stdout.strip()
            if state.startswith("Z"):
                return False
    return True


def _process_command_line(pid):
    if not _is_process_running(pid):
        return ""
    if os.name == "nt":
        command = [
            "powershell",
            "-NoProfile",
            "-Command",
            f"(Get-CimInstance Win32_Process -Filter \"ProcessId = {pid}\").CommandLine",
        ]
    else:
        command = ["ps", "-p", str(pid), "-o", "command="]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    return result.stdout.strip()


def _is_managed_dcbf_process(pid):
    return MANAGED_RUN_CHILD_FLAG in _process_command_line(pid)


def _list_posix_descendants(root_pid):
    result = subprocess.run(["ps", "-eo", "pid=,ppid="], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return []
    children_by_parent = {}
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        pid, ppid = int(parts[0]), int(parts[1])
        children_by_parent.setdefault(ppid, []).append(pid)

    descendants = []
    stack = list(children_by_parent.get(root_pid, []))
    while stack:
        current = stack.pop()
        if current in descendants:
            continue
        descendants.append(current)
        stack.extend(children_by_parent.get(current, []))
    return descendants


def _kill_process_tree(pid, timeout_seconds=8.0):
    if not _is_process_running(pid):
        return True

    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], check=False, capture_output=True, text=True)
        return not _is_process_running(pid)

    descendants = _list_posix_descendants(pid)
    process_group_id = None
    try:
        process_group_id = os.getpgid(pid)
    except ProcessLookupError:
        return True

    def _send(sig):
        if process_group_id is not None:
            try:
                os.killpg(process_group_id, sig)
            except (ProcessLookupError, PermissionError):
                pass
        for child_pid in sorted(descendants + [pid], reverse=True):
            try:
                os.kill(child_pid, sig)
            except (ProcessLookupError, PermissionError):
                continue

    _send(signal.SIGTERM)
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if not _is_process_running(pid):
            return True
        time.sleep(0.25)

    _send(signal.SIGKILL)
    deadline = time.time() + 4.0
    while time.time() < deadline:
        if not _is_process_running(pid):
            return True
        time.sleep(0.25)
    return not _is_process_running(pid)


def _prompt_yes_no(message):
    if not sys.stdin.isatty():
        raise RuntimeError(message)
    answer = input(message).strip().lower()
    return answer in {"y", "yes"}


def _count_xyz_structures(path):
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return 0
    return sum(1 for _ in iread(str(path), index=":"))


def _count_main_scf_structures(run_dir, main_index):
    main_dir = Path(run_dir) / f"main_{main_index}"
    if not main_dir.exists():
        return 0
    total = 0
    for gen_dir in sorted(
        [path for path in main_dir.iterdir() if path.is_dir() and path.name.startswith("gen_")],
        key=lambda path: int(path.name.replace("gen_", "")),
    ):
        total += _count_xyz_structures(gen_dir / DFT_WORK_DIR / "scf_filter.xyz")
    return total


def _emit_run_summary(run_dir, message, context=None):
    timestamp = time.strftime("%Y-%m-%d %H:%M")
    prefix = f"{context} " if context else ""
    line = f"{timestamp} INFO: {prefix}{message}"
    print(line)
    app_log = _app_log_file_path(run_dir)
    app_log.parent.mkdir(parents=True, exist_ok=True)
    with open(app_log, "a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _example_sample_candidates(root: Path):
    root = Path(root).resolve()
    yield root / "example" / "sample"
    yield root / "source" / "DCBF" / "example" / "sample"
    yield root / "source" / "OCBF_stability" / "example" / "sample"


def _resolve_example_sample_dir():
    checked = set()

    def _should_check(candidate: Path):
        try:
            resolved = candidate.resolve()
        except OSError:
            resolved = candidate
        if resolved in checked:
            return False
        checked.add(resolved)
        return True

    for env_name in ("DCBF_ROOT", "DCBF_HOME"):
        raw_root = os.environ.get(env_name)
        if not raw_root:
            continue
        for candidate in _example_sample_candidates(Path(raw_root)):
            if _should_check(candidate) and candidate.is_dir():
                return candidate

    for parent in Path(__file__).resolve().parents:
        for candidate in _example_sample_candidates(parent):
            if _should_check(candidate) and candidate.is_dir():
                return candidate

    checked_text = "\n".join(str(path) for path in sorted(checked))
    raise FileNotFoundError(
        "Could not locate the bundled example/sample directory. Checked:\n"
        f"{checked_text}"
    )


def handle_create_init_command(_args):
    sample_dir = _resolve_example_sample_dir()
    target_dir = Path.cwd().resolve()
    source_items = sorted(sample_dir.iterdir(), key=lambda path: path.name)
    if not source_items:
        raise FileNotFoundError(f"No template files found under {sample_dir}")

    conflicts = [item.name for item in source_items if (target_dir / item.name).exists()]
    if conflicts:
        conflict_text = ", ".join(conflicts)
        raise FileExistsError(
            f"dcbf create-init aborted because these paths already exist in {target_dir}: {conflict_text}"
        )

    for source_item in source_items:
        destination = target_dir / source_item.name
        if source_item.is_dir():
            shutil.copytree(source_item, destination)
        else:
            shutil.copy2(source_item, destination)
        print(f"copied: {destination}")

    print(f"initialized_from: {sample_dir}")
    return 0


def handle_raw_dft_command(args):
    if args.raw_dft_action == "pack":
        result = pack_raw_dft_directory(args.directory, output_path=args.output)
        print(f"Directory archived: {result['source_dir']}")
        print(f"archive: {result['archive']}")
        print(f"format: {result['archive_format']}")
        print(f"size_bytes: {result['size_bytes']}")
        print(f"sha256: {result['sha256']}")
        print(f"files: {len(result['files'])}")
        return 0
    if args.raw_dft_action == "verify":
        result = verify_raw_dft_archive(args.archive)
        print(f"Archive verified: {result['archive']}")
        print(f"format: {result['archive_format']}")
        print(f"size_bytes: {result['size_bytes']}")
        print(f"sha256: {result['sha256']}")
        print(f"files: {len(result['files'])}")
        if result["manifest"]:
            status = "matched" if result["manifest_matched"] else "not matched"
            print(f"manifest: {result['manifest']} ({status})")
        return 0
    if args.raw_dft_action == "extract":
        result = extract_raw_dft_archive(args.archive, output_dir=args.output_dir)
        print(f"Archive extracted: {result['archive']}")
        print(f"output_dir: {result['output_dir']}")
        print(f"sha256: {result['sha256']}")
        print(f"files: {len(result['files'])}")
        return 0
    raise ValueError(f"Unsupported raw-dft action: {args.raw_dft_action}")


class DCBFApplication:
    def run_from_config(self, config_path):
        from .artifact_bundle import ArtifactBundler
        from .candidate_pool import CandidatePool
        from .high_precision_training import HighPrecisionTrainer
        from .training_dataset import TrainingDatasetStore

        config_path = Path(config_path).resolve()
        bootstrapper = WorkspaceBootstrapper(config_path)
        training_cfg = bootstrapper.config.get("training", {})

        run_dir, xyz_input, elements, _, should_pause_after_build = bootstrapper.prepare_workspace()
        training_store = TrainingDatasetStore(run_dir, bootstrapper.config)
        training_store.ensure_initial(xyz_input)
        if should_pause_after_build:
            ArtifactBundler(bootstrapper.config, run_dir, bootstrapper.config_path).collect()
            return run_dir

        workflow = bootstrapper.config["workflow"]
        parameter = WorkspaceBootstrapper.apply_parameter_defaults(dict(bootstrapper.config.get("parameter", {})))
        main_loop_npt = workflow.get("main_loop_npt")
        main_loop_nvt = workflow.get("main_loop_nvt")
        loop_schedule = build_md_loop_schedule(main_loop_npt, main_loop_nvt)
        last_schedule = loop_schedule[-1] if loop_schedule else {"npt": None, "nvt": None}
        workflow_context = format_log_context(
            f"main_{max(len(loop_schedule) - 1, 0)}",
            npt=last_schedule.get("npt"),
            nvt=last_schedule.get("nvt"),
        )

        start_position, gen_num, main_num = check_run_position(str(run_dir), loop_schedule)
        sleep_time = workflow.get("sleep_time", 10)
        max_gen = workflow.get("max_gen", 10)

        last_config_mtime = config_path.stat().st_mtime_ns
        startup_dataset = bootstrapper.config.get("dataset")
        startup_run_dir = bootstrapper.config.get("run_dir")
        startup_workflow = bootstrapper.config.get("workflow")

        def _reload_runtime_config():
            nonlocal bootstrapper, training_cfg, last_config_mtime
            try:
                mtime = config_path.stat().st_mtime_ns
            except OSError:
                return False
            if mtime == last_config_mtime:
                return False
            try:
                updated = WorkspaceBootstrapper(config_path)
            except Exception as exc:
                last_config_mtime = mtime
                _emit_run_summary(
                    run_dir,
                    f"[config.reload.warning] invalid JSON/config ignored: {exc}",
                    workflow_context,
                )
                return False
            if (
                updated.config.get("dataset") != startup_dataset
                or updated.config.get("run_dir") != startup_run_dir
            ):
                last_config_mtime = mtime
                _emit_run_summary(
                    run_dir,
                    "[config.reload.warning] dataset/run_dir changes require a restart; hot reload skipped",
                    workflow_context,
                )
                return False
            try:
                updated.refresh_runtime_config(run_dir, elements, xyz_input)
            except Exception as exc:
                last_config_mtime = mtime
                _emit_run_summary(
                    run_dir,
                    f"[config.reload.warning] failed to refresh runtime config: {exc}",
                    workflow_context,
                )
                return False
            bootstrapper = updated
            training_cfg = updated.config.get("training", {})
            last_config_mtime = mtime
            _emit_run_summary(
                run_dir,
                "[config.reload] main JSON changed; refreshed init/parameter.yaml and "
                "dcbf.runtime.json for the next generation",
                workflow_context,
            )
            if updated.config.get("workflow") != startup_workflow:
                _emit_run_summary(
                    run_dir,
                    "[config.reload.warning] workflow/main-loop changes are not hot-reloaded; "
                    "restart the run to apply them",
                    workflow_context,
                )
            return True

        for index in range(main_num, len(loop_schedule)):
            npt = loop_schedule[index]["npt"]
            nvt = loop_schedule[index]["nvt"]
            start_position, gen_num, new_main_num = check_run_position(str(run_dir), loop_schedule)
            gen_while_loop(
                str(run_dir), npt, nvt, start_position, gen_num, sleep_time, max_gen,
                reload_config=_reload_runtime_config,
            )
            total_scf_structures = _count_main_scf_structures(run_dir, index)
            _emit_run_summary(
                run_dir,
                f"[workflow.summary] main_{index} completed. total_scf_structures={total_scf_structures}",
                format_log_context(f"main_{index}", npt=npt, nvt=nvt),
            )
            if new_main_num < len(loop_schedule) - 1:
                next_path = run_dir / ("main_" + str(new_main_num + 1)) / "gen_0"
                mkdir(str(next_path))

        candidate_pool = CandidatePool(run_dir)
        remaining_candidates = candidate_pool.count()
        if remaining_candidates > 0:
            _emit_run_summary(
                run_dir,
                f"[workflow.summary] final loop reached with remaining candidate structures. dropped={remaining_candidates}",
                workflow_context,
            )
            candidate_pool.clear()

        WorkspaceBootstrapper.export_final_xyz(str(run_dir), bootstrapper.config)
        output_xyz_name = workflow.get("output_xyz_name", "sampled_training_dataset.xyz")
        total_sampled_structures = _count_xyz_structures(run_dir / output_xyz_name)
        _emit_run_summary(
            run_dir,
            f"[workflow.summary] sampling completed. total_structures={total_sampled_structures} output_xyz={run_dir / output_xyz_name}",
            workflow_context,
        )
        if training_cfg.get("enabled") or bool(
            (training_cfg.get("final") or {}).get("enabled", False)
        ):
            trainer = HighPrecisionTrainer(bootstrapper.config, bootstrapper.config_path, run_dir)
            trainer.run()
        coverage_cfg = dict(bootstrapper.config.get("coverage_plot") or {})
        if coverage_cfg.get("enabled"):
            try:
                from .coverage_pca import run_coverage_pca_from_config

                # Run after optional final training so the automatic plot uses
                # the newest complete checkpoint in the workspace.
                run_coverage_pca_from_config(bootstrapper.config, run_dir)
                _emit_run_summary(
                    run_dir,
                    "[coverage_pca] SUS2 PCA coverage analysis completed",
                    workflow_context,
                )
            except Exception as exc:
                _emit_run_summary(run_dir, f"[coverage_pca.warning] {exc}", workflow_context)
        ArtifactBundler(bootstrapper.config, run_dir, bootstrapper.config_path).collect()
        training_store.mark_complete_and_cleanup()
        return run_dir

    @staticmethod
    def run_generation(workspace):
        from .generation import GenerationRunner

        runner = GenerationRunner(workspace)
        runner.run()
        return runner.workspace

    @staticmethod
    def benchmark_selection(output_path):
        from .selection.benchmark import SelectionBenchmark

        benchmark = SelectionBenchmark()
        output_path, results = benchmark.write_report(output_path)
        return output_path, results

    @staticmethod
    def calibrate_selection(output_path, cases):
        from .selection.calibrate import SelectionCalibrator

        calibrator = SelectionCalibrator()
        output_path, results = calibrator.write_report(output_path, cases=cases)
        return output_path, results

    @staticmethod
    def reduce_from_config(config_path):
        from .reduce import DCBFReducer

        config_path = Path(config_path).resolve()
        if os.environ.get(REDUCE_WORKER_ENV) == "1":
            reducer = DCBFReducer(config_path)
            return reducer.run()

        from .memory_guard import run_monitored_generation

        work_dir = _resolve_reduce_work_dir(config_path)
        work_dir.mkdir(parents=True, exist_ok=True)
        child_environment = os.environ.copy()
        child_environment[REDUCE_WORKER_ENV] = "1"
        command = [sys.executable, "-m", "dcbf.cli", "reduce", str(config_path)]
        run_monitored_generation(command, work_dir, poll_seconds=2, env=child_environment)
        return None

    def run_from_config_managed(self, config_path):
        config_path = Path(config_path).resolve()
        run_dir = _resolve_run_dir_from_config(config_path)
        run_dir.mkdir(parents=True, exist_ok=True)
        pid_path = _pid_file_path(run_dir)
        log_path = _log_file_path(run_dir)
        last_pid = _read_last_pid(pid_path)

        if last_pid is not None and _is_process_running(last_pid):
            managed = _is_managed_dcbf_process(last_pid)
            descriptor = "managed DCBF run" if managed else "running process"
            if not _prompt_yes_no(
                f"Detected {descriptor} PID {last_pid}. Kill it and start a new run? [y/N]: "
            ):
                print(f"Kept existing PID {last_pid}. No new run was started.")
                return 1
            if not _kill_process_tree(last_pid):
                raise RuntimeError(f"Failed to terminate PID {last_pid} and its child processes cleanly")

        child_argv = [sys.executable, "-m", "dcbf.cli", "run", str(config_path), MANAGED_RUN_CHILD_FLAG]

        log_handle = open(log_path, "a", encoding="utf-8")
        try:
            if os.name == "nt":
                creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
                process = subprocess.Popen(
                    child_argv,
                    cwd=str(run_dir),
                    stdin=subprocess.DEVNULL,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    creationflags=creationflags,
                    close_fds=True,
                )
            else:
                process = subprocess.Popen(
                    child_argv,
                    cwd=str(run_dir),
                    stdin=subprocess.DEVNULL,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    close_fds=True,
                )
        finally:
            log_handle.close()

        _append_pid(pid_path, process.pid)
        print(f"Started background DCBF run PID {process.pid}")
        print(f"log: {log_path}")
        print(f"pid history: {pid_path}")
        return 0

    def kill_managed_run(self, target=None):
        run_dir = _resolve_kill_target_dir(target)
        pid_path = _pid_file_path(run_dir)
        last_pid = _read_last_pid(pid_path)
        if last_pid is None:
            print(f"No PID found in {pid_path}")
            return 1
        if not _is_process_running(last_pid):
            print(f"PID {last_pid} from {pid_path} is not running")
            return 1
        if not _kill_process_tree(last_pid):
            raise RuntimeError(f"Failed to terminate PID {last_pid} and its child processes cleanly")
        print(f"Killed PID {last_pid}")
        print(f"pid history: {pid_path}")
        return 0


def build_parser(include_advanced_commands=True):
    parser = argparse.ArgumentParser(
        prog="dcbf",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=top_level_epilog(include_advanced_commands),
    )
    parser.add_argument(
        "--traceback",
        action="store_true",
        help="print the full traceback when a command fails",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    add_high_precision_subparsers(subparsers)
    add_coverage_pca_subparser(subparsers)

    subparsers.add_parser(
        "create-init",
        help="copy the bundled example/sample contents into the current directory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_required_epilog(
            "no positional input; run inside the target directory. Existing template files must not already exist.",
            command="create-init",
        ),
    )

    mp_search_defaults = get_section_defaults("mp_search")
    mp_search_parser = subparsers.add_parser(
        "mp-search",
        help="fetch Materials Project structures for element combinations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_required_epilog(
            "one or more element symbols; API key via --api-key or `dcbf mp-search set api_key=...`.",
            "mp_search",
            "mp-search",
        ),
    )
    mp_search_parser.add_argument("items", nargs="*", help='element symbols, or `set key=value ...`')
    mp_search_parser.add_argument(
        "--api-key",
        dest="api_key",
        help=f"Materials Project API key (current: {mp_search_defaults['api_key']})",
    )
    mp_search_parser.add_argument(
        "--output-dir",
        dest="output_dir",
        help=f"root directory for combination outputs (current: {mp_search_defaults['output_dir']})",
    )
    mp_search_parser.add_argument(
        "--csv-name",
        dest="csv_name",
        help=f"summary CSV filename written inside each combination directory (current: {mp_search_defaults['csv_name']})",
    )

    run_parser = subparsers.add_parser(
        "run",
        help="prepare and launch a DCBF workflow from one JSON file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_required_epilog("exactly one DCBF JSON config file.", command="run"),
    )
    run_parser.add_argument("config", help="path to the JSON config file")
    run_parser.add_argument("--foreground", action="store_true", help="run in the foreground without managed background mode")
    run_parser.add_argument(MANAGED_RUN_CHILD_FLAG, action="store_true", help=argparse.SUPPRESS)

    if include_advanced_commands:
        generation_parser = subparsers.add_parser(
            "run-generation",
            help="run one generated workspace iteration",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=advanced_command_example_epilog("run-generation"),
        )
        generation_parser.add_argument("--workspace", default=".", help="generation workspace path")

    reduce_parser = subparsers.add_parser(
        "reduce",
        help="reduce redundant database structures from one JSON file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_required_epilog(
            "exactly one reduce JSON config file. "
            "Use reduce.selection_mode=candidate_only for candidate-only self-reduction, "
            "or reduce.selection_mode=reference_guided to select new structures against an existing reference set. "
            "reduce.xyz_io_mode accepts fast_extxyz, auto, or ase; fast_extxyz natively "
            "supports non-periodic EXTXYZ without Lattice (the CFG omits Supercell). "
            "Inputs with other schemas the fast path cannot interpret use ASE fallback. "
            "Reference-guided chunk_size defaults to 1000000; reduce.starting_model_path "
            "must point to a SUS2-SH .sus2 model, or use builtin:sus2_87 for the bundled 87-element model. "
            "Optional reduce.run_scf_labeling labels accepted candidates using dft and scheduler settings. "
            "reduce.update_model_after_selection continues the checkpoint after each accepted round. "
            "reduce.selection_budget_by_coverage and reduce.coverage_target_by_stage drive multi-round update/SCF: "
            "each round uses the first unmet stage budget, and the final target stops the run with coverage_target_reached; "
            "reduce.max_update_rounds is only a safety limit. max_total_new_structures is removed (deprecated and ignored). "
            "Every round is retained under work_dir/updates. "
            "Outputs are selected_structures_xyz and remaining_candidates_xyz.",
            command="reduce",
        ),
    )
    reduce_parser.add_argument("config", help="path to the JSON config file")

    raw_dft_parser = subparsers.add_parser(
        "raw-dft",
        help="pack, verify, or safely extract raw DFT files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  dcbf raw-dft pack task_1\n"
            "  dcbf raw-dft verify summary_bundle/raw_dft/vasp/main_0/gen_0/dir_1/task_1.tar.zst\n"
            "  dcbf raw-dft extract summary_bundle/raw_dft/vasp/main_0/gen_0/dir_1/task_1.tar.zst "
            "--output-dir restored/task_1"
        ),
    )
    raw_dft_subparsers = raw_dft_parser.add_subparsers(
        dest="raw_dft_action",
        required=True,
    )
    raw_dft_pack_parser = raw_dft_subparsers.add_parser(
        "pack",
        help="archive every regular file in one directory with bundled zstd -19 -T1",
    )
    raw_dft_pack_parser.add_argument(
        "directory",
        help="directory to archive in full",
    )
    raw_dft_pack_parser.add_argument(
        "--output",
        help="new .tar.zst path; defaults to DIRECTORY.tar.zst beside the directory",
    )
    raw_dft_verify_parser = raw_dft_subparsers.add_parser(
        "verify",
        help="verify compression, member safety, SHA-256, and matching manifest metadata",
    )
    raw_dft_verify_parser.add_argument(
        "archive",
        help="path to a task_*.tar.zst or legacy task_*.tar.gz archive",
    )
    raw_dft_extract_parser = raw_dft_subparsers.add_parser(
        "extract",
        help="verify and safely extract a raw DFT archive",
    )
    raw_dft_extract_parser.add_argument(
        "archive",
        help="path to a task_*.tar.zst or legacy task_*.tar.gz archive",
    )
    raw_dft_extract_parser.add_argument(
        "--output-dir",
        help="new extraction directory; defaults beside the archive and must not already exist",
    )

    kill_parser = subparsers.add_parser(
        "kill",
        help="kill the current managed DCBF run from pid.txt",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_required_epilog(
            "optional run directory or run config path; defaults to the current directory.",
            command="kill",
        ),
    )
    kill_parser.add_argument(
        "target",
        nargs="?",
        help="optional run directory or run config path; defaults to the current directory",
    )

    if include_advanced_commands:
        benchmark_parser = subparsers.add_parser(
            "benchmark-selection",
            help="profile the structure-selection hot path",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=advanced_command_example_epilog("benchmark-selection"),
        )
        benchmark_parser.add_argument(
            "--output",
            default="benchmark_outputs/selection_profile.json",
            help="where to write the JSON benchmark report",
        )

        calibrate_parser = subparsers.add_parser(
            "calibrate-selection",
            help="verify exact equality with the original min_cover",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=advanced_command_example_epilog("calibrate-selection"),
        )
        calibrate_parser.add_argument(
            "--output",
            default="benchmark_outputs/selection_calibration.json",
            help="where to write the JSON calibration report",
        )
        calibrate_parser.add_argument(
            "--cases",
            type=int,
            default=120,
            help="number of randomized calibration cases",
        )

    return parser


def _dispatch_command(app, args):
    if args.command == "train":
        return handle_train_command(args)
    if args.command == "efs-distri":
        return handle_efs_distri_command(args)
    if args.command == "predict-xyz":
        return handle_predict_xyz_command(args)
    if args.command == "plot-errors":
        return handle_plot_errors_command(args)
    if args.command == "coverage-pca":
        return handle_coverage_pca_command(args)
    if args.command == "create-init":
        return handle_create_init_command(args)
    if args.command == "mp-search":
        return handle_mp_search_command(args)
    if args.command == "run":
        if args.foreground or getattr(args, "managed_run_child", False) or not sys.stdin.isatty():
            app.run_from_config(args.config)
            return 0
        return app.run_from_config_managed(args.config)
    if args.command == "run-generation":
        app.run_generation(args.workspace)
        return 0
    if args.command == "benchmark-selection":
        app.benchmark_selection(args.output)
        return 0
    if args.command == "calibrate-selection":
        app.calibrate_selection(args.output, args.cases)
        return 0
    if args.command == "reduce":
        app.reduce_from_config(args.config)
        return 0
    if args.command == "raw-dft":
        return handle_raw_dft_command(args)
    if args.command == "kill":
        return app.kill_managed_run(args.target)
    raise ValueError(f"Unsupported command: {args.command}")


def _report_command_error(args, exc):
    print(f"dcbf {args.command}: error: {exc}", file=sys.stderr)
    if getattr(args, "traceback", False) or os.environ.get("DCBF_TRACEBACK") == "1":
        import traceback

        traceback.print_exc()


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv in [["-h"], ["--help"]]:
        build_parser(include_advanced_commands=False).print_help()
        return 0
    if argv in [["-hh"], ["--help-all"]]:
        build_parser(include_advanced_commands=True).print_help()
        return 0

    parser = build_parser()
    args = parser.parse_args(argv)
    app = DCBFApplication()
    try:
        return _dispatch_command(app, args)
    except KeyboardInterrupt:
        print("dcbf: interrupted", file=sys.stderr)
        return 130
    except (
        ValueError,
        RuntimeError,
        FileNotFoundError,
        NotADirectoryError,
        PermissionError,
        subprocess.CalledProcessError,
    ) as exc:
        _report_command_error(args, exc)
        return 1
    except Exception as exc:
        _report_command_error(args, exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
