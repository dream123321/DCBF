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

from .bootstrap import WorkspaceBootstrapper
from .coverage_pca import add_coverage_pca_subparser, handle_coverage_pca_command
from .high_precision_cli import (
    add_high_precision_subparsers,
    handle_efs_distri_command,
    handle_plot_errors_command,
    handle_predict_xyz_command,
    handle_relax_command,
    handle_train_command,
)
from .mp_search import handle_mp_search_command
from .path_names import DFT_WORK_DIR
from .runtime_config import load_json_config

MANAGED_RUN_CHILD_FLAG = "--managed-run-child"
PID_FILE_NAME = "pid.txt"
LOG_FILE_NAME = "logout"
APP_LOG_FILE_NAME = "app.log"
ADVANCED_TOP_LEVEL_COMMANDS = {"run-generation", "benchmark-selection", "calibrate-selection"}


def _required_epilog(required_text, defaults_section=None):
    sections = [f"Required input:\n  {required_text}"]
    if defaults_section:
        sections.append(defaults_epilog(defaults_section))
    return "\n\n".join(sections)


def _resolve_run_dir_from_config(config_path):
    config_path = Path(config_path).resolve()
    raw_config = load_json_config(config_path)
    normalized = WorkspaceBootstrapper.normalize_config_layout(raw_config)
    return WorkspaceBootstrapper.resolve_path(config_path.parent, normalized.get("run_dir", ".")).resolve()


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


def _emit_run_summary(run_dir, message):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"{timestamp} INFO: {message}"
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

    for env_name in ("DCBF_ROOT", "DCBF_V3_ROOT"):
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


class DCBFApplication:
    def run_from_config(self, config_path, prepare_only=False):
        from .artifact_bundle import ArtifactBundler
        from .candidate_pool import CandidatePool
        from .high_precision_training import HighPrecisionTrainer

        bootstrapper = WorkspaceBootstrapper(config_path)
        training_cfg = bootstrapper.config.get("training", {})

        run_dir, _, _, _, should_pause_after_build = bootstrapper.prepare_workspace()
        if prepare_only:
            return run_dir
        if should_pause_after_build:
            ArtifactBundler(bootstrapper.config, run_dir, bootstrapper.config_path).collect()
            return run_dir

        workflow = bootstrapper.config["workflow"]
        parameter = WorkspaceBootstrapper.apply_parameter_defaults(dict(bootstrapper.config.get("parameter", {})))
        main_loop_npt = workflow.get("main_loop_npt")
        main_loop_nvt = workflow.get("main_loop_nvt")
        selected_loop = self._select_loop_reference(main_loop_npt, main_loop_nvt)

        start_position, gen_num, main_num = check_run_position(str(run_dir), selected_loop)
        mode = workflow.get("mode", "full-automatic")
        sleep_time = workflow.get("sleep_time", 10)
        max_gen = workflow.get("max_gen", 10)

        if mode == "semi-automatic":
            for index in range(main_num, len(selected_loop)):
                npt = main_loop_npt[index] if main_loop_npt is not None and index < len(main_loop_npt) else None
                nvt = main_loop_nvt[index] if main_loop_nvt is not None and index < len(main_loop_nvt) else None
                gen_while_loop(str(run_dir), npt, nvt, start_position, gen_num, sleep_time, max_gen)
                total_scf_structures = _count_main_scf_structures(run_dir, index)
                _emit_run_summary(
                    run_dir,
                    f"[workflow.summary] main_{index} completed. total_scf_structures={total_scf_structures}",
                )
        elif mode == "full-automatic":
            for index in range(main_num, len(selected_loop)):
                npt = main_loop_npt[index] if main_loop_npt is not None and index < len(main_loop_npt) else None
                nvt = main_loop_nvt[index] if main_loop_nvt is not None and index < len(main_loop_nvt) else None
                start_position, gen_num, new_main_num = check_run_position(str(run_dir), selected_loop)
                gen_while_loop(str(run_dir), npt, nvt, start_position, gen_num, sleep_time, max_gen)
                total_scf_structures = _count_main_scf_structures(run_dir, index)
                _emit_run_summary(
                    run_dir,
                    f"[workflow.summary] main_{index} completed. total_scf_structures={total_scf_structures}",
                )
                if new_main_num < len(selected_loop) - 1:
                    next_path = run_dir / ("main_" + str(new_main_num + 1)) / "gen_0"
                    mkdir(str(next_path))
        else:
            raise ValueError("workflow.mode must be 'semi-automatic' or 'full-automatic'")

        candidate_pool = CandidatePool(run_dir)
        remaining_candidates = candidate_pool.count()
        if remaining_candidates > 0:
            _emit_run_summary(
                run_dir,
                f"[workflow.summary] final loop reached with remaining candidate structures. dropped={remaining_candidates}",
            )
            candidate_pool.clear()

        WorkspaceBootstrapper.export_final_xyz(str(run_dir), bootstrapper.config)
        output_xyz_name = workflow.get("output_xyz_name", "all_sample_data.xyz")
        total_sampled_structures = _count_xyz_structures(run_dir / output_xyz_name)
        _emit_run_summary(
            run_dir,
            f"[workflow.summary] sampling completed. total_structures={total_sampled_structures} output_xyz={run_dir / output_xyz_name}",
        )
        coverage_cfg = dict(bootstrapper.config.get("coverage_plot") or {})
        if coverage_cfg.get("enabled"):
            try:
                from .coverage_pca import run_coverage_pca_from_config

                run_coverage_pca_from_config(bootstrapper.config, run_dir)
            except Exception as exc:
                _emit_run_summary(run_dir, f"[coverage_pca.warning] {exc}")
        if training_cfg.get("enabled"):
            trainer = HighPrecisionTrainer(bootstrapper.config, bootstrapper.config_path, run_dir)
            trainer.run()
        ArtifactBundler(bootstrapper.config, run_dir, bootstrapper.config_path).collect()
        return run_dir

    @staticmethod
    def _select_loop_reference(main_loop_npt, main_loop_nvt):
        if main_loop_npt is not None:
            return main_loop_npt
        if main_loop_nvt is not None:
            return main_loop_nvt
        raise ValueError("workflow.main_loop_npt and workflow.main_loop_nvt cannot both be null")

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

        reducer = DCBFReducer(config_path)
        return reducer.run()

    def run_from_config_managed(self, config_path, prepare_only=False):
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
        if prepare_only:
            child_argv.append("--prepare-only")

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
    parser = argparse.ArgumentParser(prog="dcbf")
    if not include_advanced_commands:
        parser.epilog = "Use `dcbf -hh` to show advanced commands."
    subparsers = parser.add_subparsers(dest="command", required=True)
    add_high_precision_subparsers(subparsers)
    add_coverage_pca_subparser(subparsers)

    subparsers.add_parser(
        "create-init",
        help="copy the bundled example/sample contents into the current directory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_required_epilog(
            "no positional input; run inside the target directory. Existing template files must not already exist."
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
        epilog=_required_epilog("exactly one DCBF JSON config file."),
    )
    run_parser.add_argument("config", help="path to the JSON config file")
    run_parser.add_argument("--prepare-only", action="store_true", help="only materialize the workspace files")
    run_parser.add_argument("--foreground", action="store_true", help="run in the foreground without managed background mode")
    run_parser.add_argument(MANAGED_RUN_CHILD_FLAG, action="store_true", help=argparse.SUPPRESS)

    if include_advanced_commands:
        generation_parser = subparsers.add_parser("run-generation", help="run one generated workspace iteration")
        generation_parser.add_argument("--workspace", default=".", help="generation workspace path")

    reduce_parser = subparsers.add_parser(
        "reduce",
        help="reduce redundant database structures from one JSON file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_required_epilog(
            "exactly one reduce JSON config file. "
            "Use reduce.mode=candidate_only for candidate-only self-reduction, "
            "or reduce.mode=reference_guided to select new structures against an existing reference set."
        ),
    )
    reduce_parser.add_argument("config", help="path to the JSON config file")

    kill_parser = subparsers.add_parser(
        "kill",
        help="kill the current managed DCBF run from pid.txt",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_required_epilog("optional run directory or run config path; defaults to the current directory."),
    )
    kill_parser.add_argument(
        "target",
        nargs="?",
        help="optional run directory or run config path; defaults to the current directory",
    )

    if include_advanced_commands:
        benchmark_parser = subparsers.add_parser("benchmark-selection", help="profile the structure-selection hot path")
        benchmark_parser.add_argument(
            "--output",
            default="benchmark_outputs/selection_profile.json",
            help="where to write the JSON benchmark report",
        )

        calibrate_parser = subparsers.add_parser("calibrate-selection", help="verify exact equality with the original min_cover")
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
    if args.command == "train":
        return handle_train_command(args)
    if args.command == "relax":
        return handle_relax_command(args)
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
            app.run_from_config(args.config, prepare_only=args.prepare_only)
            return 0
        return app.run_from_config_managed(args.config, prepare_only=args.prepare_only)
        return 0
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
    if args.command == "kill":
        return app.kill_managed_run(args.target)
    parser.error(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
