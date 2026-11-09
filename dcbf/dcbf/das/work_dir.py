import os
import time

from ..path_names import DFT_WORK_DIR, MD_WORK_DIR

COMPLETION_MARKERS = ("__ok__", "__start__", "__failed__")
EMPTY_QUEUE_POLLS = 3

'''获取目录的最大深度'''
def get_directory_depth(directory):
    max_depth = 0
    for root, dirs, _ in os.walk(directory):
        # Hidden work directories (for example .dcbf_md_parts descriptor shards)
        # are not MD case directories and must not define the deepest level.
        dirs[:] = [name for name in dirs if not name.startswith(".")]
        depth = root[len(directory):].count(os.sep)  # 计算当前目录相对于基础目录的层级
        max_depth = max(max_depth, depth)
    return max_depth

'''获取npy和tpye的路径'''
def search_directories_in_directory(directory,depth):
    npy_path = []
    for root, dirs, files in os.walk(directory):
        dirs[:] = [name for name in dirs if not name.startswith(".")]
        temp_depth = root[len(directory):].count(os.sep)
        if temp_depth == depth:
            npy_path.append(root)
    return npy_path

def work_deepest_dir(pwd):
    work = os.path.join(pwd, MD_WORK_DIR)
    depth = get_directory_depth(work)
    calc_list = search_directories_in_directory(work, depth)
    return calc_list

def deepest_dir(pwd,dir):
    work = os.path.join(pwd,dir)
    depth = get_directory_depth(work)
    calc_list = search_directories_in_directory(work, depth)
    return calc_list

def scf_dir(pwd):
    work = os.path.join(pwd, DFT_WORK_DIR, 'scf', 'filter')
    dir_path = [os.path.join(work,a) for a in os.listdir(work) if a != 'time.txt']
    temp_list = []
    for sub_dir in dir_path:
        temp_list = [os.path.join(sub_dir,a) for a in os.listdir(sub_dir)] + temp_list
    return temp_list

def _collect_overdue_dirs(dirs, pending_warning_hours, now_ts):
    if pending_warning_hours is None:
        return []
    try:
        threshold_hours = float(pending_warning_hours)
    except (TypeError, ValueError):
        return []
    if threshold_hours <= 0:
        return []

    overdue = []
    for directory in dirs:
        ok_file = os.path.join(directory, '__ok__')
        if os.path.exists(ok_file):
            continue
        start_file = os.path.join(directory, '__start__')
        if not os.path.exists(start_file):
            continue
        elapsed_hours = (now_ts - os.path.getmtime(start_file)) / 3600.0
        if elapsed_hours >= threshold_hours:
            overdue.append((directory, elapsed_hours))
    return overdue


def _format_directory_detail(dirs, limit=10):
    detail = ", ".join(str(path) for path in dirs[:limit])
    if len(dirs) > limit:
        detail += f", ... (+{len(dirs) - limit} more)"
    return detail


def _no_completion_markers(dirs):
    """True when none of ``dirs`` shows any sign of a started job."""
    for directory in dirs:
        for marker in COMPLETION_MARKERS:
            if os.path.exists(os.path.join(directory, marker)):
                return False
    return True


def _probe_reports_no_job(probe_result):
    """Interpret a submitted_probe() value; None means "cannot determine".

    False / empty schedule listings mean no scheduler job exists.  Anything
    else (a job id, a non-empty listing, True) means the queue still owns
    work, so the empty-queue fast fail must not trigger.
    """
    if probe_result is None:
        return False
    if isinstance(probe_result, bool):
        return not probe_result
    try:
        return len(probe_result) == 0
    except TypeError:
        return not probe_result


def _timeout_error_message(dirs, waited_seconds, pending_dirs):
    pending = pending_dirs or dirs
    return (
        f"Timed out after {waited_seconds / 3600.0:.2f} h ({waited_seconds:.0f} s) "
        f"waiting for scheduler tasks; "
        f"{len(pending)} of {len(dirs)} directories are unfinished: "
        f"{_format_directory_detail(pending)}"
    )


def _empty_queue_error_message(dirs, poll_seconds, polls):
    return (
        "no scheduler job and no completion markers: "
        f"{len(dirs)} task directories have no __start__/__ok__/__failed__ marker "
        f"and the scheduler queue reported no job for {polls} consecutive "
        f"polls ({polls * poll_seconds}s); submissions were probably lost. "
        f"Directories: {_format_directory_detail(dirs)}"
    )


class SchedulerQueueProbe:
    """Best-effort scheduler liveness probe for check_finish/check_dft_finish.

    ``name_filter`` limits the query to jobs whose name contains that text; when
    it is None every job of the current user counts, so an empty listing means
    "this user owns no scheduler job at all".  Callers that cannot build a
    reliable name filter should leave it None and rely on the conservative
    whole-user query.  Any failure to query the scheduler returns None, which
    disables the empty-queue fast fail instead of raising a false alarm.
    """

    def __init__(self, backend="bsub", name_filter=None, user=None, timeout=60):
        self.backend = "sbatch" if str(backend).lower() == "sbatch" else "bsub"
        self.name_filter = str(name_filter) if name_filter else None
        self.user = user or os.environ.get("USER") or os.environ.get("LOGNAME") or ""
        self.timeout = timeout
        self.last_output = ""

    def _command(self):
        if self.backend == "bsub":
            cmd = ["bjobs", "-a", "-noheader", "-o", "job_name"]
            if self.user:
                cmd.extend(["-u", self.user])
            return cmd
        cmd = ["squeue", "--noheader", "--format=%j"]
        if self.user:
            cmd.extend(["-u", self.user])
        return cmd

    def __call__(self):
        import subprocess

        try:
            result = subprocess.run(
                self._command(), capture_output=True, text=True, timeout=self.timeout
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if result.returncode != 0:
            self.last_output = (result.stderr or "")[-2000:]
            return None
        self.last_output = (result.stdout or "")[-2000:]
        names = [line.strip() for line in (result.stdout or "").splitlines() if line.strip()]
        if self.name_filter is not None:
            names = [name for name in names if self.name_filter in name]
        return names


def build_submitted_probe(scheduler=None, name_filter=None, backend=None):
    """Build the default scheduler probe used by generation.py."""
    if backend is None:
        backend = getattr(scheduler, "submission_backend", None) or "bsub"
    return SchedulerQueueProbe(backend=backend, name_filter=name_filter)


def check_finish(dirs, logger, log, pending_warning_hours=None, timeout_hours=None,
                 poll_seconds=10, submitted_probe=None):
    dirs = list(dirs)
    last_report_signature = None
    started_at = time.monotonic()
    empty_queue_streak = 0
    while True:
        count = 0
        failed_dirs = []
        pending_dirs = []
        now_ts = time.time()
        for a in dirs:
            failed_file = os.path.join(a, '__failed__')
            if os.path.exists(failed_file):
                failed_dirs.append(a)
            ok_file = os.path.join(a, '__ok__')
            if os.path.exists(ok_file):
                count += 1
            else:
                pending_dirs.append(a)
        if failed_dirs:
            detail = ", ".join(failed_dirs[:10])
            if len(failed_dirs) > 10:
                detail += f", ... (+{len(failed_dirs) - 10} more)"
            logger.error("Task failure markers found: %s", detail)
            raise RuntimeError(f"Task failure markers found: {detail}")
        overdue = _collect_overdue_dirs(dirs, pending_warning_hours, now_ts)
        if overdue:
            signature = tuple(sorted(path for path, _ in overdue))
            if signature != last_report_signature:
                detail = ", ".join(f"{path} ({hours:.2f} h)" for path, hours in overdue)
                logger.warning(
                    "SCF tasks exceeding pending_warning_hours=%s: %s",
                    pending_warning_hours,
                    detail,
                )
                last_report_signature = signature
        else:
            last_report_signature = None
        if count == len(dirs):
            logger.info(log)
            break

        waited = time.monotonic() - started_at
        if timeout_hours is not None and waited >= float(timeout_hours) * 3600.0:
            raise RuntimeError(_timeout_error_message(dirs, waited, pending_dirs))

        probe_result = submitted_probe() if submitted_probe is not None else None
        if (_no_completion_markers(pending_dirs)
                and _probe_reports_no_job(probe_result)):
            empty_queue_streak += 1
            if empty_queue_streak >= EMPTY_QUEUE_POLLS:
                raise RuntimeError(
                    _empty_queue_error_message(pending_dirs, poll_seconds, empty_queue_streak)
                )
        else:
            empty_queue_streak = 0
        time.sleep(poll_seconds)

def _read_tail_text(path, max_chars=20000):
    if not os.path.exists(path):
        return ""
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as handle:
            text = handle.read()
    except OSError:
        return ""
    return text[-max_chars:]

def is_dft_invalid_dir(path):
    invalid_file = os.path.join(path, "__cp2k_invalid__")
    if os.path.exists(invalid_file):
        return True

    failed_file = os.path.join(path, "__failed__")
    if not os.path.exists(failed_file):
        return False

    text = "\n".join(
        _read_tail_text(os.path.join(path, name))
        for name in ("logout", "cp2k.out", "cp2k.err")
    ).lower()
    invalid_markers = (
        "invalid",
        "hard_and_fallback_failed",
    )
    return any(marker in text for marker in invalid_markers)

def check_dft_finish(dirs, logger, log, pending_warning_hours=None, timeout_hours=None,
                     poll_seconds=10, submitted_probe=None):
    dirs = list(dirs)
    last_report_signature = None
    started_at = time.monotonic()
    empty_queue_streak = 0
    while True:
        count = 0
        failed_dirs = []
        invalid_dirs = []
        pending_dirs = []
        now_ts = time.time()
        for a in dirs:
            ok_file = os.path.join(a, "__ok__")
            if os.path.exists(ok_file):
                count += 1
                continue
            if is_dft_invalid_dir(a):
                # Invalid-input directories count as completed and never take
                # part in the empty-queue judgement below.
                invalid_dirs.append(a)
                count += 1
                continue
            failed_file = os.path.join(a, "__failed__")
            if os.path.exists(failed_file):
                failed_dirs.append(a)
                continue
            pending_dirs.append(a)

        if failed_dirs:
            detail = ", ".join(failed_dirs[:10])
            if len(failed_dirs) > 10:
                detail += f", ... (+{len(failed_dirs) - 10} more)"
            logger.error("Fatal DFT task failure markers found: %s", detail)
            raise RuntimeError(f"Fatal DFT task failure markers found: {detail}")

        overdue = _collect_overdue_dirs(dirs, pending_warning_hours, now_ts)
        overdue = [(path, hours) for path, hours in overdue if not is_dft_invalid_dir(path)]
        if overdue:
            signature = tuple(sorted(path for path, _ in overdue))
            if signature != last_report_signature:
                detail = ", ".join(f"{path} ({hours:.2f} h)" for path, hours in overdue)
                logger.warning(
                    "SCF tasks exceeding pending_warning_hours=%s: %s",
                    pending_warning_hours,
                    detail,
                )
                last_report_signature = signature
        else:
            last_report_signature = None

        if count == len(dirs):
            if invalid_dirs:
                detail = ", ".join(invalid_dirs[:10])
                if len(invalid_dirs) > 10:
                    detail += f", ... (+{len(invalid_dirs) - 10} more)"
                logger.warning("DFT invalid tasks skipped: %s", detail)
            logger.info(log)
            break

        waited = time.monotonic() - started_at
        if timeout_hours is not None and waited >= float(timeout_hours) * 3600.0:
            raise RuntimeError(_timeout_error_message(dirs, waited, pending_dirs))

        probe_result = submitted_probe() if submitted_probe is not None else None
        if (_no_completion_markers(pending_dirs)
                and _probe_reports_no_job(probe_result)):
            empty_queue_streak += 1
            if empty_queue_streak >= EMPTY_QUEUE_POLLS:
                raise RuntimeError(
                    _empty_queue_error_message(pending_dirs, poll_seconds, empty_queue_streak)
                )
        else:
            empty_queue_streak = 0
        time.sleep(poll_seconds)

#与check_finish区别，不用while循环
def check_scf(pwd):
    count = 0
    dirs = scf_dir(pwd)
    for a in dirs:
        ok_file = os.path.join(a, '__ok__')
        if os.path.exists(ok_file):
            count += 1
    if count == len(dirs):
        return True
    else:
        return False

#每个结构用bsub提交，所以叫bsub_dir，即每个结构的主目录
def bsub_dir(pwd):
    work = os.path.join(pwd, MD_WORK_DIR)
    dirs = [os.path.join(work,f) for f in os.listdir(work) if os.path.isdir(os.path.join(work,f))]
    return dirs

def submit_lammps_task(pwd,logger,method):
    for dir in bsub_dir(pwd):
        os.chdir(dir)
        count_1 = 0
        name = os.path.basename(dir)
        for a in work_deepest_dir(pwd):
            ok_file = os.path.join(a, '__ok__')
            if not os.path.exists(ok_file):
                count_1 += 1
        if count_1 != 0:
            os.system(f'{method}')
            logger.info(f'{name}: Task is submitted')

#检测分歧阈值筛选出来的每个结构的xyz,是否数量为0,为0说明收敛
def check_filter_xyz_0(pwd):
    count = 0
    dirs_2 = bsub_dir(pwd)
    for dir in dirs_2:
        os.chdir(dir)
        if os.path.getsize('filter.xyz') == 0:
            count = count + 1
    if count == len(dirs_2):
        return False
    else:
        return True

def delete_dump(dirs):
    for dir in dirs:
        file = os.path.join(dir,'force.0.dump')
        if os.path.exists(file):
            os.remove(file)
