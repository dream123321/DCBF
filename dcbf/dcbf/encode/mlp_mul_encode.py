import os
from pathlib import Path
import concurrent.futures
import shlex
from .file_conversion import dump2cfg, merge_cfg_out
from .cfg_descriptor_encode import _build_calc_descriptors_command
import subprocess
import tempfile
import threading
import multiprocessing
from .dimension_min_cover import detect_local_cpu_limit, stop_pool
from ..memory_guard import MIB, current_guard, work_memory, stage_progress


def main_dump2cfg(path, cfg_name, volume_guard=None):
    input_path = os.path.join(path, 'force.0.dump')
    output_path = os.path.join(path, cfg_name)
    return dump2cfg(input_path, output_path, volume_guard)


def _build_calc_descriptors_shell(sus2_mlp_exe, mtp_path, md_cfg, md_out, train_env=None):
    command, shell_mode = _build_calc_descriptors_command(
        sus2_mlp_exe,
        mtp_path,
        md_cfg,
        md_out,
        train_env=train_env,
    )
    if shell_mode:
        return command, command
    rendered = " ".join(shlex.quote(str(item)) for item in command)
    return command, rendered


def mul_encode(pwd, mtp_path, dirs, cfg_name, out_name, sus2_mlp_exe, train_env=None, workers=None,
               volume_guards=None, volume_stats=None):
    cpu_limit, _ = detect_local_cpu_limit()
    worker_limit = min(len(dirs), cpu_limit, cpu_limit if workers is None else max(1, int(workers)))
    if not dirs:
        return []
    if volume_guards is None:
        volume_guards = [None] * len(dirs)
    if current_guard() is not None:
        worker_limit = min(worker_limit, max(1, work_memory() // (256 * MIB)))
    stage_progress('md_dump_conversion', input_path=dirs[0], workers=worker_limit)
    results = [None] * len(dirs)
    if worker_limit == 1:
        results = [main_dump2cfg(path, cfg_name, guard) for path, guard in zip(dirs, volume_guards)]
    else:
        before = {p.pid for p in multiprocessing.active_children()}
        executor = concurrent.futures.ProcessPoolExecutor(max_workers=worker_limit)
        children = []
        pending = {}
        iterator = iter(enumerate(zip(dirs, volume_guards)))
        try:
            for _ in range(worker_limit):
                index, (path, guard) = next(iterator)
                pending[executor.submit(main_dump2cfg, path, cfg_name, guard)] = index
            children = [p for p in multiprocessing.active_children() if p.pid not in before]
            while pending:
                done, _ = concurrent.futures.wait(pending, return_when=concurrent.futures.FIRST_COMPLETED)
                for future in done:
                    index = pending.pop(future)
                    try:
                        results[index] = future.result()
                    except Exception as exc:
                        raise RuntimeError(f'MD dump conversion failed for {dirs[index]}: {exc}') from exc
                    item = next(iterator, None)
                    if item is not None:
                        index, (path, guard) = item
                        pending[executor.submit(main_dump2cfg, path, cfg_name, guard)] = index
        except BaseException:
            children = [p for p in multiprocessing.active_children() if p.pid not in before]
            stop_pool(executor, children)
            raise
        else:
            executor.shutdown(wait=True)

    counts = []
    for path, guard, result in zip(dirs, volume_guards, results):
        kept, detail = result
        counts.append(int(kept))
        if volume_stats is not None and guard is not None:
            volume_stats[str(path)] = detail

    commands = []
    for path, count in zip(dirs, counts):
        md_cfg = os.path.join(path, cfg_name)
        md_out = os.path.join(path, out_name)
        if count == 0:
            # Every frame of this case dir was truncated: calc-descriptors cannot read an
            # empty cfg, so leave an empty md.out behind for the merge and skip the case.
            Path(md_out).touch()
            continue
        commands.append(_build_calc_descriptors_shell(sus2_mlp_exe, mtp_path, md_cfg, md_out, train_env=train_env))

    cancelled = threading.Event()
    stage_progress('md_descriptor_encoding', input_path=dirs[0], workers=worker_limit)

    def run_command(item):
        command, rendered = item
        if cancelled.is_set():
            return
        with tempfile.TemporaryFile() as output:
            try:
                process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT)
            except Exception:
                cancelled.set()
                raise
            while process.poll() is None:
                if cancelled.wait(0.2):
                    process.terminate()
                    try:
                        process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=3)
                    return
            if process.returncode:
                cancelled.set()
                output.seek(max(0, output.tell() - 2000))
                raise RuntimeError(f'calc-descriptors failed during mul_encode with exit code {process.returncode}: {rendered}\n'
                                   + output.read().decode('utf-8', 'replace'))

    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_limit) as executor:
        try:
            list(executor.map(run_command, commands))
        finally:
            cancelled.set()

    merge_cfg_out(pwd, dirs, cfg_name, out_name)
    return counts


if __name__ == '__main__':
    pwd = os.getcwd()
    dirs = ''
    mul_encode(pwd, mtp_path, dirs, cfg_name, out_name)
