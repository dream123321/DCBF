import os
from pathlib import Path
import concurrent.futures
import shlex
from .file_conversion import dump2cfg, dump2cfg_sharded, merge_cfg_out
from .cfg_descriptor_encode import _build_calc_descriptors_command
import subprocess
import tempfile
import threading
import multiprocessing
from .dimension_min_cover import detect_local_cpu_limit, stop_pool
from ..memory_guard import MIB, current_guard, work_memory, stage_progress


def main_dump2cfg(path, cfg_name, volume_guard=None, descriptor_chunk_size=None,
                  elements=None, sort_ele=True):
    input_path = os.path.join(path, 'force.0.dump')
    if descriptor_chunk_size is None:
        output_path = os.path.join(path, cfg_name)
        kept, detail = dump2cfg(input_path, output_path, volume_guard, elements, sort_ele)
        return {
            "kept": int(kept),
            "detail": detail,
            "cfg_paths": [output_path],
            "cfg_counts": [int(kept)],
        }
    part_dir = Path(path) / f".dcbf_{Path(cfg_name).stem}_parts"
    kept, detail, cfg_paths, cfg_counts = dump2cfg_sharded(
        input_path,
        part_dir,
        Path(cfg_name).stem,
        volume_guard,
        descriptor_chunk_size,
        elements,
        sort_ele,
    )
    return {
        "kept": int(kept),
        "detail": detail,
        "cfg_paths": cfg_paths,
        "cfg_counts": cfg_counts,
    }


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
               volume_guards=None, volume_stats=None, merge_outputs=True,
               return_artifacts=False, descriptor_chunk_size=None, elements=None, sort_ele=True):
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
        results = [
            main_dump2cfg(path, cfg_name, guard, descriptor_chunk_size, elements, sort_ele)
            for path, guard in zip(dirs, volume_guards)
        ]
    else:
        before = {p.pid for p in multiprocessing.active_children()}
        executor = concurrent.futures.ProcessPoolExecutor(max_workers=worker_limit)
        children = []
        pending = {}
        iterator = iter(enumerate(zip(dirs, volume_guards)))
        try:
            for _ in range(worker_limit):
                index, (path, guard) = next(iterator)
                pending[executor.submit(
                    main_dump2cfg,
                    path,
                    cfg_name,
                    guard,
                    descriptor_chunk_size,
                    elements,
                    sort_ele,
                )] = index
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
                        pending[executor.submit(
                            main_dump2cfg,
                            path,
                            cfg_name,
                            guard,
                            descriptor_chunk_size,
                            elements,
                            sort_ele,
                        )] = index
        except BaseException:
            children = [p for p in multiprocessing.active_children() if p.pid not in before]
            stop_pool(executor, children)
            raise
        else:
            executor.shutdown(wait=True)

    counts = []
    for path, guard, result in zip(dirs, volume_guards, results):
        counts.append(int(result["kept"]))
        if volume_stats is not None and guard is not None:
            volume_stats[str(path)] = result["detail"]

    commands = []
    cfg_paths = []
    out_paths = []
    for path, result in zip(dirs, results):
        if descriptor_chunk_size is None and int(result["kept"]) == 0:
            # Preserve the historical empty-shard behavior for callers that
            # still request a merged output.
            md_cfg = Path(path) / cfg_name
            md_out = Path(path) / out_name
            cfg_paths.append(md_cfg)
            out_paths.append(md_out)
            md_out.touch()
            continue
        for md_cfg in result["cfg_paths"]:
            md_cfg = Path(md_cfg)
            md_out = md_cfg.with_suffix(".out")
            cfg_paths.append(md_cfg)
            out_paths.append(md_out)
            commands.append(_build_calc_descriptors_shell(
                sus2_mlp_exe,
                mtp_path,
                md_cfg,
                md_out,
                train_env=train_env,
            ))

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

    if merge_outputs and descriptor_chunk_size is None:
        merge_cfg_out(pwd, dirs, cfg_name, out_name)
    if return_artifacts:
        return {
            "counts": counts,
            "cfg_paths": [str(path) for path in cfg_paths],
            "out_paths": [str(path) for path in out_paths],
            "descriptor_chunk_size": descriptor_chunk_size,
        }
    return counts


if __name__ == '__main__':
    pwd = os.getcwd()
    dirs = ''
    mul_encode(pwd, mtp_path, dirs, cfg_name, out_name)
