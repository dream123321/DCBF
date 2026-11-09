import os

from tqdm import tqdm

from .qe_collect_efs import collect_efs
from .file_conversion import write_normalized_extxyz
from .scf_filter_sources import collection_failure_reason, finalize_scf_collection


def remove(file):
    if os.path.exists(file):
        os.remove(file)


def ok(path):
    ok_path = os.path.join(path, "__ok__")
    return 1 if os.path.exists(ok_path) else 0


def qe_main_xyz(current, out_name, ori_out_name, force_threshold):
    dirs = [file for file in os.listdir(current) if os.path.isdir(os.path.join(current, file)) and file != "__pycache__"]
    remove(out_name)
    remove(ori_out_name)
    ok_count = 0
    len_count = 0
    force_count = 0
    no_success_bsub_path = []
    collected_tasks = []
    excluded_records = []

    for directory in tqdm(dirs):
        path = os.path.join(current, directory)
        for sub_dir in [file for file in os.listdir(path) if os.path.isdir(os.path.join(path, file))]:
            sub_dir_path = os.path.join(path, sub_dir)
            try:
                task_ok = ok(sub_dir_path)
                ok_count += task_ok
                if task_ok == 1:
                    atom = collect_efs(sub_dir_path)
                    write_normalized_extxyz(ori_out_name, atom, append=True)
                    collected_tasks.append(sub_dir_path)
                    len_count += 1
            except Exception as exc:
                no_success_bsub_path.append(sub_dir_path)
                excluded_records.append({
                    "task": sub_dir_path,
                    "reason": collection_failure_reason(exc),
                    "detail": str(exc),
                })

    return finalize_scf_collection(
        current,
        out_name,
        ori_out_name,
        force_threshold,
        ok_count,
        collected_tasks,
        no_success_bsub_path,
        excluded_records,
    )
