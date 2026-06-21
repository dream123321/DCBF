import os

from ase.io import iread
import numpy as np
from tqdm import tqdm

from .qe_collect_efs import collect_efs
from .file_conversion import write_normalized_extxyz


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

    for directory in tqdm(dirs):
        path = os.path.join(current, directory)
        for sub_dir in [file for file in os.listdir(path) if os.path.isdir(os.path.join(path, file))]:
            sub_dir_path = os.path.join(path, sub_dir)
            try:
                ok_count += ok(sub_dir_path)
                if ok(sub_dir_path) == 1:
                    atom = collect_efs(sub_dir_path)
                    write_normalized_extxyz(ori_out_name, atom, append=True)
                    len_count += 1
            except Exception:
                no_success_bsub_path.append(sub_dir_path)

    if not os.path.exists(ori_out_name):
        return ok_count, len_count, no_success_bsub_path, force_count, "None"

    data = list(iread(ori_out_name))
    if not data:
        return ok_count, len_count, no_success_bsub_path, force_count, "None"

    max_list = []
    for atom in data:
        max_force = np.linalg.norm(atom.get_forces(), axis=1).max()
        max_list.append(max_force)
        if max_force < force_threshold:
            write_normalized_extxyz(out_name, atom, append=True)
            force_count += 1

    force_of_force_count_0 = "None"
    if force_count == 0:
        min_index = max_list.index(min(max_list))
        write_normalized_extxyz(out_name, data[min_index], append=False)
        force_of_force_count_0 = min(max_list)

    return ok_count, len_count, no_success_bsub_path, force_count, force_of_force_count_0
