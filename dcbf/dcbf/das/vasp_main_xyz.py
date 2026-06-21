import os
from .file_conversion import write_normalized_extxyz
from .vasp_collect_efs import collect_efs
from .no_success_bsub import no_success_bsub
from ase.io import iread, read
from tqdm import tqdm
import numpy as np
import re
from pathlib import Path
from ..path_names import DFT_WORK_DIR


def remove(file):
    if os.path.exists(file):
        os.remove(file)

def ok(path):
    t = 0
    ok_path = os.path.join(path,'__ok__')
    if os.path.exists(ok_path):
        t = 1
    return t

def INCAR_NELM(INCAR):
    # 正则表达式匹配 NELM 的值
    pattern = re.compile(r"NELM\s*=\s*(\d+)")

    # 打开文件并提取 NELM 的值
    nelm_value = 60
    with open(INCAR, "r") as file:
        for line in file:
            match = pattern.search(line)
            #print(match)
            if match:
                nelm_value = int(match.group(1))
    return nelm_value

def nmax_label(logout,nmax):
    with open(logout, 'r') as file:
        for line in file:
            if 'DAV:' in line:
                string = line.split()
            elif 'RMM:' in line:
                string = line.split()
        log_nmax = int(string[1])
    label = True
    #print(log_nmax)
    if int(log_nmax) == nmax:
        label = False
    return label


def get_parent_directory(target_dir, levels=1):
    """获取指定目录的上级目录"""
    path = Path(target_dir)
    for _ in range(levels):
        path = path.parent
    return path

def vasp_main_xyz(current, out_name, ori_out_name, force_threshold):
    dirs = [file for file in os.listdir(current) if os.path.isdir(os.path.join(current,file)) and file != '__pycache__']
    INCAR_path = os.path.join(get_parent_directory(current, levels=5), 'init', 'INCAR')
    nmax = INCAR_NELM(INCAR_path)
    #print(nmax)
    remove(out_name)
    remove(ori_out_name)
    ok_count = 0
    len_count = 0
    force_count = 0

    no_success_bsub_path = []
    for dir in tqdm(dirs):
        path = os.path.join(current, dir)
        for sub_dir in [file for file in os.listdir(path) if os.path.isdir(os.path.join(path, file))]:
            sub_dir_path = os.path.join(path, sub_dir)

            try:
                ok_count = ok_count + ok(sub_dir_path)
                if ok(sub_dir_path)==1:
                    if nmax_label(os.path.join(sub_dir_path,'logout'), nmax):
                        atom = collect_efs(sub_dir_path)
                        write_normalized_extxyz(ori_out_name, atom, append=True)
                        len_count = len_count + 1
            except:
                server = None #(废弃的功能)
                no_success_bsub(server, sub_dir_path)
                no_success_bsub_path.append(sub_dir_path)
                #print('Collecting structure unsuccessful, please check! ', sub_dir_path)

    data = list(iread(ori_out_name))

    max_list = []
    for atom in data:
        max_ = np.linalg.norm(atom.get_forces(), axis=1).max()
        max_list.append(max_)
        if max_ < force_threshold:
            write_normalized_extxyz(out_name, atom, append=True)
            force_count = force_count + 1

    force_of_force_count_0 = 'None'
    if force_count == 0:
        min_index = max_list.index(min(max_list))
        write_normalized_extxyz(out_name, data[min_index], append=False)
        force_of_force_count_0 = min(max_list)

    return ok_count, len_count, no_success_bsub_path, force_count, force_of_force_count_0

if __name__ =='__main__':
    pwd = os.getcwd()
    current = os.path.join(pwd, DFT_WORK_DIR, 'scf', 'filter')
    out_name = os.path.join(pwd, DFT_WORK_DIR, 'scf_filter.xyz')
    ori_out_name = os.path.join(pwd, DFT_WORK_DIR, 'ori_scf_filter.xyz')
    force_threshold = 10
    ok_count, len_count, no_success_bsub_path, force_count, force_of_force_count_0 = vasp_main_xyz(current, out_name, ori_out_name, force_threshold)

    print(ok_count, len_count, no_success_bsub_path, force_count)
