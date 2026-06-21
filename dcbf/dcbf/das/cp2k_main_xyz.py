import os
from .cp2k_collect_efs import collect_efs
from .file_conversion import write_normalized_extxyz
from .no_success_bsub import no_success_bsub
from ase.io import iread, read
from tqdm import tqdm
import numpy as np


def remove(file):
    if os.path.exists(file):
        os.remove(file)

def ok(path):
    t = 0
    ok_path = os.path.join(path,'__ok__')
    if os.path.exists(ok_path):
        t = 1
    return t

def cp2k_main_xyz(current, out_name, ori_out_name, force_threshold):
    dirs = [file for file in os.listdir(current) if os.path.isdir(os.path.join(current,file)) and file != '__pycache__']
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
    current = os.path.join(pwd,'scf','filter')
    out_name =os.path.join(pwd,'out.xyz')
    ori_out_name = os.path.join(pwd, 'ori_out.xyz')
    force_threshold = 10
    ok_count,len_count,no_success_bsub_path, force_count = cp2k_main_xyz(current, out_name, ori_out_name, force_threshold)
    print(ok_count,len_count,no_success_bsub_path, force_count)
