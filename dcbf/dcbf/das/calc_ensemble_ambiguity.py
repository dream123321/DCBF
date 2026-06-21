#!/usr/bin/env python
import os
import os.path
import sys
import numpy as np
from ase import Atoms
from path import Path
from ase.io import iread
import glob
from ase.data import atomic_numbers,chemical_symbols
from ..path_names import MD_WORK_DIR


def _extract_lammps_error_message(lines):
    error_line = None
    last_command = None
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("ERROR:"):
            error_line = stripped
        elif stripped.startswith("Last command:"):
            last_command = stripped
    if error_line is None:
        return False, ""

    message = error_line
    if "ERROR: Lost atoms:" in error_line:
        message = "Lost atoms"
        detail = error_line.split("ERROR: Lost atoms:", 1)[1].strip()
        if detail:
            message = f"{message} | {detail}"
    if last_command:
        message = f"{message} | {last_command}"
    return True, message


def _format_lammps_error_context(path):
    path_parts = os.path.normpath(path).split(os.sep)
    structure_name = os.path.basename(path) + ".vasp"
    case_name = os.path.basename(path)

    work_indices = [index for index, part in enumerate(path_parts) if part == MD_WORK_DIR]
    if work_indices:
        work_index = work_indices[-1]
        if work_index + 1 < len(path_parts):
            structure_name = path_parts[work_index + 1] + ".vasp"
        if work_index + 2 < len(path_parts):
            case_name = "/".join(path_parts[work_index + 2:])

    return structure_name, case_name


def check_lmp_error(path):
    output_files = sorted(glob.glob(os.path.join(path, 'jobs_mlip_*.ini.out')))
    for output_file in output_files:
        with open(output_file, 'r', encoding='utf-8', errors='ignore') as handle:
            lines = handle.readlines()[-50:]
        error, message = _extract_lammps_error_message(lines)
        if error:
            structure_name, case_name = _format_lammps_error_context(path)
            return True, f"LAMMPS error | structure={structure_name} | case={case_name} | {message}"
    return False, ""

'''不收集丢原子的结构'''
def entire_structure_index(work_path):
    fn=f"{work_path}/force.0.nc"
    data = list(iread(fn))
    std_num = data[0].get_global_number_of_atoms()
    index = 0
    for temp in data:
        if temp.get_global_number_of_atoms() == std_num:
            index = index + 1
    return index

def read_force_from_nc(fn,entire_structure_index):
    forces = []
    data = list(iread(fn))[:entire_structure_index]
    for a in data:
        forces.append(a.get_forces().tolist())
    forces = np.array(forces)
    atomic_types = data[0].get_atomic_numbers()
    return forces, atomic_types


def ambiguity_distribution(threshold_low,first_column,end,num_elements):
    data = [float(value) for value in first_column]
    start = 0

    # 使用linspace创建等差数列
    temp_list = np.linspace(start, end, num_elements)
    temp_list = temp_list.tolist()
    for index, i in enumerate(temp_list):
        if threshold_low < i:
            gg = index
            #print(index)
            break

    temp_list.insert(gg, round(threshold_low, 3))
    hist, bin_edges = np.histogram(data, bins=temp_list)
    return temp_list,hist

def get_force_ambiguity(work_path):
    fns = glob.glob(f"{work_path}/force.*.nc")
    if not fns:
        raise RuntimeError(f"No ensemble force dumps were found in {work_path}")
    fns = sorted(fns)
    index = entire_structure_index(work_path)
    atomic_types = read_force_from_nc(fns[0],index)[1]
    unique_types = sorted(list(set(atomic_types.tolist())))
    unique_types.append(-1)

    force_arrays = []
    shape_summary = []
    for fn in fns:
        forces, _ = read_force_from_nc(fn, index)
        force_arrays.append(forces)
        shape_summary.append(f"{os.path.basename(fn)}={forces.shape[0]}")

    reference_shape = force_arrays[0].shape
    mismatched = [arr.shape for arr in force_arrays if arr.shape != reference_shape]
    if mismatched:
        raise RuntimeError(
            "Inconsistent ensemble force dumps in "
            f"{work_path}: {', '.join(shape_summary)}"
        )

    force_set = np.stack(force_arrays, axis=0).astype(np.float64, copy=False)
    force_mean = np.mean(force_set, axis=0)
    force_diff = force_set - force_mean
    ee = np.sqrt(np.mean(np.sum(force_diff ** 2, axis=3), axis=0))
    # print(ee.shape)

    for ii in unique_types:
        filter_index = np.arange(len(atomic_types)) if ii < 0 else np.where(atomic_types == ii)[0]
        # print(ii, filter_index, atomic_types)
        res_ee = np.zeros((ee.shape[0], 3))
        res_ee[:, 0] = np.max(ee[:, filter_index], axis=1)
        res_ee[:, 1] = np.min(ee[:, filter_index], axis=1)
        res_ee[:, 2] = np.mean(ee[:, filter_index], axis=1)
        fout = f"af.out" if ii < 0 else f"af_{ii}.out"
        fout = os.path.join(work_path,fout)
        np.savetxt(fout, res_ee, header="af: max min mean")
    return index

def ambiguity_extract(work_path,dump,ambiguity,threshold_low,threshold_high,ele,sort_ele,end, num_elements):
    dump_path = os.path.join(work_path,dump)
    data = list(iread(dump_path))
    ambiguity_path = os.path.join(work_path,ambiguity)

    map_dic = {}
    if sort_ele:
        ele = [atomic_numbers[i] for i in ele]
        ele = sorted(ele)
        for i in range(len(ele)):
            map_dic.update({i+1: chemical_symbols[ele[i]]})
    else:
        for i in range(len(ele)):
            map_dic.update({i+1: ele[i]})
    #print(map_dic)

    with open(ambiguity_path, 'r') as f:
        first_column = [line.split()[0] for line in f]
        first_column = first_column[1:]
        indexes = [[index,value] for index, value in enumerate(first_column) if float(value) > threshold_low and float(value) < threshold_high]

    structure = []
    temp_list,hist = ambiguity_distribution(threshold_low, first_column,end, num_elements)
    for index,value in indexes:
        an = data[index].get_atomic_numbers()
        new_ele = [map_dic[i] for i in an]
        data[index].set_chemical_symbols(new_ele)
        data[index].info['ambiguity']=value
        structure.append(data[index])

    # out_path = os.path.join(work_path,out)
    # write(out_path,structure,format='extxyz')
    length = len(structure)
    #logger.info(f'{work_path}: {length} structures have been selected')
    return length,structure,temp_list,hist

if __name__ == "__main__":
    work_path = os.getcwd()
    total_stru = get_force_ambiguity(work_path)
    ele = ['C']
    out = 'filter.xyz'
    sort_ele = False
    ambiguity_extract(work_path,'force.0.dump', 'af.out', 0.05,0.4, ele,sort_ele)
