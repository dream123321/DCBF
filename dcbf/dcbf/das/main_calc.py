import os

from ase.io import write
from tqdm import trange
import yaml

from .gen_calc_file import INPUT, create_potcar, poscar2STRU, prepare_qe_input


def check_and_modify_calc_dir(pwd, xyz_list, calc_dir_num):
    parameter_yaml = os.path.join(pwd, "parameter.yaml")
    end_yaml = os.path.join(pwd, "end.yaml")
    xyz_num = len(xyz_list)

    if xyz_num == 0:
        return 0

    if calc_dir_num <= 0:
        new_calc_dir_num = xyz_num
    elif xyz_num < calc_dir_num:
        new_calc_dir_num = xyz_num
    else:
        return calc_dir_num

    with open(parameter_yaml, "r") as file:
        data_1 = yaml.safe_load(file)
    dft = data_1["dft"]
    dft["calc_dir_num"] = new_calc_dir_num
    with open(parameter_yaml, "w") as file:
        yaml.safe_dump(data_1, file, default_flow_style=False)

    if os.path.exists(end_yaml):
        with open(end_yaml, "r") as file:
            data_2 = yaml.safe_load(file)
        dft = data_2["dft"]
        dft["calc_dir_num"] = new_calc_dir_num
        with open(end_yaml, "w") as file:
            yaml.safe_dump(data_2, file, default_flow_style=False)
    return new_calc_dir_num


def mkdir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)


def mkdir_Placefile(dir, num_posacr, calc_dir_num, xyz, scheduler, gen_potcar_mode="", potcar_dir=""):
    quotient = num_posacr // calc_dir_num
    remainder = num_posacr % calc_dir_num
    temp = [quotient] * calc_dir_num
    divide = [value + 1 if index < remainder else value for index, value in enumerate(temp)]
    range_list = [[sum(divide[: idx + 1]) - divide[idx] + 1, sum(divide[: idx + 1])] for idx in range(len(divide))]

    for i in trange(1, calc_dir_num + 1):
        sub_dir = os.path.join(dir, "dir_" + str(i))
        mkdir(sub_dir)
        index_poscar = [value for value in range(range_list[i - 1][0], range_list[i - 1][1] + 1)]

        for ii in index_poscar:
            sub_sub_dir = os.path.join(sub_dir, str(ii))
            mkdir(sub_sub_dir)
            scf_cal_engine = scheduler.scf_cal_engine
            if scf_cal_engine == "abacus":
                write(os.path.join(sub_sub_dir, "POSCAR"), xyz[ii - 1], format="vasp")
                os.chdir(sub_sub_dir)
                poscar2STRU(dir)
                INPUT(dir, sub_sub_dir, scf_cal_engine)
                os.remove("ase_sort.dat")
            elif scf_cal_engine == "cp2k":
                atoms = xyz[ii - 1]
                write(os.path.join(sub_sub_dir, "cp2k.xyz"), atoms, format="extxyz")
                llist = atoms.get_cell()
                os.chdir(sub_sub_dir)
                INPUT(dir, sub_sub_dir, scf_cal_engine)
                cell_A = f"      A    {llist[0][0]:.8f}     {llist[0][1]:.8f}     {llist[0][2]:.8f}\n"
                cell_B = f"      B    {llist[1][0]:.8f}     {llist[1][1]:.8f}     {llist[1][2]:.8f}\n"
                cell_C = f"      C    {llist[2][0]:.8f}     {llist[2][1]:.8f}     {llist[2][2]:.8f}\n"

                cp2k_inp = "cp2k.inp"
                with open(cp2k_inp, "r") as f:
                    lines = f.readlines()
                    for index, line in enumerate(lines):
                        if "&CELL" in line:
                            cell_index = index
                            break
                    lines[cell_index + 1] = cell_A
                    lines[cell_index + 2] = cell_B
                    lines[cell_index + 3] = cell_C
                with open(cp2k_inp, "w") as file:
                    file.writelines(lines)
            elif scf_cal_engine == "qe":
                os.chdir(sub_sub_dir)
                prepare_qe_input(sub_sub_dir, xyz[ii - 1], dir)
            elif scf_cal_engine == "vasp":
                atoms = xyz[ii - 1]
                sorted_indices = sorted(range(len(atoms)), key=lambda index: atoms[index].symbol)
                sorted_atoms = atoms.__class__()
                sorted_atoms.cell = atoms.cell
                sorted_atoms.pbc = atoms.pbc
                for index in sorted_indices:
                    sorted_atoms.append(atoms[index])
                write(os.path.join(sub_sub_dir, "POSCAR"), sorted_atoms, format="vasp")
                os.chdir(sub_sub_dir)
                INPUT(dir, sub_sub_dir, scf_cal_engine)
                if gen_potcar_mode == "my_potcar":
                    create_potcar("POSCAR", potcar_dir, output_file="POTCAR")
                else:
                    os.system("(echo 103)|vaspkit")
            else:
                raise ValueError(f"{scf_cal_engine} is not exist!")


def main_calc(atom_list, calc_dir_num, path_main, scheduler):
    if len(atom_list) == 0 or calc_dir_num <= 0:
        return 0

    path = os.path.dirname(os.path.dirname(path_main))
    jobname = os.path.basename(path)
    if os.path.exists(os.path.join(path, "init", "POTCAR_dir")):
        gen_potcar_mode = "my_potcar"
        potcar_dir = os.path.join(path, "init", "POTCAR_dir")
    else:
        gen_potcar_mode = "vaspkit"
        potcar_dir = ""

    pwd = os.getcwd()
    dir = os.path.join(pwd, "filter")
    mkdir(dir)

    num_posacr = len(atom_list)
    mkdir_Placefile(dir, num_posacr, calc_dir_num, atom_list, scheduler, gen_potcar_mode, potcar_dir)

    os.chdir(pwd)
    jobs_script = os.path.join(pwd, "jobs_script")
    mkdir(jobs_script)

    quotient = num_posacr // calc_dir_num
    remainder = num_posacr % calc_dir_num
    temp = [quotient] * calc_dir_num
    divide = [value + 1 if index < remainder else value for index, value in enumerate(temp)]
    range_list = [[sum(divide[: idx + 1]) - divide[idx] + 1, sum(divide[: idx + 1])] for idx in range(len(divide))]

    for i in range(1, calc_dir_num + 1):
        range1, range2 = range_list[i - 1][0], range_list[i - 1][1]
        content_1 = f"""#!/bin/bash
{scheduler.bsub_script_scf_job_name} {jobname}_{str(i)} 
"""
        content_1 = content_1 + scheduler.bsub_script_scf

        content_2 = f"""dir_1="filter"
dir_2="{'dir_' + str(i)}"
cd ../$dir_1/$dir_2

path=$(pwd)

for item in {{{range1}..{range2}}}; do
    cd $path/$item
    start_time=$(date +%s.%N)
    touch __start__
    if $COMMAND_std > logout 2>&1; then
        touch __ok__
    else
        status=$?
        echo "COMMAND_std failed with exit code $status" >> logout
        touch __failed__
        exit $status
    fi

    end_time=$(date +%s.%N)
    runtime=$(echo "$end_time - $start_time" | bc)
    cd $path
    cd ..
    echo "job_{str(i)}_$item total_runtime:$runtime s" >> time.txt
done"""
        with open(os.path.join(jobs_script, f"bsub_{i}.lsf"), "w") as file:
            file.write(content_1 + content_2)
        os.chdir(jobs_script)

    os.chdir(pwd)

    start_calc = f"""import os
start = 1
end = {calc_dir_num}

pwd = os.getcwd()
jobs_script = os.path.join(pwd,'jobs_script')
os.chdir(jobs_script)
for i in range(start,end+1):
    if not os.path.exists(f'bsub_{{i}}.lsf'):
        print('errors')
    os.system(f'{scheduler.start_calc_command} bsub_{{i}}.lsf')"""

    with open("start_calc.py", "w") as f:
        f.write(start_calc)
    return num_posacr
