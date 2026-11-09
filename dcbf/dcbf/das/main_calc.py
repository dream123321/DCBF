import os

from ase.io import write
from tqdm import trange
import yaml

from .gen_calc_file import INPUT, create_potcar, poscar2STRU, prepare_qe_input


# Generated as jobs_script/../start_calc.py.  It must be a standalone script
# (executed with the deployment interpreter from the SCF workspace) and it must
# return a non-zero exit status when any scheduler submission fails, because
# generation.py relies on subprocess.run(..., check=True) instead of guessing
# from job counts.
_START_CALC_TEMPLATE = '''import json
import os
import re
import subprocess
import sys
import time

start = 1
end = __CALC_DIR_NUM__
submit_command = __SUBMIT_COMMAND__
summary_name = "jobs_submitted.json"

_JOB_ID_PATTERNS = (
    re.compile(r"Submitted batch job\\s+(\\d+)"),
    re.compile(r"Job\\s*<\\s*(\\d+)\\s*>"),
    re.compile(r"job(?:id)?\\s*[=:]\\s*<?(\\d+)>?", re.IGNORECASE),
)


def extract_job_id(text):
    text = text or ""
    for pattern in _JOB_ID_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(1)
    for token in text.replace("<", " ").replace(">", " ").split():
        if token.isdigit():
            return token
    return None


def main():
    pwd = os.getcwd()
    jobs_script = os.path.join(pwd, "jobs_script")
    os.chdir(jobs_script)
    records = []
    failures = []
    for i in range(start, end + 1):
        script_name = "bsub_%d.lsf" % i
        command = "%s %s" % (submit_command, script_name)
        record = {
            "index": i,
            "script": script_name,
            "command": command,
            "returncode": None,
            "job_id": None,
            "stdout": "",
            "stderr": "",
        }
        if not os.path.exists(script_name):
            record["returncode"] = 127
            record["stderr"] = "missing job script"
            records.append(record)
            failures.append(record)
            print("[start_calc] missing job script: %s" % script_name)
            continue
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        stdout = (result.stdout or "").strip()
        stderr = (result.stderr or "").strip()
        record["returncode"] = int(result.returncode)
        record["stdout"] = stdout[-2000:]
        record["stderr"] = stderr[-2000:]
        record["job_id"] = extract_job_id(stdout) or extract_job_id(stderr)
        records.append(record)
        if record["returncode"] != 0:
            failures.append(record)
            print(
                "[start_calc] submission FAILED: %s rc=%s stderr=%s"
                % (command, record["returncode"], stderr[-400:])
            )
        elif record["job_id"] is None:
            print(
                "[start_calc] WARNING: submission returned rc=0 but no job id was "
                "parsed for %s (stdout=%s)" % (command, stdout[-400:])
            )
        else:
            print("[start_calc] submitted %s job_id=%s" % (script_name, record["job_id"]))

    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "submit_command": submit_command,
        "total": len(records),
        "succeeded": len(records) - len(failures),
        "failed": len(failures),
        "missing_job_id": sum(
            1 for item in records
            if item["returncode"] == 0 and item["job_id"] is None
        ),
        "jobs": records,
    }
    with open(summary_name, "w") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\\n")

    if failures:
        detail = ", ".join(
            "%s(rc=%s)" % (item["script"], item["returncode"]) for item in failures[:10]
        )
        if len(failures) > 10:
            detail += ", ... (+%d more)" % (len(failures) - 10)
        print(
            "[start_calc] %d/%d scheduler submissions failed: %s"
            % (len(failures), len(records), detail)
        )
        print("[start_calc] details written to %s" % os.path.join(jobs_script, summary_name))
        return 1
    print(
        "[start_calc] all %d scheduler submissions were accepted; details in %s"
        % (len(records), os.path.join(jobs_script, summary_name))
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''


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
                if os.path.exists("ase_sort.dat"):
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


def main_calc(atom_list, calc_dir_num, path_main, scheduler, resume_tasks=False):
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

        resume_guard = ('    if [ -f __ok__ ]; then continue; fi\n'
                        '    if [ -f __failed__ ]; then mv __failed__ __failed__.previous; fi\n') if resume_tasks else ''
        content_2 = f"""dir_1="filter"
dir_2="{'dir_' + str(i)}"
cd ../$dir_1/$dir_2

path=$(pwd)
batch_failed=0

for item in {{{range1}..{range2}}}; do
    cd $path/$item
{resume_guard}    start_time=$(date +%s.%N)
    touch __start__
    $COMMAND_std > logout 2>&1
    status=$?
    if [ "$status" -eq 0 ]; then touch __ok__; else touch __failed__; batch_failed=1; fi
    end_time=$(date +%s.%N)
    runtime=$(echo "$end_time - $start_time" | bc)
    cd $path
    cd ..
    echo "job_{str(i)}_$item total_runtime:$runtime s" >> time.txt
done
exit $batch_failed"""
        with open(os.path.join(jobs_script, f"bsub_{i}.lsf"), "w") as file:
            file.write(content_1 + content_2)
        os.chdir(jobs_script)

    os.chdir(pwd)

    start_calc = _START_CALC_TEMPLATE.replace(
        "__CALC_DIR_NUM__", str(int(calc_dir_num))
    ).replace(
        "__SUBMIT_COMMAND__", repr(str(scheduler.start_calc_command))
    )

    with open("start_calc.py", "w") as f:
        f.write(start_calc)
    return num_posacr
