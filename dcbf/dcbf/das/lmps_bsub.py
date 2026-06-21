import os


def lammps_bsub(dir, mlp_nums, mlp_encode_model, scheduler):
    script_content_1 = f"""#!/bin/bash
{scheduler.bsub_script_lmp_job_name}{dir}
"""
    script_content_1 = script_content_1 + scheduler.bsub_script_lmp

    lmp_env = str(getattr(scheduler, "lmp_env", "") or "").strip()
    lmp_env_text = f"{lmp_env}\n" if lmp_env else ""

    content_1 = f"""pwd=$(pwd)
touch __start__
{lmp_env_text}# Loop through directories in the current working directory
run_and_record() {{
    local label="$1"
    local command="$2"
    local output_file="$3"
    local time_file="$4"
    local start_time
    local end_time
    local runtime
    local status
    start_time=$(date +%s.%N)
    eval "$command" > "$output_file" 2>&1
    status=$?
    end_time=$(date +%s.%N)
    runtime=$(echo "$end_time - $start_time" | bc)
    echo "$label total_runtime:$runtime s" >> "$time_file"
    return $status
}}

for dir in "$pwd"/*; do
    if [ -d "$dir" ]; then
        structure_time_file="$dir/time.txt"
        # Loop through subdirectories
        for sub_dir in "$dir"/*; do
            if [ -d "$sub_dir" ]; then
                echo "$sub_dir"
                cd "$sub_dir"
                touch __start__

                # --- PLUMED setup: generate input.plumed from template ---
                _T=$(grep "variable T equal" lmp.in 2>/dev/null | awk '{{print $NF}}')
                _INIT="$(cd ../../../../../../init 2>/dev/null && pwd)"
                if [ -n "$_INIT" ] && [ -f "$_INIT/input.plumed.tmpl" ]; then
                    sed "s/__TEMP__/$_T/g" "$_INIT/input.plumed.tmpl" > input.plumed
                    [ -f "$_INIT/water.ndx" ] && cp -f "$_INIT/water.ndx" ./water.ndx
                fi

                # initial run
                COMMAND="${{COMMAND_0}} -in lmp.in -var out_dump_file force.0.nc"
                run_and_record "$(basename "$sub_dir") lmp.in" "$COMMAND" "jobs_mlip_0.ini.out" "$structure_time_file"

                # rerun
"""

    content_temp = ""

    if mlp_encode_model is False:
        for index in range(1, mlp_nums):
            content_temp += f"""
                COMMAND="${{COMMAND_0}} -in in_rerun_{index}.lmp -var out_dump_file force.{index}.nc"
                run_and_record "$(basename "$sub_dir") in_rerun_{index}.lmp" "$COMMAND" "jobs_mlip_{index}.ini.out" "$structure_time_file"
"""

    content_temp += """
                touch __ok__"""

    content_2 = """
            fi
        done
    fi
done
cd $pwd
touch __ok__
"""

    with open("bsub.lsf", "w", encoding="utf-8") as file:
        file.write(script_content_1 + content_1 + content_temp + content_2)


if __name__ == "__main__":
    lmp_exe = "mpirun -machinefile $LSB_DJOB_HOSTFILE -np $NP /work/phy-huangj/app/interface-lammps-mlip-2/lmp_intel_cpu_intelmpi"
    lammps_bsub("qiming", 3, False, lmp_exe)
