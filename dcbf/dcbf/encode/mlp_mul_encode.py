import os
import concurrent.futures
import shlex
from .file_conversion import dump2cfg, merge_cfg_out
from .cfg_descriptor_encode import _build_calc_descriptors_command
import subprocess


def main_dump2cfg(path, cfg_name):
    input_path = os.path.join(path, 'force.0.dump')
    output_path = os.path.join(path, cfg_name)
    length = dump2cfg(input_path, output_path)
    return length


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


def mul_encode(pwd, mtp_path, dirs, cfg_name, out_name, sus2_mlp_exe, train_env=None):
    with concurrent.futures.ProcessPoolExecutor() as executor:
        cfg_names = [cfg_name for _ in dirs]
        results = list(executor.map(main_dump2cfg, dirs, cfg_names))

    commands = []
    for path in dirs:
        md_cfg = os.path.join(path, cfg_name)
        md_out = os.path.join(path, out_name)
        commands.append(_build_calc_descriptors_shell(sus2_mlp_exe, mtp_path, md_cfg, md_out, train_env=train_env))

    processes = [
        (
            subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True),
            rendered,
        )
        for command, rendered in commands
    ]

    for process, rendered in processes:
        _, stderr = process.communicate()
        if process.returncode != 0:
            raise RuntimeError(
                f"calc-descriptors failed during mul_encode with exit code {process.returncode}: {rendered}\n"
                f"{stderr[-2000:]}"
            )

    merge_cfg_out(pwd, dirs, cfg_name, out_name)
    return results


if __name__ == '__main__':
    pwd = os.getcwd()
    dirs = ''
    mul_encode(pwd, mtp_path, dirs, cfg_name, out_name)
