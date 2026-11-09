"""LAMMPS error diagnostics shared by ordinary and enhanced MD."""
import glob
import os
from .path_names import MD_WORK_DIR


def _extract_last_lammps_step(lines, error_index):
    for line in reversed(lines[:error_index]):
        fields = line.split()
        if len(fields) < 2:
            continue
        try:
            step = int(fields[0])
            float(fields[1])
        except ValueError:
            continue
        if step >= 0:
            return step
    return None


def _extract_lammps_error_message(lines):
    error_line = error_index = last_command = None
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('ERROR:'):
            error_line, error_index = stripped, index
        elif stripped.startswith('Last command:'):
            last_command = stripped
    if error_line is None:
        return False, ''
    message = error_line
    if 'ERROR: Lost atoms:' in message:
        step = _extract_last_lammps_step(lines, error_index)
        message = f'Lost atoms | step={step if step is not None else "unknown"}'
    if last_command:
        message += f' | {last_command}'
    return True, message


def check_lmp_error(path):
    parts = os.path.normpath(path).split(os.sep)
    structure = os.path.basename(path) + '.vasp'
    case = os.path.basename(path)
    indices = [i for i, part in enumerate(parts) if part == MD_WORK_DIR]
    if indices:
        i = indices[-1]
        if i + 1 < len(parts):
            structure = parts[i + 1] + '.vasp'
        if i + 2 < len(parts):
            case = '/'.join(parts[i + 2:])
    for output in sorted(glob.glob(os.path.join(path, 'jobs_mlip_*.ini.out'))):
        with open(output, encoding='utf-8', errors='ignore') as handle:
            error, message = _extract_lammps_error_message(handle.readlines()[-200:])
        if error:
            return True, f'LAMMPS error | structure={structure} | case={case} | {message}'
    return False, ''
