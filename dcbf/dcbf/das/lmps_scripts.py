from ase.io import read, write
import os
from ase.data import atomic_numbers
from ase.data import chemical_symbols
import random
import importlib.util
from pathlib import Path
import re


def main_pos2lmp(elements, size, sort_ele):
    if os.path.exists("POSCAR"):
        poscar = "POSCAR"
    else:
        poscar = [f for f in os.listdir() if f.endswith(".vasp")][0]

    atoms = read(poscar)
    atoms = atoms.repeat(size)

    if sort_ele:
        atomic_nums = sorted(atomic_numbers[element] for element in elements)
        specorder = [chemical_symbols[number] for number in atomic_nums]
    else:
        specorder = elements

    write("data.in", atoms, format="lammps-data", masses=True, specorder=specorder, force_skew=True)


def _inject_sus2_pair_style(script_text, mtp_filename):
    lines = script_text.splitlines()
    normalized = []
    inserted = False

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("# variable mlip_ini"):
            continue
        if stripped.startswith("pair_style") and ("mlip" in stripped or "sus2mtp" in stripped):
            continue
        if stripped == "pair_coeff * *":
            continue

        normalized.append(line)
        if not inserted and stripped.startswith("read_data data.in"):
            normalized.append("")
            normalized.append(f"pair_style sus2mtp {mtp_filename}")
            normalized.append("pair_coeff * *")
            inserted = True

    if not inserted:
        raise ValueError("Could not find 'read_data data.in' in the LAMMPS template")

    return "\n".join(normalized) + "\n"


def _extract_nevery_expression(lmp_input):
    match = re.search(r"^\s*variable\s+nevery\s+equal\s+(.+?)\s*$", lmp_input, flags=re.MULTILINE)
    if match is None:
        raise ValueError(
            "Could not extract variable nevery from generated LAMMPS input; "
            "DAS rerun requires the same dump cadence as force.0.dump"
        )
    return match.group(1).strip()


def _load_optional_extra_input(init_dir, ensemble):
    path = Path(init_dir) / f"lmp_{ensemble}_extra.in"
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8")
    return text.rstrip() + "\n"


def lammps_scripts(ensemble, temp, mlp_nums, mlp_encode_model, pwd):
    init_dir = Path(pwd).resolve().parents[1] / "init"
    spec = importlib.util.spec_from_file_location("ocbf_lmp_in", init_dir / "lmp_in.py")
    lmp_in = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(lmp_in)
    random_number = random.randint(1000, 1000000)

    pre_data_in = f"""variable T equal {temp}
    variable random equal {random_number}
    """

    if ensemble == "npt":
        lmp_input = pre_data_in + lmp_in.npt
    elif ensemble == "nvt":
        lmp_input = pre_data_in + lmp_in.nvt
    else:
        raise ValueError(f"{ensemble}: error")

    nevery_expression = _extract_nevery_expression(lmp_input)
    extra_input_text = _load_optional_extra_input(init_dir, ensemble)

    rerun_template = f"""variable input_dump_file string "force.0.dump"
variable nevery equal {nevery_expression}
# variable out_dump_file string

units metal
boundary p p p
atom_style atomic

box tilt large
read_data data.in

neighbor 2.0 bin
neigh_modify delay 10 check yes

reset_timestep 0

compute pe all pe/atom
dump 1 all custom ${{nevery}} ${{out_dump_file}} id type x y z fx fy fz c_pe
dump_modify 1 sort id append yes

rerun ${{input_dump_file}} dump x y z"""

    main_input = _inject_sus2_pair_style(lmp_input, "current_0.mtp")
    if extra_input_text:
        marker = "pair_coeff * *\n"
        if marker not in main_input:
            raise ValueError("Could not find 'pair_coeff * *' while injecting LAMMPS extra input")
        main_input = main_input.replace(marker, marker + "\n" + extra_input_text + "\n", 1)

    with open("lmp.in", "w", encoding="utf-8") as file:
        file.write(main_input)

    if mlp_encode_model is False:
        for index in range(1, mlp_nums):
            with open(f"in_rerun_{index}.lmp", "w", encoding="utf-8") as file:
                file.write(_inject_sus2_pair_style(rerun_template, f"current_{index}.mtp"))
