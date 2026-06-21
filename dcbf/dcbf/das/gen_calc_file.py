import os
import re
import shutil
import glob

from ase.calculators.calculator import kpts2sizeandoffsets
from ase.data import atomic_masses, atomic_numbers
from ase.io import read, write
import yaml


QE_CARD_PREFIXES = (
    "ATOMIC_SPECIES",
    "ATOMIC_POSITIONS",
    "K_POINTS",
    "CELL_PARAMETERS",
    "CONSTRAINTS",
    "OCCUPATIONS",
    "HUBBARD",
)


def _copy_if_exists(source_path, destination_folder):
    if os.path.exists(source_path):
        shutil.copy(source_path, destination_folder)
        return True
    return False


def _strip_key_value_lines(path, keys):
    if not os.path.exists(path):
        return
    key_set = {key.lower() for key in keys}
    with open(path, "r", encoding="utf-8") as handle:
        lines = handle.readlines()
    filtered = []
    for line in lines:
        stripped = line.strip()
        lowered = stripped.lower()
        if any(lowered.startswith(f"{key} ") or lowered.startswith(f"{key}=") for key in key_set):
            continue
        filtered.append(line)
    with open(path, "w", encoding="utf-8") as handle:
        handle.writelines(filtered)


def find_vasp_files_in_current_directory():
    current_directory = os.getcwd()
    return [file for file in os.listdir(current_directory) if file.endswith(".vasp")]


def get_upper_directory(level, current_path):
    for _ in range(level):
        current_path = os.path.dirname(os.path.abspath(current_path))
    return current_path


def poscar2STRU(dir):
    if not os.path.exists("POSCAR"):
        poscar = find_vasp_files_in_current_directory()[0]
        shutil.copy(poscar, "POSCAR")
    path = get_upper_directory(5, dir)
    pp_orb_dic = os.path.join(path, "init", "pp_orb_dic.yaml")

    with open(pp_orb_dic, "rb") as file:
        yaml_data = yaml.safe_load(file)
        pp_dic = yaml_data["pp_dic"]
        orb_dic = yaml_data["orb_dic"]

    pwd = os.getcwd()
    cs_vasp = os.path.join(pwd, "POSCAR")
    cs_atoms = read(cs_vasp, format="vasp")
    cs_stru = os.path.join(pwd, "STRU")

    atom_symbols = list(set(cs_atoms.get_chemical_symbols()))
    pp = {}
    basis = {}
    for atom in atom_symbols:
        pp[atom] = pp_dic[atom]
        basis[atom] = orb_dic[atom]

    write(cs_stru, cs_atoms, format="abacus", pp=pp, basis=basis)


def chek_file_exist(file_path, scf_cal_engine):
    if not os.path.exists(file_path):
        raise ValueError(f"{scf_cal_engine}:{file_path} is not exist!")


def read_poscar(poscar_file="POSCAR"):
    try:
        with open(poscar_file, "r", encoding="utf-8") as handle:
            lines = handle.readlines()

        elements_line = None
        for index in range(3, min(10, len(lines))):
            line = lines[index].strip()
            if line and any(char.isupper() for char in line) and not all(char.isdigit() or char.isspace() for char in line):
                elements_line = line
                break

        if not elements_line:
            raise ValueError("Can't find the element row in the POSCAR file!")

        return elements_line.split()
    except Exception as exc:
        raise ValueError(f"There is an error reading POSCAR: {exc}!")


def find_potcar_file(element, potcar_dir):
    search_patterns = [
        f"{element}*/POTCAR",
        f"POTCAR_{element}",
        f"*/POTCAR_{element}",
        f"{element}*",
        f"*{element}*",
    ]

    for pattern in search_patterns:
        search_path = os.path.join(potcar_dir, pattern)
        matches = glob.glob(search_path)
        file_matches = [match for match in matches if os.path.isfile(match)]
        if not file_matches:
            continue

        for match in file_matches:
            filename = os.path.basename(match)
            parent_dir = os.path.basename(os.path.dirname(match))
            if filename == "POTCAR" and parent_dir.startswith(element):
                return match
            if filename == f"POTCAR_{element}":
                return match
        return file_matches[0]
    return None


def create_potcar(poscar_file, potcar_dir, output_file="POTCAR"):
    if not os.path.exists(potcar_dir):
        raise ValueError(f"error! pseudo potential ('{potcar_dir}') isn't exist !")

    elements = read_poscar(poscar_file)
    with open(output_file, "wb") as outfile:
        for element in elements:
            potcar_file = find_potcar_file(element, potcar_dir)
            if potcar_file and os.path.isfile(potcar_file):
                with open(potcar_file, "rb") as infile:
                    outfile.write(infile.read())
            else:
                raise ValueError(f"error!: Can't find {element} POTCAR file in {potcar_dir}")


def _replace_qe_scalar(line, key, value):
    pattern = rf"(\b{key}\s*=\s*)([^,\s/]+)"
    return re.sub(pattern, rf"\g<1>{value}", line)


def _find_qe_block_end(lines, start_index):
    for index in range(start_index + 1, len(lines)):
        stripped = lines[index].strip()
        upper = stripped.upper()
        if stripped.startswith("&") or stripped == "/":
            return index
        if upper.startswith(QE_CARD_PREFIXES):
            return index
    return len(lines)


def _replace_or_append_qe_block(lines, card_name, new_block_lines):
    for index, line in enumerate(lines):
        if line.strip().upper().startswith(card_name):
            end_index = _find_qe_block_end(lines, index)
            return lines[:index] + new_block_lines + lines[end_index:]
    if lines and lines[-1].strip():
        lines.append("\n")
    return lines + new_block_lines


def _replace_or_insert_qe_scalar(lines, namelist_name, key, value):
    start_index = None
    end_index = None
    for index, line in enumerate(lines):
        if line.strip().upper() == f"&{namelist_name.upper()}":
            start_index = index
            continue
        if start_index is not None and line.strip() == "/":
            end_index = index
            break
    if start_index is None or end_index is None:
        raise ValueError(f"qe.in must contain &{namelist_name.upper()} namelist")

    key_pattern = re.compile(rf"\b{key}\s*=")
    for index in range(start_index + 1, end_index):
        if key_pattern.search(lines[index]):
            lines[index] = _replace_qe_scalar(lines[index], key, value)
            return lines

    lines.insert(end_index, f"   {key:<16} = {value}\n")
    return lines


def _load_yaml_if_exists(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _extract_pseudo_dir(lines, base_dir):
    pattern = re.compile(r"pseudo_dir\s*=\s*['\"]([^'\"]+)['\"]", re.IGNORECASE)
    for line in lines:
        match = pattern.search(line)
        if match:
            pseudo_dir = match.group(1)
            if os.path.isabs(pseudo_dir):
                return pseudo_dir
            return os.path.normpath(os.path.join(base_dir, pseudo_dir))
    return None


def _normalize_pseudo_entry(symbol, value):
    if isinstance(value, str):
        return {"filename": value}
    if isinstance(value, dict) and "filename" in value:
        return dict(value)
    raise ValueError(f"Invalid qe pseudo mapping for {symbol}: {value!r}")


def _load_qe_pseudo_map(workspace_root):
    pseudo_map_path = os.path.join(workspace_root, "init", "qe_pseudo_map.yaml")
    data = _load_yaml_if_exists(pseudo_map_path)
    if not data:
        raise ValueError("QE mode requires init/qe_pseudo_map.yaml to map element symbols to UPF files")
    return {symbol: _normalize_pseudo_entry(symbol, value) for symbol, value in data.items()}


def _build_species_block(atoms, pseudo_map, pseudo_dir):
    seen = set()
    symbols = []
    for symbol in atoms.get_chemical_symbols():
        if symbol not in seen:
            seen.add(symbol)
            symbols.append(symbol)

    lines = ["ATOMIC_SPECIES\n"]
    for symbol in symbols:
        if symbol not in pseudo_map:
            raise ValueError(
                f"Element {symbol} is missing in init/qe_pseudo_map.yaml. "
                f"Please add a pseudopotential filename for it."
            )
        entry = pseudo_map[symbol]
        filename = entry["filename"]
        if pseudo_dir is not None:
            pseudo_path = os.path.join(pseudo_dir, filename)
            chek_file_exist(pseudo_path, "qe")
        mass = entry.get("mass")
        if mass is None:
            mass = atomic_masses[atomic_numbers[symbol]]
        lines.append(f"{symbol} {mass:.10f} {filename}\n")
    return lines


def _load_qe_kpoints_config(workspace_root):
    config = _load_yaml_if_exists(os.path.join(workspace_root, "init", "qe_kpoints.yaml"))
    return config or {}


def _build_kpoints_block(atoms, kpoints_config):
    if not kpoints_config:
        return None

    if "kspacing" in kpoints_config:
        kspacing = float(kpoints_config["kspacing"])
        if kspacing <= 0:
            raise ValueError("init/qe_kpoints.yaml: kspacing must be > 0")
        gamma = bool(kpoints_config.get("gamma", True))
        even = kpoints_config.get("even", False)
        density = 1.0 / kspacing
        size, offsets = kpts2sizeandoffsets(
            density=density,
            gamma=gamma,
            even=even,
            atoms=atoms,
        )
        shift = [1 if abs(offset) > 1e-12 else 0 for offset in offsets]
        return [
            "K_POINTS automatic\n",
            f"{int(size[0])} {int(size[1])} {int(size[2])} {int(shift[0])} {int(shift[1])} {int(shift[2])}\n",
        ]

    if "mesh" in kpoints_config:
        mesh = kpoints_config["mesh"]
        shift = kpoints_config.get("shift", [0, 0, 0])
        if len(mesh) != 3 or len(shift) != 3:
            raise ValueError("init/qe_kpoints.yaml: mesh and shift must have length 3")
        return [
            "K_POINTS automatic\n",
            f"{int(mesh[0])} {int(mesh[1])} {int(mesh[2])} {int(shift[0])} {int(shift[1])} {int(shift[2])}\n",
        ]

    raise ValueError("init/qe_kpoints.yaml must define either kspacing or mesh")


def patch_qe_input(template_path, atoms, workspace_root):
    with open(template_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    lines = _replace_or_insert_qe_scalar(lines, "SYSTEM", "nat", len(atoms))
    lines = _replace_or_insert_qe_scalar(lines, "SYSTEM", "ntyp", len(set(atoms.get_chemical_symbols())))

    pseudo_dir = _extract_pseudo_dir(lines, os.path.dirname(template_path))
    pseudo_map = _load_qe_pseudo_map(workspace_root)
    species_block = _build_species_block(atoms, pseudo_map, pseudo_dir)

    cell_block = ["CELL_PARAMETERS angstrom\n"]
    for vector in atoms.get_cell():
        cell_block.append(f"{vector[0]:.14f} {vector[1]:.14f} {vector[2]:.14f}\n")

    pos_block = ["ATOMIC_POSITIONS angstrom\n"]
    for symbol, position in zip(atoms.get_chemical_symbols(), atoms.get_positions()):
        pos_block.append(f"{symbol} {position[0]:.10f} {position[1]:.10f} {position[2]:.10f}\n")

    lines = _replace_or_append_qe_block(lines, "ATOMIC_SPECIES", species_block)
    has_native_kpoints = any(line.strip().upper().startswith("K_POINTS") for line in lines)
    kpoints_block = None if has_native_kpoints else _build_kpoints_block(atoms, _load_qe_kpoints_config(workspace_root))
    if kpoints_block is not None:
        lines = _replace_or_append_qe_block(lines, "K_POINTS", kpoints_block)
    lines = _replace_or_append_qe_block(lines, "CELL_PARAMETERS", cell_block)
    lines = _replace_or_append_qe_block(lines, "ATOMIC_POSITIONS", pos_block)

    with open(template_path, "w", encoding="utf-8") as file:
        file.writelines(lines)


def prepare_qe_input(destination_folder, atoms, dir):
    workspace_root = get_upper_directory(5, dir)
    qe_input_path = os.path.join(workspace_root, "init", "qe.in")
    chek_file_exist(qe_input_path, "qe")
    qe_atoms = atoms.copy()
    qe_atoms.pbc = [True, True, True]
    write(os.path.join(destination_folder, "POSCAR"), qe_atoms, format="vasp")
    local_qe_input = os.path.join(destination_folder, "qe.in")
    shutil.copy(qe_input_path, local_qe_input)
    patch_qe_input(local_qe_input, qe_atoms, workspace_root)


def INPUT(dir, destination_folder, scf_cal_engine):
    path = get_upper_directory(5, dir)
    if scf_cal_engine == "abacus":
        input_path = os.path.join(path, "init", "INPUT")
        chek_file_exist(input_path, scf_cal_engine)
        shutil.copy(input_path, destination_folder)
        if _copy_if_exists(os.path.join(path, "init", "KPT"), destination_folder):
            _strip_key_value_lines(os.path.join(destination_folder, "INPUT"), {"kspacing"})
    elif scf_cal_engine == "cp2k":
        input_path = os.path.join(path, "init", "cp2k.inp")
        chek_file_exist(input_path, scf_cal_engine)
        shutil.copy(input_path, destination_folder)
    elif scf_cal_engine == "qe":
        input_path = os.path.join(path, "init", "qe.in")
        chek_file_exist(input_path, scf_cal_engine)
        shutil.copy(input_path, destination_folder)
    elif scf_cal_engine == "vasp":
        input_path = os.path.join(path, "init", "INCAR")
        chek_file_exist(input_path, scf_cal_engine)
        shutil.copy(input_path, destination_folder)
        if _copy_if_exists(os.path.join(path, "init", "KPOINTS"), destination_folder):
            _strip_key_value_lines(os.path.join(destination_folder, "INCAR"), {"KSPACING", "KGAMMA"})
