from ase.io import iread
from ase.data import atomic_numbers,chemical_symbols
import numpy as np

from ..tensor_utils import extract_virial_matrix

CFG_CELL_LINE = "{:16.8f} {:16.8f} {:16.8f}\n"
CFG_ATOM_LINE_WITH_FORCES = " {:6d} {:6d} {:16.8f} {:16.8f} {:16.8f} {:16.8f} {:16.8f} {:16.8f}\n"
CFG_ATOM_LINE_POS_ONLY = " {:6d} {:6d} {:16.8f} {:16.8f} {:16.8f}\n"
CFG_SCALAR_LINE = "{:16.8f}\n"
CFG_STRESS_LINE = "{:16.8f} {:16.8f} {:16.8f} {:16.8f} {:16.8f} {:16.8f}\n"
EXTXYZ_MATRIX_FORMAT = " ".join(["{: .8f}"] * 9)
EXTXYZ_POS_FORMAT = "{} {: .8f} {: .8f} {: .8f}\n"
EXTXYZ_POS_FORCE_FORMAT = "{} {: .8f} {: .8f} {: .8f} {: .8f} {: .8f} {: .8f}\n"
EXTXYZ_PRESERVED_INFO_KEYS = ("main", "label")


def _as_3x3_array(value):
    return np.asarray(value, dtype=float).reshape(3, 3)


def _format_extxyz_info_value(value):
    if isinstance(value, (bool, np.bool_)):
        return "T" if bool(value) else "F"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.8f}"
    text = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{text}"'


def _normalized_extxyz_payload(atoms):
    positions = np.asarray(atoms.get_positions(), dtype=float)
    payload = {
        "symbols": list(atoms.get_chemical_symbols()),
        "lattice": list(np.asarray(atoms.get_cell(), dtype=float).reshape(-1)),
        "positions": positions,
        "forces": None,
        "energy": None,
        "virial": None,
        "extra_info": {},
    }

    if "forces" in atoms.arrays:
        payload["forces"] = np.asarray(atoms.arrays["forces"], dtype=float)
    else:
        try:
            payload["forces"] = np.asarray(atoms.get_forces(), dtype=float)
        except Exception:
            payload["forces"] = None

    if "energy" in atoms.info:
        payload["energy"] = float(atoms.info["energy"])
    else:
        try:
            payload["energy"] = float(atoms.get_potential_energy())
        except Exception:
            payload["energy"] = None

    if "virial" in atoms.info:
        payload["virial"] = list(_as_3x3_array(atoms.info["virial"]).reshape(-1))
    for key in EXTXYZ_PRESERVED_INFO_KEYS:
        if key in atoms.info:
            payload["extra_info"][key] = atoms.info[key]
    return payload


def write_normalized_extxyz(path, images, append=False):
    if hasattr(images, "get_positions"):
        images = [images]

    mode = "a" if append else "w"
    with open(path, mode, encoding="utf-8") as handle:
        for atoms in images:
            payload = _normalized_extxyz_payload(atoms)
            handle.write(str(len(payload["symbols"])) + "\n")
            header = (
                f'Lattice="{EXTXYZ_MATRIX_FORMAT.format(*payload["lattice"])}" '
                "Properties=species:S:1:pos:R:3"
            )
            if payload["forces"] is not None:
                header += ":forces:R:3"
            if payload["energy"] is not None:
                header += f" energy={payload['energy']:.8f}"
            if payload["virial"] is not None:
                header += f' virial="{EXTXYZ_MATRIX_FORMAT.format(*payload["virial"])}"'
            for key, value in payload["extra_info"].items():
                header += f" {key}={_format_extxyz_info_value(value)}"
            header += ' pbc="T T T" \n'
            handle.write(header)

            for index, symbol in enumerate(payload["symbols"]):
                position = payload["positions"][index]
                if payload["forces"] is not None:
                    handle.write(EXTXYZ_POS_FORCE_FORMAT.format(symbol, *position, *payload["forces"][index]))
                else:
                    handle.write(EXTXYZ_POS_FORMAT.format(symbol, *position))


def _cfg_block_iter(lines):
    index = 0
    total = len(lines)
    while index < total:
        if lines[index].strip() != "BEGIN_CFG":
            index += 1
            continue

        size = int(lines[index + 2].split()[0])
        lattice = [
            float(item)
            for item in (lines[index + 4].split() + lines[index + 5].split() + lines[index + 6].split())
        ]
        atom_header = lines[index + 7].strip()
        atom_rows = []
        cursor = index + 8
        for _ in range(size):
            words = lines[cursor].split()
            row = {
                "type_index": int(words[1]),
                "symbol_data": words,
                "position": [float(words[2]), float(words[3]), float(words[4])],
            }
            if len(words) >= 8:
                row["forces"] = [float(words[5]), float(words[6]), float(words[7])]
            atom_rows.append(row)
            cursor += 1

        energy = None
        stress = None
        while cursor < total:
            stripped = lines[cursor].strip()
            if stripped == "Energy":
                cursor += 1
                if cursor < total:
                    energy = float(lines[cursor].split()[0])
            elif stripped.startswith("PlusStress"):
                cursor += 1
                if cursor < total:
                    stress = [float(item) for item in lines[cursor].split()]
            elif stripped == "END_CFG":
                break
            cursor += 1

        yield {
            "size": size,
            "lattice": lattice,
            "atom_header": atom_header,
            "atom_rows": atom_rows,
            "energy": energy,
            "stress": stress,
        }
        index = cursor + 1


def _write_extxyz_blocks(blocks, map_dic, out_path):
    with open(out_path, "w", encoding="utf-8") as handle:
        for block in blocks:
            handle.write(str(block["size"]) + "\n")
            header = (
                f'Lattice="{EXTXYZ_MATRIX_FORMAT.format(*block["lattice"])}" '
                "Properties=species:S:1:pos:R:3"
            )

            has_forces = all("forces" in row for row in block["atom_rows"])
            if has_forces:
                header += ":forces:R:3"
            if block["energy"] is not None:
                header += f" energy={block['energy']:.8f}"
            if block["stress"] is not None:
                stress = block["stress"]
                virial = [
                    stress[0], stress[5], stress[4],
                    stress[5], stress[1], stress[3],
                    stress[4], stress[3], stress[2],
                ]
                header += f' virial="{EXTXYZ_MATRIX_FORMAT.format(*virial)}"'
            header += ' pbc="T T T" \n'
            handle.write(header)

            for row in block["atom_rows"]:
                symbol = map_dic[row["type_index"]]
                if has_forces:
                    handle.write(EXTXYZ_POS_FORCE_FORMAT.format(symbol, *row["position"], *row["forces"]))
                else:
                    handle.write(EXTXYZ_POS_FORMAT.format(symbol, *row["position"]))


def xyz2cfg(ele, ele_model, input, out, allow_missing_labels=False):
    fin=iread(input)
    map_dic={}
    if ele_model == 1:
        temp = sorted([atomic_numbers[i] for i in ele])
        ele = [chemical_symbols[a] for a in temp]
    elif ele_model ==2:
        ele=ele
    for i in range(len(ele)):
        map_dic.update({ele[i]:i})

    b=list(fin)
    ff = open(out,"w")
    for i in range(len(b)):
        #print(i)
        atoms=b[i]
        ele=atoms.get_chemical_symbols()
        nat=len(ele)
        cell = atoms.get_cell()
        pos = atoms.get_positions()
        payload = _normalized_extxyz_payload(atoms)
        force = payload["forces"]
        en = payload["energy"]
        if force is None:
            if not allow_missing_labels:
                force = atoms.get_forces()
            else:
                force = np.zeros((nat, 3), dtype=float)
        if en is None:
            if not allow_missing_labels:
                en = atoms.get_potential_energy()
            else:
                en = 0.0
        virial = extract_virial_matrix(atoms)
        ff.write("""BEGIN_CFG\n""")
        ff.write(""" Size\n""")
        ff.write("""  {:6} \n""".format(nat))
        ff.write(""" Supercell \n""")
        ff.write(CFG_CELL_LINE.format(cell[0, 0], cell[0, 1], cell[0, 2]))
        ff.write(CFG_CELL_LINE.format(cell[1, 0], cell[1, 1], cell[1, 2]))
        ff.write(CFG_CELL_LINE.format(cell[2, 0], cell[2, 1], cell[2, 2]))
        ff.write("""AtomData:  id type       cartes_x      cartes_y      cartes_z     fx          fy          fz\n""")
        for i in range(nat):
            ff.write(CFG_ATOM_LINE_WITH_FORCES.format(i + 1, map_dic[ele[i]], pos[i, 0], pos[i, 1], pos[i, 2], force[i,0], force[i,1], force[i,2]))
        ff.write("""Energy \n""")
        ff.write(CFG_SCALAR_LINE.format(en))
        if virial is not None:
            ff.write("""PlusStress:  xx          yy          zz          yz          xz          xy \n""")
            ff.write(CFG_STRESS_LINE.format(virial[0,0], virial[1,1], virial[2,2], virial[1,2], virial[0,2], virial[0,1]))
        ff.write("""END_CFG \n""")
    ff.close()

def sort_xyz2cfg(input,out):
    fin = iread(input)
    b = list(fin)
    map_dic = {}
    ele_set = set()
    for i in b:
        ele_set = ele_set | set(i.get_chemical_symbols())
    ele_ = list(ele_set)
    order = True
    if (order):
        ele_order = {}
        for i in ele_:
            ele_order.update({i: atomic_numbers[i]})
        ele_ = sorted(ele_, key=lambda x: ele_order[x])
    for i in range(len(ele_)):
        map_dic.update({ele_[i]: i})
    ff = open(out, "w")
    for i in range(len(b)):
        # print(i)
        atoms = b[i]
        ele = atoms.get_chemical_symbols()
        nat = len(ele)
        cell = atoms.get_cell()
        pos = atoms.get_positions()
        force = atoms.get_forces()
        en = atoms.get_potential_energy()
        virial = extract_virial_matrix(atoms)
        ff.write("""BEGIN_CFG\n""")
        ff.write(""" Size\n""")
        ff.write("""  {:6} \n""".format(nat))
        ff.write(""" Supercell \n""")
        ff.write(CFG_CELL_LINE.format(cell[0, 0], cell[0, 1], cell[0, 2]))
        ff.write(CFG_CELL_LINE.format(cell[1, 0], cell[1, 1], cell[1, 2]))
        ff.write(CFG_CELL_LINE.format(cell[2, 0], cell[2, 1], cell[2, 2]))
        ff.write("""AtomData:  id type       cartes_x      cartes_y      cartes_z     fx          fy          fz\n""")
        for i in range(nat):
            ff.write(CFG_ATOM_LINE_WITH_FORCES.format(i + 1, map_dic[ele[i]], pos[i, 0], pos[i, 1], pos[i, 2], force[i, 0], force[i, 1], force[i, 2]))
        ff.write("""Energy \n""")
        ff.write(CFG_SCALAR_LINE.format(en))
        if virial is not None:
            ff.write("""PlusStress:  xx          yy          zz          yz          xz          xy \n""")
            ff.write(CFG_STRESS_LINE.format(virial[0, 0], virial[1, 1], virial[2, 2], virial[1, 2], virial[0, 2], virial[0, 1]))
        ff.write("""END_CFG \n""")
    ff.close()
    return ele_

def dump2cfg(ele_model,input,out):
    fin= iread(input)
    b = list(fin)
    map_dic={}
    ele = list(set(b[0].get_chemical_symbols()))
    if ele_model == 1:
        temp = sorted([atomic_numbers[i] for i in ele])
        ele = [chemical_symbols[a] for a in temp]
    elif ele_model ==2:
        ele=ele
    for i in range(len(ele)):
        map_dic.update({ele[i]:i})

    ff = open(out,"w")
    for i in range(len(b)):
        #print(i)
        atoms=b[i]
        ele=atoms.get_chemical_symbols()
        nat=len(ele)
        cell = atoms.get_cell()
        pos = atoms.get_positions()
        #force= atoms.get_forces()
        #en=atoms.get_potential_energy()
        #virial=atoms.info['virial']
        ff.write("""BEGIN_CFG\n""")
        ff.write(""" Size\n""")
        ff.write("""  {:6} \n""".format(nat))
        ff.write(""" Supercell \n""")
        ff.write(CFG_CELL_LINE.format(cell[0, 0], cell[0, 1], cell[0, 2]))
        ff.write(CFG_CELL_LINE.format(cell[1, 0], cell[1, 1], cell[1, 2]))
        ff.write(CFG_CELL_LINE.format(cell[2, 0], cell[2, 1], cell[2, 2]))
        ff.write("""AtomData:  id type       cartes_x      cartes_y      cartes_z     \n""")
        # for i in range(nat):
        #     ff.write(
        #         """ {:6} {:6} {:12.6f} {:12.6f} {:12.6f} {:12.6f} {:12.6f} {:12.6f}\n""".format(i + 1, map_dic[ele[i]], pos[i, 0], pos[i, 1], pos[i, 2],
        #                                                                                      force[i,0], force[i,1], force[i,2]))
        for i in range(nat):
            ff.write(CFG_ATOM_LINE_POS_ONLY.format(i + 1, map_dic[ele[i]], pos[i, 0], pos[i, 1], pos[i, 2]))
        #ff.write("""Energy \n""")
        #ff.write(f"""\t{en} \n""")
        #ff.write("""PlusStress:  xx          yy          zz          yz          xz          xy \n""")
        #ff.write(f"\t{virial[0,0]}  \t{virial[1,1]}  \t{virial[2,2]}  \t{virial[1,2]}  \t{virial[0,2]}  \t{virial[0,1]} \n")
        ff.write("""END_CFG \n""")
    ff.close()

def merge_cfg(file_path_1,file_path_2,output_file_path):
    with open(file_path_1, 'r') as file1:
        content_1 = file1.read()
    with open(file_path_2, 'r') as file2:
        content_2 = file2.read()
    merged_content = content_1 + '\n' + content_2
    with open(output_file_path, 'w') as output_file:
        output_file.write(merged_content)

def cfg2xyz(ele,ele_model,cfgs,out):
    map_dic = {}
    if ele_model == 1:
        ele_ = [atomic_numbers[i] for i in ele]
        ele_ = sorted(ele_)
        ele = [chemical_symbols[a] for a in ele_]
        for i in range(len(ele)):
            map_dic.update({i: ele[i]})
    elif ele_model == 2:
        for i in range(len(ele)):
            map_dic.update({i: ele[i]})
    with open(cfgs) as f:
        blocks = list(_cfg_block_iter(f.readlines()))
    _write_extxyz_blocks(blocks, map_dic, out)

