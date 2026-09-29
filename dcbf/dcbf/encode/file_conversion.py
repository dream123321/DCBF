from ase.io import iread
from ase.data import atomic_numbers,chemical_symbols
import io
import itertools
import os
from tqdm import tqdm
from ..path_names import MD_WORK_DIR
from ..npt_volume_filter import (
    classify_frame_volume,
    new_frame_volume_detail,
    _frame_timestep,
)

CFG_CELL_LINE = "{:16.8f} {:16.8f} {:16.8f}\n"
CFG_ATOM_LINE_WITH_FORCES = " {:6d} {:6d} {:16.8f} {:16.8f} {:16.8f} {:16.8f} {:16.8f} {:16.8f}\n"
CFG_ATOM_LINE_POS_ONLY = " {:6d} {:6d} {:16.8f} {:16.8f} {:16.8f}\n"
CFG_SCALAR_LINE = "{:16.8f}\n"
CFG_STRESS_LINE = "{:16.8f} {:16.8f} {:16.8f} {:16.8f} {:16.8f} {:16.8f}\n"
EXTXYZ_MATRIX_FORMAT = " ".join(["{: .8f}"] * 9)
EXTXYZ_POS_FORMAT = "{} {: .8f} {: .8f} {: .8f}\n"
EXTXYZ_POS_FORCE_FORMAT = "{} {: .8f} {: .8f} {: .8f} {: .8f} {: .8f} {: .8f}\n"


def _as_3x3_array(value):
    import numpy as np

    return np.asarray(value, dtype=float).reshape(3, 3)


def _ordered_elements(elements, sort_ele):
    if sort_ele:
        atomic_order = sorted(atomic_numbers[item] for item in elements)
        return [chemical_symbols[number] for number in atomic_order]
    return list(elements)


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
        atom_rows = []
        cursor = index + 8
        for _ in range(size):
            words = lines[cursor].split()
            row = {
                "type_index": int(words[1]),
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
                symbol = map_dic[str(row["type_index"])]
                if has_forces:
                    handle.write(EXTXYZ_POS_FORCE_FORMAT.format(symbol, *row["position"], *row["forces"]))
                else:
                    handle.write(EXTXYZ_POS_FORMAT.format(symbol, *row["position"]))


def xyz2cfg(elements, sort_ele, input_path, output_path):
    fin = iread(input_path)
    map_dic = {}
    ordered_elements = _ordered_elements(elements, sort_ele)
    for index, element in enumerate(ordered_elements):
        map_dic.update({element: index})

    ff = open(output_path,"w")
    for atoms in fin:
        #print(i)
        ele=atoms.get_chemical_symbols()
        nat=len(ele)
        cell = atoms.get_cell()
        pos = atoms.get_positions()
        force= atoms.get_forces()
        en=atoms.get_potential_energy()
        virial = _as_3x3_array(atoms.info['virial'])
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
        ff.write("""PlusStress:  xx          yy          zz          yz          xz          xy \n""")
        ff.write(CFG_STRESS_LINE.format(virial[0,0], virial[1,1], virial[2,2], virial[1,2], virial[0,2], virial[0,1]))
        ff.write("""END_CFG \n""")
    ff.close()

'''不收集丢原子的结构'''
def dump2cfg(input,out,volume_guard=None):
    fin = _stream_dump_atoms(input)
    try:
        first = next(fin)
    except StopIteration as exc:
        raise ValueError(f'Empty MD trajectory: {input}') from exc
    b = itertools.chain([first], fin)

    map_dic={}
    ele = list(set(first.get_chemical_symbols()))
    # if sort_ele:
    #     temp = sorted([atomic_numbers[i] for i in ele])
    #     ele = [chemical_symbols[a] for a in temp]
    #     print(temp, ele)
    # else:
    #     ele=ele
    for i in range(len(ele)):
        map_dic.update({ele[i]:chemical_symbols.index(ele[i])-1})
    #print(map_dic)
    detail = new_frame_volume_detail()
    expected_atom_count = len(first)

    def drop_after_failure(atoms, reason):
        detail["total"] += 1
        detail["dropped"] += 1
        if not detail.get("truncated", False):
            detail["truncated"] = True
            detail["first_failed_reason"] = reason
            detail["first_failed_step"] = _frame_timestep(atoms)

    ff = open(out,"a")
    for atoms in b:
        #print(i)
        if detail.get("truncated", False) and detail.get("first_failed_reason") == "lost_atoms":
            drop_after_failure(atoms, "lost_atoms")
            continue
        if len(atoms) != expected_atom_count:
            drop_after_failure(atoms, "lost_atoms")
            continue
        if not classify_frame_volume(atoms, volume_guard, detail):
            continue
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
    return detail["kept"], detail


def _stream_dump_atoms(path):
    from ase.io.lammpsrun import read_lammps_dump_text
    with open(path, encoding='utf-8') as handle:
        first = handle.readline()
        if first.strip() != 'ITEM: TIMESTEP':
            yield from iread(path)
            return
        lines = [first]
        for line in handle:
            if line.strip() == 'ITEM: TIMESTEP':
                yield read_lammps_dump_text(io.StringIO(''.join(lines)), index=-1)
                lines = []
            lines.append(line)
        if lines:
            yield read_lammps_dump_text(io.StringIO(''.join(lines)), index=-1)

def merge_cfg(file_path_1,file_path_2,output_file_path):
    with open(file_path_1, 'r') as file1:
        content_1 = file1.read()
    with open(file_path_2, 'r') as file2:
        content_2 = file2.read()
    merged_content = content_1 + '\n' + content_2
    with open(output_file_path, 'w') as output_file:
        output_file.write(merged_content)

def cfg2xyz(elements, sort_ele, cfg_file_path, xyz_file_path):
    map_dic = {}
    ordered_elements = _ordered_elements(elements, sort_ele)
    for index, element in enumerate(ordered_elements):
        map_dic.update({str(index): element})
    with open(cfg_file_path, encoding="utf-8") as handle:
        _write_extxyz_blocks(_stream_cfg_blocks(handle), map_dic, xyz_file_path)


def _stream_cfg_blocks(handle):
    lines = []
    for line in handle:
        if line.strip() == 'BEGIN_CFG':
            if lines:
                raise RuntimeError('Truncated CFG block before BEGIN_CFG')
            lines = [line]
        elif lines:
            lines.append(line)
            if line.strip() == 'END_CFG':
                yield from _cfg_block_iter(lines)
                lines = []
    if lines:
        raise RuntimeError('Truncated CFG block at end of file')

def remove(file):
    if os.path.exists(file):
        os.remove(file)

def merge_cfg_out(pwd,merge_file_dirs,cfg_name,out_name):
    import shutil
    md_cfg = os.path.join(pwd, MD_WORK_DIR, cfg_name)
    md_out = os.path.join(pwd, MD_WORK_DIR, out_name)
    # remove(md_cfg)
    # remove(md_out)
    with open(md_cfg, 'w') as outfile:
        for path in merge_file_dirs:
            single_md_cfg = os.path.join(path, cfg_name)
            with open(single_md_cfg, 'r') as infile:
                shutil.copyfileobj(infile, outfile, 1024 * 1024)
                outfile.write('\n')
    with open(md_out, 'w') as outfile:
        for path in merge_file_dirs:
            single_md_out = os.path.join(path, out_name)
            with open(single_md_out, 'r') as infile:
                shutil.copyfileobj(infile, outfile, 1024 * 1024)
                outfile.write('\n')
    # for path in merge_file_dirs:
    #     single_md_out = os.path.join(path, 'md.out')
    #     single_md_cfg = os.path.join(path, 'md.cfg')
    #     os.remove(single_md_out)
    #     os.remove(single_md_cfg)

if __name__ == '__main__':
    sort_ele = True
    input = 'force.0.dump'
    out = 'md.cfg'
    #dump2cfg(input, out)
    ele = ['Al','As','Ga']
    cfg_file_path, xyz_file_path = 'md.cfg','md.xyz'
    cfg2xyz(ele, sort_ele, cfg_file_path, xyz_file_path)
