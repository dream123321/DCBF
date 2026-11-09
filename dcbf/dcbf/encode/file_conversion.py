from ase.io import iread
from ase.data import atomic_numbers,chemical_symbols
import io
import itertools
import os
from pathlib import Path
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


def _cfg_lattice(atoms):
    import numpy as np

    cell = np.asarray(atoms.get_cell(), dtype=float).reshape(3, 3)
    pbc = np.asarray(getattr(atoms, "pbc", (False, False, False)), dtype=bool)
    if not np.isfinite(cell).all():
        raise ValueError("ASE cell contains non-finite values")
    if pbc.shape != (3,) or not pbc.any():
        return None
    lattice = cell.copy()
    lattice[~pbc, :] = 0.0
    return lattice if np.any(np.abs(lattice) > 1.0e-12) else None


def _write_cfg_lattice(handle, atoms):
    lattice = _cfg_lattice(atoms)
    if lattice is None:
        return None
    handle.write(" Supercell\n")
    for row in lattice:
        handle.write(CFG_CELL_LINE.format(*row))
    return lattice


def _ordered_elements(elements, sort_ele):
    if sort_ele:
        atomic_order = sorted(atomic_numbers[item] for item in elements)
        return [chemical_symbols[number] for number in atomic_order]
    return list(elements)


def _dump_element_mapping(first_atoms, elements=None, sort_ele=True):
    """Map one LAMMPS dump frame onto the model's element order.

    DCBF writes dumps with only the LAMMPS ``type`` column.  ASE decodes that
    integer as an atomic number (type 1 -> H), so the decoded chemical symbols
    are not the model elements.  When the raw type array is available, map it
    directly onto the ordered elements written into data.in.
    """
    if elements is None:
        # Compatibility for direct legacy callers that do not provide a model
        # order.  The sampling path always supplies elements explicitly.
        return {
            symbol: chemical_symbols.index(symbol) - 1
            for symbol in set(first_atoms.get_chemical_symbols())
        }

    ordered = _ordered_elements(elements, sort_ele)
    types = first_atoms.arrays.get("type")
    if types is not None:
        present = {int(value) for value in types}
        invalid = sorted(
            value for value in present if value < 1 or value > len(ordered)
        )
        if invalid:
            raise ValueError(
                "MD trajectory contains LAMMPS atom types outside the model "
                f"element order ({' '.join(ordered)}): {invalid}"
            )
        return {str(value): value - 1 for value in present}

    mapping = {symbol: index for index, symbol in enumerate(ordered)}
    unknown = sorted(set(first_atoms.get_chemical_symbols()).difference(mapping))
    if unknown:
        raise ValueError(
            "MD trajectory contains elements absent from the model element order: "
            + ", ".join(unknown)
        )
    return mapping


def _cfg_type_indices(atoms, mapping, use_type_column):
    """Return the zero-based CFG type index for every atom in one MD frame."""
    if use_type_column:
        return [mapping[str(int(value))] for value in atoms.arrays["type"]]
    return [mapping[symbol] for symbol in atoms.get_chemical_symbols()]


def _cfg_block_iter(lines):
    index = 0
    total = len(lines)
    while index < total:
        if lines[index].strip() != "BEGIN_CFG":
            index += 1
            continue

        size = int(lines[index + 2].split()[0])
        cursor = index + 3
        while cursor < total and not lines[cursor].strip():
            cursor += 1
        lattice = None
        if cursor < total and lines[cursor].strip().lower() in {"supercell", "supercell:"}:
            lattice = [
                float(item)
                for item in (
                    lines[cursor + 1].split()
                    + lines[cursor + 2].split()
                    + lines[cursor + 3].split()
                )
            ]
            if not any(abs(value) > 1.0e-12 for value in lattice):
                lattice = None
            cursor += 4
        while cursor < total and not lines[cursor].strip():
            cursor += 1
        if cursor >= total:
            raise ValueError("CFG block is missing AtomData")
        atom_rows = []
        cursor += 1
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
    owns_handle = not hasattr(out_path, "write")
    handle = open(out_path, "w", encoding="utf-8") if owns_handle else out_path
    try:
        for block in blocks:
            handle.write(str(block["size"]) + "\n")
            header = ""
            if block["lattice"] is not None:
                header += f'Lattice="{EXTXYZ_MATRIX_FORMAT.format(*block["lattice"])}" '
            header += "Properties=species:S:1:pos:R:3"
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
            header += f' pbc="{"T T T" if block["lattice"] is not None else "F F F"}"\n'
            handle.write(header)

            for row in block["atom_rows"]:
                symbol = map_dic[str(row["type_index"])]
                if has_forces:
                    handle.write(EXTXYZ_POS_FORCE_FORMAT.format(symbol, *row["position"], *row["forces"]))
                else:
                    handle.write(EXTXYZ_POS_FORMAT.format(symbol, *row["position"]))
    finally:
        if owns_handle:
            handle.close()


def xyz2cfg(elements, sort_ele, input_path, output_path, allow_missing_labels=False):
    if allow_missing_labels:
        mapping = {symbol: i for i, symbol in enumerate(_ordered_elements(elements, sort_ele))}
        with open(output_path, 'w', encoding='utf-8') as stream:
            for atoms in iread(input_path):
                stream.write(f'BEGIN_CFG\n Size\n {len(atoms)}\n')
                _write_cfg_lattice(stream, atoms)
                stream.write('AtomData: id type cartes_x cartes_y cartes_z\n')
                for i, (symbol, position) in enumerate(zip(atoms.get_chemical_symbols(), atoms.positions)):
                    stream.write(CFG_ATOM_LINE_POS_ONLY.format(i+1, mapping[symbol], *position))
                stream.write('END_CFG\n')
        return
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
        pos = atoms.get_positions()
        force= atoms.get_forces()
        en=atoms.get_potential_energy()
        virial = _as_3x3_array(atoms.info['virial'])
        ff.write("""BEGIN_CFG\n""")
        ff.write(""" Size\n""")
        ff.write("""  {:6} \n""".format(nat))
        _write_cfg_lattice(ff, atoms)
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
def dump2cfg(input,out,volume_guard=None,elements=None,sort_ele=True):
    fin = _stream_dump_atoms(input)
    try:
        first = next(fin)
    except StopIteration as exc:
        raise ValueError(f'Empty MD trajectory: {input}') from exc
    b = itertools.chain([first], fin)

    map_dic = _dump_element_mapping(first, elements, sort_ele)
    use_type_column = elements is not None and "type" in first.arrays
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
        nat=len(atoms)
        pos = atoms.get_positions()
        cfg_types=_cfg_type_indices(atoms, map_dic, use_type_column)
        #force= atoms.get_forces()
        #en=atoms.get_potential_energy()
        #virial=atoms.info['virial']
        ff.write("""BEGIN_CFG\n""")
        ff.write(""" Size\n""")
        ff.write("""  {:6} \n""".format(nat))
        _write_cfg_lattice(ff, atoms)
        ff.write("""AtomData:  id type       cartes_x      cartes_y      cartes_z     \n""")
        # for i in range(nat):
        #     ff.write(
        #         """ {:6} {:6} {:12.6f} {:12.6f} {:12.6f} {:12.6f} {:12.6f} {:12.6f}\n""".format(i + 1, map_dic[ele[i]], pos[i, 0], pos[i, 1], pos[i, 2],
        #                                                                                      force[i,0], force[i,1], force[i,2]))
        for i in range(nat):
            ff.write(CFG_ATOM_LINE_POS_ONLY.format(i + 1, cfg_types[i], pos[i, 0], pos[i, 1], pos[i, 2]))
        #ff.write("""Energy \n""")
        #ff.write(f"""\t{en} \n""")
        #ff.write("""PlusStress:  xx          yy          zz          yz          xz          xy \n""")
        #ff.write(f"\t{virial[0,0]}  \t{virial[1,1]}  \t{virial[2,2]}  \t{virial[1,2]}  \t{virial[0,2]}  \t{virial[0,1]} \n")
        ff.write("""END_CFG \n""")
    ff.close()
    return detail["kept"], detail


def dump2cfg_sharded(input_path, output_dir, prefix, volume_guard=None, chunk_size=1000000,
                     elements=None, sort_ele=True):
    """Stream a LAMMPS dump into ordered CFG shards.

    The filtering rules intentionally mirror ``dump2cfg``.  Only the file
    boundary changes, so the kept frame count and frame order remain identical
    to the historical single-file path.
    """
    chunk_size = max(1, int(chunk_size))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fin = _stream_dump_atoms(input_path)
    try:
        first = next(fin)
    except StopIteration as exc:
        raise ValueError(f'Empty MD trajectory: {input_path}') from exc
    blocks = itertools.chain([first], fin)

    map_dic = _dump_element_mapping(first, elements, sort_ele)
    use_type_column = elements is not None and "type" in first.arrays
    detail = new_frame_volume_detail()
    expected_atom_count = len(first)
    part_paths = []
    part_counts = []
    handle = None
    current_count = 0

    def drop_after_failure(atoms, reason):
        detail["total"] += 1
        detail["dropped"] += 1
        if not detail.get("truncated", False):
            detail["truncated"] = True
            detail["first_failed_reason"] = reason
            detail["first_failed_step"] = _frame_timestep(atoms)

    def open_part():
        nonlocal handle, current_count
        if handle is not None:
            handle.close()
        path = output_dir / f"{prefix}.part_{len(part_paths):04d}.cfg"
        handle = path.open("w", encoding="utf-8")
        part_paths.append(str(path))
        part_counts.append(0)
        current_count = 0

    try:
        for atoms in blocks:
            if detail.get("truncated", False) and detail.get("first_failed_reason") == "lost_atoms":
                drop_after_failure(atoms, "lost_atoms")
                continue
            if len(atoms) != expected_atom_count:
                drop_after_failure(atoms, "lost_atoms")
                continue
            if not classify_frame_volume(atoms, volume_guard, detail):
                continue
            if handle is None or current_count >= chunk_size:
                open_part()

            nat = len(atoms)
            positions = atoms.get_positions()
            cfg_types = _cfg_type_indices(atoms, map_dic, use_type_column)
            handle.write("BEGIN_CFG\n")
            handle.write(" Size\n")
            handle.write("  {:6} \n".format(nat))
            _write_cfg_lattice(handle, atoms)
            handle.write("AtomData:  id type       cartes_x      cartes_y      cartes_z     \n")
            for index in range(nat):
                handle.write(
                    CFG_ATOM_LINE_POS_ONLY.format(
                        index + 1,
                        cfg_types[index],
                        positions[index, 0],
                        positions[index, 1],
                        positions[index, 2],
                    )
                )
            handle.write("END_CFG \n")
            current_count += 1
            part_counts[-1] += 1
    finally:
        if handle is not None:
            handle.close()

    return detail["kept"], detail, part_paths, part_counts


def _dump_type_column(block):
    """Return the raw LAMMPS type ids in one dump block, or None.

    ASE 3.28 stores the dump ``type`` column on the returned Atoms object,
    while ASE 3.26 (the ase-abacus fork) does not.  Parsing it here keeps the
    model-order mapping independent of the ASE build.
    """
    import numpy as np

    lines = block.splitlines()
    natoms = None
    columns = None
    start = None
    for index, line in enumerate(lines):
        if line.startswith('ITEM: NUMBER OF ATOMS') and index + 1 < len(lines):
            natoms = int(lines[index + 1].split()[0])
        elif line.startswith('ITEM: ATOMS'):
            columns = line.split()[2:]
            start = index + 1
    if natoms is None or columns is None or 'type' not in columns:
        return None
    type_index = columns.index('type')
    rows = lines[start:start + natoms]
    if len(rows) != natoms:
        return None
    try:
        return np.array(
            [int(float(row.split()[type_index])) for row in rows], dtype=int
        )
    except (IndexError, ValueError):
        return None


def _stream_dump_atoms(path):
    from ase.io.lammpsrun import read_lammps_dump_text

    def _atoms_from_block(block):
        atoms = read_lammps_dump_text(io.StringIO(block), index=-1)
        types = _dump_type_column(block)
        if types is not None and len(types) == len(atoms):
            atoms.set_array('type', types)
        return atoms

    with open(path, encoding='utf-8') as handle:
        first = handle.readline()
        if first.strip() != 'ITEM: TIMESTEP':
            yield from iread(path)
            return
        lines = [first]
        for line in handle:
            if line.strip() == 'ITEM: TIMESTEP':
                yield _atoms_from_block(''.join(lines))
                lines = []
            lines.append(line)
        if lines:
            yield _atoms_from_block(''.join(lines))

def merge_cfg(file_path_1,file_path_2,output_file_path):
    with open(file_path_1, 'r') as file1:
        content_1 = file1.read()
    with open(file_path_2, 'r') as file2:
        content_2 = file2.read()
    merged_content = content_1 + '\n' + content_2
    with open(output_file_path, 'w') as output_file:
        output_file.write(merged_content)

def cfg2xyz(elements, sort_ele, cfg_file_path, xyz_file_path):
    from contextlib import ExitStack

    map_dic = {}
    ordered_elements = _ordered_elements(elements, sort_ele)
    for index, element in enumerate(ordered_elements):
        map_dic.update({str(index): element})
    if isinstance(cfg_file_path, (str, os.PathLike)):
        cfg_paths = [cfg_file_path]
    else:
        cfg_paths = list(cfg_file_path)
    with open(xyz_file_path, "w", encoding="utf-8") as output:
        with ExitStack() as stack:
            for cfg_path in cfg_paths:
                handle = stack.enter_context(open(cfg_path, encoding="utf-8"))
                _write_extxyz_blocks(_stream_cfg_blocks(handle), map_dic, output)


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
