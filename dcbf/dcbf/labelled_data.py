"""Strict EF/EFS readers and CFG export, without inventing missing labels."""
from pathlib import Path

import numpy as np
from ase.io import iread
from ase.stress import voigt_6_to_full_3x3_stress

from .das.file_conversion import _cfg_lattice


def labels(atoms, require_stress=False):
    try:
        energy = float(atoms.get_potential_energy())
        forces = np.asarray(atoms.get_forces(), dtype=float)
    except Exception as exc:
        raise ValueError("A labelled structure must have energy and forces") from exc
    if not np.isfinite(energy) or forces.shape != (len(atoms), 3) or not np.isfinite(forces).all():
        raise ValueError("Invalid/nonfinite energy or forces; expected forces shape (N,3)")
    virial = None
    if "virial" in atoms.info:
        raw = np.asarray(atoms.info['virial'], dtype=float)
        virial = voigt_6_to_full_3x3_stress(raw) if raw.size == 6 else raw.reshape(3, 3)
    elif atoms.calc is not None and 'stress' in atoms.calc.results:
        virial = -np.asarray(atoms.get_stress(voigt=False), dtype=float) * atoms.get_volume()
    elif 'stress' in atoms.info:
        raw = np.asarray(atoms.info['stress'], dtype=float)
        stress = voigt_6_to_full_3x3_stress(raw) if raw.size == 6 else raw.reshape(3, 3)
        virial = -stress * atoms.get_volume()
    if virial is not None and not np.isfinite(virial).all():
        raise ValueError('Nonfinite stress/virial label')
    if require_stress and virial is None:
        raise ValueError('EFS training requires a stress or virial label on every structure')
    if not np.isfinite(atoms.positions).all() or not np.isfinite(atoms.cell.array).all():
        raise ValueError('Nonfinite geometry')
    return energy, forces, virial


def inspect_labels(path, mode='auto'):
    if mode not in ('auto', 'ef', 'efs'):
        raise ValueError('training.labels must be auto, ef, or efs')
    count = stress_count = 0
    for i, atoms in enumerate(iread(str(path), format='extxyz')):
        try:
            _, _, virial = labels(atoms, require_stress=mode == 'efs')
        except Exception as exc:
            raise ValueError(f'Invalid labels in {path}, frame {i}: {exc}') from exc
        count += 1
        stress_count += virial is not None
    if not count:
        raise ValueError(f'Empty labelled dataset: {path}')
    effective = 'efs' if mode != 'ef' and stress_count == count else 'ef'
    return {'frames': count, 'stress_frames': stress_count, 'requested': mode, 'effective': effective}


def write_training_cfg(path, output, elements, mode='auto'):
    """Keep the checkpoint's complete species order, even for sparse datasets."""
    stats = inspect_labels(path, mode)
    mapping = {element: i for i, element in enumerate(elements)}
    if len(mapping) != len(elements):
        raise ValueError('Duplicate elements in checkpoint mapping')
    with Path(output).open('w', encoding='utf-8') as stream:
        for frame, atoms in enumerate(iread(str(path), format='extxyz')):
            energy, forces, virial = labels(atoms)
            symbols = atoms.get_chemical_symbols()
            unknown = set(symbols) - set(mapping)
            if unknown:
                raise ValueError(f'Dataset elements missing from checkpoint mapping: {unknown}')
            stream.write(f'BEGIN_CFG\n Size\n {len(atoms)}\n')
            lattice = _cfg_lattice(atoms)
            if lattice is not None:
                stream.write(' Supercell\n')
                for row in lattice:
                    stream.write(' '.join(f'{x:.16g}' for x in row) + '\n')
            stream.write('AtomData: id type cartes_x cartes_y cartes_z fx fy fz\n')
            for i, (symbol, position, force) in enumerate(zip(symbols, atoms.positions, forces)):
                stream.write(f'{i+1} {mapping[symbol]} ' + ' '.join(f'{x:.16g}' for x in [*position, *force]) + '\n')
            stream.write(f'Energy\n {energy:.16g}\n')
            if stats['effective'] == 'efs':
                stream.write('PlusStress: xx yy zz yz xz xy\n')
                stream.write(' '.join(f'{x:.16g}' for x in
                    (virial[0, 0], virial[1, 1], virial[2, 2], virial[1, 2], virial[0, 2], virial[0, 1])) + '\n')
            stream.write('END_CFG\n')
    return stats
