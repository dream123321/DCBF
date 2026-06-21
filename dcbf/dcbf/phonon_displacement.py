from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Union

from ase import Atoms
import numpy as np
import spglib


AXIS_DIRECTIONS = np.array(
    [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ],
    dtype=np.int64,
)

DIAGONAL_DIRECTIONS = np.array(
    [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1],
        [1, -1, 0],
        [1, 0, -1],
        [0, 1, -1],
        [1, 1, 1],
        [1, 1, -1],
        [1, -1, 1],
        [-1, 1, 1],
    ],
    dtype=np.int64,
)


@dataclass
class PhononDisplacementSpec:
    atom_index: int
    direction: np.ndarray
    cartesian_displacement: np.ndarray


def generate_phonopy_like_supercells(
    atoms: Atoms,
    supercell_matrix,
    distance: float = 0.01,
    plusminus: Union[Literal["auto"], bool] = "auto",
    is_diagonal: bool = True,
    is_trigonal: bool = False,
    symprec: float = 1e-5,
):
    """Generate finite-displacement supercells compatible with phonopy.

    This reproduces the same displacement-direction search logic as phonopy's
    finite-displacement workflow without depending on phonopy itself.
    """
    supercell_matrix = _normalize_supercell_matrix(supercell_matrix)
    supercell = build_phonopy_like_supercell(atoms, supercell_matrix, symprec=symprec)
    directions = get_phonopy_like_displacement_directions(
        supercell,
        plusminus=plusminus,
        is_diagonal=is_diagonal,
        is_trigonal=is_trigonal,
        symprec=symprec,
    )
    specs = directions_to_displacement_specs(directions, distance, supercell)

    displaced_supercells = []
    for spec in specs:
        displaced = supercell.copy()
        displaced.positions[spec.atom_index] += spec.cartesian_displacement
        displaced.pbc = [True, True, True]
        displaced.info = dict(displaced.info)
        displaced.info["phonon_atom_index"] = int(spec.atom_index)
        displaced.info["phonon_direction"] = spec.direction.astype(int).tolist()
        displaced.info["phonon_displacement"] = spec.cartesian_displacement.tolist()
        displaced_supercells.append(displaced)
    return displaced_supercells, specs, supercell


def build_phonopy_like_supercell(atoms: Atoms, supercell_matrix, symprec: float = 1e-5):
    supercell_matrix = _normalize_supercell_matrix(supercell_matrix)
    multiplicities = _get_surrounding_frame(supercell_matrix)
    trim_frame = np.array(
        [
            supercell_matrix[0] / float(multiplicities[0]),
            supercell_matrix[1] / float(multiplicities[1]),
            supercell_matrix[2] / float(multiplicities[2]),
        ],
        dtype=float,
    )
    simple_supercell = _build_simple_supercell(atoms, multiplicities, supercell_matrix)
    return _trim_cell_like_phonopy(trim_frame, simple_supercell, symprec=symprec)


def get_phonopy_like_displacement_directions(
    supercell: Atoms,
    plusminus: Union[Literal["auto"], bool] = "auto",
    is_diagonal: bool = True,
    is_trigonal: bool = False,
    symprec: float = 1e-5,
):
    if plusminus not in {"auto", True, False}:
        raise ValueError("plusminus must be 'auto', true, or false")

    symmetry_dataset = spglib.get_symmetry_dataset(_ase_atoms_to_spglib_cell(supercell), symprec=symprec)
    if symmetry_dataset is None:
        raise RuntimeError("spglib failed to determine symmetry for phonon displacements")

    rotations = np.asarray(symmetry_dataset.rotations, dtype=np.int64, order="C")
    translations = np.asarray(symmetry_dataset.translations, dtype=float, order="C")
    equivalent_atoms = np.asarray(symmetry_dataset.equivalent_atoms, dtype=np.int64)
    independent_atoms = np.array([index for index, mapped in enumerate(equivalent_atoms) if index == mapped], dtype=np.int64)
    search_directions = DIAGONAL_DIRECTIONS if is_diagonal else AXIS_DIRECTIONS

    displacements = []
    for atom_index in independent_atoms:
        site_symmetry = _get_site_symmetry(
            atom_index,
            supercell.cell.array,
            supercell.get_scaled_positions(),
            rotations,
            translations,
            symprec,
        )
        for displacement_direction in _get_displacements_for_one_site(
            site_symmetry,
            search_directions,
            is_trigonal=is_trigonal,
        ):
            direction_list = [int(atom_index), int(displacement_direction[0]), int(displacement_direction[1]), int(displacement_direction[2])]
            displacements.append(direction_list)
            if plusminus == "auto":
                if _needs_minus_displacement(displacement_direction, site_symmetry):
                    displacements.append([int(atom_index), -int(displacement_direction[0]), -int(displacement_direction[1]), -int(displacement_direction[2])])
            elif plusminus is True:
                displacements.append([int(atom_index), -int(displacement_direction[0]), -int(displacement_direction[1]), -int(displacement_direction[2])])
    return displacements


def directions_to_displacement_specs(displacement_directions, distance: float, supercell: Atoms):
    lattice = np.asarray(supercell.cell.array, dtype=float)
    specs = []
    for displacement in displacement_directions:
        atom_index = int(displacement[0])
        direction = np.asarray(displacement[1:], dtype=np.int64)
        cartesian = direction @ lattice
        norm = np.linalg.norm(cartesian)
        if norm < 1e-12:
            raise ValueError("encountered a zero-length displacement direction")
        cartesian = cartesian * (float(distance) / norm)
        specs.append(
            PhononDisplacementSpec(
                atom_index=atom_index,
                direction=direction,
                cartesian_displacement=np.asarray(cartesian, dtype=float),
            )
        )
    return specs


def _get_displacements_for_one_site(site_symmetry, directions, is_trigonal: bool = False):
    found_one = _find_single_generator(site_symmetry, directions)
    if found_one is not None:
        return [found_one]

    found_two = _find_two_generators(site_symmetry, directions)
    if found_two is not None:
        first_direction, second_direction, rotation_index = found_two
        if is_trigonal and _is_trigonal_axis(site_symmetry[rotation_index]):
            second_rotated = first_direction @ site_symmetry[rotation_index].T
            third_rotated = second_rotated @ site_symmetry[rotation_index].T
            return [first_direction, second_rotated, third_rotated, second_direction]
        return [first_direction, second_direction]

    return [directions[0], directions[1], directions[2]]


def _find_single_generator(site_symmetry, directions):
    for direction in directions:
        rotated_directions = [direction @ rotation.T for rotation in site_symmetry]
        for first_index in range(len(site_symmetry)):
            for second_index in range(first_index + 1, len(site_symmetry)):
                determinant = _determinant3([direction, rotated_directions[first_index], rotated_directions[second_index]])
                if determinant != 0:
                    return direction
    return None


def _find_two_generators(site_symmetry, directions):
    for direction in directions:
        rotated_directions = [direction @ rotation.T for rotation in site_symmetry]
        for rotation_index in range(len(site_symmetry)):
            rotated_direction = rotated_directions[rotation_index]
            for second_direction in directions:
                determinant = _determinant3([direction, rotated_direction, second_direction])
                if determinant != 0:
                    return direction, second_direction, rotation_index
    return None


def _get_site_symmetry(atom_index, lattice, scaled_positions, rotations, translations, symprec):
    position = scaled_positions[atom_index]
    site_symmetry = []
    if len(rotations) != len(translations):
        raise ValueError("rotations and translations must have the same length")
    for rotation, translation in zip(rotations, translations):
        rotated_position = position @ rotation.T + translation
        diff = position - rotated_position
        diff -= np.rint(diff)
        cartesian_diff = diff @ lattice
        if np.linalg.norm(cartesian_diff) < symprec:
            site_symmetry.append(rotation)
    return np.asarray(site_symmetry, dtype=np.int64)


def _needs_minus_displacement(direction, site_symmetry):
    for rotation in site_symmetry:
        rotated_direction = direction @ rotation.T
        if not (rotated_direction + direction).any():
            return False
    return True


def _is_trigonal_axis(rotation):
    return bool(np.array_equal(rotation @ rotation @ rotation, np.eye(3, dtype=int)))


def _determinant3(vectors):
    matrix = np.asarray(vectors, dtype=float)
    return int(round(np.linalg.det(matrix)))


def _get_surrounding_frame(supercell_matrix):
    matrix = np.asarray(supercell_matrix, dtype=int)
    axes = np.array(
        [
            [0, 0, 0],
            matrix[:, 0],
            matrix[:, 1],
            matrix[:, 2],
            matrix[:, 1] + matrix[:, 2],
            matrix[:, 2] + matrix[:, 0],
            matrix[:, 0] + matrix[:, 1],
            matrix[:, 0] + matrix[:, 1] + matrix[:, 2],
        ],
        dtype=int,
    )
    return [int(max(axes[:, index]) - min(axes[:, index])) for index in range(3)]


def _build_simple_supercell(atoms: Atoms, multiplicities, supercell_matrix):
    mat = np.diag(np.asarray(multiplicities, dtype=int))
    scaled_positions = np.asarray(atoms.get_scaled_positions(), dtype=float)
    symbols = atoms.get_chemical_symbols()
    masses = atoms.get_masses()
    magmoms = atoms.get_initial_magnetic_moments() if atoms.has("initial_magmoms") else None
    lattice = np.asarray(atoms.cell.array, dtype=float)

    b_grid, c_grid, a_grid = np.meshgrid(
        range(int(multiplicities[1])),
        range(int(multiplicities[2])),
        range(int(multiplicities[0])),
    )
    lattice_points = np.c_[a_grid.ravel(), b_grid.ravel(), c_grid.ravel()]
    positions_multi = (
        np.tile(lattice_points, (len(scaled_positions), 1))
        + np.repeat(scaled_positions, len(lattice_points), axis=0)
    ) @ np.linalg.inv(mat).T

    symbols_multi = [symbol for symbol in symbols for _ in range(len(lattice_points))]
    masses_multi = np.repeat(masses, len(lattice_points))
    supercell = Atoms(
        symbols=symbols_multi,
        cell=np.dot(mat, lattice),
        scaled_positions=positions_multi,
        masses=masses_multi,
        pbc=[True, True, True],
    )
    if magmoms is not None and len(magmoms) == len(symbols):
        supercell.set_initial_magnetic_moments(np.repeat(magmoms, len(lattice_points)))
    return supercell


def _trim_cell_like_phonopy(relative_axes, cell: Atoms, symprec: float = 1e-5):
    trimmed_lattice = np.dot(relative_axes.T, cell.cell.array)
    positions_in_new_lattice = np.dot(cell.get_scaled_positions(), np.linalg.inv(relative_axes).T)
    positions_in_new_lattice -= np.floor(positions_in_new_lattice)

    trimmed_positions = []
    trimmed_symbols = []
    trimmed_masses = []
    trimmed_magmoms = [] if cell.has("initial_magmoms") else None

    masses = cell.get_masses()
    magmoms = cell.get_initial_magnetic_moments() if cell.has("initial_magmoms") else None
    for index, position in enumerate(positions_in_new_lattice):
        found_overlap = False
        if trimmed_positions:
            diff = np.asarray(trimmed_positions, dtype=float) - position
            diff -= np.rint(diff)
            distances = np.sqrt(np.sum(np.dot(diff, trimmed_lattice) ** 2, axis=1))
            if np.any(distances < symprec):
                found_overlap = True

        if not found_overlap:
            trimmed_positions.append(position)
            trimmed_symbols.append(cell[index].symbol)
            trimmed_masses.append(masses[index])
            if trimmed_magmoms is not None and magmoms is not None:
                trimmed_magmoms.append(magmoms[index])

    trimmed = Atoms(
        symbols=trimmed_symbols,
        cell=trimmed_lattice,
        scaled_positions=np.asarray(trimmed_positions, dtype=float),
        masses=np.asarray(trimmed_masses, dtype=float),
        pbc=[True, True, True],
    )
    if trimmed_magmoms is not None:
        trimmed.set_initial_magnetic_moments(np.asarray(trimmed_magmoms, dtype=float))
    expected_atoms = len(cell) * np.linalg.det(relative_axes)
    if len(trimmed) != int(np.rint(expected_atoms)):
        raise RuntimeError("phonopy-like supercell trimming failed")
    return trimmed


def _normalize_supercell_matrix(raw_supercell):
    matrix = np.asarray(raw_supercell, dtype=int)
    if matrix.shape == (3,):
        return np.diag(matrix)
    if matrix.shape == (3, 3):
        return matrix
    raise ValueError("phonon displacement supercell must be [a, b, c] or a 3x3 matrix")


def _ase_atoms_to_spglib_cell(atoms: Atoms):
    lattice = np.asarray(atoms.cell.array, dtype=float)
    scaled_positions = np.asarray(atoms.get_scaled_positions(), dtype=float)
    numbers = np.asarray(atoms.numbers, dtype=int)
    if atoms.has("initial_magmoms"):
        magmoms = np.asarray(atoms.get_initial_magnetic_moments(), dtype=float)
        return lattice, scaled_positions, numbers, magmoms
    return lattice, scaled_positions, numbers
