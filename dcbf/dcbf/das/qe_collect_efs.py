import os
import shutil

from ase.io import read
import numpy as np


def _voigt_to_matrix(stress):
    return np.array(
        [
            [stress[0], stress[5], stress[4]],
            [stress[5], stress[1], stress[3]],
            [stress[4], stress[3], stress[2]],
        ]
    )


def collect_efs(input_path):
    logout = os.path.join(input_path, "logout")
    atoms = read(logout, format="espresso-out", index=-1)
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    stress_matrix = _voigt_to_matrix(atoms.get_stress())

    atoms.calc = None
    atoms.info["energy"] = energy
    atoms.info["stress"] = stress_matrix
    atoms.info["virial"] = -1.0 * stress_matrix * atoms.get_volume()
    atoms.info["pbc"] = "T T T"
    atoms.arrays["forces"] = forces
    atoms.pbc = [True, True, True]

    tmp_qe = os.path.join(input_path, "tmp_qe")
    if os.path.isdir(tmp_qe):
        shutil.rmtree(tmp_qe)

    return atoms
