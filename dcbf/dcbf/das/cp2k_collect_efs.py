import os
import numpy as np
from ase.io import read,write,iread


# Constants definition
HA_TO_EV = 27.211386245988  # 1 Ha = 27.211 eV
HA_BOHR_TO_EV_ANGSTROM = 51.422067476326  # 1 Ha/Bohr = 51.422 eV/Å
GPA_TO_EV_A3 = 0.006241509073  # 1 GPa = 0.00624 eV/Å³


def parse_forces_and_stress(logout_path):
    """ Parse the logout file to extract energy, atomic forces, and stress tensor """
    energy, forces, stress = None, [], None
    with open(logout_path, 'r') as f:
        lines = f.readlines()

    # Extract energy
    for line in lines:
        if 'ENERGY| Total FORCE_EVAL ( QS ) energy [a.u.]' in line:
            energy = float(line.split()[-1]) * HA_TO_EV  # Convert to eV
            break

    # Extract atomic forces
    force_start, force_end = None, None
    for i, line in enumerate(lines):
        if 'ATOMIC FORCES in [a.u.]' in line:
            force_start = i + 2
        if 'SUM OF ATOMIC FORCES' in line:
            force_end = i
            break
    forces = [
        [float(x) * HA_BOHR_TO_EV_ANGSTROM for x in line.split()[3:6]]
        for line in lines[force_start:force_end]
        if line.split()[0].isdigit()
    ]

    # Extract stress tensor
    for i, line in enumerate(lines):
        if 'STRESS| Analytical stress tensor [GPa]' in line:
            stress_lines = lines[i + 2:i + 5]
            stress = np.array([
                [float(x) for x in row.split()[2:]]
                for row in stress_lines
            ]) * GPA_TO_EV_A3  # Convert to eV/Å³
            break

    return energy, forces, stress

def collect_efs(input_path):
    xyz = os.path.join(input_path,'cp2k.xyz')
    logout_path = os.path.join(input_path, 'logout')
    energy, forces, stress = parse_forces_and_stress(logout_path)
    old_atoms = read(xyz)
    temp_poscar = os.path.join(input_path,'temp.vasp')
    write(temp_poscar,old_atoms ,format='vasp')
    atoms = read(temp_poscar)
    atoms.info['energy'] = energy
    atoms.info['stress'] = np.array(stress)
    atoms.info['virial'] = np.array(stress) * atoms.get_volume()
    atoms.info['pbc'] = "T T T"
    atoms.arrays['forces'] = np.array(forces)
    atoms.pbc = [True, True, True]
    if os.path.exists(temp_poscar):
        os.remove(temp_poscar)
    return atoms


if __name__ == '__main__':
    input_path = os.getcwd()
    atoms = cp2k2xyz(input_path)
    print(atoms)
    write('test.xyz',atoms,format='extxyz')

    # """ Main function to read logout, process data, and update cp2k.xyz """
    # energy, forces, stress = parse_forces_and_stress(logout_path)
    # update_cp2k_file(cp2k_xyz_path, energy, forces, stress, 'cp2knew.xyz')
