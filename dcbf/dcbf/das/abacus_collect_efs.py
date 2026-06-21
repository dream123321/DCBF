import os
import re
from ase import Atoms
from ase.io import write
import numpy as np

KBAR2eV_ANG3 = 0.0006241509073

class collect_efs():
    def __init__(self, running, out, label):
        self.running = running
        self.out = out
        self.label = label

    def read_abacus_out(self):
        string_0 = 'TOTAL ATOM NUMBER'
        string_1 = 'TOTAL-FORCE (eV/Angstrom)'
        string_2 = 'TOTAL-STRESS (KBAR)'
        string_3 = 'Volume (A^3)'
        string_4 = 'final'
        no = 'Relaxation is converged'
        #ele = []
        lattice = []
        position = []
        force = []
        stress = []
        energy = []

        label = False
        with open(self.running,'r') as f:
            lines = f.readlines()
            num = 0
            for index,line in enumerate(lines):
                if string_0 in line:
                    num = int(line.split()[4])
                if string_1 in line:
                    temp_force = []
                    temp_ele = []
                    for i in range(num):
                        temp_force.append([float(i) for i in lines[index+i+2].split()[1:]])
                        temp_ele.append(re.sub(r'\d+', '', lines[index + i + 2].split()[0]))
                    # print('force',temp_force)
                    # print('ele',temp_ele)
                    force.append(temp_force)

                if string_2 in line:
                    #temp_stress = []
                    temp_stress = [[float(i) for i in lines[index+2].split()],[float(i) for i in lines[index+3].split()],[float(i) for i in lines[index+4].split()]]
                    stress.append(temp_stress)
                    #print('stress',temp_stress)
                    #print('______________________________')
                if string_3 in line:
                    g=1
                    #print(line.split()[3])
                if string_4 in line:
                    energy.append(float(line.split()[3]))
                if 'Lattice vectors: (Cartesian coordinate: in unit of a_0)' in line:
                    temp_lattice = [[float(i) for i in lines[index+1].split()],[float(i) for i in lines[index+2].split()],[float(i) for i in lines[index+3].split()]]
                    lattice.append(temp_lattice)
                    #print('lattice',temp_lattice)
                if line.strip() == "DIRECT COORDINATES":
                    temp_position = []
                    for i in range(num):
                        temp_position.append([float(i) for i in lines[index+i+2].split()[1:4]])
                    #print('position',temp_position)
                    position.append(temp_position)
                if no in line:
                    label = True
                '''temp_energy 最后输出的一个能量'''
        return label,lattice,temp_ele,position,force,stress,energy

    def last_xyz(self):
        label,lattice,ele,position,force,stress,energy = self.read_abacus_out()

        atoms = Atoms(ele,
                      scaled_positions=position[-1],
                      cell=lattice[-1])
        atoms.info['energy'] = energy[-1]
        atoms.info['virial'] = np.array(stress[-1]) * KBAR2eV_ANG3 * atoms.get_volume()
        atoms.info['pbc'] = "T T T"
        atoms.info['label'] = self.label
        atoms.arrays['forces'] = np.array(force[-1])
        atoms.pbc = [True, True, True]
        write(self.out, atoms, format='extxyz')
        return label

    def traj_xyz(self):
        label, lattice, ele, position, force, stress, energy = self.read_abacus_out()
        print(len(lattice),len(energy),len(stress))
        atoms_list = []

        for i in range(len(energy)):
            atoms = Atoms(ele,
                          scaled_positions=position[i],
                          cell=lattice[i])
            atoms.info['energy'] = energy[i]
            atoms.info['virial'] = np.array(stress[i]) * KBAR2eV_ANG3 * atoms.get_volume()
            atoms.info['pbc'] = "T T T"
            atoms.info['label'] = self.label
            atoms.arrays['forces'] = np.array(force[i])
            atoms.pbc = [True, True, True]
            atoms_list.append(atoms)
        #write(self.out, atoms_list, format='extxyz')
        return atoms_list,label

    def last_atoms(self):
        label,lattice,ele,position,force,stress,energy = self.read_abacus_out()

        atoms = Atoms(ele,
                      scaled_positions=position[-1],
                      cell=lattice[-1])
        atoms.info['energy'] = energy[-1]
        atoms.info['virial'] = np.array(stress[-1]) * KBAR2eV_ANG3 * atoms.get_volume()
        atoms.info['pbc'] = "T T T"
        atoms.info['label'] = self.label
        atoms.arrays['forces'] = np.array(force[-1])
        atoms.pbc = [True, True, True]

        return atoms,label


if __name__=='__main__':
    collect_efs('running_cell-relax.log','out.xyz','2').traj_xyz()
