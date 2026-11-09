import os
from ase.io import read, write, iread
import numpy as np

def remove(file):
    if os.path.exists(file):
        os.remove(file)

def collect_efs(input_path):
    vasp_xml = os.path.join(input_path, 'vasprun.xml')
    atom = read(vasp_xml,format='vasp-xml')
    s = atom.get_stress()
    six2nine = np.array([s[0], s[5], s[4], s[5], s[1], s[3], s[4], s[3], s[2]])
    atom.info['virial'] = -1 * six2nine * atom.get_volume()
    return atom

if __name__ == '__main__':
    pwd = os.getcwd()
    input_path = pwd
    collect_efs(input_path)





