import os
from ase.io import read, write, iread
import numpy as np
import glob

def remove(file):
    if os.path.exists(file):
        os.remove(file)

def collect_efs(input_path):
    vasp_xml = os.path.join(input_path, 'vasprun.xml')
    keep_files = ['vasprun.xml', 'POSCAR','__ok__','logout']
    all_files = glob.glob(os.path.join(input_path, '*'))
    # 过滤出需要删除的文件
    files_to_delete = [file for file in all_files if os.path.basename(file) not in keep_files]
    for file in files_to_delete:
        remove(file)
    atom = read(vasp_xml,format='vasp-xml')
    s = atom.get_stress()
    six2nine = np.array([s[0], s[5], s[4], s[5], s[1], s[3], s[4], s[3], s[2]])
    atom.info['virial'] = -1 * six2nine * atom.get_volume()
    return atom

if __name__ == '__main__':
    pwd = os.getcwd()
    input_path = pwd
    collect_efs(input_path)





