import os
import shutil
from .lmps_scripts import *
from .lmps_bsub import lammps_bsub
from ase.io import read,write
import pickle
from ..path_names import MD_WORK_DIR, SUS2_MODEL_DIR

def mkdir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)

def Lattice_scaling(structure,scale_factor,out):
    structure = read(structure)
    scaled_positions = structure.get_scaled_positions()
    new_cell = [scale_factor * vec for vec in structure.cell]
    structure.set_cell(new_cell)
    structure.set_scaled_positions(scaled_positions)
    write(out,structure,format='vasp')

def mkdir_vasp(pwd,mlp_MD, ele, size, mlp_nums,sort_ele,nvt_lattice_scaling_factor, mlp_encode_model, scheduler):
    #scripts_path = os.path.join(pwd,'utils',calc_script)
    tt = os.path.dirname(os.path.dirname(pwd))
    stru_path = os.path.join(tt, 'stru')
    work_path = os.path.join(pwd, MD_WORK_DIR)
    mtp_path = os.path.join(pwd, SUS2_MODEL_DIR)
    mkdir(work_path)

    gen_num = os.path.basename(pwd).replace('gen_', '')
    last_gen = 'gen_' + str((int(gen_num) - 1))
    last_gen_path = os.path.join(os.path.dirname(pwd), last_gen)
    stru_pkl = os.path.join(last_gen_path, 'stru.pkl')
    if int(gen_num) == 0:
        t = [f for f in os.listdir(stru_path) if f.endswith('.vasp')]
    else:
        with open(stru_pkl, 'rb') as f:
            t = pickle.load(f)
    for a in t:
        file = os.path.join(pwd,stru_path,a)
        dir = os.path.join(pwd,work_path,a.replace('.vasp',''))
        mkdir(dir)
        os.chdir(dir)
        lammps_bsub(dir, mlp_nums, mlp_encode_model, scheduler)
        #os.system('dos2unix bsub.lsf')
        for md in mlp_MD:
            k = list(md.keys())
            k_dir = os.path.join(dir, k[0])
            mkdir(k_dir)
            v = list(md.values())
            model = 1
            if model == 1:
                for temp_press in v[0]:
                    if k[0] == 'nvt':
                        for factor in nvt_lattice_scaling_factor:
                            dir_temp_press = os.path.join(k_dir, str(factor) +'_'+ str(temp_press))
                            out = os.path.join(dir_temp_press, str(factor) +'_'+a)
                            mkdir(dir_temp_press)
                            os.chdir(dir_temp_press)
                            if not os.path.exists('__ok__'):
                                #shutil.copy(file, dir_temp_press)
                                Lattice_scaling(file, factor, out)
                                main_pos2lmp(ele, size, sort_ele)
                                lammps_scripts(k[0], temp_press, mlp_nums, mlp_encode_model,pwd)
                                for mtp in [i for i in os.listdir(mtp_path)]:
                                    sub_mtp_path = os.path.join(mtp_path, mtp)
                                    shutil.copy(sub_mtp_path, dir_temp_press)
                                    # os.system('dos2unix bsub.lsf')
                            else:
                                pass
                    elif k[0] == 'npt':
                        dir_temp_press = os.path.join(k_dir, str(temp_press))
                        mkdir(dir_temp_press)
                        os.chdir(dir_temp_press)
                        if not os.path.exists('__ok__'):
                            shutil.copy(file, dir_temp_press)
                            main_pos2lmp(ele, size, sort_ele)
                            lammps_scripts(k[0], temp_press, mlp_nums, mlp_encode_model,pwd)
                            for mtp in [i for i in os.listdir(mtp_path)]:
                                sub_mtp_path = os.path.join(mtp_path, mtp)
                                shutil.copy(sub_mtp_path, dir_temp_press)
                                # os.system('dos2unix bsub.lsf')
                        else:
                            pass

                        #print('__ok__ file exists')
            # elif model == 2:
            #     shutil.copy(file, k_dir)
            #     os.chdir(k_dir)
            #     INPUT_bsub(k[0],v[0][0],v[0][-1],server,bsub['queue'],bsub['cores'])
