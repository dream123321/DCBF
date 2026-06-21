import os
import glob
import sys
import shutil
from pathlib import Path
import yaml
from .file_conversion import xyz2cfg, merge_cfg
from .work_dir import deepest_dir,check_finish
from ..path_names import DFT_WORK_DIR, SUS2_MODEL_DIR
from ..mtp import write_mtp_template


BATCH_MERGED_MARKER = "__candidate_batch_merged__"

def mkdir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)

def check(file):
    if not os.path.exists(file):
        sys.exit(f'{file} train.cfg no exists')

def _format_current_md_schedule(parameter_yaml_path):
    try:
        with open(parameter_yaml_path, 'r', encoding='utf-8') as handle:
            data = yaml.safe_load(handle) or {}
    except Exception:
        return ""
    mlp_md = data.get('mlp_MD')
    if not mlp_md:
        return ""
    return f" | current_mlp_MD={mlp_md}"

def train_mlp_lsf(current_num,out,COMMAND,pwd, scheduler):
    script_text = f"""#!/bin/bash
{scheduler.bsub_script_train_sus_job_name}{current_num}
"""
    script_text = script_text + scheduler.bsub_script_train_sus

    train_env = str(getattr(scheduler, "train_env", "") or "").strip()
    train_env_text = f"{train_env}\n" if train_env else ""

    content = f"""
start_time=$(date +%s.%N)
touch __start__
$COMMAND_std > logout 2>&1
end_time=$(date +%s.%N)
touch __ok__
runtime=$(echo "$end_time - $start_time" | bc)
echo "total_runtime：$runtime s" >> time.txt"""
    COMMAND_std = 'COMMAND_std="' + COMMAND + f'../../{SUS2_MODEL_DIR}/current_{current_num}.mtp"'
    total = script_text + train_env_text + COMMAND_std + content
    with open(out,'w') as file:
        file.write(total)


def _copy_reference_generation(reference_path, train_mlp_path, mtp_path):
    reference_path = Path(reference_path)
    shutil.copy(reference_path / 'train_mlp' / 'train.cfg', train_mlp_path)
    for source_mtp in (reference_path / SUS2_MODEL_DIR).glob('current*'):
        shutil.copy(source_mtp, mtp_path)


def _reset_training_dir_state(train_dir):
    for file_name in ('__start__', '__ok__', 'logout', 'time.txt'):
        file_path = Path(train_dir) / file_name
        if file_path.exists():
            file_path.unlink()

def pre_train_mlp(pwd, mlp_nums, ele, ele_model,logger, scheduler):
    mtp_path = os.path.join(pwd, SUS2_MODEL_DIR)
    mkdir(mtp_path)
    os.chdir(mtp_path)
    names = glob.glob('current*')

    if len(names) != mlp_nums:
        train_mlp_path = os.path.join(pwd,'train_mlp')
        mkdir(train_mlp_path)

        gen_num = os.path.basename(pwd).replace('gen_','')
        main_dir = os.path.dirname(os.path.dirname(pwd))
        main_num = os.path.basename(os.path.dirname(pwd)).replace('main_','')

        if int(gen_num) == 0 and int(main_num) != 0:
            last_main_num = 'main_' + str((int(main_num) - 1))
            last_main_path = os.path.join(main_dir, last_main_num)
            last_main_gen_num = sorted(
                [
                    int(name.replace('gen_', ''))
                    for name in os.listdir(last_main_path)
                    if name.startswith('gen_') and os.path.isdir(os.path.join(last_main_path, name))
                ]
            )[-1]
            reference_generation = os.path.join(last_main_path, 'gen_'+str(last_main_gen_num))
            _copy_reference_generation(reference_generation, train_mlp_path, mtp_path)
            current_parameter_yaml = os.path.join(pwd, 'parameter.yaml')
            logger.info(f"main_{main_num} | gen_0: start{_format_current_md_schedule(current_parameter_yaml)}")
            return False
        elif int(gen_num) == 0 and int(main_num) == 0:
            parameter_yaml = os.path.join(main_dir, 'init', 'parameter.yaml')
            with open(parameter_yaml, 'r', encoding='utf-8') as handle:
                data = yaml.safe_load(handle)
            xyz2cfg(ele, ele_model, data['dataset_xyz_input'], os.path.join(train_mlp_path, 'train.cfg'))
            for i in range(mlp_nums):
                sub_train_mlp_path = os.path.join(train_mlp_path, str(i))
                mkdir(sub_train_mlp_path)
                os.chdir(sub_train_mlp_path)
                write_mtp_template(Path('hyx.mtp'), data['mtp_type'], len(ele))
                train_mlp_lsf(i, 'bsub.lsf',scheduler.original_COMMAND,pwd, scheduler)
                #os.system('dos2unix bsub.lsf')
            current_parameter_yaml = os.path.join(pwd, 'parameter.yaml')
            logger.info(f"main_0 | gen_0: start{_format_current_md_schedule(current_parameter_yaml)}")
            return True
        else:
            last_gen = 'gen_'+str((int(gen_num)-1))
            last_gen_path = os.path.join(os.path.dirname(pwd),last_gen)
            _copy_reference_generation(last_gen_path, train_mlp_path, mtp_path)
            logger.info(f'main_{main_num} | gen_{gen_num}: start')
            return False
    else:
        dirs = deepest_dir(pwd, 'train_mlp')
        if len(dirs) != 1:
            check_finish(dirs, logger, 'All training_mlp tasks have been completed')
        else:
            logger.info(f'SUS2 model already exists')
        return False


def update_mlp_from_current_batch(pwd, mlp_nums, ele, ele_model, logger, scheduler):
    train_mlp_path = Path(pwd) / 'train_mlp'
    mtp_path = Path(pwd) / SUS2_MODEL_DIR
    current_train_cfg = train_mlp_path / 'train.cfg'
    scf_filter_xyz = Path(pwd) / DFT_WORK_DIR / 'scf_filter.xyz'
    scf_filter_cfg = train_mlp_path / 'scf_filter.cfg'
    merged_marker = train_mlp_path / BATCH_MERGED_MARKER

    check(str(current_train_cfg))
    check(str(scf_filter_xyz))

    if not merged_marker.exists():
        xyz2cfg(ele, ele_model, str(scf_filter_xyz), str(scf_filter_cfg))
        merged_train_cfg = train_mlp_path / 'train_batch_merged.cfg'
        merge_cfg(str(current_train_cfg), str(scf_filter_cfg), str(merged_train_cfg))
        shutil.move(str(merged_train_cfg), str(current_train_cfg))
        merged_marker.touch()

    for source_mtp in mtp_path.glob('current*'):
        current_num = source_mtp.name.split('_')[1].split('.')[0]
        sub_train_mlp_path = train_mlp_path / str(current_num)
        mkdir(str(sub_train_mlp_path))
        _reset_training_dir_state(sub_train_mlp_path)
        shutil.copy(source_mtp, sub_train_mlp_path / 'hyx.mtp')
        os.chdir(sub_train_mlp_path)
        train_mlp_lsf(current_num, 'bsub.lsf', scheduler.subsequent_COMMAND, pwd, scheduler)

    logger.info('candidate batch trigger: start mlip update')
    start_train(pwd, scheduler.task_submission_method, mlp_nums, logger)

def start_train(pwd,task_submission_method, mlp_nums, logger):
    dirs = deepest_dir(pwd,'train_mlp')
    for a in dirs:
        if not os.path.exists(os.path.join(a,'__start__')):
            os.chdir(a)
            os.system(f"{task_submission_method}")
            logger.info(f'mlp_{os.path.basename(a)} has been submitted')
        else:
            logger.info(f'mlp_{os.path.basename(a)} is already in training')
    check_finish(dirs, logger, 'All training_mlp tasks have been completed')

    mtp_path = os.path.join(pwd, SUS2_MODEL_DIR)
    mkdir(mtp_path)
    os.chdir(mtp_path)
    names = glob.glob('current*')
    if len(names) != mlp_nums:
        sys.exit(f'Please check the training process! Successful training={len(names)}')
        logger.error(f'Please check the training process! Successful training={len(names)}')
if __name__=='__main__':
    path =''
    train_mlp(path, 3, 'taiying')
