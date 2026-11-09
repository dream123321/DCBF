import os
import glob
import hashlib
import json
import yaml
import time
import sys
import subprocess
from ..memory_guard import run_monitored_generation

def mkdir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)

def _mlp_md(npt, nvt):
    mlp_md = []
    if npt:
        mlp_md.append({'npt': npt})
    if nvt:
        mlp_md.append({'nvt': nvt})
    if not mlp_md:
        raise ValueError('npt and nvt cannot both be empty for the same MD loop')
    return mlp_md

def modify_yaml(input,output,npt,nvt):
    with open(input, 'rb') as file:
        data = yaml.safe_load(file)
    data['mlp_MD'] = _mlp_md(npt, nvt)
    with open(output, 'w') as file:
        yaml.safe_dump(data, file, default_flow_style=False)

def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def copy_init(pwd,npt,nvt,run_position):
    init_path = os.path.join(pwd,'init')
    init_yaml = os.path.join(init_path,'parameter.yaml')
    run_position_yaml = os.path.join(run_position,'parameter.yaml')
    if not os.path.isfile(init_yaml):
        raise FileNotFoundError(f'init parameter.yaml does not exist: {init_yaml}')

    source_hash = _sha256_file(init_yaml)
    snapshot_path = os.path.join(run_position, '.dcbf_parameter_source.json')
    previous_hash = None
    if os.path.isfile(snapshot_path):
        try:
            with open(snapshot_path, 'r', encoding='utf-8') as handle:
                previous_hash = json.load(handle).get('sha256')
        except (OSError, ValueError):
            previous_hash = None

    if not os.path.isfile(run_position_yaml) or previous_hash != source_hash:
        # A new/changed init/parameter.yaml is authoritative.  Rewrite the
        # generation copy so a resubmitted run uses the new parameters
        # immediately instead of replaying the old parameter.yaml.
        modify_yaml(init_yaml, run_position_yaml, npt, nvt)
        with open(snapshot_path, 'w', encoding='utf-8') as handle:
            json.dump({'source': init_yaml, 'sha256': source_hash}, handle, indent=2)
        print(f'[parameters] refreshed {run_position_yaml} from {init_yaml}', flush=True)
        return True

    # The source did not change: preserve generation-local/runtime edits
    # (for example dft.calc_dir_num), but keep mlp_MD in sync with the current
    # main-loop schedule.
    try:
        with open(run_position_yaml, 'r', encoding='utf-8') as handle:
            data = yaml.safe_load(handle) or {}
    except yaml.YAMLError:
        modify_yaml(init_yaml, run_position_yaml, npt, nvt)
        print(f'[parameters] repaired {run_position_yaml} from {init_yaml}', flush=True)
        return True
    expected_mlp_md = _mlp_md(npt, nvt)
    if data.get('mlp_MD') != expected_mlp_md:
        data['mlp_MD'] = expected_mlp_md
        with open(run_position_yaml, 'w', encoding='utf-8') as handle:
            yaml.safe_dump(data, handle, default_flow_style=False)
        print(f'[parameters] updated mlp_MD in {run_position_yaml}', flush=True)
    return False

def gen_while_loop(pwd, npt, nvt, start_position,gen_num,sleep_time,max_gen,reload_config=None):
    main_dir = os.path.dirname(start_position)
    last_allowed_gen = max_gen - 1
    if gen_num > last_allowed_gen:
        return

    while True:
        if reload_config is not None:
            reload_config()
        os.chdir(start_position)
        if not os.path.exists('__ok__'):
            copy_init(pwd, npt, nvt, start_position)
            run_monitored_generation([sys.executable, '-m', 'dcbf.cli', 'run-generation', '--workspace', start_position], start_position)
            while True:
                if os.path.exists('__ok__'):
                    break
                if os.path.exists('__error__'):
                    sys.exit(1)
                time.sleep(sleep_time)
        if os.path.exists('__end__') or gen_num >= last_allowed_gen:
            break
        else:
            gen_num = gen_num + 1
            start_position = os.path.join(main_dir,'gen_'+str(gen_num))
            mkdir(start_position)

# def check_run_position(pwd,main_loop_npt):
#     for i in range(len(main_loop_npt)):
#         main_path = os.path.join(pwd,'main_'+str(i))
#         mkdir(main_path)
#     os.chdir(pwd)
#     main_list = glob.glob('main_*')
#     sorted_main_list = sorted(['main_'+a.replace('main_','') for a in main_list])
#     main_gen_len = []
#     for main in sorted_main_list:
#         os.chdir(os.path.join(pwd,main))
#         gen_list = glob.glob('gen_*')
#         main_gen_len.append(len(gen_list))
#     temp = ''
#     if all(element == 0 for element in main_gen_len):
#         temp = 'initialization'
#     else:
#         temp_list = []
#         for index, number in enumerate(main_gen_len):
#             if number == 0:
#                 temp_list.append(index)
#         if len(temp_list) != 0:
#             temp = sorted_main_list[temp_list[0]-1]
#         else:
#             temp = sorted_main_list[-1]
#
#     if temp == 'initialization':
#         os.mkdir(os.path.join(pwd,'main_0','gen_0'))
#         return os.path.join(pwd,'main_0','gen_0'),0,0
#     else:
#         path = os.path.join(pwd,temp)
#         os.chdir(path)
#         gen_list = glob.glob('gen_*')
#         sorted_gen_list = sorted(['gen_' + a.replace('gen_', '') for a in gen_list])
#         run_position = os.path.join(path, sorted_gen_list[-1])
#         gen_num = int(os.path.basename(run_position).replace('gen_',''))
#         main_num = int(os.path.basename(path).replace('main_',''))
#         return run_position,gen_num,main_num

def check_run_position(pwd,main_loop_npt):
    for i in range(len(main_loop_npt)):
        main_path = os.path.join(pwd,'main_'+str(i))
        mkdir(main_path)
    os.chdir(pwd)
    main_list = glob.glob('main_*')
    sorted_main_list = sorted([int(a.replace('main_','')) for a in main_list])
    sorted_main_list = ['main_' + str(a) for a in sorted_main_list]
    main_gen_len = []
    for main in sorted_main_list:
        os.chdir(os.path.join(pwd,main))
        gen_list = glob.glob('gen_*')
        main_gen_len.append(len(gen_list))

    if all(element == 0 for element in main_gen_len):
        temp = 'initialization'
    else:
        index = [index for index, value in enumerate(main_gen_len) if value != 0][-1]
        temp = 'main_'+str(index)

    if temp == 'initialization':
        os.mkdir(os.path.join(pwd,'main_0','gen_0'))
        return os.path.join(pwd,'main_0','gen_0'),0,0
    else:
        path = os.path.join(pwd,temp)
        os.chdir(path)
        gen_list = glob.glob('gen_*')
        sorted_gen_list = sorted([int(a.replace('gen_', '')) for a in gen_list])
        sorted_gen_list = ['gen_' + str(a) for a in sorted_gen_list]
        run_position = os.path.join(path, sorted_gen_list[-1])
        gen_num = int(os.path.basename(run_position).replace('gen_',''))
        main_num = int(os.path.basename(path).replace('main_',''))
        return run_position,gen_num,main_num
