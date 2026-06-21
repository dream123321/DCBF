import os
import glob
import yaml
import time
import sys
import subprocess

def mkdir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)

def modify_yaml(input,output,npt,nvt):
    with open(input, 'rb') as file:
        data = yaml.safe_load(file)
    if npt is not None and nvt is not None:
        data['mlp_MD'] = [{'npt': npt}, {'nvt': nvt}]
    elif npt is None and nvt is not None:
        data['mlp_MD'] = [{'nvt': nvt}]
    elif npt is not None and nvt is None:
        data['mlp_MD'] = [{'npt': npt}]
    with open(output, 'w') as file:
        yaml.safe_dump(data, file, default_flow_style=False)

def copy_init(pwd,npt,nvt,run_position):
    init_path = os.path.join(pwd,'init')
    yaml = os.path.join(init_path,'parameter.yaml')
    run_position_yaml = os.path.join(run_position,'parameter.yaml')
    if not os.path.exists(run_position_yaml):
        modify_yaml(yaml, run_position_yaml, npt, nvt)

def gen_while_loop(pwd, npt, nvt, start_position,gen_num,sleep_time,max_gen):
    main_dir = os.path.dirname(start_position)
    last_allowed_gen = max_gen - 1
    if gen_num > last_allowed_gen:
        return

    while True:
        os.chdir(start_position)
        if not os.path.exists('__ok__'):
            copy_init(pwd, npt, nvt, start_position)
            subprocess.run([sys.executable, '-m', 'dcbf.cli', 'run-generation', '--workspace', start_position], check=True)
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
