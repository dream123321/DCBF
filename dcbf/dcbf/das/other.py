import os
import yaml
import shutil

def mkdir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)

def remove(file_name):
    if os.path.exists(file_name) and os.path.isfile(file_name):
        os.remove(file_name)
    elif os.path.exists(file_name) and os.path.isdir(file_name):
        shutil.rmtree(file_name)

def end_yaml(end):
    with open(end, 'rb') as file:
        end_yaml_data = yaml.safe_load(file)
    threshold_low = end_yaml_data['threshold_low']
    threshold_high = end_yaml_data['threshold_high']
    sample = end_yaml_data['sample']
    n = sample['n']
    cluster_threshold_init = sample['cluster_threshold_init']
    k = sample['k']
    return threshold_low, threshold_high, n, cluster_threshold_init, k

def touch(path,filename):
    file_path = os.path.join(path,filename)
    with open(file_path, 'w') as f:
        pass