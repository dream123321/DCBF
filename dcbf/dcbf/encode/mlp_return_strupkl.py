import glob
import os
import pickle

def mlp_return_strupkl(pwd, dirs,dirs_stru_counts,all_select_index):
    select_no_convergence = True
    if select_no_convergence == True:
        index_ranges = {}
        for i, path in enumerate(dirs):
            dir_name = os.path.basename(os.path.dirname(os.path.dirname(path)))
            if dir_name not in index_ranges:
                index_ranges[dir_name] = [i, i]
            else:
                index_ranges[dir_name][1] = i
        for dir_name, value in index_ranges.items():
            index_ranges[dir_name] = [sum(dirs_stru_counts[:value[0]]),sum(dirs_stru_counts[:(value[1]+1)])-1]
        result_keys = []
        for key, value_range in index_ranges.items():
            if any(num in range(value_range[0], value_range[1] + 1) for num in all_select_index):
                result_keys.append(key +'.vasp')
        stru_pkl = os.path.join(pwd, 'stru.pkl')
        #print(result_keys)
        with open(stru_pkl, 'wb') as f:
            pickle.dump(result_keys, f)
    else:
        stru_dirs = os.path.join(os.path.dirname(os.path.dirname(pwd)),'stru')
        stru_files = glob.glob(os.path.join(stru_dirs,'*.vasp'))
        result_keys = [os.path.basename(a) for a in stru_files]
        stru_pkl = os.path.join(pwd, 'stru.pkl')
        # print(result_keys)
        with open(stru_pkl, 'wb') as f:
            pickle.dump(result_keys, f)

