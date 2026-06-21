from .descriptor_encoder_xyz import *
from ase.io import iread,write,read
from collections import Counter
import pickle
import os
import sys
import time

def load(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    return data

def sample_main(path, n=15, threshold_init=0.1, k=2, clustering_by_ambiguity = True):
    MD_out = 'filter.xyz'
    data = list(iread(MD_out))
    ambiguity_list = [a.info['ambiguity'] for a in data]
    g = sample_plot()
    pkl = os.path.join(path,'filter.pkl')
    MD_sample_out = 'sample_'+MD_out
    MD_remaining_out = 'remain_'+MD_out

    '''encoder'''
    if not os.path.exists(pkl):
        start = time.time()
        encoder(f_in=MD_out, save_path=pkl).MBTR()
        temp_1 = time.time()
        print(f'encoder time:{temp_1-start}')
    else:
        print('filter.pkl already exists!')

    '''sample'''
    temp_2 = time.time()

    coverage_score, sample_index, remaining_index, sample_data, remaining_data, clustering_label, new_threshold_init = g.sample_plot(load(pkl), f_in=MD_out, n=n, threshold_init=threshold_init,
                                      k=k, b_bins=1000, sample_out=MD_sample_out, remaining_out=MD_remaining_out)
    if new_threshold_init != threshold_init:
        coverage_score, sample_index, remaining_index, sample_data, remaining_data, clustering_label, new_threshold_init = g.sample_plot(
            load(pkl), f_in=MD_out, n=n, threshold_init=new_threshold_init,
            k=k, b_bins=1000, sample_out=MD_sample_out, remaining_out=MD_remaining_out)
    end = time.time()

    if clustering_by_ambiguity == False:
        rename_sample = str(len(sample_index)) + '_' + 'sample'+'_'+MD_out
        write(rename_sample, sample_data, format='extxyz')
        # rename_remaining = str(len(remain_indexes)) + '_' + remaining_out
        # write(rename_remaining, remaining_data, format='extxyz')
    elif clustering_by_ambiguity == True:
        #print(clustering_label)
        sample_classes = [clustering_label[a] for a in sample_index]
        element_counts = Counter(sample_classes)
        #print(f"Element counts: {element_counts},{element_counts[1]}")
        indexs_stru = []
        indexes_of_elements = {}
        for index, value in enumerate(clustering_label):
            if value in indexes_of_elements:
                indexes_of_elements[value].append(index)
            else:
                indexes_of_elements[value] = [index]
        for element in sorted(indexes_of_elements):
            temp_index = indexes_of_elements[element]
            #print(element,temp_index)
            ambiguity = [ambiguity_list[a] for a in temp_index]
            #print(element, ambiguity)
            sorted_pairs = sorted(enumerate(ambiguity), key=lambda x: x[1],reverse=True)
            #print(sorted_pairs)
            top_with_indexes = sorted_pairs[:element_counts[element]]
            #print(top_with_indexes)
            indexes = [temp_index[index_value[0]] for index_value in top_with_indexes]
            indexs_stru = indexes + indexs_stru
            #print(element,[ambiguity_list[a] for a in indexes])
        rename_sample = str(len(sample_index)) + '_' + 'sample' + '_' + MD_out
        write(rename_sample, [data[a] for a in indexs_stru], format='extxyz')
        #print(indexs_stru)
    else:
        print('error: please check clustering_by_ambiguity!')
        sys.exit(1)
    print(f'sample time:{end-temp_2}')
    return len(sample_index),len(sample_index+remaining_index)
if __name__ == '__main__':
    path = os.getcwd()
    sample_main(path,n=3, threshold_init=10, k=2, clustering_by_ambiguity = True)

