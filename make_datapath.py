import warnings
warnings.filterwarnings('ignore')

import os
import sys
import json
import torchaudio
from tqdm import tqdm
from os.path import join as opj

train_json_path  = './data_path/train/'
clean_train_path = 'resampled train clean path'
noisy_train_path = 'resampled train noisy path'

test_json_path  = './data_path/test/'
clean_test_path = 'resampled test clean path'
noisy_test_path = 'resampled test noisy path'

def get_info(data_path, data_list):
    
    data_path_list = []
    for file in tqdm(data_list):
        file  = opj(data_path, file)
        infos = [file, torchaudio.load(file)[0].shape[-1]]
        data_path_list.append(infos)  
    return data_path_list

def make_json(clean_path, noisy_path, json_path):
    
    clean_list = os.listdir(clean_path)
    noisy_list = os.listdir(noisy_path)

    clean_list.sort(); noisy_list.sort()

    clean_list = get_info(clean_path, clean_list)
    noisy_list = get_info(noisy_path, noisy_list)

    if not os.path.exists(json_path): 
        os.makedirs(json_path)
    with open(opj(json_path, 'clean.json'), 'w') as file:
        json.dump(clean_list, file, indent=4)
    with open(opj(json_path, 'noisy.json'), 'w') as file:
        json.dump(noisy_list, file, indent=4)
    print('---json is generated---')

if __name__ == "__main__":
    print('--- Trainset Json ---')
    make_json(clean_train_path, noisy_train_path, train_json_path)
    print('--- Testset Json ---')
    make_json(clean_test_path, noisy_test_path, test_json_path)

