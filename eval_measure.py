import os
import json
import librosa
from tqdm import tqdm
from src.metric import eval_composite
from os.path import join as opj

clean_path    = 'resampled clean test path'
enhanced_path = './enhanced'

def eval_all_measure():

    clean_list    = os.listdir(clean_path)
    enhanced_list = [i for i in os.listdir(enhanced_path) if '_enhanced.wav' in i]
    clean_list.sort(); enhanced_list.sort()

    csig, cbak, covl, pesq_score, count = 0, 0, 0, 0, 0
    for clean_file, enhanced_file in tqdm(zip(clean_list, enhanced_list)):
        
        assert clean_file[:-4] in enhanced_file,'Not matched clean and enhanced wav'

        clean      = librosa.load(opj(clean_path, clean_file), sr=None)[0]
        enhance    = librosa.load(opj(enhanced_path, enhanced_file), sr=None)[0]
        res        = eval_composite(clean, enhance)
        csig       += res['csig']
        cbak       += res['cbak']
        covl       += res['covl']
        pesq_score += res['pesq']
        count      += 1        
        
    print(f'CSIG: {csig/count}, CBAK: {cbak/count}, COVL: {covl/count}, PESQ:{pesq_score/count}')

if __name__ == "__main__":
    eval_all_measure()