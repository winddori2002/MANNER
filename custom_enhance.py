import warnings
warnings.filterwarnings('ignore')

import os
import json
import yaml
import torch
import argparse
from tqdm import tqdm

from src.dataset import *
from src.utils import *
from src.models import MANNER as MANNER_BASE
from src.models_small import MANNER as MANNER_SMALL 

def main(args):
    
    seed_init()
    
    # Select model version
    if 'base' in args.model_name:
        model = MANNER_BASE(in_channels=1, out_channels=1, hidden=60, depth=4, kernel_size=8, stride=4, growth=2, head=1, segment_len=64).to(args.device)
    elif 'large' in args.model_name:
        model = MANNER_BASE(in_channels=1, out_channels=1, hidden=120, depth=4, kernel_size=8, stride=4, growth=2, head=1, segment_len=64).to(args.device)
    elif 'small' in args.model_name:
        model = MANNER_SMALL(in_channels=1, out_channels=1, hidden=60, depth=4, kernel_size=8, stride=4, growth=2, head=1, segment_len=64).to(args.device)
        
    checkpoint = torch.load(f'./weights/{args.model_name}')
    model.load_state_dict(checkpoint['state_dict'])
    print(f'--- Load {args.model_name} weights ---')
        
    model.eval()
    with torch.no_grad():
        
        output_path = args.noisy_path # you can change the output path
        noisy_list  = os.listdir(args.noisy_path)
        for n_file in tqdm(noisy_list):
            
            noisy, sr = torchaudio.load(f'{args.noisy_path}/{n_file}')
            if sr != 16000:
                tf    = torchaudio.transforms.Resample(sr, 16000)
                noisy = tf(noisy)
            
            noisy    = noisy.unsqueeze(0).to(args.device)
            enhanced = model(noisy)
            enhanced = enhanced.squeeze(0).detach().cpu()

            save_name = n_file.split('.')[0] + '_enhanced.wav' 
            torchaudio.save(f'{output_path}/{save_name}', enhanced, 16000)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0', help='Cuda device')
    parser.add_argument('--noisy_path', type=str, default='./examples', help='Noisy input folder')
    parser.add_argument('--model_name', type=str, default='manner_base.pth', help='Model name')
    
    args = parser.parse_args()
    main(args)
