import numpy as np
import os
import torch
import torchaudio
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader 

from pesq import pesq
from pystoi import stoi
import neptune

from .dataset import *
from .time_loss import *
from .stft_loss import *
from .utils import *
from .models import *
from .metric import *

class Tester:
    def __init__(self, args):
        
        self.args        = args
        self.criterion   = self.select_loss().to(args.device)
        self.stft_loss   = MultiResolutionSTFTLoss(factor_sc=args.stft_sc_factor, factor_mag=args.stft_mag_factor).to(args.device)
        self.kwargs      = {"matching": args.dataset['matching'], "sample_rate":16000}
        self.valset      = TestDataset(args.dataset['train'], valid=args.dataset['val'], **self.kwargs)
        self.val_loader  = DataLoader(self.valset, batch_size=1, shuffle=False)
        self.testset     = TestDataset(args.dataset['test'], **self.kwargs)
        self.test_loader = DataLoader(self.testset, batch_size=1, shuffle=False)
        self.model       = MANNER(**args.manner).to(args.device)
        
    def select_loss(self):
        
        if self.args.loss == 'l1':
            criterion = L1Loss()
        elif self.args.loss == 'l2':
            criterion = nn.MSELoss()
        elif self.args.loss == 'ch':
            criterion = L1CharbonnierLoss()
                           
        return criterion
        
    def test(self, test_type='Test'):
        
        checkpoint = torch.load(self.args.model_path + self.args.model_name)
        self.model.load_state_dict(checkpoint['state_dict'])
        
        total_pesq = 0
        total_stoi = 0
        total_loss = 0
        total_cnt  = 0        
        
        if test_type == 'Test':
            data_loader = self.test_loader
        else:
            data_loader = self.val_loader
        
        self.model.eval()
        with torch.no_grad():
            for i, (noise, clean, l1, l2, file) in enumerate(tqdm(data_loader)):

                noise, clean = noise.to(self.args.device), clean.to(self.args.device)
                noise_label  = noise - clean

                estimate       = self.model(noise)
                noise_estimate = noise - estimate

                loss       = self.criterion(clean.squeeze(1), estimate.squeeze(1))
                noise_loss = self.criterion(noise_label.squeeze(1), noise_estimate.squeeze(1))
                
                if self.args.stft_loss:
                    sc_loss, mag_loss = self.stft_loss(estimate.squeeze(1), clean.squeeze(1))
                    loss += sc_loss + mag_loss  
                    sc_loss, mag_loss = self.stft_loss(noise_estimate.squeeze(1), noise_label.squeeze(1))
                    noise_loss += sc_loss + mag_loss  
                    
                loss = WeightedLoss(clean.squeeze(1), noise_label.squeeze(1), loss, noise_loss)
                    
                estimate = estimate.cpu()
                clean    = clean.cpu()
                noise    = noise.cpu()

                total_loss += loss.item()   
                total_cnt  += clean.shape[0]
                temp_pesq, temp_stoi = get_scores(clean, estimate, self.args)
                total_pesq += temp_pesq
                total_stoi += temp_stoi
                if (test_type =='Test') and (self.args.save_enhanced==True):
                    write_result(estimate, noise, file, self.args)

            pesq_score = total_pesq/total_cnt
            stoi_score = total_stoi/total_cnt 
            loss_score = total_loss/total_cnt
            
            print("Set: {} | Loss: {:.4f} | PESQ: {:.4f} | STOI: {:.4f} ".format(test_type, loss_score, pesq_score, stoi_score)) 
                            
            # only logging for test
            if (test_type == 'Test') and (self.args.logging == True):
                neptune.log_metric('test loss', loss_score)
                neptune.log_metric('test PESQ', pesq_score)
                neptune.log_metric('test STOI', stoi_score)                

