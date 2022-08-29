import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm
from .dataset import *
from .utils import *
from .time_loss import *
from .stft_loss import *
# from .models import *
from .evaluation import *
from .augment import *
from .models import MANNER as MANNER_BASE
from .models_small import MANNER as MANNER_SMALL 


class Trainer:

    def __init__(self, data, args):
        seed_init()
        self.args      = args
#         self.model     = MANNER(**args.manner).to(args.device)
        if 'small' in args.model_name:
            print('--- Load MANNER Small ---')
            self.model = MANNER_SMALL(**args.manner).to(args.device)
        else:
            print('--- Load MANNER BASE or Large ---')
            self.model = MANNER_BASE(**args.manner).to(args.device)  
        self.criterion = self.select_loss().to(args.device)
        self.stft_loss = MultiResolutionSTFTLoss(factor_sc=args.stft_sc_factor, factor_mag=args.stft_mag_factor).to(args.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=0.001, steps_per_epoch=len(data['train']), epochs=args.epoch)
        
        self.train_loader = data['train']
        self.val_loader   = data['val']
        
        self.tester       = Tester(args)
        
        # augment
        if args.aug:
            print('---augmentation applied---')
            augments = []
            augments.append(self.select_aug())
            self.augment = torch.nn.Sequential(*augments)

        # logging
        if args.logging:
            print('---logging start---')
            neptune_load(get_params(args))
            
        # checkpoint
        if args.checkpoint:
            self._load_checkpoint()

    def _load_checkpoint(self):
        checkpoint = torch.load(self.args.model_path + self.args.model_name)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer']) 
        print('---load previous weigths and optimizer---')
       
    def select_loss(self):
        if self.args.loss == 'l1':
            criterion = L1Loss()
        elif self.args.loss == 'l2':
            criterion = nn.MSELoss()   
        elif self.args.loss == 'ch':
            criterion = L1CharbonnierLoss()
        return criterion

    def select_aug(self):
        if self.args.aug_type == 'tempo':
            augmentation = TempoAugment()
        elif self.args.aug_type == 'speed':
            augmentation = SpeedAugment()
        elif self.args.aug_type == 'shift':
            augmentation = Shift()
        return augmentation
        
    def train(self):
        
        best_pesq = 0
        for epoch in range(self.args.epoch):
            
            self.model.train()
            train_loss = self._run_epoch(self.train_loader)
            if self.args.logging == True:
                neptune.log_metric('train loss', train_loss)

            if epoch > self.args.logging_cut:
            
                self.model.eval()
                with torch.no_grad():
                    val_loss, v_pesq, v_stoi = self._run_epoch(self.val_loader, valid=True)

                if v_pesq > best_pesq:
                    best_pesq  = v_pesq
                    checkpoint = {'loss': best_pesq,
                                  'state_dict': self.model.state_dict(),
                                  'optimizer': self.optimizer.state_dict()}
                    torch.save(checkpoint, self.args.model_path + self.args.model_name)                     
            
                print("epoch: {:03d} | trn loss: {:.4f} | val loss: {:.4f}".format(epoch, train_loss, val_loss))
                print("Set: {} | PESQ: {:.4f} | STOI: {:.4f} ".format('Val', v_pesq, v_stoi)) 

                if self.args.logging == True:
                    neptune.log_metric('val loss', val_loss)
                    neptune.log_metric('val PESQ', v_pesq)
                    neptune.log_metric('val STOI', v_stoi)                    
                
    def _run_epoch(self, data_loader, valid=False):
        
        total_loss = 0 
        total_pesq = 0
        total_stoi = 0
        total_cnt  = 0
        
        for i, (noise, clean, l1, l2) in enumerate(tqdm(data_loader)):

            if self.args.aug and not valid:              
                sources = torch.stack([noise - clean, clean])
                sources = self.augment((sources, l1))
                noise, clean = sources
                noise = noise + clean     

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
                                 
            if not valid:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

            if valid:
                estimate = estimate.cpu()
                clean    = clean.cpu()
                noise    = noise.cpu()

                total_cnt  += clean.shape[0]
                temp_pesq, temp_stoi = get_scores(clean, estimate, self.args)
                total_pesq += temp_pesq
                total_stoi += temp_stoi               
                    
            total_loss += loss.item()

        if valid:
            return total_loss/(i+1), total_pesq/total_cnt, total_stoi/total_cnt
        else:
            return total_loss/(i+1)



