import random
import numpy as np
import torch
import torchaudio
from torch import nn
from torch.nn import functional as F

class Shift(nn.Module):
    # -*- coding: utf-8 -*-
    # Shift augmentation
    #  
    # Original copyright:
    # Copyright (c) Facebook, Inc. and its affiliates.
    # Demucs (https://github.com/facebookresearch/denoiser) / author: adefossez
    """Shift."""

    def __init__(self, shift=8192, same=False):
        """
        :param shift: randomly shifts the signals up to a given factor
        :param same: shifts both clean and noisy files by the same factor
        """
        super().__init__()
        self.shift = shift
        self.same  = same

    def forward(self, wav):
        wav, _ = wav
        sources, batch, channels, length = wav.shape
        length = length - self.shift
        if self.shift > 0:
            if not self.training:
                wav = wav[..., :length]
            else:
                offsets = torch.randint(
                    self.shift,
                    [1 if self.same else sources, batch, 1, 1], device=wav.device)
                offsets = offsets.expand(sources, -1, channels, -1)
                indexes = torch.arange(length, device=wav.device)
                wav     = wav.gather(3, indexes + offsets)
        return wav
    
class TempoAugment(nn.Module):
    
    def __init__(self):
        super().__init__()

    def tempo_effect(self, input):
        tempo   = np.random.uniform(0.9 ,1.1)
        effects = [['tempo', f'{tempo:.5f}'] ,['rate', '16000']]
        output  = torchaudio.sox_effects.apply_effects_tensor(input, 16000, effects)[0]
        return output
        
    def forward(self, wav):
        wav, wav_length = wav
        sources, batch, channels, length = wav.shape
        noise, clean = wav[0], wav[1]
        
        clean        = [c[...,:l] for c,l in zip(clean, wav_length)]
        clean        = [self.tempo_effect(c) for c in clean]
        clean_length = [c.shape[-1] for c in clean]
        max_length   = max(clean_length)
        
        clean2 = []
        noise2 = []
        for i in range(len(wav_length)):
            cl, nl = clean_length[i], wav_length[i]
            if cl >= nl:
                clean2.append(clean[i][...,:nl])
                noise2.append(noise[i])
            else:
                clean2.append(clean[i])
                noise2.append(noise[i][...,:cl])

        clean2 = [F.pad(c, (0, length-c.shape[-1])) for c in clean2]
        clean2 = torch.stack(clean2)
        noise2 = [F.pad(n, (0, length-n.shape[-1])) for n in noise2]
        noise2 = torch.stack(noise2)
        wav    = torch.stack([noise2, clean2])
        wav    = wav[...,:max_length]
        
        return wav

class SpeedAugment(nn.Module):
    
    def __init__():
        super().__init__()
    
    def speed_effect(self, input):
        speed   = np.random.uniform(0.9 ,1.1)
        effects = [['speed', f'{speed:.5f}'] ,['rate', '16000']]
        output  = torchaudio.sox_effects.apply_effects_tensor(input, 16000, effects)[0]
        return output 

    def forward(self, wav):
        wav, wav_length = wav
        sources, batch, channels, length = wav.shape
        noise, clean = wav[0], wav[1]
        
        clean        = [c[...,:l] for c,l in zip(clean, wav_length)]
        clean        = [self.speed_effect(c) for c in clean]
        clean_length = [c.shape[-1] for c in clean]
        max_length   = max(clean_length)
        
        clean2 = []
        noise2 = []
        for i in range(len(wav_length)):
            cl, nl = clean_length[i], wav_length[i]
            if cl >= nl:
                clean2.append(clean[i][...,:nl])
                noise2.append(noise[i])
            else:
                clean2.append(clean[i])
                noise2.append(noise[i][...,:cl])

        clean2 = [F.pad(c, (0, length-c.shape[-1])) for c in clean2]
        clean2 = torch.stack(clean2)
        noise2 = [F.pad(n, (0, length-n.shape[-1])) for n in noise2]
        noise2 = torch.stack(noise2)
        wav    = torch.stack([noise2, clean2])
        wav    = wav[...,:max_length]
        
        return wav