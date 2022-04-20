# -*- coding: utf-8 -*-
# Modifications in Dataset
# 
# Original copyright:
# Copyright (c) Facebook, Inc. and its affiliates.
# The copyright is under the CC-BY-NC 4.0 from Demucs.
# Demucs (https://github.com/facebookresearch/denoiser) / author: adefossez


import re
import random
import numpy as np
import json
from pathlib import Path
import math
import os
import sys

import torchaudio
from torch.nn import functional as F

def match_files(noisy, clean, matching="sort"):
    """match_files.
    Sort files to match noisy and clean filenames.
    :param noisy: list of the noisy filenames
    :param clean: list of the clean filenames
    :param matching: the matching function, at this point only sort is supported
    """
    if matching == "sort":
        noisy.sort()
        clean.sort()
    else:
        raise ValueError(f"Invalid value for matching {matching}")

def train_select(valid_list, json_list):
    
    new_json = []
    for file in json_list:
        if ('p'+str(valid_list[0])+'_' not in file[0]) and ('p'+str(valid_list[1])+'_' not in file[0]) :
            new_json.append(file)
            
    return new_json
        
def valid_select(valid_list, json_list):
    
    new_json = []
    for file in json_list:
        if ('p'+str(valid_list[0])+'_' in file[0]) or ('p'+str(valid_list[1])+'_' in file[0]) :
            new_json.append(file)
                
    return new_json


class TrainDataset:
    def __init__(self, json_dir, matching="sort", valid=None, length=None, stride=None,
                 pad=True, sample_rate=None):
        """
        TrainDataset:
            json_dir: directory containing both clean.json and noisy.json.
            matching: matching function for the files.
            valid: valid speaker list.
            length: maximum sequence length.
            stride: the stride used for splitting audio sequences.
            pad: pad the end of the sequence with zeros.
            sample_rate: the signals sampling rate.
        """
        noisy_json = os.path.join(json_dir, 'noisy.json')
        clean_json = os.path.join(json_dir, 'clean.json')
        with open(noisy_json, 'r') as f:
            noisy = json.load(f)
        with open(clean_json, 'r') as f:
            clean = json.load(f)
       
       # select speakers execpt valid speaker from trainset
        if valid != None:
            noisy = train_select(valid, noisy)
            clean = train_select(valid, clean)       

        match_files(noisy, clean, matching)
        kw = {'length': length, 'stride': stride, 'pad': pad, 'sample_rate': sample_rate}
        self.clean_set = Audioset(clean, **kw)
        self.noisy_set = Audioset(noisy, **kw)

        assert len(self.clean_set) == len(self.noisy_set)

    def __getitem__(self, index):
        return self.noisy_set[index][0], self.clean_set[index][0], self.noisy_set[index][1], self.clean_set[index][1]

    def __len__(self):
        return len(self.noisy_set)
    
class ValDataset:
    def __init__(self, json_dir, matching="sort", valid=None, length=None, stride=None,
                 pad=True, sample_rate=None):
        """
        ValDataset:
            json_dir: directory containing both clean.json and noisy.json.
            matching: matching function for the files.
            valid: valid speaker list.
            length: maximum sequence length.
            stride: the stride used for splitting audio sequences.
            pad: pad the end of the sequence with zeros.
            sample_rate: the signals sampling rate.
        """
        noisy_json = os.path.join(json_dir, 'noisy.json')
        clean_json = os.path.join(json_dir, 'clean.json')
        with open(noisy_json, 'r') as f:
            noisy = json.load(f)
        with open(clean_json, 'r') as f:
            clean = json.load(f)
       
       # select only valid speakers from trainset
        if valid != None:
            noisy = valid_select(valid, noisy)
            clean = valid_select(valid, clean)       

        match_files(noisy, clean, matching)
        kw = {'length': length, 'stride': stride, 'pad': pad, 'sample_rate': sample_rate}
        self.clean_set = Audioset(clean, **kw)
        self.noisy_set = Audioset(noisy, **kw)

        assert len(self.clean_set) == len(self.noisy_set)

    def __getitem__(self, index):
        return self.noisy_set[index][0], self.clean_set[index][0], self.noisy_set[index][1], self.clean_set[index][1]

    def __len__(self):
        return len(self.noisy_set)
    
class TestDataset:
    def __init__(self, json_dir, matching="sort", valid=None, length=None, stride=None,
                 pad=True, sample_rate=None):
        """
        ValDataset:
            json_dir: directory containing both clean.json and noisy.json.
            matching: matching function for the files.
            valid: valid speaker list.
            length: maximum sequence length.
            stride: the stride used for splitting audio sequences.
            pad: pad the end of the sequence with zeros.
            sample_rate: the signals sampling rate.
        """
        noisy_json = os.path.join(json_dir, 'noisy.json')
        clean_json = os.path.join(json_dir, 'clean.json')
        with open(noisy_json, 'r') as f:
            noisy = json.load(f)
        with open(clean_json, 'r') as f:
            clean = json.load(f)
        
        # select valid speaker from trainset if you want to use valid set for test phase
        if valid != None:
            noisy = valid_select(valid, noisy)
            clean = valid_select(valid, clean)

        match_files(noisy, clean, matching)
        kw = {'length': length, 'stride': stride, 'pad': pad, 'sample_rate': sample_rate, 'with_path': True}
        self.clean_set = Audioset(clean, **kw)
        self.noisy_set = Audioset(noisy, **kw)

        assert len(self.clean_set) == len(self.noisy_set)

    def __getitem__(self, index):
        return self.noisy_set[index][0], self.clean_set[index][0], self.noisy_set[index][1], self.clean_set[index][1], self.clean_set[index][2] 

    def __len__(self):
        return len(self.noisy_set)
            
class Audioset:
    def __init__(self, files=None, length=None, stride=None,
                 pad=True, with_path=False, sample_rate=None):
        """
        files should be a list [(file, length)]
        """
        self.files        = files
        self.num_examples = []
        self.length       = length
        self.stride       = stride or length
        self.with_path    = with_path
        self.sample_rate  = sample_rate
        for file, file_length in self.files:
            if length is None:
                examples = 1
            elif file_length < length:
                examples = 1 if pad else 0
            elif pad:
                examples = int(math.ceil((file_length - self.length) / self.stride) + 1)
            else:
                examples = (file_length - self.length) // self.stride + 1
            self.num_examples.append(examples)

    def __len__(self):
        return sum(self.num_examples)

    def __getitem__(self, index):
        
        for (file, _), examples in zip(self.files, self.num_examples):
            if index >= examples:
                index -= examples
                continue
            num_frames = 0
            offset     = 0
            if self.length is not None:
                offset     = self.stride * index
                num_frames = self.length
            out, sr    = torchaudio.load(str(file), offset=offset, num_frames=num_frames)
            out_length = out.shape[-1]
            
            if self.sample_rate is not None:
                if sr != self.sample_rate:
                    raise RuntimeError(f"Expected {file} to have sample rate of "
                                       f"{self.sample_rate}, but got {sr}")
            if num_frames:
                out = F.pad(out, (0, num_frames - out.shape[-1]))
            if self.with_path:
                return out, out_length,file
            else:
                return out, out_length
