import numpy as np
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import *
from .conv_modules import *
from .utils import *


class MaskGate(nn.Module):

    def __init__(self, channels):
        super().__init__()
        
        self.output      = nn.Sequential(nn.Conv1d(channels, channels, kernel_size=1), nn.Tanh())
        self.output_gate = nn.Sequential(nn.Conv1d(channels, channels, kernel_size=1), nn.Sigmoid())
        self.mask        = nn.Sequential(nn.Conv1d(channels, channels, kernel_size=1), nn.ReLU())

    def forward(self, x):

        mask = self.output(x) * self.output_gate(x)
        mask = self.mask(mask)
    
        return mask
    
class Encoder(nn.Module):
  
    def __init__(self, in_channels, out_channels, kernel_size, stride, segment_len, head):
        super().__init__()
        
        self.down_conv  = nn.Sequential(nn.Conv1d(in_channels, in_channels, kernel_size, stride),
                                        nn.BatchNorm1d(in_channels), nn.ReLU())
        self.conv_block = ResConBlock(in_channels, growth1=2, growth2=2)
        self.attn_block = MultiviewAttentionBlock(out_channels, segment_len, head)

    def forward(self, x):

        x = self.down_conv(x)        
        x = self.conv_block(x)
        x = self.attn_block(x)

        return x
    
class Decoder(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride, segment_len, head):
        super().__init__()
        
        self.conv_block = ResConBlock(in_channels, growth1=2, growth2=1/2)
        self.up_conv    = nn.Sequential(nn.ConvTranspose1d(out_channels, out_channels, kernel_size, stride),
                                        nn.BatchNorm1d(out_channels), nn.ReLU())
        self.attn_block = MultiviewAttentionBlock(out_channels, segment_len, head)

    def forward(self, x):
        
        x = self.conv_block(x)
        x = self.up_conv(x)
        x = self.attn_block(x)
        
        return x
    
class MANNER_Block(nn.Module):
    """
    MANNER block.
        in_channels  :  inital in channel.
        out_channels :  inital out channel.
        hidden       :  hidden channel size.
        depth        :  depth of encoder and decoder layer.
        kernel_size  :  kernel size for down and up convolution layer.
        stride       :  stride for down and up convolution layer.
        growth       :  growth rate of the channel size.
        head         :  number of heads in global attention.
        segment_len  :  chunk size for overlapped chunking in global and local attention.
        
    in_conv   :  initial convolution layer to expand channel from 1 to N (hidden).
    out_conv  :  last convolution layer to reduce channel from N to 1, estimating enhanced speech.
    encoder   :  encoder number of depth layers.
    decoder   :  decoder number of depth layers.
    linear    :  linear transformation to pass decoder.
    mask_gate :  estimate mask to apply on the output of the initial convolution layer.
    """
    def __init__(self, in_channels, out_channels, hidden, depth, kernel_size, stride, growth, head, segment_len):
        super().__init__()
        
        self.depth    = depth      
        self.in_conv  = nn.Sequential(nn.Conv1d(in_channels, hidden, kernel_size=3, stride=1, padding=1),
                                 nn.BatchNorm1d(hidden), nn.ReLU())
        self.out_conv = nn.Sequential(nn.Conv1d(hidden, in_channels, kernel_size=3, stride=1, padding=1))      
        in_channels   = in_channels*hidden
        out_channels  = out_channels*growth
        
        encoder = []
        decoder = []
        for layer in range(depth):
            encoder.append(Encoder(in_channels, out_channels*hidden, kernel_size, stride, segment_len, head))
            decoder.append(Decoder(out_channels*hidden, in_channels, kernel_size, stride, segment_len, head))

            in_channels  = hidden*(2**(layer+1))
            out_channels *= growth
            
        decoder.reverse()
        
        self.encoder = nn.ModuleList(encoder)
        self.decoder = nn.ModuleList(decoder)      
        
        hdim           = (hidden*growth**(layer+1))
        self.linear    = nn.Sequential(nn.Linear(hdim, hdim, bias = False), nn.ReLU())
        self.mask_gate = MaskGate(hidden)
            
    def forward(self, x):
        """
        input X : [B, 1, T]
        output X: [B, 1, T]
        """    
        x       = self.in_conv(x) # [B,1,T] -> [B,N,T]
        enc_out = x # for applying mask   
        
        skips = []
        for encoder in self.encoder:
            x = encoder(x)
            skips.append(x)
            
        x = x.permute(0, 2, 1) # [B,N,T] -> [B,T,N]
        x = self.linear(x)
        x = x.permute(0, 2, 1) # [B,N,T]

        for decoder in self.decoder:
            skip = skips.pop(-1)
            x    = x + skip[..., :x.shape[-1]]
            x    = decoder(x)        

        mask = self.mask_gate(x) # [B,N,T]
        x    = enc_out * mask
        x    = self.out_conv(x) # [B,1,T]
        
        return x
    
class MANNER(nn.Module):
    """
    MANNER for speech enhancement in time-domain.
        in_channels  :  inital in channel.
        out_channels :  inital out channel.
        hidden       :  hidden channel size.
        depth        :  depth of encoder and decoder layer.
        kernel_size  :  kernel size for down and up convolution layer.
        stride       :  stride for down and up convolution layer.
        growth       :  growth rate of the channel size.
        head         :  number of heads in global attention.
        segment_len  :  chunk size for overlapped chunking in global and local attention.
    """
    eps = 1e-3
    rescale = 0.1
    def __init__(self, in_channels, out_channels, hidden, depth, kernel_size, stride, growth, 
                       head, segment_len):
        super().__init__()
        
        self.kernel_size  = kernel_size
        self.stride       = stride
        self.depth        = depth
        
        self.manner_block = MANNER_Block(in_channels, out_channels, hidden, depth, kernel_size, stride, 
                                        growth, head, segment_len)
        print('---rescale applied---')
        rescale_module(self, reference=self.rescale)

    def padding(self, length):

        length = math.ceil(length)
        for idx in range(self.depth):
            length = math.ceil((length - self.kernel_size) / self.stride) + 1
            length = max(length, 1)
        for idx in range(self.depth):
            length = (length - 1) * self.stride + self.kernel_size
        length = int(math.ceil(length))
        
        return int(length)

    def forward(self, x):

        """
        Input X: [B, 1, T]
        B: batch size
        1: channel size
        T: signal length
        """

        # input normalization by its standardiviation
        x2  = x.mean(dim=1, keepdim=True)
        std = x2.std(dim=-1, keepdim=True)
        x   = x / (self.eps + std)
        
        # estimate enhanced speech
        length = x.shape[-1]
        x      = F.pad(x, (0, self.padding(length) - length))
        x      = self.manner_block(x)
        x      = x[..., :length]
        
        return std * x


##########The codes of Encoder, Decoder, MANNER block, MANNER for MANNER (small) are as below: ##############
    
# class Encoder(nn.Module):
    
#     def __init__(self, in_channels, out_channels, kernel_size, stride, segment_len, head, layer, depth):
#         super().__init__()
        
#         self.layer      = layer
#         self.depth      = depth
#         self.down_conv  = nn.Sequential(nn.Conv1d(in_channels, in_channels, kernel_size, stride),
#                                        nn.BatchNorm1d(in_channels), nn.ReLU())
#         self.conv_block = ResConBlock(in_channels, growth1=2, growth2=2)
#         if layer==(depth-1):
#             self.attn_block = MultiviewAttentionBlock(out_channels, segment_len, head)

#     def forward(self, x):

#         x = self.down_conv(x)        
#         x = self.conv_block(x)
#         if self.layer==(self.depth-1):
#             x = self.attn_block(x)

#         return x
    
# class Decoder(nn.Module):
    
#     def __init__(self, in_channels, out_channels, kernel_size, stride, segment_len, head, layer, depth):
#         super().__init__()
        
#         self.layer      = layer
#         self.depth      = depth
#         self.conv_block = ResConBlock(in_channels, growth1=2, growth2=1/2)
#         self.up_conv    = nn.Sequential(nn.ConvTranspose1d(out_channels, out_channels, kernel_size, stride),
#                                      nn.BatchNorm1d(out_channels), nn.ReLU())
#         if layer==(depth-1):
#             self.attn_block = MultiviewAttentionBlock(out_channels, segment_len, head)

#     def forward(self, x):
        
#         x = self.conv_block(x)
#         x = self.up_conv(x)
#         if self.layer==(self.depth-1):
#             x = self.attn_block(x)
        
#         return x
    
# class MANNER_Block(nn.Module):
    
#     def __init__(self, in_channels, out_channels, hidden, depth, kernel_size, stride, growth, head, segment_len):
#         super().__init__()
        
#         self.depth    = depth      
#         self.in_conv  = nn.Sequential(nn.Conv1d(in_channels, hidden, kernel_size=3, stride=1, padding=1),
#                                  nn.BatchNorm1d(hidden), nn.ReLU())
#         self.out_conv = nn.Sequential(nn.Conv1d(hidden, in_channels, kernel_size=3, stride=1, padding=1))      
#         in_channels   = in_channels*hidden
#         out_channels  = out_channels*growth
        
#         encoder = []
#         decoder = []
#         for layer in range(depth):
#             encoder.append(Encoder(in_channels, out_channels*hidden, kernel_size, stride, segment_len, head, layer, depth))
#             decoder.append(Decoder(out_channels*hidden, in_channels, kernel_size, stride, segment_len, head, layer, depth))

#             in_channels  = hidden*(2**(layer+1))
#             out_channels *= growth
            
#         decoder.reverse()
        
#         self.encoder = nn.ModuleList(encoder)
#         self.decoder = nn.ModuleList(decoder)      
        
#         hdim           = (hidden*growth**(layer+1))
#         self.linear    = nn.Sequential(nn.Linear(hdim, hdim, bias = False), nn.ReLU())
#         self.mask_gate = MaskGate(hidden)
            
#     def forward(self, x):
        
#         x       = self.in_conv(x)
#         enc_out = x
        
#         skips = []
#         for encoder in self.encoder:
#             x = encoder(x)
#             skips.append(x)
            
#         x = x.permute(0, 2, 1) # (B,N,T) -> (B,T, N)
#         x = self.linear(x)
#         x = x.permute(0, 2, 1) # (B,N,T)

#         for decoder in self.decoder:
#             skip = skips.pop(-1)
#             x    = x + skip[..., :x.shape[-1]]
#             x    = decoder(x)        

#         mask = self.mask_gate(x)
#         x    = enc_out * mask
#         x    = self.out_conv(x)
        
#         return x
    
# class MANNER(nn.Module):
#     eps = 1e-3
#     rescale = 0.1
    
#     def __init__(self, in_channels, out_channels, hidden, depth, kernel_size, stride, growth, 
#                        head, segment_len):
#         super().__init__()
        
#         self.kernel_size = kernel_size
#         self.stride      = stride
#         self.depth       = depth
        
#         self.manner_block = MANNER_Block(in_channels, out_channels, hidden, depth, kernel_size, stride, 
#                                         growth, head, segment_len)

#         print('---rescale applied---')
#         rescale_module(self, reference=self.rescale)

#     def padding(self, length):

#         length = math.ceil(length)
#         for idx in range(self.depth):
#             length = math.ceil((length - self.kernel_size) / self.stride) + 1
#             length = max(length, 1)
#         for idx in range(self.depth):
#             length = (length - 1) * self.stride + self.kernel_size
#         length = int(math.ceil(length))
        
#         return int(length)

#     def forward(self, x):
        
#         x2  = x.mean(dim=1, keepdim=True)
#         std = x2.std(dim=-1, keepdim=True)
#         x   = x / (self.eps + std)
        
#         # x (B, 1, T)
#         length = x.shape[-1]
#         x      = F.pad(x, (0, self.padding(length) - length))
#         x      = self.manner_block(x)
#         x      = x[..., :length]
        
#         return std * x