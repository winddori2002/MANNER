import numpy as np
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .models import *
from .conv_modules import *
from .chunk import *

class ChannelAttention(nn.Module):
    
    def __init__(self, channels):
        super().__init__()
        
        self.fc = nn.Sequential(nn.Linear(channels, channels//2), nn.ReLU(),
                               nn.Linear(channels//2, channels))        
        
    def forward(self, x):
        """
        Input X: [B,N,T]
        Ouput X: [B,N,T]
        """
        
        # [B,N,T] -> [B,N,1]
        attn_max = F.adaptive_max_pool1d(x, 1)
        attn_avg = F.adaptive_avg_pool1d(x, 1)
        
        attn_max = self.fc(attn_max.squeeze())
        attn_avg = self.fc(attn_avg.squeeze())
        
        # [B,N,1]
        attn = attn_max + attn_avg
        attn = F.sigmoid(attn).unsqueeze(-1)
        
        # [B,N,T]
        x = x * attn
        
        return x 
    
class ScaledDotProductAttention(nn.Module):
    
    def __init__(self, temperature):
        super().__init__()
        
        self.temperature = temperature

    def forward(self, q, k, v):

        attn   = torch.matmul(q / self.temperature, k.transpose(2, 3))
        attn   = F.softmax(attn, dim=-1)
        output = torch.matmul(attn, v)

        return output, attn

class GlobalAttention(nn.Module):

    def __init__(self, n_head, d_model, d_k, d_v):
        super().__init__()

        self.n_head = n_head
        self.d_k    = d_k
        self.d_v    = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc   = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

    def forward(self, q, k, v):
        
        """
        Input X: [B*N/3,P,C]
        Output X: [B*N/3,P,C]
        """
        # [B*N,P,C]
        b, p, c     = q.shape 
        d_k, n_head = self.d_k, self.n_head

        # [B*N,P,C] -> [B*N,P,N_head,D_k]
        q = self.w_qs(q).view(b, p, n_head, d_k) 
        k = self.w_ks(k).view(b, p, n_head, d_k)
        v = self.w_vs(v).view(b, p, n_head, d_k)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        q, attn = self.attention(q, k, v)

        q = q.transpose(1, 2).contiguous().view(b, p, -1)
        q = self.fc(q)

        return q

class LocalAttention(nn.Module):

    def __init__(self, channels):
        super().__init__()
        kernel_size1    = 31
        kernel_size2    = 7
        self.depth_conv = nn.Sequential(DepthwiseConv(channels, channels, kernel_size1, stride=1, padding=(kernel_size1 - 1) // 2),
                                        nn.BatchNorm1d(channels), Swish()) 
        self.conv       = BasicConv(2, 1, kernel_size2, stride=1, padding=(kernel_size2-1) // 2, relu=False)
        
    def forward(self, x):
        """
        Input X: [B,N/3,P,C]
        Output X: [B,N/3,P,C]
        """

        b, n, p, c = x.size()
        # [B,N/3,P,C] -> [B*P,N/3,C]
        attn = x.permute(0,2,1,3).contiguous().view(b*p, n, c)
        attn = self.depth_conv(attn)
        attn = torch.cat([torch.max(attn, dim=1)[0].unsqueeze(1), torch.mean(attn, dim=1).unsqueeze(1)], dim=1)
        attn = self.conv(attn)
        # [B*P,1,C]
        attn = F.sigmoid(attn)
        attn = attn.view(b, p, 1, c).permute(0,2,1,3).contiguous()
        x = x * attn    

        return x

class MultiviewAttentionBlock(nn.Module):
    """
    Multiview Attention block.
        channels     :  in channel in encoder.
        head         :  number of heads in global attention.
        segment_len  :  chunk size for overlapped chunking in global and local attention.
    """
    def __init__(self, channels, segment_len, head):
        super().__init__()
        
        self.inter = int(channels / 3)
        d_k        = int(segment_len * head)
        
        self.dsp   = DualPathProcessing(segment_len, segment_len//2)
        
        self.in_branch0 = BasicConv(channels, self.inter, kernel_size=1, stride=1)
        self.in_branch1 = BasicConv(channels, self.inter, kernel_size=1, stride=1)
        self.in_branch2 = BasicConv(channels, self.inter, kernel_size=1, stride=1)
        
        self.channel_attn = ChannelAttention(self.inter)
        self.global_attn  = GlobalAttention(head, segment_len, d_k, d_k)
        self.local_attn   = LocalAttention(self.inter)
        
        self.out_branch0 = BasicConv(self.inter, self.inter, kernel_size=3, stride=1, padding=1)
        self.out_branch1 = BasicConv(self.inter, self.inter, kernel_size=3, stride=1, padding=1)
        self.out_branch2 = BasicConv(self.inter, self.inter, kernel_size=3, stride=1, padding=1)

        self.conv     = BasicConv(self.inter*3, channels, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(channels, channels, kernel_size=1, stride=1, relu=False)
        
        # mask gate as activation unit
        self.output_tanh    = nn.Sequential(nn.Conv1d(channels, channels, kernel_size=1), nn.Tanh())
        self.output_sigmoid = nn.Sequential(nn.Conv1d(channels, channels, kernel_size=1), nn.Sigmoid())
        self.gate_conv      = nn.Sequential(nn.Conv1d(channels, channels, kernel_size=1), nn.ReLU())
        
    def forward(self, x):
        """
        Input X: [B,N,T] pass through three-path with x:[B,N/3,T], respectively.
        Ouput X: [B,N,T]
        C: chunk size
        P: number of chunks
        {x0: channel path, x1: global path, x2: local path}
        """
        # [B,N,T] -> [B,N/3,T]
        x0 = self.in_branch0(x)
        x1 = self.in_branch1(x)
        x2 = self.in_branch2(x)
        
        # overlapped chunking / (B,N/3,C,P) -> [B,N/3,P,C]
        x1 = self.dsp.unfold(x1).transpose(2,3)
        x2 = self.dsp.unfold(x2).transpose(2,3)
        
        b,n,p,c = x1.size()
        
        # [B*N/3,P,C]
        x1 = x1.view(b*n,p,c) 
        
        x0 = self.channel_attn(x0)
        x1 = self.global_attn(x1, x1, x1)
        x2 = self.local_attn(x2)
        
        x1 = x1.view(b,n,p,c)
        
        # [B,N/3,P,C] -> [B,N/3,T]
        x1 = self.dsp.fold(x1.transpose(2,3))
        x2 = self.dsp.fold(x2.transpose(2,3))
        
        x0 = self.out_branch0(x0)
        x1 = self.out_branch1(x1)
        x2 = self.out_branch2(x2)

        # Concat: [B,N/3,T]*3 -> [B,N,T]
        out   = torch.cat([x0, x1, x2], dim=1)
        out   = self.conv(out)
        short = self.shortcut(x)
        
        # mask gate
        gated_tanh = self.output_tanh(out)
        gated_sig  = self.output_sigmoid(out)
        gated      = gated_tanh * gated_sig
        out        = self.gate_conv(gated)
        
        x = short + out
     
        return x
