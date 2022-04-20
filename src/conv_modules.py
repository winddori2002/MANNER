import numpy as np
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import *
from .utils import *


class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super().__init__()

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn   = nn.BatchNorm1d(out_channels, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
            
        return x
    
class Swish(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        
        return x * self.sigmoid(x)

class DepthwiseConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0, bias=False):
        super().__init__()

        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, 
                              groups=in_channels, stride=stride, padding=padding, bias=bias)

    def forward(self, x):
        
        x = self.conv(x)
        
        return x
    
class PointwiseConv(nn.Module):
    
    def __init__(self, in_channels, out_channels, stride=1, padding=0, bias=True):
        super(PointwiseConv, self).__init__()
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                              stride=stride, padding=padding, bias=bias)

    def forward(self, x):
        
        x = self.conv(x)
        
        return x


class ResConBlock(nn.Module):
    """
    Residual Conformer block.
        in_channels  :  in channel in encoder and decoder.
        kernel_size  :  kernel size for depthwise convolution.
        growth1      :  expanding channel size and reduce after GLU.
        growth2      :  decide final channel size in the block, encoder for 2, decoder for 1/2.
    """
    def __init__(self, in_channels, kernel_size=31, growth1=2, growth2=2):
        super().__init__()
        
        out_channels1 = int(in_channels*growth1)
        out_channels2 = int(in_channels*growth2)
        
        self.point_conv1 = nn.Sequential(
                                PointwiseConv(in_channels, out_channels1, stride=1, padding=0, bias=True),
                                nn.BatchNorm1d(out_channels1), nn.GLU(dim=1))
        self.depth_conv  = nn.Sequential(
                                DepthwiseConv(in_channels, in_channels, kernel_size, stride=1, padding=(kernel_size - 1) // 2),
                                nn.BatchNorm1d(in_channels), Swish())
        self.point_conv2 = nn.Sequential(
                                PointwiseConv(in_channels, out_channels2, stride=1, padding=0, bias=True),
                                nn.BatchNorm1d(out_channels2), Swish())
        self.conv     = BasicConv(out_channels2, out_channels2, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_channels, out_channels2, kernel_size=1, stride=1, relu=False)

    def forward(self, x):
        
        out = self.point_conv1(x)
        out = self.depth_conv(out)
        out = self.point_conv2(out)
        out = self.conv(out)
        out += self.shortcut(x)
        out = F.relu(out)
        
        return out