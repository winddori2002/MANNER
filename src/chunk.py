import numpy as np
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import *

class SinglePathProcessing(nn.Module):
    """
    Chunking without overlapp.
    segment_len: chunk size
    """
    def __init__(self, segment_len):
        super().__init__()
        
        self.segment_len = segment_len
        
    def pad_segment(self, input):
        """This pad_segment is for direct segment pad without stride"""
        # input size : (B, N, T)
        b, dim, s = input.shape
        rest      = s % self.segment_len
        
        if rest > 0 :
            rest  = self.segment_len - rest
            pad   = (torch.zeros(b, dim, rest)).type(input.type()).to(input.device)
            input = torch.cat([input, pad], dim = -1)
            
        return input, rest
    
    def segmentation(self, input):
        """This segment is for direct segment without stride"""
        # (B, N, T)
        input, rest = self.pad_segment(input)
        b, dim, s   = input.shape
        
        # (B, N, L, T)
        segments = input.view(b, dim, -1,  self.segment_len)
        
        return segments, rest
    
class DualPathProcessing(nn.Module):

    # -*- coding: utf-8 -*-
    # Original copyright:
    # Asteroid (https://github.com/asteroid-team/asteroid)
    """
    Overlapped chunking.
    chunk_size: chunk size.
    hop_size: overlapping size.
    """

    def __init__(self, chunk_size, hop_size):
        super(DualPathProcessing, self).__init__()
        self.chunk_size    = chunk_size
        self.hop_size      = hop_size
        self.n_orig_frames = None

    def unfold(self, x):
        
        # x is (batch, chan, frames)
        batch, chan, frames = x.size()
        assert x.ndim == 3
        self.n_orig_frames = x.shape[-1]
        unfolded = torch.nn.functional.unfold(
            x.unsqueeze(-1),
            kernel_size=(self.chunk_size, 1),
            padding=(self.chunk_size, 0),
            stride=(self.hop_size, 1),
        )

        return unfolded.reshape(
            batch, chan, self.chunk_size, -1
        )  # (batch, chan, chunk_size, n_chunks)

    def fold(self, x, output_size=None):
        r"""
        Folds back the spliced feature tensor.
        Input shape $(batch, channels, chunksize, nchunks)$ to original shape
        $(batch, channels, time)$ using overlap-add.
        Args:
            x (:class:`torch.Tensor`): spliced feature tensor of shape
                $(batch, channels, chunksize, nchunks)$.
            output_size (int, optional): sequence length of original feature tensor.
                If None, the original length cached by the previous call of
                :meth:`unfold` will be used.
        Returns:
            :class:`torch.Tensor`:  feature tensor of shape $(batch, channels, time)$.
        .. note:: `fold` caches the original length of the input.
        """
        output_size = output_size if output_size is not None else self.n_orig_frames
        # x is (batch, chan, chunk_size, n_chunks)
        batch, chan, chunk_size, n_chunks = x.size()
        to_unfold = x.reshape(batch, chan * self.chunk_size, n_chunks)
        x = torch.nn.functional.fold(
            to_unfold,
            (output_size, 1),
            kernel_size=(self.chunk_size, 1),
            padding=(self.chunk_size, 0),
            stride=(self.hop_size, 1),
        )

        # force float div for torch jit
        x /= float(self.chunk_size) / self.hop_size

        return x.reshape(batch, chan, self.n_orig_frames)

    @staticmethod
    def intra_process(x, module):
        r"""Performs intra-chunk processing.
        Args:
            x (:class:`torch.Tensor`): spliced feature tensor of shape
                (batch, channels, chunk_size, n_chunks).
            module (:class:`torch.nn.Module`): module one wish to apply to each chunk
                of the spliced feature tensor.
        Returns:
            :class:`torch.Tensor`: processed spliced feature tensor of shape
            $(batch, channels, chunksize, nchunks)$.
        .. note:: the module should have the channel first convention and accept
            a 3D tensor of shape $(batch, channels, time)$.
        """

        # x is (batch, channels, chunk_size, n_chunks)
        batch, channels, chunk_size, n_chunks = x.size()
        # we reshape to batch*chunk_size, channels, n_chunks
        x = x.transpose(1, -1).reshape(batch * n_chunks, chunk_size, channels).transpose(1, -1)
        x = module(x)
        x = x.reshape(batch, n_chunks, channels, chunk_size).transpose(1, -1).transpose(1, 2)
        return x

    @staticmethod
    def inter_process(x, module):
        r"""Performs inter-chunk processing.
        Args:
            x (:class:`torch.Tensor`): spliced feature tensor of shape
                $(batch, channels, chunksize, nchunks)$.
            module (:class:`torch.nn.Module`): module one wish to apply between
                each chunk of the spliced feature tensor.
        Returns:
            x (:class:`torch.Tensor`): processed spliced feature tensor of shape
            $(batch, channels, chunksize, nchunks)$.
        .. note:: the module should have the channel first convention and accept
            a 3D tensor of shape $(batch, channels, time)$.
        """

        batch, channels, chunk_size, n_chunks = x.size()
        x = x.transpose(1, 2).reshape(batch * chunk_size, channels, n_chunks)
        x = module(x)
        x = x.reshape(batch, chunk_size, channels, n_chunks).transpose(1, 2)
        return x