import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import permutations

class L1Loss(nn.Module):
    """L1 loss for weighted loss"""
    def __init__(self):
        super(L1Loss, self).__init__()

    def forward(self, clean, clean_est, dim=2):
        if dim==2:
            return torch.abs(clean-clean_est).mean(dim=-1)
        elif dim==3:
            return torch.abs(clean-clean_est).mean(dim=-1).mean(dim=-1)

class L1CharbonnierLoss(nn.Module):
    """L1 Charbonnierloss for weighted loss."""
    def __init__(self):
        super(L1CharbonnierLoss, self).__init__()
        self.eps = 1e-6

    def forward(self, x, y, dim=2):
        
        diff = x - y
        loss = torch.sqrt(diff*diff + self.eps)
        if dim==2:
            return loss.mean(dim=-1)
        elif dim==3:
            return loss.mean(dim=-1).mean(dim=-1)
    
def WeightedLoss(clean, noise_label, clean_loss, noise_loss, eps=2e-7):
    bsum = lambda x: torch.sum(x, dim=1)
    a    = bsum(clean**2) / (bsum(clean**2) + bsum(noise_label**2) + eps)
    
    return torch.mean(a*clean_loss + (1-a)*noise_loss)