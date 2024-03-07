import torch
import torch.nn as nn
import torch.nn.functional as F


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, weight, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.weight = weight
        self.eps = eps

    def forward(self, out, data):
        x = out['pred']
        y = data['gt']
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = self.weight * torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss


class TVLoss(nn.Module):
    def __init__(self, weight: float=1) -> None:
        """Total Variation Loss
        Args:
            weight (float): weight of TV loss
        """
        super().__init__()
        self.weight = weight
    
    def forward(self, out, data):
        x = out['pred']
        y = data['gt']
        batch_size, c, h, w = x.size()
        tv_h = torch.abs(x[:,:,1:,:] - x[:,:,:-1,:]).sum()
        tv_w = torch.abs(x[:,:,:,1:] - x[:,:,:,:-1]).sum()
        return self.weight * (tv_h + tv_w) / (batch_size * c * h * w)