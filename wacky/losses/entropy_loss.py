import torch
import torch.nn as nn

from wacky.losses import WackyLoss


class EntropyLoss(WackyLoss, nn.Module):
    def __init__(self, coef=0.01, reduction='sum'):
        super(EntropyLoss, self).__init__()
        self.coef = coef
        self.reduction = reduction

    def forward(self, probs):
        entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
        loss = -self.coef * entropy
        return self._maybe_reduction(loss)
