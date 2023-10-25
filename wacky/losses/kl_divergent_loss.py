import torch
import torch.nn as nn

from wacky.losses import WackyLoss


class KLDivergenceLoss(WackyLoss, nn.Module):
    def __init__(self, target, coef=1.0, reduction='sum'):
        super(KLDivergenceLoss, self).__init__()
        self.target = target
        self.coef = coef
        self.reduciton = reduction

    def forward(self, log_probs, old_log_probs):
        kl_div = torch.sum(log_probs.exp() * (log_probs - old_log_probs), dim=-1)
        loss = self.coef * ((kl_div - self.target) ** 2)
        return loss
