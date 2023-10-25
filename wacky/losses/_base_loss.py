from abc import ABC
import torch


class WackyLoss(ABC):
    reduction: str

    def _maybe_reduction(self, loss):
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:
            raise ValueError(
                f"Invalid reduction. Expected 'none', 'mean', or 'sum', got {self.reduction}"
            )
