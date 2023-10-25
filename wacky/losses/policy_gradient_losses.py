import torch
import torch.nn as nn

from wacky.losses import WackyLoss


class PolicyGradientLoss(WackyLoss, nn.Module):
    def __init__(self, reduction='sum'):
        super(PolicyGradientLoss, self).__init__()
        self.reduction = reduction

    def forward(self, log_probs, policy_gradient_term):
        loss = -log_probs * policy_gradient_term
        return self._maybe_reduction(loss)


class SurrogateLoss(WackyLoss, nn.Module):
    def __init__(self, reduction='sum'):
        super(SurrogateLoss, self).__init__()
        self.reduction = reduction

    def forward(self, log_probs, old_log_probs, policy_gradient_term):
        # Compute importance sampling ratio
        ratio = torch.exp(log_probs - old_log_probs)

        # Compute surrogate objective
        loss = - ratio * policy_gradient_term
        return self._maybe_reduction(loss)


class ClippedSurrogateLoss(WackyLoss, nn.Module):
    def __init__(self, epsilon=0.2, reduction='sum'):
        """
        Initialize the clipped surrogate loss function.

        Args:
            epsilon (float): Clip parameter.
            reduction (str, optional): Specifies the reduction to apply to the output:
                'none' | 'mean' | 'sum'. Default: 'mean'
        """
        super(ClippedSurrogateLoss, self).__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, log_probs, old_log_probs, policy_gradient_term):
        """
        Compute the clipped surrogate loss.

        Args:
            log_probs (Tensor): New log probabilities of the actions.
            old_log_probs (Tensor): Old log probabilities of the actions.
            policy_gradient_term (Tensor): Estimated advantages.

        Returns:
            loss (Tensor): Loss value.
        """
        # Compute importance sampling ratio
        ratio = torch.exp(log_probs - old_log_probs)
        # Compute clipped surrogate objective
        surrogate_objective = ratio * policy_gradient_term
        clipped_surrogate_objective = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * policy_gradient_term
        loss = -torch.min(surrogate_objective, clipped_surrogate_objective)
        loss = self._maybe_reduction(loss)

        clip_fraction = (torch.abs(ratio - 1) > self.epsilon).float()

        return loss, ratio, clip_fraction, surrogate_objective, clipped_surrogate_objective
