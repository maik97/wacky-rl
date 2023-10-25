import torch
import torch.nn as nn

from wacky.returns.episode_returns import compute_episode_returns


class REINFORCELoss(nn.Module):
    def __init__(self, discount_factor=0.99, reduction='none', normalize=True):
        """
        Initialize the REINFORCE Loss function.

        Args:
            discount_factor (float): The discount factor gamma.
            reduction (str, optional): Specifies the reduction to apply to the output:
                'none' | 'mean' | 'sum'. Default: 'mean'
        """
        super(REINFORCELoss, self).__init__()
        self.discount_factor = discount_factor
        self.reduction = reduction
        self.normalize = normalize


    def forward(self, log_probs, rewards, done_flags):
        """
        Compute the REINFORCE Loss.

        Args:
            log_probs (Tensor): A tensor of shape [..., num_steps] containing log probabilities of actions.
            rewards (Tensor): A tensor of shape [..., num_steps] containing rewards.
            dones (Tensor): A tensor of shape [..., num_steps] indicating if the episode ended at each time step.

        Returns:
            loss (Tensor): A tensor containing the loss value.
        """

        # Compute returns
        returns = compute_episode_returns(
            rewards=rewards,
            discount_factor=self.discount_factor,
            done_flags=done_flags,
            normalize=self.normalize
        )

        # Compute the REINFORCE loss
        loss = -log_probs * returns

        # Apply reduction
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:
            raise ValueError("Invalid reduction. Expected 'none', 'mean', or 'sum', got {}".format(self.reduction))
