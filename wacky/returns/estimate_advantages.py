import torch
from torch import nn


def compute_gae_v1(
        rewards,
        values,
        done_flags,
        discount_factor,
        smoothing_factor,
        normalize_advantage=True,
        norm_dim_advantage=None,
        normalize_returns=False,
        norm_dim_returns=None,
):
    """
    Compute the Generalized Advantage Estimation (GAE).
    Args:
        rewards (Tensor): A tensor of shape [..., num_steps] containing rewards.
        values (Tensor): A tensor of shape [..., num_steps] containing value estimates.
        done_flags (Tensor): A tensor of shape [..., num_steps] indicating if the episode ended at each time step.
        discount_factor (float): The discount factor.
        smoothing_factor (float): The smoothing factor for GAE.
    Returns:
        advantages (Tensor): A tensor of shape [..., num_steps] containing the computed GAE.
    """
    if rewards.shape != values.shape or rewards.shape != done_flags.shape:
        raise ValueError("The shape of rewards, values, and done_flags must match.")

    gae = 0
    advantages = torch.zeros_like(rewards)
    next_value = 0

    #returns = torch.zeros_like(rewards)
    #next_return = 0

    for t in reversed(range(rewards.shape[-1])):
        delta = rewards[..., t] + discount_factor * next_value * (1 - done_flags[..., t]) - values[..., t]
        gae = delta + discount_factor * smoothing_factor * (1 - done_flags[..., t]) * gae
        advantages[..., t] = gae
        next_value = values[..., t]
        #returns[..., t] = rewards[..., t] + discount_factor * next_return * (1 - done_flags[..., t])
        #next_return = returns[..., t]

    returns = advantages + values

    # Normalize if required
    if normalize_advantage:
        advantages = (advantages - advantages.mean(norm_dim_advantage)) / (advantages.std(norm_dim_advantage) + 1e-9)

    if normalize_returns:
        returns = (returns - returns.mean(norm_dim_returns)) / (returns.std(norm_dim_returns) + 1e-9)

    return advantages, returns

def compute_gae(
        rewards,
        values,
        done_flags,
        next_value,
        discount_factor,
        smoothing_factor,
        normalize_advantage=False,
        norm_dim_advantage=None,
        normalize_returns=False,
        norm_dim_returns=None,
):
    """
    Compute the Generalized Advantage Estimation (GAE).
    Args:
        rewards (Tensor): A tensor of shape [..., num_steps] containing rewards.
        values (Tensor): A tensor of shape [..., num_steps] containing value estimates.
        done_flags (Tensor): A tensor of shape [..., num_steps] indicating if the episode ended at each time step.
        discount_factor (float): The discount factor.
        smoothing_factor (float): The smoothing factor for GAE.
    Returns:
        advantages (Tensor): A tensor of shape [..., num_steps] containing the computed GAE.
    """
    if rewards.shape != values.shape or rewards.shape != done_flags.shape:
        raise ValueError("The shape of rewards, values, and done_flags must match.")

    gae = 0
    advantages = torch.zeros_like(rewards)

    for t in reversed(range(rewards.shape[-1])):
        if t == rewards.shape[-1] - 1:
            next_value = next_value.squeeze()
            next_non_terminal = 1.0 - done_flags[..., t]
        else:
            next_value = values[..., t + 1]
            next_non_terminal = 1.0 - done_flags[..., t + 1]
        delta = rewards[..., t] + discount_factor * next_value * next_non_terminal - values[..., t]
        gae = delta + discount_factor * smoothing_factor * next_non_terminal * gae
        advantages[..., t] = gae

    returns = advantages + values

    # Normalize if required
    if normalize_advantage:
        advantages = (advantages - advantages.mean(norm_dim_advantage)) / (advantages.std(norm_dim_advantage) + 1e-9)

    if normalize_returns:
        returns = (returns - returns.mean(norm_dim_returns)) / (returns.std(norm_dim_returns) + 1e-9)

    return advantages, returns


class GAE(nn.Module):

    def __init__(
            self,
            discount_factor=0.99,
            smoothing_factor=0.95,
            normalize_advantage=False,
            norm_dim_advantage=None,
            normalize_returns=False,
            norm_dim_returns=None,
    ):
        super(GAE, self).__init__()
        self.discount_factor = discount_factor
        self.smoothing_factor = smoothing_factor
        self.normalize_advantage = normalize_advantage
        self.norm_dim_advantage = norm_dim_advantage
        self.normalize_returns = normalize_returns
        self.norm_dim_returns = norm_dim_returns

    def forward(self, rewards, values, done_flags, next_value):
        return compute_gae(
            rewards=rewards,
            values=values,
            done_flags=done_flags,
            next_value=next_value,
            discount_factor=self.discount_factor,
            smoothing_factor=self.smoothing_factor,
            normalize_advantage=self.normalize_advantage,
            norm_dim_advantage=self.norm_dim_advantage,
            normalize_returns=self.normalize_returns,
            norm_dim_returns=self.norm_dim_returns,
        )


# Test your code
if __name__ == "__main__":
    rewards = torch.tensor([[1.0, 1.0, 1.0], [2.0, 0.0, 1.0]], dtype=torch.float32)
    values = torch.tensor([[1.2, 0.8, 1.0], [1.9, 0.2, 1.0]], dtype=torch.float32)
    dones = torch.tensor([[0, 0, 1], [0, 1, 1]], dtype=torch.float32)
    gamma = 0.99
    tau = 0.95

    advantages, returns = compute_gae(rewards, values, dones, gamma, tau)
    print("Computed advantages:", advantages)
    print("Computed returns:", returns)
