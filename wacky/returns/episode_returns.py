import torch
from torch import nn


def compute_episode_returns(rewards, discount_factor, done_flags, normalize=False):
    """
    Compute the Monte Carlo returns in PyTorch.

    Args:
        rewards (Tensor): A tensor of shape [..., episode_length] containing rewards.
        discount_factor (float or Tensor): Discount factor gamma. Can be a float or a tensor of shape [batch_size, episode_length].
        done_flags (Tensor): A tensor of shape [..., episode_length] containing done flags (0 or 1).
        normalize (bool): Whether to normalize the returns.

    Returns:
        returns (Tensor): A tensor of shape [..., episode_length] containing the calculated returns.
    """

    *_, episode_length = rewards.shape

    # Initialize returns tensor
    returns = torch.zeros_like(rewards)

    # Last step return is simply the last reward for each episode
    returns[..., -1] = rewards[..., -1]

    # Calculate the returns using dynamic programming
    for t in reversed(range(episode_length - 1)):
        returns[..., t] = rewards[..., t] + discount_factor * returns[..., t + 1] * (1 - done_flags[..., t])

    # Normalize the returns if required
    if normalize:
        returns = (returns - returns.mean()) / (returns.std(dim=-1) + 1e-9)

    return returns


class EpisodeReturns(nn.Module):

    def __init__(self, discount_factor=0.99, normalize=True):
        super(EpisodeReturns, self).__init__()
        self.discount_factor = discount_factor
        self.normalize = normalize

    def from_memory(self, memory, rewards='rewards', done_flags='done_flags'):
        return self.forward(
            rewards=memory[rewards],
            done_flags=memory[done_flags]
        )

    def forward(self, rewards, done_flags):
        return compute_episode_returns(
            rewards=rewards,
            discount_factor=self.discount_factor,
            done_flags=done_flags,
            normalize=self.normalize
        )


if __name__ == "__main__":

    def test_example():
        # Example usage
        rewards = torch.tensor([[[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]])
        discount_factor = 0.9
        done_flags = torch.tensor([[[0, 0, 1], [0, 0, 1]]], dtype=torch.float32)

        mc_return = compute_episode_returns(rewards, discount_factor, done_flags)
        print(mc_return)

    test_example()

