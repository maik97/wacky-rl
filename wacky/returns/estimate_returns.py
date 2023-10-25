import torch


def n_step_returns(
        future_rewards,
        discount_factor,
        value_estimates=None,
        sequence_flag=True,
):
    """
    Compute the estimated returns based on future rewards and value estimates,
    with optional n-step.

    Args:
    - future_rewards (Tensor): A tensor of shape [..., num_samples, n] containing future rewards
                               starting from R_{t+1} up to R_{t+n+1}.
    - discount_factor (float): The discount factor (γ) to use for future rewards.
    - value_estimates (Tensor, optional): A tensor of shape [..., num_samples] containing the estimated state
                                          values (V(S_{t+n+1})) for each sample. If not provided, it won't be
                                          considered in the n-step return calculation.
    - sequence_flag (bool): True if future_rewards and value_estimates are sequences, False otherwise.

    Returns:
    - n_step_returns (Tensor): A tensor of shape [..., num_samples] containing the computed estimated returns.

    Example Usage:
    >>> future_rewards = torch.tensor([[[0.5, 0.6], [0.5, 0.5]], [[0.4, 0.5], [0.5, 0.5]]])
    >>> discount_factor = 0.9
    >>> value_estimates = torch.tensor([[0.5, 0.6], [0.5, 0.5]])
    >>> compute_returns(future_rewards, discount_factor, value_estimates)

    Notes:
        The function allows for future_rewards and value_estimates to be sequences (n-step) or
        single values (TD-Learning). The sequence_flag controls this behavior. The tensors future_rewards and
        value_estimates must match shapes, except in their last dim when using sequence_flag.
    """

    # Initialize returns tensor with the same shape
    if sequence_flag:
        if future_rewards.dim() < 2 or value_estimates.dim() < 2:
            raise ValueError(
                "Both future_rewards and value_estimates must have at least 2 dimensions when sequence_flag is True."
            )
        if future_rewards.shape[:-1] != value_estimates.shape[:-1]:
            raise ValueError(
                "The shape of future_rewards must match with value_estimates except for the last "
                "dimension when sequence_flag is True."
            )
        n_step_returns = torch.zeros(future_rewards.shape[:-1])

    else:
        if future_rewards.shape != value_estimates.shape:
            raise ValueError(
                "The shape of future_rewards must match with value_estimates when sequence_flag is False."
            )
        n_step_returns = torch.zeros_like(future_rewards)

    if future_rewards.shape[-1] > 1 and sequence_flag:
        # Assuming is a sequence of shape [..., n]
        n_steps = future_rewards.shape[-1]
        discount_vector = (torch.ones(n_steps) * discount_factor) ** torch.arange(0, n_steps)
        # Perform dot product along the last dimension
        discounted_rewards = torch.tensordot(future_rewards, discount_vector, dims=([-1], [0]))
        # Add to n_step_returns
        n_step_returns += discounted_rewards
    else:
        # No sequence, assuming is shape [..., 1, num_samples]
        n_step_returns += future_rewards
        n_steps = 0

    if value_estimates is not None:

        if value_estimates.shape[-1] > 1 and sequence_flag:
            # Assuming is a sequence of shape [..., n]
            next_n_steps = value_estimates.shape[-1]
            discount_vector = (
                    (torch.ones(next_n_steps) * discount_factor)
                    ** (torch.arange(n_steps + 1, n_steps + next_n_steps + 1))
            )
            # Perform dot product along the last dimension
            discounted_values = torch.tensordot(value_estimates, discount_vector, dims=([-1], [0]))
            # Add to n_step_returns
            n_step_returns += discounted_values
        else:
            n_step_returns += (discount_factor ** n_steps) * value_estimates

    return n_step_returns


if __name__ == '__main__':
    # Example usage:
    future_rewards = torch.tensor([[[0.5, 0.6], [0.5, 0.5]], [[0.4, 0.5], [0.5, 0.5]]])
    discount_factor = 0.9
    value_estimates = torch.tensor([[[0.5, 0.6, 0.3], [0.5, 0.5, 0.2]], [[0.4, 0.5, 0.1], [0.5, 0.5, 0.34]]])
    estimated_returns = n_step_returns(future_rewards, discount_factor, value_estimates)
    print(estimated_returns)  # We expect two batches, each with two samples
