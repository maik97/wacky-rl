import torch
import torch.nn as nn

from wacky.returns.estimate_returns import n_step_returns


def td_error(rewards, values, next_values, dones, discount_factor):
    """
    Calculate the Temporal Difference (TD) error.
    Args:
        rewards (Tensor): A tensor of shape [..., num_steps] containing rewards.
        values (Tensor): A tensor of shape [..., num_steps] containing value estimates.
        dones (Tensor): A tensor of shape [..., num_steps] indicating if the episode ended at each time step.
        discount_factor (float): The discount factor gamma.
        next_values (Tensor, optional): A tensor of shape [..., num_steps] containing value estimates for next states.
    Returns:
        td_errors (Tensor): A tensor of shape [..., num_steps] containing TD errors.
    """
    if rewards.shape != values.shape or rewards.shape != dones.shape or rewards.shape != next_values.shape:
        raise ValueError("The shape of rewards, values, and dones must match.")

    # Compute returns using the n_step_returns function, but essentially with 1 step, hence sequence_flag is False
    returns = n_step_returns(rewards, discount_factor, value_estimates=next_values, sequence_flag=False)
    # Calculate TD error
    td_errors = returns - values
    return td_errors


class TDErrorLoss(nn.Module):
    def __init__(self, discount_factor, reduction='none', error_type='squared'):
        """
        Initialize the TD Error Loss function.

        Args:
            discount_factor (float): The discount factor gamma.
            reduction (str, optional): Specifies the reduction to apply to the output:
                'none' | 'mean' | 'sum'. Default: 'mean'
            error_type (str, optional): Specifies the type of error to compute:
                'squared' | 'abs'. Default: 'squared'
        """
        super(TDErrorLoss, self).__init__()
        self.discount_factor = discount_factor
        self.reduction = reduction
        self.error_type = error_type

    def forward(self, rewards, values, dones, next_values=None):
        """
        Compute the TD Error Loss.

        Args:
            rewards (Tensor): A tensor of shape [..., num_steps] containing rewards.
            values (Tensor): A tensor of shape [..., num_steps] containing value estimates.
            dones (Tensor): A tensor of shape [..., num_steps] indicating if the episode ended at each time step.
            next_values (Tensor, optional): A tensor of shape [..., num_steps] containing value estimates for next states.

        Returns:
            loss (Tensor): A tensor containing the loss value.
        """

        # Compute TD errors
        td_errors = td_error(
            rewards=rewards,
            values=values,
            next_values=next_values,
            dones=dones,
            discount_factor=self.discount_factor
        )

        # Apply either absolute or squared error
        if self.error_type == 'squared':
            loss = td_errors ** 2
        elif self.error_type == 'abs':
            loss = torch.abs(td_errors)
        else:
            raise ValueError("Invalid error_type. Expected 'squared' or 'abs', got {}".format(self.error_type))

        # Apply reduction
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:
            raise ValueError("Invalid reduction. Expected 'none', 'mean', or 'sum', got {}".format(self.reduction))


if __name__ == "__main__":
    rewards = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
    values = torch.tensor([1.2, 0.8, 1.0], dtype=torch.float32)
    next_values = torch.tensor([0.9, 0.9, 0.0], dtype=torch.float32)
    dones = torch.tensor([0, 0, 1], dtype=torch.float32)  # Episode ends at the last step
    gamma = 0.99

    td_errors_result = td_error(rewards, values, next_values, dones, gamma)
    print("TD errors:", td_errors_result)

    td_loss_fn = TDErrorLoss(discount_factor=gamma, reduction='mean', error_type='squared')
    loss_value = td_loss_fn(rewards, values, dones, next_values)
    print(f"TD error loss (squared, mean reduction): {loss_value.item()}")

    td_loss_fn = TDErrorLoss(discount_factor=gamma, reduction='sum', error_type='abs')
    loss_value = td_loss_fn(rewards, values, dones, next_values)
    print(f"TD error loss (abs, sum reduction): {loss_value.item()}")
