from typing import Type, Callable, Union, Optional, Tuple

import torch
from torch import nn
from torch.distributions import Distribution, Normal, Categorical
from torch import Tensor

from wacky.modules import WackyModule
from wacky.activations import ActivationConstructor
from wacky.modules._layer_constructor import LayerConstructor


class CategoricalAtoms(nn.Module):
    """
    Class for creating a distribution using categorical atoms for reinforcement learning actions.

    Args:
    - in_features (int): The number of input features.
    - action_n (int): The number of possible actions.
    - activation (Union[nn.Module, str, dict], optional):
        Activation function for the logits layer. Can be a PyTorch class, a string,
        or a dictionary specifying type and arguments.
    - atom_size (int, optional): The size of the atom for value approximation. Defaults to 51.
    - layer (nn.Module, optional): The layer type for the network. Defaults to nn.Linear.
    - v_min (float, optional): The minimum value of support for the value approximation. Defaults to -10.0.
    - v_max (float, optional): The maximum value of support for the value approximation. Defaults to 10.0.
    - init_method (callable, optional): Initialization method for layer weights.

    Example usage:
        CategoricalAtoms(64, 3, activation="ReLU", atom_size=51, v_min=-10, v_max=10)

    This class is designed to facilitate the creation of distributions over discrete action values
    using 'atoms'. An atom is essentially a discrete representation of a continuous value.
    The use of atoms allows us to discretize a continuous distribution into a finite number of
    categories or 'atoms'.

    While the class can be a component in algorithms like [C51 Categorical DQN](https://arxiv.org/abs/1707.06887),
    it is general enough to be used wherever a discretized representation of continuous values is required.
    By approximating a continuous distribution with atoms, this class allows for a more nuanced understanding
    of environments where the returns are not single-valued but form a distribution.
    """

    def __init__(
            self,
            in_features: int,
            action_n: int,
            activation: Union[nn.Module, str, dict, None] = None,
            atom_size: int = 51,
            layer: Type[nn.Module] = nn.Linear,
            v_min: float = -10.0,
            v_max: float = 10.0,
            init_method: Optional[Callable] = None,
            x_clamp_min: float = 1e-3,
            *args, **kwargs
    ) -> None:
        super(CategoricalAtoms, self).__init__()

        self.atom_size = atom_size
        self.action_n = action_n
        self.in_features = in_features
        self.x_clamp_min = x_clamp_min
        self.layer = layer(in_features, (action_n * atom_size), *args, **kwargs)
        self.activation = ActivationConstructor.construct(activation)

        if init_method:
            init_method(self.layer.weight)

        self.support = torch.linspace(v_min, v_max, self.atom_size)

    def latent(self, x: Tensor) -> Tensor:
        """
        Compute and return the logits.

        Args:
        - x (Tensor): The input tensor.

        Returns:
        - Tensor: The logits tensor.
        """
        x = self.layer(x)
        if self.activation:
            x = self.activation(x)
        return x

    def make_dist(self, x: Tensor) -> Tensor:
        """
        Create and return the distribution.

        Args:
        - x (Tensor): The logits tensor.

        Returns:
        - Tensor: The distribution tensor.
        """
        x = x.view(-1, self.action_n, self.atom_size)
        return x.clamp(min=self.x_clamp_min)

    def forward(self, x: Tensor) -> Tensor:
        """
        Compute and return the weighted sum of the support and distribution.

        Args:
        - x (Tensor): The input tensor.

        Returns:
        - Tensor: The weighted sum tensor.
        """
        x = self.latent(x)
        distribution = self.make_dist(x)
        return torch.sum(distribution * self.support, dim=2)
