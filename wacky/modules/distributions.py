from typing import Type, Union, Tuple, Dict, Any

from gym import spaces

import torch
from torch import nn, Tensor
from torch.distributions import Distribution, Normal, Categorical, Bernoulli

from wacky.modules import WackyModule
from wacky.modules import ModuleConstructor
from wacky.modules.router import ModuleListRouter, ModuleDictRouter


class ContinuousDistribution(WackyModule):
    """
    Class for creating a continuous distribution (e.g. for actions in reinforcement learning).

    This class provides a framework for generating continuous action distributions.
    It is typically used in reinforcement learning algorithms like Policy Gradients, SAC, or DDPG
    where the action space is continuous.

    Attributes:
        activation_mu (Union[nn.Module, Type[nn.Module], nn.Module, str, Dict[str, Any]]):
            Activation function for the mean. Can be:
            - An uninstantiated PyTorch class (e.g., nn.ReLU)
            - An instantiated PyTorch class (e.g., nn.ReLU())
            - A string name of the PyTorch layer (e.g., "ReLU")
            - A dictionary with keys "type" and "args" specifying the type and arguments for the activation function

        activation_sigma (Union[nn.Module, Type[nn.Module], nn.Module, str, Dict[str, Any]]):
            Activation function for the standard deviation. Can be in the same formats as activation_mu.

        distribution_type (Type[Distribution]): The type of distribution to use.
        min_sigma (float): The minimum value for the standard deviation.


    Methods:
        make_dist: Makes a distribution object based on mean and standard deviation.
        forward: Returns a sampled action and its log probability from the distribution.
        eval_action: Evaluates the log probability of a given action based on the distribution.

    Example Usage:
        >>> from torch.distributions import Normal
        >>> import torch.nn.functional as F

        >>> # Different ways to define activation functions
        >>> activation_mu1 = torch.nn.Tanh()
        >>> activation_mu2 = "Tanh"
        >>> activation_mu3 = {"type": "Tanh", "args": {}}
        >>> activation_mu4 = nn.Tanh

        >>> # Instantiate the class
        >>> dist1 = ContinuousDistribution(activation_mu1, None, Normal)
        >>> dist2 = ContinuousDistribution(activation_mu2, None, Normal)
        >>> dist3 = ContinuousDistribution(activation_mu3, None, Normal)
        >>> dist4 = ContinuousDistribution(activation_mu4, None, Normal)

        >>> # Sample mean and std tensors (These are usually obtained from a neural network)
        >>> mu = torch.Tensor([1.0])
        >>> sigma = torch.Tensor([0.5])

        >>> # Forward pass to get an action and its log probability
        >>> action, log_prob = dist1((mu, sigma))

        >>> # Evaluate the log probability of a specific action
        >>> log_prob = dist1.eval_action((mu, sigma), torch.Tensor([0.8]))
    """

    def __init__(
            self,
            activation_mu: Union[nn.Module, Type[nn.Module], str, Dict[str, Any]] = None,
            activation_sigma: Union[nn.Module, Type[nn.Module], str, Dict[str, Any]] = None,
            distribution_type: Type[Distribution] = Normal,
            min_sigma: float = 0.4,
    ) -> None:
        """
        Initialize the ContinuousDistribution.

        Args:
            activation_mu (Union[nn.Module, Type[nn.Module], nn.Module, str, Dict[str, Any]]):
                Activation function for the mean. Can be:
                - An uninstantiated PyTorch class (e.g., nn.ReLU)
                - An instantiated PyTorch class (e.g., nn.ReLU())
                - A string name of the PyTorch layer (e.g., "ReLU")
                - A dictionary with keys "type" and "args" specifying the type and arguments for the activation function

            activation_sigma (Union[nn.Module, Type[nn.Module], nn.Module, str, Dict[str, Any]]):
                Activation function for the standard deviation. Can be in the same formats as activation_mu.

            distribution_type (Type[Distribution]): The type of distribution to use.
            min_sigma (float): The minimum value for the standard deviation.

        """
        super(ContinuousDistribution, self).__init__()

        self.activation_mu = ModuleConstructor.construct(activation_mu)
        self.activation_sigma = ModuleConstructor.construct(activation_sigma)

        self.distribution_type = distribution_type
        self.min_sigma = min_sigma

    def make_dist(self, x: Tuple[Tensor, Tensor], *args: Any, **kwargs: Any) -> Distribution:
        """
        Create a distribution based on mean and standard deviation.

        Args:
            x: Tuple of tensor for mean and tensor for standard deviation.
            args: Additional positional arguments.
            kwargs: Additional keyword arguments.

        Returns:
            Distribution: A PyTorch distribution object.
        """
        mu, sigma = x

        if self.activation_mu:
            mu = self.activation_mu(mu)

        if self.activation_sigma:
            sigma = self.activation_sigma(sigma)

        if self.min_sigma:
            sigma = torch.clamp(sigma, min=self.min_sigma)
        return self.distribution_type(mu, sigma, *args, **kwargs)

    def forward(self, x: Tuple[Tensor, Tensor], deterministic: bool = False, *args: Any, **kwargs: Any) -> Tuple[
        Tensor, Tensor]:
        """
        Sample an action and compute its log probability.

        Args:
            x: Tuple of tensor for mean and tensor for standard deviation.
            deterministic: Whether to use deterministic sampling.
            args: Additional positional arguments.
            kwargs: Additional keyword arguments.

        Returns:
            tuple: A tuple containing the action and its log probability.
        """
        distribution = self.make_dist(x, *args, **kwargs)
        action = distribution.rsample() if not deterministic else x[0]
        log_prob = distribution.log_prob(action)
        return action, log_prob

    def eval_action(self,  x: Tuple[Tensor, Tensor], action: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        """
        Evaluate the log probability of a given action.

        Args:
            x: Tuple of tensor for mean and tensor for standard deviation.
            action: The action to evaluate.
            args: Additional positional arguments.
            kwargs: Additional keyword arguments.

        Returns:
            Tensor: Log probability of the action.
        """
        distribution = self.make_dist(x, *args, **kwargs)
        log_prob = distribution.log_prob(action)
        entropy = distribution.entropy()
        return log_prob, entropy

    def __repr__(self):
        return super().__repr__()[:-1] + f'  (distribution_type): {self.distribution_type.__name__}'


class DiscreteDistribution(WackyModule):
    """
    Class for creating a discrete distribution (e.g. for actions in reinforcement learning).

    This class facilitates the creation of discrete distributions for action selection
    in reinforcement learning tasks. Such distributions are useful in policy gradient
    methods that operate on environments with a finite, discrete action space.
    The class is highly customizable, allowing for different types of distributions and
    activation functions for logits.

    Attributes:
        activation (Union[nn.Module, Type[nn.Module], str, Dict[str, Any]]):
            Activation function for the logits. Can be:
            - An instantiated PyTorch class (e.g., nn.Softmax(dim=1))
            - An uninstantiated PyTorch class (e.g., nn.Softmax)
            - A string name of the PyTorch layer (e.g., "Softmax")
            - A dictionary with keys "type" and "args" specifying the type and arguments for the activation function

        distribution_type (Type[Distribution]):
            The type of discrete distribution to use. Defaults to Categorical.

    Methods:
        make_dist: Constructs a distribution object based on the provided logits.
        forward: Samples an action from the distribution, returns it and its log probability.
        eval_action: Evaluates the log probability of a given action based on the distribution.

    Example Usage:
        >>> from torch.distributions import Categorical
        >>> import torch.nn.functional as F

        >>> # Different ways to define activation functions
        >>> activation1 = torch.nn.Softmax(dim=1)
        >>> activation2 = "Softmax"
        >>> activation3 = {"type": "Softmax", "args": {"dim": 1}}
        >>> activation4 = nn.Softmax

        >>> # Instantiate the class
        >>> dist1 = DiscreteDistribution(activation1, Categorical)
        >>> dist2 = DiscreteDistribution(activation2, Categorical)
        >>> dist3 = DiscreteDistribution(activation3, Categorical)
        >>> dist4 = DiscreteDistribution(activation4, Categorical)

        >>> # Sample logits tensor (usually obtained from a neural network)
        >>> logits = torch.Tensor([[0.2, 0.8]])

        >>> # Forward pass to get an action and its log probability
        >>> action, log_prob = dist1(logits)

        >>> # Evaluate the log probability of a specific action
        >>> log_prob_eval = dist1.eval_action(logits, torch.Tensor([1]))
    """

    def __init__(
            self,
            activation: Union[nn.Module, Type[nn.Module], str, Dict[str, Any]] = nn.Softmax,
            distribution_type: Type[Distribution] = Categorical,
    ) -> None:
        """
        Initialize the DiscreteDistribution.

        Args:
            activation (Union[nn.Module, Type[nn.Module], str, Dict[str, Any]]):
                Activation function for the logits. Can be:
                - An instantiated PyTorch class (e.g., nn.Softmax(dim=1))
                - An uninstantiated PyTorch class (e.g., nn.Softmax)
                - A string name of the PyTorch layer (e.g., "Softmax")
                - A dictionary with keys "type" and "args" specifying the type and arguments for the activation function

            distribution_type (Type[Distribution]):
                The type of discrete distribution to use. Defaults to Categorical.
        """
        super(DiscreteDistribution, self).__init__()

        self.activation = None #ModuleConstructor.construct(activation)
        self.distribution_type = distribution_type

    def make_dist(self, x: Tensor, *args, **kwargs) -> Distribution:
        """
        Constructs a distribution object based on the provided logits.

        Args:
            x (Tensor): The logits for the actions.
            *args: Additional positional arguments for the distribution.
            **kwargs: Additional keyword arguments for the distribution.

        Returns:
            Distribution: A PyTorch Distribution object.
        """
        if self.activation:
            x = self.activation(x)
        return self.distribution_type(logits=x, *args, **kwargs)

    def forward(self, x: Tensor, deterministic: bool = False, *args, **kwargs) -> Tuple[Tensor, Tensor]:
        """
        Sample an action from the distribution and return it along with its log probability.

        Args:
            x (Tensor): The logits for the actions.
            deterministic (bool): Whether to sample deterministically.
                If True, returns the action with the highest probability. Defaults to False.
            *args: Additional positional arguments for the distribution.
            **kwargs: Additional keyword arguments for the distribution.

        Returns:
            Tuple[Tensor, Tensor]: A tuple containing the sampled action and its log probability.
        """
        distribution = self.make_dist(x, *args, **kwargs)
        action = distribution.sample() if not deterministic else torch.argmax(x, dim=-1)
        log_prob = distribution.log_prob(action)
        return action, log_prob

    def eval_action(self, x: Tensor, action: Tensor, *args, **kwargs) -> Tensor:
        """
        Evaluate the log probability of a given action based on the distribution.

        Args:
            x (Tensor): The logits for the actions.
            action (Tensor): The action to evaluate.
            *args: Additional positional arguments for the distribution.
            **kwargs: Additional keyword arguments for the distribution.

        Returns:
            Tensor: The log probability of the action.
        """
        distribution = self.make_dist(x, *args, **kwargs)
        log_prob = distribution.log_prob(action)
        entropy = distribution.entropy()
        return log_prob, entropy

    def __repr__(self):
        return super().__repr__()[:-1] + f'  (distribution_type): {self.distribution_type.__name__}'


class BinaryDistribution(WackyModule):
    """
    Class for creating a binary or multi-binary distribution for actions in reinforcement learning.

    This class facilitates the creation of binary or multi-binary distributions
    for action selection in policy gradient methods. It's particularly useful for
    problems with a binary or multiple independent binary action spaces.

    Attributes:
        activation (Union[nn.Module, Type[nn.Module], str, Dict[str, Any]]):
            Activation function for the logits. Can be:
            - An instantiated PyTorch class (e.g., nn.Sigmoid())
            - An uninstantiated PyTorch class (e.g., nn.Sigmoid)
            - A string name of the PyTorch layer (e.g., "Sigmoid")
            - A dictionary with keys "type" and "args" specifying the type and arguments for the activation function

        distribution_type (Type[Distribution]):
            The type of binary distribution to use. Defaults to Bernoulli.

        independent_variables (bool):
            Sums up the log propabilities of the output if set to true.

    Methods:
        make_dist: Constructs a distribution object based on the provided logits or probabilities.
        forward: Samples an action from the distribution, returns it and its log probability.
        eval_action: Evaluates the log probability of a given action based on the distribution.

    Example Usage:
        >>> from torch.distributions import Bernoulli

        >>> # Instantiate the class with default activation (Sigmoid) and default distribution (Bernoulli)
        >>> dist = BinaryDistribution(activation='sigmoid', distribution_type=Bernoulli)

        >>> # Sample logits tensor (usually obtained from a neural network)
        >>> logits = torch.Tensor([[0.2, -0.5]])

        >>> # Forward pass to get an action and its log probability
        >>> action, log_prob = dist(logits)
    """
    def __init__(
            self,
            activation: Union[nn.Module, Type[nn.Module], str, Dict[str, Any]] = nn.Sigmoid,
            distribution_type: Type[Distribution] = Bernoulli,
            independent_variables: bool = True,
    ) -> None:
        """
        Initialize the BinaryDistribution.

        Args:
            activation (Union[nn.Module, Type[nn.Module], str, Dict[str, Any]]):
                Activation function for the logits. Can be:
                - An instantiated PyTorch class (e.g., nn.Sigmoid())
                - An uninstantiated PyTorch class (e.g., nn.Sigmoid)
                - A string name of the PyTorch layer (e.g., "Sigmoid")
                - A dictionary with keys "type" and "args" specifying the type and arguments for the activation function

            distribution_type (Type[Distribution]):
                The type of binary distribution to use. Defaults to Bernoulli.

            independent_variables (bool):
                Sums up the log propabilities of the output if set to true.
        """
        super(BinaryDistribution, self).__init__()
        self.activation = ModuleConstructor.construct(activation)
        self.distribution_type = distribution_type
        self.independent_variables = independent_variables

    def make_dist(self, x: Tensor, *args, **kwargs) -> Distribution:
        """
        Constructs a distribution object based on the provided logits or probabilities.

        Args:
            x (Tensor): The logits or probabilities for the actions.
            *args, **kwargs: Additional arguments for the distribution constructor.

        Returns:
            Distribution: A PyTorch Distribution object.
        """
        if self.activation:
            x = self.activation(x)
        return self.distribution_type(probs=x, *args, **kwargs)

    def forward(self, x: Tensor, deterministic: bool = False, *args, **kwargs) -> Tuple[Tensor, Tensor]:
        """
        Sample an action from the distribution and return it along with its log probability.

        Args:
            x (Tensor): The logits or probabilities for the actions.
            deterministic (bool): If True, returns the action with the highest probability.
            *args, **kwargs: Additional arguments for the distribution constructor.

        Returns:
            Tuple[Tensor, Tensor]: A tuple containing the sampled action and its log probability.
        """
        distribution = self.make_dist(x, *args, **kwargs)
        if deterministic:
            action = (x >= 0.5).float()
        else:
            action = distribution.sample()
        log_prob = distribution.log_prob(action)
        if self.independent_variables:
            log_prob = log_prob.sum(dim=-1)  # Summing log_prob as we are treating each binary var as independent
        return action, log_prob

    def eval_action(self, x: Tensor, action: Tensor, *args, **kwargs) -> Tensor:
        """
        Evaluate the log probability of a given action based on the distribution.

        Args:
            x (Tensor): The logits or probabilities for the actions.
            action (Tensor): The action to evaluate.
            *args, **kwargs: Additional arguments for the distribution constructor.

        Returns:
            Tensor: The log probability of the action.
        """
        distribution = self.make_dist(x, *args, **kwargs)
        log_prob = distribution.log_prob(action)
        if self.independent_variables:
            log_prob = log_prob.sum(dim=-1)  # Summing log_prob as we are treating each binary var as independent
        return log_prob, None

    def __repr__(self):
        return super().__repr__()[:-1] + f'  (distribution_type): {self.distribution_type.__name__}'


class DistributionConstructor:

    @staticmethod
    def get_total_output_dimension(space):
        if isinstance(space, spaces.Discrete):
            return space.n

        elif isinstance(space, spaces.Box):
            return space.shape[0] * 2

        elif isinstance(space, spaces.MultiBinary):
            return space.n

        elif isinstance(space, spaces.MultiDiscrete):
            return sum(space.nvec)

        elif isinstance(space, spaces.Tuple):
            return sum(
                DistributionConstructor.get_total_output_dimension(s)
                for s in space.spaces
            )

        elif isinstance(space, spaces.Dict):
            return sum(
                DistributionConstructor.get_total_output_dimension(s)
                for s in space.spaces.values()
            )

        else:
            raise ValueError(f"Unsupported action space type: {type(space)}")

    @staticmethod
    def construct_layer(space, module, module_kwargs=None):
        out_features = DistributionConstructor.get_total_output_dimension(space)
        if module_kwargs is None:
            module_kwargs = {}
        module_kwargs['out_features'] = out_features
        if not isinstance(module, dict):
            module = {'type': module}
        if 'kwargs' not in module:
            module['kwargs'] = {}
        module['kwargs'].update(module_kwargs)
        return ModuleConstructor.construct(module)

    @staticmethod
    def construct_distribution(
            space,
            continuous_dist_kwargs=None,
            discrete_dist_kwargs=None,
            binary_dist_kwargs=None,
    ):

        continuous_dist_kwargs = continuous_dist_kwargs if continuous_dist_kwargs is not None else {}
        discrete_dist_kwargs = discrete_dist_kwargs if discrete_dist_kwargs is not None else {}
        binary_dist_kwargs = binary_dist_kwargs if binary_dist_kwargs is not None else {}

        if isinstance(space, spaces.Discrete):
            return DiscreteDistribution(**discrete_dist_kwargs)

        elif isinstance(space, spaces.Box):
            return ContinuousDistribution(**continuous_dist_kwargs)

        elif isinstance(space, spaces.MultiBinary):
            return BinaryDistribution(**binary_dist_kwargs)

        elif isinstance(space, spaces.MultiDiscrete):
            return ModuleListRouter([DiscreteDistribution() for _ in space.nvec])

        elif isinstance(space, spaces.Tuple):
            layers = [DistributionConstructor.construct_distribution(s) for s in space.spaces]
            return ModuleListRouter(layers)

        elif isinstance(space, spaces.Dict):
            layers = {key: DistributionConstructor.construct_distribution(s) for key, s in space.spaces.items()}
            return ModuleDictRouter(layers)

        else:
            raise ValueError(f"Unsupported action space type: {type(space)}")


class DistributionLayer(WackyModule):

    def __init__(
            self,
            space,
            module=nn.LazyLinear,
            module_kwargs=None,
            continuous_dist_kwargs=None,
            discrete_dist_kwargs=None,
            binary_dist_kwargs=None,
    ):
        super(DistributionLayer, self).__init__()

        self._layer = DistributionConstructor.construct_layer(
            space=space,
            module=module,
            module_kwargs=module_kwargs,
        )
        self.distribution = DistributionConstructor.construct_distribution(
            space=space,
            continuous_dist_kwargs=continuous_dist_kwargs,
            discrete_dist_kwargs=discrete_dist_kwargs,
            binary_dist_kwargs=binary_dist_kwargs,
        )

        self.total_out_features = DistributionConstructor.get_total_output_dimension(space)

        self.space = space

    def layer(self, x):
        x = self._layer(x)
        return self.split_and_route_output(x, self.space)

    def forward(self, x: Tensor, deterministic: bool = False, *args, **kwargs) -> Tuple[Tensor, Tensor]:
        return self.distribution(self.layer(x), deterministic=deterministic, *args, **kwargs)

    def split_and_route_output(self, output, space):
        """
        Splits the output tensor according to space and returns a data structure that
        mirrors the structure of action_space but contains slices of output.
        """

        # TODO: Implement precomputed slicers.

        last_dim = -1  # The last dimension is our target for splitting

        if isinstance(space, (spaces.Discrete, spaces.MultiBinary)):
            return output  # No splitting needed

        elif isinstance(space, spaces.Box):
            dim = space.shape[0]
            return torch.split(output, [dim, dim], dim=last_dim)  # Split into mu and sigma

        elif isinstance(space, spaces.MultiDiscrete):
            splits = space.nvec.tolist()
            return torch.split(output, splits, dim=last_dim)

        elif isinstance(space, spaces.Tuple):
            splits = [DistributionConstructor.get_total_output_dimension(s) for s in space.spaces]
            output_slices = torch.split(output, splits, dim=last_dim)
            return tuple(self.split_and_route_output(slice, s) for slice, s in zip(output_slices, space.spaces))

        elif isinstance(space, spaces.Dict):
            output_slices = {}
            start_idx = 0
            end_idx = 0
            for key, s in space.spaces.items():
                end_idx += DistributionConstructor.get_total_output_dimension(s)
                output_slices[key] = output[..., start_idx:end_idx]
                start_idx = end_idx
            return {key: self.split_and_route_output(slice, s) for key, slice, s in zip(output_slices.keys(), output_slices.values(), space.spaces.values())}

        else:
            raise ValueError(f"Unsupported space type: {type(space)}")
