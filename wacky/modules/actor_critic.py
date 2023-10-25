from typing import Type, Union, Tuple, Dict, Any
from wacky.modules import DistributionLayer, WackyModule, ModuleConstructor
from torch import nn, Tensor


class Actor(WackyModule):

    def __init__(
            self,
            action_space,
            module=None,
            module_kwargs=None,
            action_layer_kwargs=None,
    ):
        super(Actor, self).__init__()

        self.network = ModuleConstructor.construct(module, module_kwargs)

        action_layer_kwargs = action_layer_kwargs if action_layer_kwargs is not None else {}
        self.action_layer = DistributionLayer(
            space=action_space,
            **action_layer_kwargs
        )

    def forward(self, x, deterministic: bool = False, *args, **kwargs):
        if self.network is not None:
            x = self.network(x)
        return self.action_layer(x, deterministic=deterministic, *args, **kwargs)

    def eval_action(self, x, action):
        if self.network is not None:
            x = self.network(x)
        x = self.action_layer.layer(x)
        return self.action_layer.distribution.eval_action(x, action)


class Critic(WackyModule):

    def __init__(
            self,
            critic_layer=None,
            critic_layer_kwargs=None,
            module=None,
            module_kwargs=None,
    ):
        super(Critic, self).__init__()

        self.network = ModuleConstructor.construct(module, module_kwargs)
        self.critic_layer = ModuleConstructor.construct(critic_layer, critic_layer_kwargs)

    def forward(self, x):
        if self.network is not None:
            x = self.network(x)
        return self.critic_layer(x)


class ActorCritic(WackyModule):

    def __init__(
            self,
            actor,
            critic,
            shared_module=None,
            shared_module_kwargs=None,
    ):
        super(ActorCritic, self).__init__()

        self.shared_network = ModuleConstructor.construct(shared_module, shared_module_kwargs)
        self.actor = actor
        self.critic = critic

    def _maybe_network(self, x):
        if self.shared_network is not None:
            x = self.shared_network(x)
        return x

    def forward(self, x, deterministic: bool = False, *args, **kwargs):
        x = self._maybe_network(x)
        return self.actor(x, deterministic=deterministic, *args, **kwargs), self.critic(x)

    def actor_forward(self, x, deterministic: bool = False, *args, **kwargs):
        x = self._maybe_network(x)
        return self.actor(x, deterministic=deterministic, *args, **kwargs)

    def critic_forward(self, x):
        x = self._maybe_network(x)
        return self.critic(x)

    def eval_action(self, x, action):
        x = self._maybe_network(x)
        return self.actor.eval_action(x, action), self.critic(x)


if __name__ == '__main__':

    import torch

    import gym

    # Create a Gym action space
    action_space = gym.spaces.Discrete(3)

    # Create an Actor object
    actor = Actor(action_space, module=[5, nn.ReLU, 10, 'Tanh'])
    print(actor)

    # Generate some fake data to feed into the model, assuming input features are of size 5
    x = torch.randn(2, 5)

    # Test forward method
    forward_output = actor(x)
    print(f'Forward output: {forward_output}')

    # Test eval_action method
    # Sample some actions from the Gym action space
    actions = torch.tensor([action_space.sample() for _ in range(2)], dtype=torch.long)
    eval_output = actor.eval_action(x, actions)
    print(f'Eval output: {eval_output}')

    print(actor)
