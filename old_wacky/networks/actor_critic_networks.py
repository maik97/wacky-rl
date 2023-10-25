import torch as th
import torch.nn as nn
from wacky import backend
from wacky.networks import WackyNetwork
from wacky import functional as funky


class ActorNetwork(WackyNetwork):

    def __init__(self, action_space, network=None, in_features=None, hidden_activation=th.nn.ReLU()):
        super(ActorNetwork, self).__init__()

        if network is not None:
            self.network = funky.maybe_make_network(network, in_features, hidden_activation)
            self.action_layer = funky.make_distribution_network(
                in_features=self.network.out_features,
                space=action_space
            )
        else:
            self.network = network
            self.action_layer = funky.make_distribution_network(
                in_features=in_features,
                space=action_space
            )

    def latent(self, x):
        if self.network is not None:
            x = self.network(x)
        return x

    def forward(self, x):
        if self.network is not None:
            x = self.network(x)
        return self.action_layer(x)

    def eval_action(self, x, action):
        return self.action_layer.eval_action(x, action)

    def learn(self, *args, **kwargs):
        if self.network is not None:
            self.network.learn(*args, **kwargs)

    def reset(self):
        self.network.reset()


class ActorCriticNetwork(nn.Module):

    def __init__(self, actor_network, critic_network, shared_network=None):
        super(ActorCriticNetwork, self).__init__()

        backend.check_type(actor_network, nn.Module, 'actor_network')
        backend.check_type(critic_network, nn.Module, 'critic_network')
        backend.check_type(shared_network, nn.Module, 'shared_network', allow_none=True)

        self.shared_network = shared_network
        self.actor_network = actor_network
        self.critic_network = critic_network

    def forward(self, x):
        if self.shared_network is not None:
            x = self.shared_network(x)
        return self.actor_network(x), self.critic_network(x)

    def actor(self, x):
        if self.shared_network is not None:
            x = self.shared_network(x)
        return self.actor_network(x)

    def critic(self, x):
        if self.shared_network is not None:
            x = self.shared_network(x)
        return self.critic_network(x)

    def eval_action(self, x, action):
        """if len(self.actor_net_module.layers) > 1:
            for layer in self.actor_net_module.layers[:-2]:
                x = layer(x)
        log_prob = self.actor_net_module.layers[-1].eval_action(x, action)"""

        if self.shared_network is not None:
            x = self.shared_network(x)
        val = self.critic_network(x)

        latent_x = self.actor_network.latent(x)
        log_prob = self.actor_network.eval_action(latent_x, action)

        return log_prob, val

    def reset(self, *args, **kwargs):
        if self.shared_network is not None:
            self.shared_network.reset(*args, **kwargs)
        self.actor_network.reset(*args, **kwargs)
        self.critic_network.reset(*args, **kwargs)


class ActorCriticNetworkConstructor:

    def __init__(
            self,
            observation_space,
            action_space,
            shared_network=None,
            actor_network=None,
            critic_network=None,
            share_some=True,
    ):
        self.observation_space = observation_space
        self.in_features = observation_space
        self.action_space = action_space
        self.share_some = share_some

        self.shared_network = None if shared_network is None else self.custom_shared_network(shared_network)
        self.actor_network = None if actor_network is None else self.custom_shared_network(actor_network)
        self.critic_network = None if critic_network is None else self.custom_shared_network(critic_network)

    def custom_shared_network(self, network, hidden_activation=nn.ReLU()):
        self.shared_network = funky.maybe_make_network(network, self.observation_space, hidden_activation)
        self.in_features = self.shared_network.out_features

    def custom_actor_network(self, network, hidden_activation=nn.ReLU()):
        self.actor_network = ActorNetwork(
            action_space=self.action_space,
            in_features=self.in_features,
            network=network,
            hidden_activation=hidden_activation
        )

    def custom_critic_network(self, network, hidden_activation=nn.ReLU(), out_activation=None):
        self.critic_network = funky.maybe_make_network(network, self.in_features, hidden_activation)
        self.critic_network.append_layer(1, activation=out_activation)

    def build(self):
        if self.shared_network is None and self.share_some:
            self.custom_shared_network([64, 64])
        if self.actor_network is None:
            self.custom_actor_network([])
        if self.critic_network is None:
            self.custom_critic_network([])

        return ActorCriticNetwork(self.actor_network, self.critic_network, self.shared_network)
