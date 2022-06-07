import torch as th
import torch.nn as nn

from gym import spaces

from wacky.networks.layered_networks import MultiLayerPerceptron
from wacky.networks.actor_critic_networks import ActorCriticNetwork
from wacky.functional.gym_space_decoder import decode_gym_space
from wacky.functional.distributions import make_distribution_network
from wacky.backend import WackyTypeError

from wacky import functional as funky
from wacky.networks.q_networks import DuellingQNetwork


def maybe_make_network(network, in_features=None, activation=None, *args, **kwargs):

    if network is None:
        network = [64, 64]

    elif isinstance(network, int):
        network = [network]

    if isinstance(in_features, spaces.Space):
        in_features = funky.decode_gym_space(in_features)[0]

    if isinstance(network, list):

        if in_features is None and len(network) >= 2:
            in_features = network.pop(0)
            print("Warning - Using first first element of network layer list as in_features:", in_features,
                  "\nReduced layer list:", network)

        elif in_features is None:
            raise ValueError("List 'network' has len 1. When defining network as list,\n"
                             "either define in_features or include in list as the first element.")

        if not isinstance(activation, list):
            activation = [activation] * len(network)

        network_module = MultiLayerPerceptron(in_features=in_features)
        for units, activ in zip(network, activation):
            network_module.append_layer(units, activ, *args, **kwargs)

    elif isinstance(network, nn.Module):
        network_module = network

    else:
        raise WackyTypeError(network, (int, list, nn.Module), parameter='network', optional=True)

    return network_module


def make_q_net(in_features, out_features, net=None, hidden_activ=th.nn.ReLU(), out_activ=None, *args, **kwargs):
    q_net_module = maybe_make_network(net, in_features, hidden_activ, *args, **kwargs)

    if isinstance(out_features, int):
        n_units = out_features
    elif isinstance(out_features, spaces.Space):
        n_units = funky.decode_gym_space(out_features, allowed_spaces=[spaces.Discrete])
    else:
        raise WackyTypeError(out_features, (int, spaces.Space), parameter='out_features', optional=False)

    q_net_module.append_layer(n_units, out_activ, module=nn.Linear)
    return q_net_module


def make_duelling_q_net(
        in_features,
        out_features,
        net=None,
        val_net=None,
        adv_net=None,
        hidden_activ=th.nn.ReLU(),
        out_activ=None,
        *args, **kwargs
):
    if val_net is None:
        val_net = []

    if adv_net is None:
        adv_net = []

    shared_module = maybe_make_network(net, in_features, hidden_activ, *args, **kwargs)
    value_module = maybe_make_network(val_net, shared_module.out_features, hidden_activ)
    adv_module = maybe_make_network(adv_net, shared_module.out_features, hidden_activ)
    adv_module.append_layer(1, out_activ)

    if isinstance(out_features, int):
        units = out_features
    elif isinstance(out_features, spaces.Space):
        units = funky.decode_gym_space(out_features, allowed_spaces=[spaces.Discrete])
    else:
        raise WackyTypeError(out_features, (int, spaces.Space), parameter='out_features', optional=False)

    value_module.append_layer(units, out_activ)

    return DuellingQNetwork(value_module, adv_module, shared_module)


def make_shared_net_for_actor_critic(observation_space, shared_net, activation_shared):
    if shared_net is None:
        shared_net_module = None
        in_features = int(decode_gym_space(observation_space, allowed_spaces=[spaces.Box])[0])

    elif isinstance(shared_net, list):
        shared_net_module = MultiLayerPerceptron(
            in_features=decode_gym_space(observation_space, allowed_spaces=[spaces.Box]),
            layer_units=shared_net,
            activation_hidden=activation_shared,
            activation_out=activation_shared
        )
        in_features = shared_net_module.out_features

    elif isinstance(shared_net, nn.Module):
        shared_net_module = shared_net
        in_features = shared_net_module.out_features

    else:
        raise TypeError("'shared_net' type must be either [None, list, nn.Module], not", type(shared_net))

    return in_features, shared_net_module


def make_actor_net(action_space, in_features, actor_net=None, activation_actor=th.nn.ReLU()):
    if actor_net is None:
        actor_net = [64, 64]

    elif isinstance(actor_net, int):
        actor_net = [actor_net]

    if isinstance(actor_net, list):
        actor_net_module = MultiLayerPerceptron(
            in_features=in_features,
            layer_units=actor_net,
            activation_hidden=activation_actor,
            activation_out=activation_actor
        )
    elif isinstance(actor_net, nn.Module):
        actor_net_module = actor_net
    else:
        raise WackyTypeError(actor_net, (list, nn.Module), parameter='actor_net', optional=True)

    action_layer = make_distribution_network(in_features=actor_net_module.out_features, space=action_space)
    actor_net_module.layers.append(action_layer)

    return actor_net_module


def make_critic_net(in_features, critic_net, activation_critic):
    if critic_net is None:
        critic_net = [64, 64]

    if isinstance(critic_net, list):
        critic_net_module = MultiLayerPerceptron(
            in_features=in_features,
            layer_units=critic_net,
            activation_hidden=activation_critic,
            activation_out=activation_critic
        )
    elif isinstance(critic_net, nn.Module):
        critic_net_module = critic_net
    else:
        raise WackyTypeError(critic_net, (list, nn.Module), parameter='critic_net', optional=True)

    critic_net_module.append_layer(1, activation=None)

    return critic_net_module


def actor_critic_net_arch(
        observation_space,
        action_space,
        shared_net=None,
        actor_net=None,
        critic_net=None,
        activation_shared=nn.ReLU(),
        activation_actor=nn.ReLU(),
        activation_critic=nn.ReLU(),
):
    in_features, shared_net_module = make_shared_net_for_actor_critic(observation_space, shared_net, activation_shared)
    actor_net_module = make_actor_net(action_space, in_features, actor_net, activation_actor)
    critic_net_module = make_critic_net(in_features, critic_net, activation_critic)

    return ActorCriticNetwork(actor_net_module, critic_net_module, shared_net_module)
