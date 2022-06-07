from torch import optim
from torch import nn

from wacky.backend import WackyValueError
from wacky.networks import OffPolicyNetworkWrapper


def maybe_get_network_params(maybe_is_network):
    '''
    Calls parameter() method if argument is network.

    :param maybe_is_network: Either nn.Module or module parameter iterator.
    :return: Iterator over module parameters
    '''
    if isinstance(maybe_is_network, OffPolicyNetworkWrapper):
        return maybe_is_network.behavior.parameters()
    elif isinstance(maybe_is_network, nn.Module):
        return maybe_is_network.parameters()
    else:
        return maybe_is_network


def get_optim(optimizer, params, lr, *args, **kwargs):
    """
    Creates torch optimizer.

    :param optimizer:
    :param params:
    :param lr:
    :param args:
    :param kwargs:
    :return:
    """

    if isinstance(optimizer, optim.Optimizer):
        return optimizer(params, lr, *args, **kwargs)

    if optimizer == 'Adadelta':
        return optim.Adadelta(params, lr, *args, **kwargs)
    elif optimizer == 'Adagrad':
        return optim.Adagrad(params, lr, *args, **kwargs)
    elif optimizer == 'Adam':
        return optim.Adam(params, lr, *args, **kwargs)
    elif optimizer == 'AdamW':
        return optim.AdamW(params, lr, *args, **kwargs)
    elif optimizer == 'SparseAdam':
        return optim.SparseAdam(params, lr, *args, **kwargs)
    elif optimizer == 'Adamax':
        return optim.Adamax(params, lr, *args, **kwargs)
    elif optimizer == 'ASGD':
        return optim.ASGD(params, lr, *args, **kwargs)
    elif optimizer == 'LBFGS':
        return optim.LBFGS(params, lr, *args, **kwargs)
    elif optimizer == 'NAdam':
        return optim.NAdam(params, lr, *args, **kwargs)
    elif optimizer == 'RAdam':
        return optim.RAdam(params, lr, *args, **kwargs)
    elif optimizer == 'RMSprop':
        return optim.RMSprop(params, lr, *args, **kwargs)
    elif optimizer == 'Rprop':
        return optim.Rprop(params, lr, *args, **kwargs)
    elif optimizer == 'SGD':
        return optim.SGD(params, lr, *args, **kwargs)
    else:
        raise WackyValueError(
            optimizer,
            ('Adadelta', 'Adagrad', 'Adam', 'AdamW', 'SparseAdam', 'Adamax',
             'ASGD', 'LBFGS', 'NAdam', 'RAdam', 'RMSprop', 'Rprop', 'SGD'),
            parameter='name',
            optional=False
        )
