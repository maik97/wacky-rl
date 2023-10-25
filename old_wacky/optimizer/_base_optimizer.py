from torch import optim
from wacky import functional as funky


class WackyOptimizer(optim.Optimizer):

    def __init__(self, network_parameter, defaults):
        params = funky.maybe_get_network_params(network_parameter)
        super(WackyOptimizer, self).__init__(params, defaults)

    def apply_loss(self, loss, set_to_none: bool = False, *args, **kwargs):
        self.zero_grad(set_to_none)
        loss.backward()
        return self.step(*args, **kwargs)
