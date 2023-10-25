from abc import ABC

import torch as th
from wacky import functional as funky
from wacky.backend import WackyValueError


def scale_on_call(scale_factor, x):
    return scale_factor * x


def scale_and_reduce_mean_on_call(scale_factor, x):
    return scale_factor * th.mean(x)


def scale_and_reduce_sum_on_call(scale_factor, x):
    return scale_factor * th.sum(x)


class BaseWackyLoss(funky.MemoryBasedFunctional, ABC):

    def __init__(self, scale_factor=1.0, wacky_reduce=None, *args, **kwargs):
        super(BaseWackyLoss, self).__init__(*args, **kwargs)
        self.scale_factor = scale_factor

        if wacky_reduce is None:
            self.on_call = wacky_reduce
        elif wacky_reduce == 'mean':
            self.on_call = scale_and_reduce_mean_on_call
        elif wacky_reduce == 'sum':
            self.on_call = scale_and_reduce_sum_on_call
        else:
            raise WackyValueError(wacky_reduce, ('mean', 'sum'), parameter='wacky_reduce', optional=True)

    def __call__(self, *args, **kwargs) -> th.Tensor:
        return self.on_call(self.scale_factor, super(BaseWackyLoss, self).__call__(*args, **kwargs))
