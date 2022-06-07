from wacky import memory as mem
from wacky.losses import BaseWackyLoss

class ValueLossWrapper(BaseWackyLoss):
    def __init__(self, loss_fn, scale_factor=1.0, wacky_reduce='mean', *args, **kwargs):
        super(ValueLossWrapper, self).__init__(scale_factor, wacky_reduce, *args, **kwargs)
        self.loss_fn = loss_fn

    def call(self, memory: [dict, mem.MemoryDict], *args, **kwargs):
        return self.loss_fn(memory['returns'], memory['values'], *args, **kwargs)
