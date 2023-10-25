from torch import nn
from wacky import functional as funky

class WackyNetwork(nn.Module):

    def __init__(self):
        super(WackyNetwork, self).__init__()

    def learn(self, *args, **kwargs):
        for layer in self.layers:
            if funky.has_method(layer, 'learn'):
                layer.learn( *args, **kwargs)

    def reset(self, *args, **kwargs):
        for layer in self.layers:
            if funky.has_method(layer, 'reset'):
                layer.reset( *args, **kwargs)
