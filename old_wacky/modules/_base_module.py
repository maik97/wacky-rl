from torch import nn


class WackyModule(nn.Module):

    def __init__(self):
        super(WackyModule, self).__init__()


class WackyLayer(WackyModule):

    def __init__(self, in_features, out_features, module=None, activation=None, *args, **kwargs):
        super(WackyLayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        if module is None:
            module = nn.Linear
        print(in_features)
        print(out_features)
        self.module = module(in_features, out_features, *args, **kwargs)
        self.activation = activation

    def forward(self, x):
        x = self.module(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class AdderLayer(WackyModule):

    def __init__(self, in_features_a, infeatures_b, out_features, module=None, activation=None):
        super(AdderLayer, self).__init__()

        if module is None:
            module = nn.Linear
        self.layer_a = module(in_features_a, out_features)
        self.layer_b = module(infeatures_b, out_features)
        self.activation = activation

    def forward(self, a, b=None):
        if b is None:
            b = a
        x = self.layer_a(a) + self.layer_b(b)
        if self.activation is not None:
            x = self.activation(x)
        return x
