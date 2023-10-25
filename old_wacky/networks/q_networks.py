from wacky.networks import WackyNetwork


class DuellingQNetwork(WackyNetwork):

    def __init__(self, value_net_module, adv_net_module, shared_net_module=None):
        super(DuellingQNetwork, self).__init__()

        self.shared_net_module = shared_net_module
        self.value_net_module = value_net_module
        self.adv_net_module = adv_net_module

    def forward(self, x):
        if self.shared_net_module is not None:
            x = self.shared_net_module(x)
        value = self.value_net_module(x)
        adv = self.adv_net_module(x)
        return value + adv - adv.mean(dim=-1, keepdim=True)
