from torch import nn

from wacky.modules import WackyModule


class ModuleDictRouter(WackyModule):
    def __init__(self, layers):
        super().__init__()
        self.module_dict = nn.ModuleDict(layers)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            # Loop through each child module to check if `name` exists
            for module in self.module_dict.values():
                if hasattr(module, name):
                    def wrapper(*args, **kwargs):
                        return getattr(module, name)(*args, **kwargs)
                    return wrapper
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def forward(self, x, *args, **kwargs):
        return {k: m(x[k], *args, **kwargs) for k, m in self.module_dict.items()}


class ModuleListRouter(WackyModule):
    def __init__(self, layers):
        super().__init__()
        self.module_list = nn.ModuleList(layers)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            # Loop through each child module to check if `name` exists
            for module in self.module_list:
                if hasattr(module, name):
                    def wrapper(*args, **kwargs):
                        return getattr(module, name)(*args, **kwargs)
                    return wrapper
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def forward(self, x, *args, **kwargs):
        return [m(x_i, *args, **kwargs) for x_i, m in zip(x, self.module_list)]
