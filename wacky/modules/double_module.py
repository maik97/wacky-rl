from wacky.modules import WackyModule, ModuleConstructor
from torch import nn
import warnings


class DoubleModule(WackyModule):

    def __init__(self, module, polyak=0.995):
        super().__init__()

        self.behavior = ModuleConstructor.construct(module)
        self.target = ModuleConstructor.construct(module)
        self.polyak = polyak

        for param in self.target.parameters():
            param.requires_grad = False

        self.target_is_initialized = False

    def forward(self, x, *args, **kwargs):
        if not self.target_is_initialized:
            _ = self.target(x)
            self.target_is_initialized = True
        return self.behavior(x, *args, **kwargs)

    def check_uninitialized_parameters(self):
        if self.target_is_initialized:
            return False
        uninitialized_params = []
        for param in self.target.parameters():
            if isinstance(param, nn.parameter.UninitializedParameter):
                uninitialized_params.append(param)
        return uninitialized_params

    def update_target_weights(self):
        uninitialized_params = self.check_uninitialized_parameters()
        if uninitialized_params:
            warnings.warn(
                "Uninitialized parameters detected in the target module. "
                "Please run one forward pass to initialize all parameters before overriding "
                "or avoid using Lazy Modules."
            )

        for target_param, behavior_param in zip(self.target.parameters(), self.behavior.parameters()):
            target_param.data.copy_(self.polyak * behavior_param.data + (1.0 - self.polyak) * target_param.data)

    def override_target(self):
        uninitialized_params = self.check_uninitialized_parameters()
        if uninitialized_params:
            warnings.warn(
                "Uninitialized parameters detected in the target module. "
                "Please run one forward pass to initialize all parameters before overriding "
                "or avoid using Lazy Modules."
            )

        for target_param, behavior_param in zip(self.target.parameters(), self.behavior.parameters()):
            target_param.data.copy_(behavior_param.data)


if __name__ == '__main__':

    import torch


    class SimpleModel(nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.layer = nn.LazyLinear(10)

        def forward(self, x):
            return self.layer(x)


    # Initialize DoublePolicy with a simple model that contains lazy layers.
    double_policy = DoubleModule(SimpleModel)

    # Override target weights
    double_policy.override_target()  # We expect a user warning.

    # Create a random tensor
    x = torch.rand((1, 5))

    # Forward pass to ensure lazy layers are initialized
    _ = double_policy(x)

    # Override target weights
    double_policy.override_target()  # no user warning now

    # Update target weights
    double_policy.update_target_weights()  # no user warning now

    # Check if requires_grad is False for target network
    for param in double_policy.target.parameters():
        print(param.requires_grad)  # Should print False
