from abc import ABC
import torch.nn as nn


class WackyModule(nn.Module, ABC):
    """
    Base class that is a specialized form of nn.Module for subclasses that may contain
    specialized modules with methods for experimental learning and resetting.
    """

    def __init__(self):
        """Initialize the WackyModule instance."""
        super(WackyModule, self).__init__()

    def get_children_with_method(self, method_name: str) -> list:
        """
        Return child layers that possess a specific method.

        Args:
            method_name (str): The name of the method to look for.

        Returns:
            list: List of child layers that have the specified method.
        """
        return [child for child in self.children() if hasattr(child, method_name)]

    def experimental_learn(self, *args, **kwargs):
        """
        Apply experimental learning techniques to applicable child layers.

        Args and kwargs are passed directly to the 'experimental_learn' method of the child layers.
        """
        for child in self.get_children_with_method('experimental_learn'):
            child.experimental_learn(*args, **kwargs)

    def experimental_reset(self, *args, **kwargs):
        """
        Reset the state of applicable child layers in an experimental manner.

        Args and kwargs are passed directly to the 'experimental_reset' method of the child layers.
        """
        for child in self.get_children_with_method('experimental_reset'):
            child.experimental_reset(*args, **kwargs)
