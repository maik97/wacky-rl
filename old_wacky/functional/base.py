import abc

from wacky import backend
from wacky import memory as mem


class WackyBase(metaclass=abc.ABCMeta):

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)

    @abc.abstractmethod
    def call(self, *args, **kwargs):
        raise NotImplementedError()


class MemoryBasedFunctional(WackyBase):

    def __init__(self, *args, **kwargs):
        """
        Wrapper for some functions from wacky.functional that uses a Dict or MemoryDict to look up arguments.
        Initializes all necessary function hyperparameters.
        """
        super(MemoryBasedFunctional, self).__init__(*args, **kwargs)
        self._check_type = True

    def __call__(self, memory: [dict, mem.MemoryDict]):
        if self._check_type:
            backend.check_type(memory, (dict, mem.MemoryDict), 'memory')
            self._check_type = False
        return self.call(memory)

    @abc.abstractmethod
    def call(self, memory):
        raise NotImplementedError()
