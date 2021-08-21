import numpy as np


class GreedyEpsilon:

    def __init__(
            self,
            eps: float = 1.0,
            eps_decay: float = 0.9999,
            eps_min: float = 0.0,
    ):
        self.eps = eps
        self.eps_decay = eps_decay
        self.eps_min = eps_min

    def __call__(self):
        self.eps = max(self.eps * self.eps_decay, self.eps_min)
        if np.random.random() < self.eps:
            return True
        else:
            return False
