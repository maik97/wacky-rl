import numpy as np


class GreedyWrapper:

    def __init__(
            self,
            out_function,
            eps: float = 1.0,
            eps_decay: float = 0.9999,
            eps_min: float = 0.0,
    ):
        self.out_function = out_function
        self.eps = eps
        self.eps_decay = eps_decay
        self.eps_min = eps_min

    def __call__(self, x, training=None):

        if not training is None:
            if training['act_argmax']:
                return self.out_function(x, training)

        self.eps = max(self.eps * self.eps_decay, self.eps_min)
        if np.random.random() < self.eps:
            greedy_random = True
        else:
            greedy_random = False

        return self.out_function(x, {'act_argmax': greedy_random})
