import numpy as np
import tensorflow as tf

class QAction:

    def __init__(
            self,
            eps: float = None,
            eps_decay: float = 0.9999,
            eps_min: float = 0.0,
    ):
        self.eps = eps
        self.eps_decay = eps_decay
        self.eps_min = eps_min

    def __call__(self, x, training=None):
        return self._q_value_to_action(tf.squeeze(x), training)

    def _q_value_to_action(self, x, training):

        if not training is None:
            act_argmax = training['act_argmax']
        else:
            act_argmax = False

        greedy_random = False
        if not self.eps is None:
            self.eps = max(self.eps*self.eps_decay, self.eps_min)
            if np.random.random() < self.eps:
                greedy_random = True

        if not greedy_random or act_argmax:
            return tf.math.argmax(x)
        return tf.random.uniform(shape=[], maxval=len(x), dtype=tf.int32)



class DDDQNAction:

    def __call_(self,x):

        v, a = x
        Q = v + (a - tf.math.reduce_mean(a, axis=1, keepdims=True))
        return Q



