import numpy as np
import tensorflow as tf


def transform_box_to_discrete(dims, act_space):
    n = dims^(np.sum(act_space.shape))
    return n, tf.constant(act_space.high, dtype=tf.float32), tf.constant(act_space.low, dtype=tf.float32)


def transform_discrete_to_box(act_space):
    n = tf.cast(tf.constant(act_space.n), dtype=tf.float32)
    act_shape = (1,)
    return act_shape, n


def contin_act_to_discrete(action, act_shape, highs, lows):
    for dim in act_shape:
        raise NotImplementedError


def discrete_act_to_contin(action, n):
    return tf.cast(tf.cast(action, dtype=tf.float32) * tf.cast(n, dtype=tf.float32), dtype=tf.int32)


class DynamicFactor:

    def __init__(self, tau: float = 1.0, init_factor: float = 1.0):

        self.tau = tau
        self.dynamic_factor = init_factor

    def __call__(self, alpha, max_from_sample):
        self.dynamic_factor = self.tau * max_from_sample + (1 - self.tau) * self.dynamic_factor
        return self.dynamic_factor


class TanhTransformer:

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def __call__(self, loss):
        return self.alpha * tf.math.tanh(loss)