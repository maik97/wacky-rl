import numpy as np
import tensorflow as tf


class ExpectedReturnsCalculator:

    def __init__(self, gamma: float = 0.99, standardize: str = None):

        self.gamma = tf.constant(gamma, dtype=tf.float32)
        self.eps = tf.constant(np.finfo(np.float32).eps.item(), dtype=tf.float32)

        self.static_standardize = False
        self.dynamic_standardize = False

        if standardize == 'static':
            self.static_standardize = True

        elif standardize == 'dynamic':
            self.dynamic_standardize = True

    def __call__(self, rewards):

        sample_len = tf.shape(rewards)[0]
        returns = tf.TensorArray(dtype=tf.float32, size=sample_len)

        rewards = tf.cast(rewards[::-1], dtype=tf.float32)

        discounted_sum = tf.constant(0.0, dtype=tf.float32)
        discounted_sum_shape = discounted_sum.shape

        for i in tf.range(sample_len):

            discounted_sum = rewards[i] + self.gamma * discounted_sum
            discounted_sum.set_shape(discounted_sum_shape)
            returns = returns.write(i, discounted_sum)

        returns = returns.stack()[::-1]

        if self.static_standardize:
            returns = ((returns - tf.math.reduce_mean(returns)) /
                       (tf.math.reduce_std(returns) + self.eps))

        if self.dynamic_standardize:
            self.global_mean = tf.math.reduce_mean(returns) * self.g_rate + self.global_mean * self.one_minus_g_rate
            returns = ((returns - (self.global_mean / 2)) /
                    (tf.math.reduce_std(returns) + self.eps))

        return returns


class LamdaTransformReturns:

    def __init__(
            self,
            gamma: float = 0.99,
            lamda: float = 0.95,
    ):

        self.gamma = gamma
        self.lamda = lamda

    def __call__(self, rewards, dones, values, next_value):
        rewards = tf.squeeze(rewards)
        dones = tf.squeeze(dones)
        values = tf.squeeze(values)
        next_value = tf.squeeze(next_value)
        g = 0
        returns = []
        for i in reversed(range(len(rewards))):
            try:
                delta = rewards[i] + self.gamma * values[i+1] * dones[i] - values[i]
            except:
                delta = rewards[i] + self.gamma * next_value * dones[i] - values[i]
            g = delta + self.gamma * self.lamda * dones[i] * g
            returns.append(g + values[i])

        returns.reverse()
        returns = tf.squeeze(tf.stack(returns))
        adv = returns - values
        adv = (adv - tf.reduce_mean(adv)) / (tf.math.reduce_std(adv) + 1e-10)

        adv = tf.squeeze(tf.stack(adv))
        returns = tf.stack(returns)

        #return tf.reshape(adv, [len(adv),-1]), tf.reshape(returns, [len(returns),-1])
        return adv, returns
