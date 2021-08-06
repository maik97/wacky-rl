import numpy as np
import tensorflow as tf
from gym import spaces

from wacky_rl.memory.memory import LoggingTensorArray




def get_expected_return(self, rewards):

    sample_len = tf.shape(rewards)[0]
    self.returns.reset_tensor(dtype=tf.float32, size=sample_len)

    rewards = tf.cast(rewards[::-1], dtype=tf.float32)

    discounted_sum = tf.constant(0.0, dtype=tf.float32)
    discounted_sum_shape = discounted_sum.shape

    for i in tf.range(sample_len):

        discounted_sum = rewards[i] + self.gamma * discounted_sum
        discounted_sum.set_shape(discounted_sum_shape)
        self.returns.write_to_tensor(i, discounted_sum)

    returns = self.returns.stack_tensor()[::-1]

    if self.standardize_returns:
        returns = ((returns - tf.math.reduce_mean(returns)) /
                   (tf.math.reduce_std(returns) + self.eps))

    if self.global_standardize_returns:
        self.global_mean = tf.math.reduce_mean(returns) * self.g_rate + self.global_mean * self.one_minus_g_rate
        returns = ((returns - (self.global_mean / 2)) /
                (tf.math.reduce_std(returns) + self.eps))

    self.logger.log_mean(self.name + '_transf_ret', np.mean(returns.numpy()))
    return returns

