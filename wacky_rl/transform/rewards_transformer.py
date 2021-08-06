import numpy as np
import tensorflow as tf
from gym import spaces

from wacky_rl.memory.memory import LoggingTensorArray







def simple_normalize_rewards(rewards):
    return rewards / tf.cast(tf.shape(rewards)[0], dtype=tf.float32)

class StaticRewardNormalizer:

    def __init__(self, max_pos_reward, max_neg_reward):
        self.max_pos_reward = max_pos_reward
        self.max_neg_reward = max_neg_reward

    def __call__(self, rewards):
        rewards = tf.where(rewards > 0, rewards / self.max_pos_reward, rewards)
        rewards = tf.where(rewards < 0, rewards / self.max_neg_reward, rewards)
        return rewards

def dynamic_normalize_rewards(rewards)
    rewards = static_normalize_rewards(rewards)

    min_r = tf.reduce_min(rewards)
    if min_r < 0:
        self.max_neg_reward = self.dynamic_rate * tf.abs(
            min_r) + self.one_minus_dynamic_rate * self.max_neg_reward

    max_r = tf.reduce_max(rewards)
    if max_r > 0:
        self.max_pos_reward = self.dynamic_rate * max_r + self.one_minus_dynamic_rate * self.max_pos_reward

def forward_discount_rewards(rewards):
    sample_len = tf.shape(rewards)[0]
    placeholder_rewards = tf.TensorArray(dtype=tf.float32, size=sample_len)

    for i in tf.range(sample_len):
        placeholder_rewards = placeholder_rewards.write(i, (rewards[i] / tf.cast(i + 1, dtype=tf.float32)))

    return placeholder_rewards.stack()




def backward_discount_rewards():

    sample_len = tf.shape(rewards)[0]
    placeholder_rewards = tf.TensorArray(dtype=tf.float32, size=sample_len)
    rewards = rewards[::-1]

    for i in tf.range(sample_len):
        placeholder_rewards = placeholder_rewards.write(i, (rewards[i] / tf.cast(i+1, dtype=tf.float32)))

    return placeholder_rewards.stack()[::-1]

        self.logger.log_mean(self.name + '_transf_rew', np.mean(rewards.numpy()))
        return rewards

