import tensorflow as tf


def simple_normalize_rewards(rewards):
    return rewards / tf.cast(tf.shape(rewards)[0], dtype=tf.float32)


def forward_discount_rewards(rewards):
    sample_len = tf.shape(rewards)[0]
    placeholder_rewards = tf.TensorArray(dtype=tf.float32, size=sample_len)

    for i in tf.range(sample_len):
        placeholder_rewards = placeholder_rewards.write(i, (rewards[i] / tf.cast(i + 1, dtype=tf.float32)))

    return placeholder_rewards.stack()


def backward_discount_rewards(rewards):

    sample_len = tf.shape(rewards)[0]
    placeholder_rewards = tf.TensorArray(dtype=tf.float32, size=sample_len)
    rewards = rewards[::-1]

    for i in tf.range(sample_len):
        placeholder_rewards = placeholder_rewards.write(i, (rewards[i] / tf.cast(i+1, dtype=tf.float32)))

    return placeholder_rewards.stack()[::-1]


class StaticRewardNormalizer:

    def __init__(self, max_pos_reward, max_neg_reward):
        self.max_pos_reward = max_pos_reward
        self.max_neg_reward = max_neg_reward

    def __call__(self, rewards):
        rewards = tf.where(rewards > 0, rewards / self.max_pos_reward, rewards)
        rewards = tf.where(rewards < 0, rewards / self.max_neg_reward, rewards)
        return rewards


class DynamicRewardNormalizer:

    def __init__(self, max_pos_reward, max_neg_reward, dynamic_rate):
        self.max_pos_reward = max_pos_reward
        self.max_neg_reward = max_neg_reward
        self.dynamic_rate = dynamic_rate

    def __call__(self, rewards):

        rewards = tf.where(rewards > 0, rewards / self.max_pos_reward, rewards)
        rewards = tf.where(rewards < 0, rewards / self.max_neg_reward, rewards)

        min_r = tf.reduce_min(rewards)
        if min_r < 0:
            self.max_neg_reward = self.dynamic_rate * tf.abs(min_r) + (1-self.dynamic_rate) * self.max_neg_reward

        max_r = tf.reduce_max(rewards)
        if max_r > 0:
            self.max_pos_reward = self.dynamic_rate * max_r + (1-self.dynamic_rate) * self.max_pos_reward

        return rewards
