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


class GAE:

    def __init__(
            self,
            gamma: float = 0.99,
            lamda: float = 0.95,
    ):

        self.gamma = gamma
        self.lamda = lamda



    def __call__(self, rewards, dones, values, next_value):

        #rewards = tf.squeeze(rewards)

        #rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-8)
        #rewards = (rewards) / (np.std(rewards) + 1e-8)
        rewards = tf.cast(tf.squeeze(rewards), dtype=tf.float32)
        dones = tf.cast(dones, dtype=tf.float32)
        #dones = tf.squeeze(dones).numpy()
        values = tf.squeeze(values)
        next_value = tf.squeeze(next_value)

        #returns = tf.TensorArray(size=len(rewards), dtype=tf.float32)
        returns = []
        #advantages = tf.TensorArray(size=len(rewards), dtype=tf.float32)
        advantages = []
        all_values = tf.experimental.numpy.append(values, next_value)
        g = 0.0
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * all_values[i+1] * dones[i] - all_values[i]
            g = delta + self.gamma * self.lamda * dones[i] * g
            ret =  g + all_values[i]
            returns.append(ret)
            #advantages.append(g)
            #returns = returns.write(i,ret)
            #advantages = advantages.write(i, g)


        #advantages.reverse()
        returns.reverse()
        returns = tf.stack(returns)
        returns = tf.cast(returns, dtype=tf.float32)
        advantages = returns - tf.stack(values)
        #returns = (returns - tf.reduce_mean(returns)) / (tf.math.reduce_std(returns) + 1e-8)



        #adv = advantages.stack()
        #adv_mean = tf.reduce_mean(adv)
        #adv_std = (tf.math.reduce_std(adv) + 1e-10)

        #for i in range(len(adv)):
        #    stand_adv = (adv[i] - adv_mean) / adv_std
        #    advantages = advantages.write(i, stand_adv)

        return advantages, returns

    def normalize_reward(self, reward: np.ndarray) -> np.ndarray:
        """
        Normalize rewards using this VecNormalize's rewards statistics.
        Calling this method does not update statistics.
        """
        if self.norm_reward:
            reward = np.clip(reward / np.sqrt(self.ret_rms.var + self.epsilon), -self.clip_reward, self.clip_reward)
        return reward