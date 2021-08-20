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

class RunningMeanStd(object):
    def __init__(self, epsilon: float = 1e-4, shape= ()):
        """
        Calulates the running mean and std of a data stream
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
        :param epsilon: helps with arithmetic issues
        :param shape: the shape of the data stream's output
        """
        self.mean = np.zeros(shape, np.float64)
        self.var = np.ones(shape, np.float64)
        self.count = epsilon

    def update(self, arr: np.ndarray) -> None:
        batch_mean = np.mean(arr, axis=0)
        batch_var = np.var(arr, axis=0)
        batch_count = arr.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: int) -> None:
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = m_2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count

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


        #rewards = tf.squeeze(rewards).numpy()
        #dones = tf.squeeze(dones).numpy()
        values = tf.squeeze(values).numpy()
        next_value = tf.squeeze(next_value).numpy()

        #returns = tf.TensorArray(size=len(rewards), dtype=tf.float32)
        returns = []
        #advantages = tf.TensorArray(size=len(rewards), dtype=tf.float32)
        advantages = []

        g = 0.0
        for i in reversed(range(len(rewards))):
            try:
                delta = rewards[i] + self.gamma * values[i+1] * dones[i] - values[i]
            except:
                delta = rewards[i] + self.gamma * next_value * dones[i] - values[i]
            g = delta + self.gamma * self.lamda * dones[i] * g
            ret =  g + values[i]
            returns.append(ret)
            advantages.append(g)
            #returns = returns.write(i,ret)
            #advantages = advantages.write(i, g)

        returns.reverse()
        advantages.reverse()


        #adv = advantages.stack()
        #adv_mean = tf.reduce_mean(adv)
        #adv_std = (tf.math.reduce_std(adv) + 1e-10)

        #for i in range(len(adv)):
        #    stand_adv = (adv[i] - adv_mean) / adv_std
        #    advantages = advantages.write(i, stand_adv)

        return tf.stack(advantages), tf.stack(returns)

    def normalize_reward(self, reward: np.ndarray) -> np.ndarray:
        """
        Normalize rewards using this VecNormalize's rewards statistics.
        Calling this method does not update statistics.
        """
        if self.norm_reward:
            reward = np.clip(reward / np.sqrt(self.ret_rms.var + self.epsilon), -self.clip_reward, self.clip_reward)
        return reward