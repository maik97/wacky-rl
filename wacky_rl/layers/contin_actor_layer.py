import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.keras import layers
from tensorflow.python.keras.engine.keras_tensor import KerasTensor

class NormalActionDistributions:

    def __init__(self, num_actions, min_sigma:float, max_sigma: float):

        self.num_actions = num_actions
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma

    def __call__(self, mu_list, sigma_list):
        self.x = [mu_list, sigma_list]
        self.distributions = []
        for i in range(self.num_actions):
            self.distributions.append(
                tfp.distributions.Normal(
                    self._scale_mu(mu_list[i]),
                    self._scale_sigma(sigma_list[i])
                )
            )
        #print(mu_list)
        #print(self.mean_actions())
        return self

    def _scale_mu(self, mu):
        return tf.squeeze(mu)

    def _scale_sigma(self, sigma):
        sigma = tf.exp(sigma)
        return tf.squeeze(tf.clip_by_value(sigma, self.min_sigma, self.max_sigma))
        #return sigma * (self.max_sigma - self.min_sigma) + self.min_sigma

    def sample_actions(self):
        actions = []
        for i in range(self.num_actions):
            actions.append(self.distributions[i].sample())
        return tf.math.tanh(tf.stack(actions))

    def mean_actions(self):
        actions = []
        for i in range(self.num_actions):
            actions.append(self.distributions[i].mean())
        return tf.math.tanh(tf.stack(actions))

    def calc_probs(self, actions):
        probs = []
        for i in range(self.num_actions):
            probs.append(self.distributions[i].prob(tf.math.atanh(actions[i])))
        return tf.stack(probs)

    def calc_log_probs(self, actions):
        log_probs = []
        for i in range(self.num_actions):
            log_probs.append(self.distributions[i].log_prob(tf.math.atanh(actions[i])))
        return tf.stack(log_probs)

    def calc_entropy(self, actions):
        entropies = []
        for i in range(self.num_actions):
            entropies.append(self.distributions[i].entropy())
        return tf.stack(entropies)

    def discrete_to_contin(self, actions, num_bins):
        contin_actions = []
        for i in range(self.num_actions):
            contin_actions.append(
                (tf.cast(actions[i], dtype=tf.float32) - (0.5 * (num_bins-1))) / (0.5 * (num_bins - 1))
            )
        return tf.stack(contin_actions)

    def contin_to_discrete(self, actions, num_bins):
        discrete_actions = []
        for i in range(self.num_actions):
            discrete_actions.append(
                (actions[i] * (0.5 * (num_bins - 1))) + (0.5 * (num_bins - 1))
            )
        return tf.cast(tf.stack(discrete_actions), dtype=tf.int32)


class ContinActionLayer(layers.Layer):

    def __init__(
            self,
            num_actions: int,
            mu_activation: str = 'tanh',
            sigma_activation: str = 'tanh',
            min_sigma: float = 0.001,
            max_sigma: float = 2.0,
            *args,
            **kwargs
    ):

        super().__init__()

        self.num_actions = num_actions

        self._mu_layer = [layers.Dense(1, activation=mu_activation, *args, **kwargs) for _ in range(num_actions)]
        self._sigma_layer = [layers.Dense(1, activation=sigma_activation, *args, **kwargs) for _ in range(num_actions)]
        self.distributions = NormalActionDistributions(num_actions, min_sigma, max_sigma)

        self.is_functional = False
        self.return_tensors = False

    def __call__(self, *args, **kwargs):

        try:
            return super().__call__(*args, **kwargs)
        except:
            self.last_layer = args[0]
            self._mu_layer = [l(*args, **kwargs) for l in self._mu_layer]
            self._sigma_layer = [l(*args, **kwargs) for l in self._sigma_layer]
            self.return_tensors = True
            self.is_functional = True
            return super().__call__(self._mu_layer + self._sigma_layer)

    def call(self, inputs, **kwargs):

        if not self.is_functional:
            mu_list = [mu_l(inputs) for mu_l in self._mu_layer]
            sigma_list = [sigma_l(inputs) for sigma_l in self._sigma_layer]
        else:
            out_list = super().call(inputs)
            mu_list = out_list[:self.num_actions]
            sigma_list = out_list[-self.num_actions:]

            if self.return_tensors:
                self.return_tensors = False
                return self.distributions(mu_list, sigma_list).mean_actions()

        return self.distributions(mu_list, sigma_list)

