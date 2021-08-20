import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.keras import layers

class NormalActionDistributions:

    def __init__(self, num_actions):

        self.num_actions = num_actions
        self.min_sigma = 1e-6
        self.max_sigma = 1.0

    def __call__(self, mu_list, sigma_list):
        #print(mu_list)
        #print(sigma_list)
        self.distributions = []
        for i in range(self.num_actions):
            self.distributions.append(
                tfp.distributions.Normal(
                    self._scale_mu(mu_list[i]),
                    self._scale_sigma(sigma_list[i])
                )
            )
        #actions = self.sample_actions()
        #print(actions)
        #print(self.calc_probs(actions))
        return self

    def _scale_mu(self, mu):
        return tf.squeeze(mu)

    def _scale_sigma(self, sigma):
        sigma = tf.squeeze(tf.clip_by_value(sigma, 0.0, 1.0))
        return sigma * (self.max_sigma - self.min_sigma) + self.min_sigma

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

    def calc_entropy(self):
        entropies = []
        for i in range(self.num_actions):
            entropies.append(self.distributions[i].entropy())
        return tf.stack(entropies)


class ContinActionLayer(layers.Layer):

    def __init__(
            self,
            num_actions,
            mu_activation='tanh',
            sigma_activation='sigmoid',
            *args,
            **kwargs
    ):
        super().__init__(**kwargs)

        self._mu_layer = [layers.Dense(1, activation=mu_activation, *args, **kwargs) for _ in range(num_actions)]
        self._sigma_layer = [layers.Dense(1, activation=sigma_activation, *args, **kwargs) for _ in range(num_actions)]
        self.distributions = NormalActionDistributions(num_actions)

    def call(self, inputs, **kwargs):
        mu_list = [mu_l(inputs) for mu_l in self._mu_layer]
        sigma_list = [sigma_l(inputs) for sigma_l in self._sigma_layer]
        return self.distributions(mu_list, sigma_list)
