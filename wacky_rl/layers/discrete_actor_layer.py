import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.keras import layers

class DiscreteActionDistributions:

    def __init__(self, num_multi_actions):

        self.num_multi_actions = num_multi_actions

    def __call__(self, x):

        self.distributions = x

    def sample_actions(self):
        actions = []
        for i in range(self.num_multi_actions):
            actions.append(tf.squeeze(tf.random.categorical(self.distributions[i], num_samples=1), axis=1))
        return tf.stack(actions)

    def mean_actions(self):
        actions = []
        for i in range(self.num_multi_actions):
            actions.append(tf.math.argmax(self.distributions[i], axis=-1))
        return tf.stack(actions)

    def calc_probs(self, actions):
        probs = []
        for i in range(self.num_multi_actions):
            probs.append(
                tf.gather_nd(
                    tf.nn.softmax(self.distributions[i]), tf.stack([np.arange(len(actions[i])), actions[i]], axis=1)
                )
            )
        return tf.stack(probs)

    def calc_log_probs(self, actions):
        log_probs = []
        probs = self.calc_probs(actions)
        for i in range(self.num_multi_actions):
            log_probs.append(tf.math.log(probs[i]))
        return tf.stack(log_probs)

    def calc_entropy(self):
        pass

    def actions_to_one_hot(self, actions):
        one_hots = []
        for i in range(self.num_multi_actions):
            one_hots.append(tf.one_hot(actions[i], len(actions[i])))
        return tf.stack(one_hots)

class DiscreteActionLayer(layers.Layer):

    def __init__(
            self,
            num_actions,
            num_multi_actions=1,
            activation='softmax',
            *args,
            **kwargs
    ):
        super().__init__(**kwargs)

        self._action_layer = [layers.Dense(num_actions, activation=activation, *args, **kwargs) for _ in range(num_multi_actions)]
        self.distributions = DiscreteActionDistributions(num_multi_actions)

    def call(self, inputs, **kwargs):
        action_list = [l(inputs) for l in self._action_layer]
        return self.distributions.update_distributions(action_list)
