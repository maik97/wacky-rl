import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.keras import layers

class DiscreteActionDistributions:

    def __init__(self, num_bins, num_actions, convert_to_contin):

        self.num_bins = num_bins
        self.num_actions = num_actions
        self.convert_to_contin = convert_to_contin

    @property
    def x(self):
        return self.distributions

    def __call__(self, x):

        self.distributions = x
        return self

    def sample_actions(self):
        actions = []
        for i in range(self.num_actions):
            actions.append(tf.squeeze(tf.random.categorical(self.distributions[i], num_samples=1), axis=1))
        return tf.stack(actions)

    def mean_actions(self):
        actions = []
        for i in range(self.num_actions):
            actions.append(tf.math.argmax(self.distributions[i], axis=-1))
        return tf.stack(actions)

    def calc_probs(self, actions):
        probs = []
        for i in range(self.num_actions):
            probs.append(
                tf.gather_nd(
                    tf.nn.softmax(self.distributions[i]), tf.stack([np.arange(len(actions[i])), actions[i]], axis=1)
                )
            )
        return tf.stack(probs)

    def calc_log_probs(self, actions):
        log_probs = []
        probs = self.calc_probs(actions)
        for i in range(self.num_actions):
            log_probs.append(tf.math.log(probs[i]))
        return tf.stack(log_probs)

    def calc_entropy(self):
        pass

    def actions_to_one_hot(self, actions):
        one_hots = []
        for i in range(self.num_actions):
            one_hots.append(tf.one_hot(actions[i], len(actions[i])))
        return tf.stack(one_hots)

    def discrete_to_contin(self, actions):
        contin_actions = []
        for i in range(self.num_actions):
            contin_actions.append(
                (tf.cast(actions[i], dtype=tf.float32) - (0.5 * (self.num_bins-1))) / (0.5 * (self.num_bins - 1))
            )
        return tf.stack(contin_actions)

    def contin_to_discrete(self, actions):
        discrete_actions = []
        for i in range(self.num_actions):
            discrete_actions.append(
                (actions[i] * (0.5 * (self.num_bins - 1))) + (0.5 * (self.num_bins - 1))
            )
        return tf.cast(tf.stack(discrete_actions), dtype=tf.int32)


class DiscreteActionLayer(layers.Layer):

    def __init__(
            self,
            num_bins,
            num_actions=1,
            activation='softmax',
            convert_to_contin=False,
            *args,
            **kwargs
    ):
        super().__init__(**kwargs)

        self._action_layer = [layers.Dense(num_bins, activation=activation, *args, **kwargs) for _ in range(num_actions)]
        self.distributions = DiscreteActionDistributions(num_bins, num_actions, convert_to_contin)

    def call(self, inputs, **kwargs):
        action_list = [l(inputs) for l in self._action_layer]
        return self.distributions(action_list)
