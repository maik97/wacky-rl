import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.keras import layers

from wacky_rl.layers import ContinActionLayer, DiscreteActionLayer


class CombinationActorLayer(layers.Layer):

    def __init__(
            self,
            num_bins,
            num_actions,
            discrete_activation='softmax',
            mu_activation='tanh',
            sigma_activation='sigmoid',
            *args,
            **kwargs
    ):
        super().__init__(**kwargs)

        self.contin_layer = ContinActionLayer(num_actions, mu_activation, sigma_activation, *args, **kwargs)
        self.discrete_layer = DiscreteActionLayer(num_bins, num_actions, discrete_activation)
        self.concat_layer = layers.Concatenate(axis=-1)

    def call(self, inputs, *args, **kwargs):
        return self.discrete_layer(inputs)


        contin_dist = self.contin_layer(self.concat_layer(inputs, discrete_dist))


