import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.keras import layers


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