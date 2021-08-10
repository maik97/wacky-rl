import tensorflow as tf

class NoisyLayer(tf.keras.layers.Layer):
    def __init__(
            self,
            units,
            activation=None,
            use_bias=True
    ):
        super(NoisyLayer, self).__init__()

        self.units = int(units) if not isinstance(units, int) else units
        if self.units < 0:
            raise ValueError(f'Received an invalid value for `units`, expected '
                             f'a positive integer. Received: units={units}')
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias


    def call(self, inputs):
        '''
        Based on:
        https://github.com/Parsa33033/Deep-Reinforcement-Learning-DQN/blob/f6225e464d75a1a5f75d59d22eb6f32c3f4879fd/Noisy-DQN.py#L83
        '''

        w_shape = [self.units, inputs.shape[1].value]
        mu_w = tf.Variable(initial_value=tf.random.truncated_normal(shape=w_shape))
        sigma_w = tf.Variable(initial_value=tf.constant(0.017, shape=w_shape))
        epsilon_w = tf.random.uniform(shape=w_shape)

        w = tf.add(mu_w, tf.multiply(sigma_w, epsilon_w))

        if self.use_bias:
            b_shape = [self.units]
            mu_b = tf.Variable(initial_value=tf.random.truncated_normal(shape=b_shape))
            sigma_b = tf.Variable(initial_value=tf.constant(0.017, shape=b_shape))
            epsilon_b = tf.random.uniform(shape=b_shape)

            b = tf.add(mu_b, tf.multiply(sigma_b, epsilon_b))
            outputs = tf.matmul(input, tf.transpose(w)) + b

        else:
            outputs = tf.matmul(input, tf.transpose(w))

        if self.activation is not None:
            outputs = self.activation(outputs)

        return outputs