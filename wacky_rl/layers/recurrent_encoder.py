import tensorflow as tf
from tensorflow.keras import layers
from collections import deque

class RecurrentEncoder(layers.Layer):

    def __init__(self, recurrent_layer, *args, **kwargs):
        super(RecurrentEncoder, self).__init__(*args, **kwargs)

        self.recurrent_layer = recurrent_layer

        self.past_inputs = deque(maxlen=6)

    def call(self, inputs, *args, **kwargs):
        try:
            x = self.recurrent_layer(tf.expand_dims(inputs, -1))
        except:
            x = self.recurrent_layer(inputs)
        print('x')
        print(x)
        return x