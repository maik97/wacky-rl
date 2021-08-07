import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import losses

'''
hidden_units: (list, tuple, np.ndarray, int, str) = 'auto',  # not used when model_layer provided
hidden_activation: str = None,  # not used when model_layer provided
out_units: int = None,  # not used when model_layer provided
out_activation: str = None,  # not used when model_layer provided

# Model Layer:
if model_layer is None:

    if hidden_units == 'auto':
        hidden_units = [64, 64]
    elif hidden_units is None:
        hidden_units = []
    elif not isinstance(hidden_units, (list, tuple, np.ndarray)):
        hidden_units = [hidden_units]

    self._wacky_layer = [layers.Dense(units, activation=hidden_activation) for units in hidden_units]

    if not out_units is None:
        self._wacky_layer.append(layers.Dense(out_units, activation=out_activation))
    else:
        raise Warning(
            "No output layer at {}, {}, {}: neither out_units nor out_layer was not specified.".format(
                self, self.model_name, self.model_index
            )
        )
else:

    if not isinstance(model_layer, list):
        self._wacky_layer = [model_layer]
    else:
        self._wacky_layer = model_layer
'''

class WackyModel(tf.keras.Model):

    def __init__(
            self,
            model_layer: (list, layers.Layer) = None,
            model_outputs: (list, layers.Layer) = None,
            optimizer: optimizers.Optimizer = None,
            learning_rate: float = None,  # not used when model_layer provided
            loss = None,
            loss_alpha: float = 1.0,
            out_function = None,
            model_name: str = 'UnnamedWackyModel',
            model_index: int = None,
    ):

        super().__init__()

        self.model_name = model_name
        self.model_index = model_index

        # Model Layer:
        if model_layer is None:
            self._wacky_layer = [
                layers.Flatten(),
                layers.Dense(128, activation='relu')
            ]

        else:
            if not isinstance(model_layer, list):
                self._wacky_layer = [model_layer]
            else:
                self._wacky_layer = model_layer

        # Model Outputs:
        if not model_outputs is None:
            if not isinstance(model_outputs, list):
                self._wacky_outputs = [model_outputs]
            else:
                self._wacky_outputs = model_outputs
        else:
            self._wacky_outputs = None

        # Optimizer:
        if optimizer is None:
            self._wacky_optimizer = tf.keras.optimizers.RMSprop()
        else:
            self._wacky_optimizer = optimizer

        # Learning Rate:
        if not learning_rate is None:
            self._wacky_optimizer.learning_rate.assign(learning_rate)

        # Loss Function:
        if loss is None:
            self._wacky_loss = losses.MeanSquaredError()
        else:
            self._wacky_loss = loss
        self.loss_alpha = loss_alpha

        # Output Function (e.g. to calculate actions)
        self._wacky_out_func = out_function

        # Gradient Tape:
        self._wacky_tape = tf.GradientTape(persistent=True)

    def _wacky_forward(self, x):
        #print(x)
        for l in self._wacky_layer: x = l(x)
        if not self._wacky_outputs is None:
            return [out(x) for out in self._wacky_outputs]
        return x

    def call(self, inputs, training=None, mask=None):

        # Start Tape Recording if necessary:
        if not self._wacky_tape._recording:
            self._wacky_tape._push_tape()
            self._wacky_tape.watch(self.trainable_variables)

        # Feedforward:
        x = self._wacky_forward(inputs)

        # Transform outputs if needed (for example to calculate actions):
        if not self._wacky_out_func is None:
            if training is None:
                return self._wacky_out_func(x)
            return self._wacky_out_func(x, training)
        return x

    def train_step(self, *args, **kwargs):

        self._wacky_tape._ensure_recording()

        loss = self.loss_alpha * self._wacky_loss(*args, **kwargs)

        self._wacky_tape._pop_tape()

        grads = self._wacky_tape.gradient(loss, self.trainable_variables)
        self._wacky_optimizer.apply_gradients(zip(grads, self.trainable_variables))

        self._wacky_tape._tape = None

        return loss


class TargetUpdate:

    def __init__(self, tau: float = 0.15):
        self.tau = tau

    def __call__(self, model: tf.keras.Model, target: tf.keras.Model):

        weights = model.get_weights()
        target_weights = target.get_weights()

        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)

        target.set_weights(target_weights)

        return target
