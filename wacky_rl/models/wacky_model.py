import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import wacky_rl

class WackyModel(tf.keras.Model):

    def __init__(
            self,
            model_layer: (list, layers.Layer) = None,
            optimizer: optimizers.Optimizer = None,
            loss = None,
            loss_alpha: float = 1.0,
            model_name: str = 'UnnamedWackyModel',
            model_index: int = None,
            grad_clip=False,
    ):

        super().__init__()

        self.model_name = model_name
        self.model_index = model_index
        self.grad_clip = grad_clip

        # Model Layer:
        if model_layer is None:
            self._wacky_layer = [
                layers.Flatten(),
                layers.Dense(64, activation='relu'),
                layers.Dense(64, activation='relu')
            ]

            # TODO: Add action layer based on loss

        else:
            if not isinstance(model_layer, list):
                self._wacky_layer = [model_layer]
            else:
                self._wacky_layer = model_layer

        # Optimizer:
        if optimizer is None:
            self._wacky_optimizer = tf.keras.optimizers.RMSprop()
        else:
            self._wacky_optimizer = optimizer

        # Loss Function:
        if loss is None:
            self._wacky_loss = wacky_rl.losses.MeanSquaredErrorLoss()
        else:
            self._wacky_loss = loss
        self.loss_alpha = loss_alpha

    def _wacky_forward(self, x):
        for l in self._wacky_layer: x = l(x)
        return x

    def call(self, inputs, training=False, mask=None, *args, **kwargs):
        return self._wacky_forward(inputs)

    def train_step(self, inputs, *args, **kwargs):

        with tf.GradientTape as tape:
            x = self._wacky_forward(inputs)
            loss = self.loss_alpha * self._wacky_loss(x, *args, **kwargs)

        grads = tape.gradient(loss, self.trainable_variables)
        return loss

    def predict_step(self, data, mask=None, *args, **kwargs):
        return self.call(data, training=False, mask=mask, *args, **kwargs)


class WackyDualingModel:

    def __init__(
            self,
            model_layer: (list, layers.Layer) = None,
            model_outputs: (list, layers.Layer) = None,
            optimizer: optimizers.Optimizer = None,
            learning_rate: float = None,  # not used when model_layer provided
            loss=None,
            loss_alpha: float = 1.0,
            out_function=None,
            model_name: str = 'UnnamedWackyModel',
            model_index: int = None,
    ):

        self.model_1 = WackyModel(model_layer, model_outputs, optimizer, learning_rate,
                       loss, loss_alpha, None, model_name+'_1', model_index)

        self.model_2 = WackyModel(model_layer, model_outputs, optimizer, learning_rate,
                        loss, loss_alpha, None, model_name+'_2', model_index)

        self._wacky_out_func = out_function

    @property
    def is_dualing(self):
        return True

    def __call__(self, inputs, training=True, mask=None, *args, **kwargs):

        x_1 = self.model_1(inputs, training, mask)
        x_2 = self.model_2(inputs, training, mask)

        self.model_1._continue_wacky_recording()
        self.model_2._continue_wacky_recording()

        x = tf.math.minimum(x_1, x_2)

        if not self._wacky_out_func is None:
            x = self._wacky_out_func(x, *args, **kwargs)

        if training:
            self.model_1._stop_wacky_recording()
            self.model_2._stop_wacky_recording()
        return x

    def train_step(self, *args, **kwargs):
        loss_1 = self.model_1.train_step(*args, **kwargs)
        loss_2 = self.model_2.train_step(*args, **kwargs)
        return tf.reduce_mean([loss_1, loss_2], 0)

    def predict_step(self, data, mask=None, *args, **kwargs):
        return self(data, training=False, mask=mask, *args, **kwargs)



class TargetUpdate:

    def __init__(self, tau: float = 0.15):
        self.tau = tau

    def __call__(self, model: tf.keras.Model, target: tf.keras.Model):

        if hasattr(model, 'is_dualing'):
            if model.is_dualing:
                target.model_1 = self._update_target(target.model_1, model.model_1)
                target.model_2 = self._update_target(target.model_2, model.model_2)
                return target

        return self._update_target(model, target)


    def _update_target(self, model, target):
        weights = model.get_weights()
        target_weights = target.get_weights()

        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)

        target.set_weights(target_weights)
        return target


class TargetModelWrapper:

    def __init__(self, model, tau: float = 0.15):

        import copy

        self.model = model
        self.target = copy.deepcopy(model)
        self._update_target = TargetUpdate(tau)

    def __call__(self, *args, **kwargs):
        return self.model( *args, **kwargs)

    def train_step(self, *args, **kwargs):
        return self.model.train_step(*args,**kwargs)

    def update_target(self):
        self.target_model = self._update_target(self.model, self.target)
