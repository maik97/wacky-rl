import tensorflow as tf
from tensorflow.keras import layers

import wacky_rl
from wacky_rl import losses

class WackyModel(tf.keras.Model):

    def __init__(
            self,
            model_layer: (list, tf.keras.layers.Layer) = None,
            optimizer: tf.keras.optimizers.Optimizer = None,
            loss: losses.WackyLoss = None,
            loss_alpha: float = 1.0,
            model_name: str = 'UnnamedWackyModel',
            model_index: int = None,
            grad_clip=False,
    ):

        super().__init__()

        self.model_name = model_name
        self.model_index = model_index
        self.grad_clip = grad_clip

        # Loss Function:
        self.loss_alpha = loss_alpha

        if loss is None:
            self._wacky_loss = wacky_rl.losses.MeanSquaredErrorLoss()
        else:
            self._wacky_loss = loss
            # TODO: Add keras loss wrapper and str

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
            self._optimizer = tf.keras.optimizers.RMSprop()
        else:
            self._optimizer = optimizer
            # TODO: Add keras optimizer with str

    def _wacky_forward(self, x):
        for l in self._wacky_layer: x = l(x)
        return x

    def call(self, inputs, training=False, mask=None, *args, **kwargs):
        return self._wacky_forward(inputs)

    def train_step(self, inputs, *args, **kwargs):

        with tf.GradientTape() as tape:
            x = self._wacky_forward(inputs)
            loss = self.loss_alpha * self._wacky_loss(x, *args, **kwargs)

        grads = tape.gradient(loss, self.trainable_variables)

        if self.grad_clip:
            grads = [tf.clip_by_norm(g, 0.5) for g in grads]

        self._optimizer.apply_gradients(zip(grads, self.trainable_variables))

        return loss

    def predict_step(self, data, mask=None, *args, **kwargs):
        return self.call(data, training=False, mask=mask, *args, **kwargs)


class WackyDualingModel:

    def __init__(
            self,
            model_layer: (list, tf.keras.layers.Layer) = None,
            optimizer: tf.keras.optimizers.Optimizer = None,
            loss: wacky_rl.losses.WackyLoss = None,
            loss_alpha: float = 1.0,
            model_name: str = 'UnnamedWackyModel',
            model_index: int = None,
            grad_clip=False,
    ):

        self.model_1 = WackyModel(model_layer, optimizer, loss, loss_alpha, model_name+'_1', model_index, grad_clip)
        self.model_2 = WackyModel(model_layer, optimizer, loss, loss_alpha, model_name+'_2', model_index, grad_clip)

    @property
    def is_dualing(self):
        return True

    def __call__(self, inputs, training=True, mask=None, *args, **kwargs):
        x_1 = self.model_1(inputs, training, mask)
        x_2 = self.model_2(inputs, training, mask)
        return tf.math.minimum(x_1, x_2)

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
        else:
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
