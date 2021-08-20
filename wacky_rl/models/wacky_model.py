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
            learning_rate: float = None,
            loss = None,
            loss_alpha: float = 1.0,
            out_function = None,
            model_name: str = 'UnnamedWackyModel',
            model_index: int = None,
            grad_clip=False,
            use_wacky_backprob =True,
    ):

        super().__init__()

        self.model_name = model_name
        self.model_index = model_index
        self.grad_clip = grad_clip
        self.use_wacky_backprob = use_wacky_backprob

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
        #self._wacky_tape = tf.GradientTape(persistent=True)



    def _wacky_forward(self, x, training):
        #print(x)
        for l in self._wacky_layer: x = l(x)
        if not self._wacky_outputs is None:
            return [out(x) for out in self._wacky_outputs]
        return x

    def _start_wacky_recording(self):
        if not self._wacky_tape._recording:
            self._wacky_tape._push_tape()
            self._wacky_tape.watch(self.trainable_variables)

    def _stop_wacky_recording(self):
        self._wacky_tape._recording = False
        # self._wacky_tape.stop_recording()

    #def _continue_wacky_recording(self):
        #self._wacky_tape._recording = True

    #def _ensure_wacky_recording(self):
        #self._wacky_tape._ensure_recording()


    def call(self, inputs, training=False, mask=None, *args, **kwargs):

        # Start Tape Recording if necessary:
        #if not self._wacky_tape._recording and training:
            #self._start_wacky_recording()

        # Feedforward:
        x = self._wacky_forward(inputs, training)

        # Transform outputs if needed (for example to calculate actions):
        if not self._wacky_out_func is None:
            x = self._wacky_out_func(x, *args, **kwargs)

        #if training:
            #self._stop_wacky_recording()
        return x

    def _wacky_backprob(self, *args, **kwargs):
        self._start_wacky_recording()
        losses_returns = self._wacky_loss(*args, **kwargs)

        if isinstance(losses_returns, list):
            losses = losses_returns[0]
            losses_returns.pop(0)
        else:
            losses = losses_returns

        loss = self.loss_alpha * losses
        # loss = self.loss_alpha  * tf.nn.compute_average_loss(losses)

        self._wacky_tape._pop_tape()
        return loss, losses_returns, self._wacky_tape

    def _normal_backprob(self, *args, **kwargs):

        with tf.GradientTape() as tape:
            losses_returns = self._wacky_loss(*args, **kwargs)

            if isinstance(losses_returns, list):
                losses = losses_returns[0]
                losses_returns.pop(0)
            else:
                losses = losses_returns

            loss = self.loss_alpha * losses
        return loss, losses_returns, tape


    def train_step(self, *args, **kwargs):

        if self.use_wacky_backprob:
            loss, losses_returns, tape = self._wacky_backprob(*args, **kwargs)
        else:
            loss, losses_returns, tape = self._normal_backprob(*args, **kwargs)

        grads = tape.gradient(loss, self.trainable_variables)
        if self.grad_clip:
            grads = [tf.clip_by_norm(g, 0.5) for g in grads]
        self._wacky_optimizer.apply_gradients(zip(grads, self.trainable_variables))

        #self._wacky_tape._tape = None

        if isinstance(losses_returns, list):
            return [loss] + losses_returns
        else:
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
