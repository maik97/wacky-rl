import tensorflow as tf
from tensorflow.keras import layers

from wacky_rl import losses

class WackyModel(tf.keras.Model):

    def __init__(
            self,
            inputs = None,
            outputs = None,
            model_name: str = 'UnnamedWackyModel',
            model_index: int = None,
            *args,
            **kwargs,
    ):

        super().__init__(*args, **kwargs, )

        if inputs is None != outputs is None:
            raise Exception('Both inputs and outputs must be assigned or None.')

        if not inputs is None and not outputs is None:
            self._wacky_layer = [tf.keras.Model(inputs, outputs)]
        elif not inputs is None and outputs is None:
            self._wacky_layer = inputs
        else:
            self._wacky_layer = []

        self.model_name = model_name
        self.model_index = model_index

    def add(self, layer):
        if isinstance(layer, list):
            for l in layer: self._wacky_layer.append(l)
        else:
            self._wacky_layer.append(layer)

    def pop(self, index):
        self._wacky_layer.pop(index)

    def mlp_network(self, num_units=64, activation='relu', dropout_rate=0.0):
        self.add(layers.Dense(num_units, activation=activation))
        if dropout_rate > 0.0:
            self.add(layers.Dropout(dropout_rate))
        self.add(layers.Dense(num_units, activation=activation))
        if dropout_rate > 0.0:
            self.add(layers.Dropout(dropout_rate))


    def compile(
            self,
            optimizer: (str, tf.keras.optimizers.Optimizer) = 'rmsprop',
            loss: (str, losses.WackyLoss, tf.keras.losses.Loss) = 'mse',
            metrics=None,
            loss_weights=None,
            weighted_metrics=None,
            run_eagerly=None,
            steps_per_execution=None,
            **kwargs
    ):

        # Loss Function:
        if isinstance(loss, losses.WackyLoss):
            self._wacky_loss = loss
        else:
            self._wacky_loss = None

        # Taken and modified from Keras:

        from tensorflow.python.keras.engine import compile_utils
        from tensorflow.python.keras.engine import base_layer
        base_layer.keras_api_gauge.get_cell('compile').set(True)
        with self.distribute_strategy.scope():

            # When compiling from an already-serialized model, we do not want to
            # reapply some processing steps (e.g. metric renaming for multi-output
            # models, which have prefixes added for each corresponding output name).
            from_serialized = kwargs.pop('from_serialized', False)

            self._validate_compile(optimizer, metrics, **kwargs)
            self._run_eagerly = run_eagerly

            self.optimizer = self._get_optimizer(optimizer)

            if self._wacky_loss is None:
                self.compiled_loss = compile_utils.LossesContainer(
                    loss, loss_weights, output_names=self.output_names)
            self.compiled_metrics = compile_utils.MetricsContainer(
                metrics, weighted_metrics, output_names=self.output_names,
                from_serialized=from_serialized)

            self._configure_steps_per_execution(steps_per_execution or 1)

            # Initializes attrs that are reset each time `compile` is called.
            self._reset_compile_cache()
            self._is_compiled = True

            if self._wacky_loss is None:
                self.loss = loss or {}  # Backwards compat.

    def _wacky_forward(self, x, training):
        if not len(self._wacky_layer) == 0:
            for l in self._wacky_layer:
                if not training:
                    if not isinstance(l, layers.Dropout):
                        x = l(x)
                else:
                    x = l(x)
            return x
        else:
            return super().__call__(inputs=x)

    def call(self, inputs, training=False, mask=None, *args, **kwargs):
        return self._wacky_forward(inputs, training)

    def train_step(self, inputs, *args, **kwargs):

        if self._wacky_loss is None:
            return super().train_step(inputs)

        with tf.GradientTape() as tape:
            x = self(inputs, training=True)
            loss = self._wacky_loss(x, *args, **kwargs)
            #loss = self.loss_alpha * self._wacky_loss(x, *args, **kwargs)

        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        return loss

    def predict_step(self, data, mask=None, *args, **kwargs):
        return self.call(data, training=False, mask=mask, *args, **kwargs)

    def fit(self, *args, **kwargs):
        if self._wacky_loss is None:
            return super().fit(*args, **kwargs)
        else:
            raise NotImplemented('The fit() method can not be used with a WackyLoss, use train_step() instead.')

    def train_on_batch(self, *args, **kwargs):
        if self._wacky_loss is None:
            return super().fit(*args, **kwargs)
        else:
            raise NotImplemented('The train_on_batch() method can not be used with a WackyLoss, use train_step() instead.')


class WackyDualingModel:

    def __init__(self, model: (WackyModel, tf.keras.Model)):

        self.model_1 = model
        self.model_2 = tf.keras.models.clone_model(model)

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

    def __call__(self, model: (WackyModel, tf.keras.Model), target: (WackyModel, tf.keras.Model)):

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
