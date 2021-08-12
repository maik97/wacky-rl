import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

class BaseActionLayer(tf.keras.layers.Layer):

    def __init__(
            self,
            return_action=True,
            return_act_prob=True,
            return_log_prob=True,
            *args,
            **kwargs):
        super().__init__( *args, **kwargs)
        self.return_action = return_action
        self.return_act_prob = return_act_prob
        self.return_log_prob = return_log_prob

    def call(self, inputs, act_argmax, *args, **kwargs):
        raise NotImplementedError('')

    def _return_outputs(self, action=None, act_prob=None, log_prob=None):

        outputs = []

        if not action is None and self.return_action:
            outputs.append(action)

        if not act_prob is None and self.return_act_prob:
            outputs.append(act_prob)

        if not log_prob is None and self.return_log_prob:
            outputs.append(log_prob)

        if len(outputs) == 0:
            return None
        elif len(outputs) == 1:
            return outputs[0]
        else:
            return outputs

class DiscreteActionLayer(BaseActionLayer):

    def __init__(self):
        super().__init__()


    def call(self, inputs, act_argmax=False):
        #print(inputs)

        #act_argmax = True
        if act_argmax:
            action = tf.math.argmax(inputs, axis=-1)
        else:
            action = tf.squeeze(tf.random.categorical(inputs, num_samples=1), axis=1)

        act_prob = tf.gather_nd(tf.nn.softmax(inputs), tf.stack([np.arange(len(action)), action], axis=1))
        log_prob = tf.math.log(act_prob)

        return self._return_outputs(action, act_prob, log_prob)

class SoftDiscreteActionLayer(DiscreteActionLayer):

    def __init__(self, n_actions):
        super().__init__()
        self.n_actions = n_actions

    def __call__(self, inputs, act_argmax=False):
        action, act_prob, log_prob = super().call(inputs, act_argmax)
        action_as_input = tf.one_hot(action, self.n_actions)
        return [action, act_prob, log_prob, action_as_input]

class ContinActionLayer(BaseActionLayer):

    def __init__(
            self,
            reparam: bool = False,
            rp=1e-6,
            log_prob_transform: str = 'sum',
    ):

        super().__init__()
        self.reparam = reparam
        self.rp =rp
        self.log_prob_transform = log_prob_transform

    def call(self, inputs, act_argmax=False):

        mu, sigma = inputs
        mu = tf.squeeze(mu)
        sigma = tf.clip_by_value(tf.squeeze(sigma), self.rp, 1)

        act_probs_dist = tfp.distributions.Normal(mu, sigma)

        actions = act_probs_dist.sample()

        if self.reparam:
            actions += tf.random.normal(shape=tf.shape(actions), mean=0.0, stddev=0.1)

        action = tf.math.tanh(actions)
        #action = tf.squeeze(action)
        act_probs = tf.squeeze(act_probs_dist.prob(actions))
        log_probs = tf.squeeze(act_probs_dist.log_prob(actions))
        log_probs = log_probs - tf.math.log(1 - tf.math.pow(action, 2) + self.rp)

        if self.log_prob_transform == 'sum':
            return action, act_probs, tf.math.reduce_sum(log_probs)

        if self.log_prob_transform == 'mean':
            return action, act_probs, tf.math.reduce_sum(log_probs)

        return self._return_outputs(action, act_probs, log_probs)

class SoftContinActionLayer(ContinActionLayer):

    def __init__(
            self,
            reparam: bool = False,
            rp=1e-6,
            log_prob_transform: str = 'sum'
    ):
        super().__init__(reparam, rp, log_prob_transform)

    def call(self, inputs, act_argmax=False):
        action, act_prob, log_prob = super().call(inputs, act_argmax)
        action_as_input = tf.reshape(action, [-1,tf.shape(action)[-1]])
        return [action, act_prob, log_prob, action_as_input]
