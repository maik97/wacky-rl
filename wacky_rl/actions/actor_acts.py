import tensorflow as tf
import tensorflow_probability as tfp

class DiscreteActorActionAlternative:

    def __init__(self):
        pass

    def __call__(self, x):

        dist = tfp.distributions.Categorical(probs=x.numpy(), dtype=tf.float32)

        action = dist.sample()
        act_prob = dist.prob(action)
        log_prob = dist.log_prob(action)

        return action, act_prob, log_prob


class DiscreteActorAction:

    def __init__(self):
        pass

    def __call__(self, x, training=None):

        x = tf.squeeze(x)

        if not training is None:
            act_argmax = training['act_argmax']
        else:
            act_argmax = False

        if act_argmax:
            action = tf.squeeze(tf.math.argmax(x))
        else:
            action = tf.squeeze(tf.random.categorical(tf.expand_dims(x, 0), 1))

        act_prob = tf.squeeze(tf.nn.softmax(tf.expand_dims(x, 0))[0, action])
        log_prob = tf.squeeze(tf.math.log(act_prob))

        return action, act_prob, log_prob


class ContinActorAction:

    def __init__(
            self,
            reparam: bool = False,
            rp=1e-6,
            log_prob_transform: str = 'sum',
    ):
        self.reparam = reparam
        self.rp =rp
        self.log_prob_transform = log_prob_transform

    def __call__(self, x, training=None):

        mu, sigma = x
        mu = tf.squeeze(mu)
        sigma = tf.clip_by_value(tf.squeeze(sigma), self.rp, 1)

        act_probs_dist = tfp.distributions.Normal(mu, sigma)

        actions = act_probs_dist.sample()

        if self.reparam:
            actions += tf.random.normal(shape=tf.shape(actions), mean=0.0, stddev=0.1)

        action = tf.squeeze(tf.math.tanh(actions))
        act_probs = tf.squeeze(act_probs_dist.prob(actions))
        log_probs = tf.squeeze(act_probs_dist.log_prob(actions))
        log_probs = log_probs - tf.math.log(1 - tf.math.pow(action, 2) + self.rp)

        if self.log_prob_transform == 'sum':
            return action, tf.math.reduce_mean(act_probs), tf.math.reduce_sum(log_probs)

        if self.log_prob_transform == 'mean':
            return action, tf.math.reduce_mean(act_probs), tf.math.reduce_sum(log_probs)

        return action, tf.math.reduce_mean(act_probs), log_probs
