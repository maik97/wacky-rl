import tensorflow as tf


class DiscreteActorAction:

    def __init__(self):
        pass

    def __call__(self, x):

        import tensorflow_probability as tfp
        dist = tfp.distributions.Categorical(probs=x.numpy(), dtype=tf.float32)

        action = dist.sample()
        act_prob = dist.prob(action)
        log_prob = dist.log_prob(action)

        return action, act_prob, log_prob


class DiscreteActorActionAlternative:

    def __init__(self):
        pass

    def __call__(self, x):

        action = tf.random.categorical(x[0], 1)[0, 0]
        act_prob = tf.squeeze(tf.nn.softmax(x[0])[0, action])
        log_prob = tf.math.log(act_prob)

        return action, act_prob, log_prob

