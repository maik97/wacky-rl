import tensorflow as tf

class DiscreteActorLoss:

    def __init__(
            self,
            entropy_factor: float = 0.0001,
            loss_transform: str = None,
    ):
        if not entropy_factor is None:
            self.entropy_factor = tf.constant(entropy_factor, dtype=tf.float32)

        self.loss_transform = loss_transform

    def __call__(self, act_probs, log_probs, advantage):

        act_probs = tf.squeeze(act_probs)
        log_probs = tf.squeeze(log_probs)
        advantage = tf.squeeze(advantage)

        loss_actor = tf.math.negative(tf.math.multiply(log_probs, advantage))

        if not self.entropy_factor is None:
            entropy_loss = self.entropy_factor * tf.math.multiply(act_probs, log_probs)
            loss_actor = loss_actor + entropy_loss

        if self.loss_transform == 'mean':
            return tf.expand_dims(tf.math.reduce_mean(loss_actor))

        if self.loss_transform == 'sum':
            return tf.expand_dims(tf.math.reduce_sum(loss_actor))

        return tf.expand_dims(loss_actor)

