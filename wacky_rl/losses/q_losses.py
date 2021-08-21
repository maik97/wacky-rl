import tensorflow as tf
import wacky_rl


class SoftQLoss(wacky_rl.losses.WackyLoss):

    def __init__(self, scale: float = 2.0, gamma: float = 0.99):
        super().__init__()
        self.scale = scale
        self.gamma = gamma

    def __call__(self, prediction, future_vals, rewards, dones, weights=None):

        target = self.scale * tf.squeeze(rewards) + self.gamma * tf.squeeze(future_vals) * tf.squeeze(dones)
        target = tf.reshape(target, [-1,1])

        if not weights is None:
            return weights*tf.keras.losses.MSE(target, prediction)
        return tf.keras.losses.MSE(target, prediction)