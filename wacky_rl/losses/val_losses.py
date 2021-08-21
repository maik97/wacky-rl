import tensorflow as tf
import wacky_rl


class SoftValueLoss(wacky_rl.losses.WackyLoss):

    def __init__(self):
        super().__init__()

    def __call__(self, prediction, log_probs, q):
        target = tf.reshape(tf.squeeze(q) - tf.squeeze(log_probs), [-1, 1])
        return tf.reduce_mean(tf.keras.losses.MSE(target, prediction))

class MeanSquaredErrorLoss(wacky_rl.losses.WackyLoss):

    def __init__(self):
        super().__init__()

    def __call__(self, prediction, returns):
        target = tf.reshape(returns, [-1,1])
        return tf.reduce_mean(tf.keras.losses.MSE(target, prediction))
