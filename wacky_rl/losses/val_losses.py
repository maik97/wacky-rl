import tensorflow as tf
from wacky_rl import losses


class SoftValueLoss(losses.WackyLoss):

    def __init__(self):
        super().__init__()

    def __call__(self, prediction, log_probs, q):
        target = tf.reshape(tf.squeeze(q) - tf.squeeze(log_probs), [-1, 1])
        return tf.reduce_mean(tf.keras.losses.MSE(target, prediction))

class MeanSquaredErrorLoss(losses.WackyLoss):

    def __init__(self):
        super().__init__()

    def __call__(self, prediction, target):
        target = tf.reshape(target, [-1,1])
        return tf.reduce_mean(tf.keras.losses.MSE(target, prediction))
