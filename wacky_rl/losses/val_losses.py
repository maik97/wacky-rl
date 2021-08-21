import tensorflow as tf


class SoftValueLoss:

    def __init__(self):
        pass

    def __call__(self, prediction, log_probs, q):
        target = tf.reshape(tf.squeeze(q) - tf.squeeze(log_probs), [-1, 1])
        return tf.reduce_mean(tf.keras.losses.MSE(target, prediction))

class MeanSquaredErrorLoss:

    def __init__(self):
        pass

    def __call__(self, prediction, returns):
        target = tf.reshape(returns, [-1,1])
        return tf.reduce_mean(tf.keras.losses.MSE(target, prediction))
