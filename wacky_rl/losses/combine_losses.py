import tensorflow as tf

def pre_calced_loss(loss):
    return loss

class SumMultipleLosses:

    def __init__(self, alpha_list: list = None):

        if not alpha_list is None:
            self.alphas = tf.stack([tf.constant(alpha, dtype=tf.float32) for alpha in alpha_list])
        else:
            self.alphas = None

    def __call__(self, loss_list):

        loss_list = tf.stack(tf.squeeze(loss_list))

        if not self.alphas is None:
            loss_list = tf.math.multiply(self.alphas, loss_list)

        return tf.expand_dims(tf.reduce_sum(loss_list))
