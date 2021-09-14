import tensorflow as tf
from wacky_rl import losses
from itertools import count


class SoftValueLoss(losses.WackyLoss):
    _ids = count(0)

    def __init__(self, logger=None):
        super().__init__()
        self.logger = logger
        self.id = next(self._ids)

    def __call__(self, prediction, log_probs, q):
        target = tf.reshape(tf.squeeze(q) - tf.squeeze(log_probs), [-1, 1])
        loss= tf.reduce_mean(tf.keras.losses.MSE(target, prediction))

        if not self.logger is None:
            self.logger.log_mean('soft_q_loss_' + str(self.id), loss)

        return loss

class MeanSquaredErrorLoss(losses.WackyLoss):
    _ids = count(0)

    def __init__(self, logger=None):
        super().__init__()
        self.logger = logger
        self.id = next(self._ids)

    def __call__(self, prediction, target):
        target = tf.reshape(target, [-1,1])
        loss = tf.reduce_mean(tf.keras.losses.MSE(target, prediction))

        if not self.logger is None:
            self.logger.log_mean('mse_loss_'+str(self.id), loss)

        return loss
