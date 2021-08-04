import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


def discrete_actor_loss(act_log_probs, act_probs, advantage, loss_critic, entropy_factor=tf.constant(0.0001, dtype=tf.float32)):

    loss_actor = -(tf.squeeze(act_log_probs) * tf.squeeze(advantage))

    if not entropy_factor is None:
        loss_actor = loss_actor + (entropy_factor * (tf.math.multiply(tf.squeeze(act_probs), tf.squeeze(act_log_probs))))

    return tf.expand_dims(tf.squeeze(loss_actor),1)


def discrete_actor_loss_reduce_mean(act_log_probs, act_probs, advantage, loss_critic, entropy_factor=tf.constant(0.0001, dtype=tf.float32)):

    loss_actor = -tf.math.reduce_mean(tf.squeeze(act_log_probs) * tf.squeeze(advantage))

    if not entropy_factor is None:
        loss_actor = loss_actor + (entropy_factor * (tf.math.multiply(tf.squeeze(act_probs), tf.squeeze(act_log_probs))))

    return loss_actor


def discrete_actor_loss_with_tfp(act_log_probs, act_probs, advantage, loss_critic, entropy_factor=tf.constant(0.0001, dtype=tf.float32)):

    policy_loss = tf.math.multiply(tf.squeeze(act_log_probs), tf.squeeze(advantage))
    entropy_loss = tf.math.negative(tf.math.multiply(tf.squeeze(act_probs), tf.squeeze(act_log_probs)))

    policy_loss = tf.reduce_mean(policy_loss)
    entropy_loss = tf.reduce_mean(entropy_loss)

    if not entropy_factor is None:
        return -policy_loss - entropy_factor * entropy_loss
    else:
        return -policy_loss
