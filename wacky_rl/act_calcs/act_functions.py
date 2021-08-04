import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

def act_discrete_actor_critic(actions):

    action = tf.random.categorical(actions[0], 1)[0, 0]
    act_prob = tf.squeeze(tf.nn.softmax(actions[0])[0, action])
    log_prob = tf.math.log(act_prob)


    return action, act_prob, log_prob


def act_discrete_actor_critic_with_tfp(actions):

    dist = tfp.distributions.Categorical(probs=actions.numpy(), dtype=tf.float32)

    action = dist.sample()
    act_prob = dist.prob(action)
    log_prob = dist.log_prob(action)

    return action, act_prob, log_prob