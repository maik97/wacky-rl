import tensorflow as tf

class SoftQLoss:

    def __init__(self, scale:float = 2.0, gamma:float = 0.99):
        self.scale = scale
        self.gamma = gamma

    def __call__(self, batch_input, batch_next_inputs, action_as_input, rewards, one_minus_dones, q_model, target_val_model):

        #_, _, _, action_as_input = actor_model(batch_input, {'act_argmax': True})

        future_vals = target_val_model(batch_next_inputs)

        target_q = self.scale * tf.squeeze(rewards) + self.gamma * tf.squeeze(future_vals) * tf.squeeze(one_minus_dones)
        target_q = tf.reshape(target_q, [-1,1])

        action_as_input = tf.reshape(action_as_input, [-1, tf.shape(action_as_input)[-1]])
        pred_q = q_model([batch_input, action_as_input])

        return tf.keras.losses.MSE(target_q, pred_q)