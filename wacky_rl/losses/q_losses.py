import tensorflow as tf

class SoftQLoss:

    def __init__(self, scale:float = 1.0, gamma:float = 0.99):
        self.scale = scale
        self.gamma = gamma

    def __call__(self, batch_input, batch_next_inputs, rewards, dones, actor_model, q_model, target_q_model):

        _, _, _, action_as_input = actor_model(batch_input)
        _, _, _, next_action_as_input = actor_model(batch_next_inputs)

        future_vals = tf.squeeze(target_q_model([batch_next_inputs, next_action_as_input]))

        target_q = self.scale * tf.squeeze(rewards) + self.gamma * future_vals * (1-tf.squeeze(dones))
        return tf.keras.losses.MSE(target_q, tf.squeeze(q_model([batch_input, action_as_input])))