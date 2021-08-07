import tensorflow as tf


class SoftValueLoss:

    def __init__(self):
        pass

    def __call__(self, batch_input, batch_action_as_input, log_probs, q_models, val_model):

        if not isinstance(q_models, list):
            q_models = [q_models]

        q_list = [tf.squeeze(q_model([batch_input, batch_action_as_input])) for q_model in q_models]
        q = tf.stack(q_list, axis=-1)
        q = tf.math.reduce_min(q, axis=1)

        target = q - tf.squeeze(log_probs)
        return tf.keras.losses.MSE(target, tf.squeeze(val_model(batch_input)))