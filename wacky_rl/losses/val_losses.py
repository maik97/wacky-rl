import tensorflow as tf


class SoftValueLoss:

    def __init__(self, train_with_argmax=True):
        self.train_with_argmax = train_with_argmax

    def __call__(self, batch_input, actor_model, q_models, val_model):

        _, _, log_probs, batch_action_as_input = actor_model(batch_input, {'act_argmax': self.train_with_argmax})

        if not isinstance(q_models, list):
            q_models = [q_models]

        q_list = [tf.squeeze(q_model([batch_input, batch_action_as_input])) for q_model in q_models]
        #q = tf.stack(q_list, axis=-1)
        #q = tf.math.reduce_min(q, axis=1)
        q = tf.math.minimum(tf.squeeze(q_list[0]), tf.squeeze(q_list[1]))

        target = tf.reshape(q - tf.squeeze(log_probs), [-1,1])
        pred = val_model(batch_input)

        return tf.keras.losses.MSE(target, pred)