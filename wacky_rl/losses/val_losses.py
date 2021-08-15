import tensorflow as tf
import wacky_rl


class SoftValueLoss:

    def __init__(self, train_with_argmax=False):
        self.train_with_argmax = train_with_argmax

    def __call__(self, batch_input, actor_model, val_model, q_model, dual_q_model=None):

        _, _, log_probs, batch_action_as_input = actor_model.predict_step(batch_input, act_argmax=self.train_with_argmax)

        q = q_model.predict_step([batch_input, batch_action_as_input])

        if not dual_q_model is None:
            dual_q = dual_q_model.predict_step([batch_input, batch_action_as_input])
            q = tf.math.minimum(q, dual_q)

        target = tf.reshape(tf.squeeze(q) - tf.squeeze(log_probs), [-1,1])
        pred = val_model.predict_step(batch_input)

        return tf.keras.losses.MSE(target, pred)

class PPOCriticLoss:

    def __init__(self):
        pass

    def __call__(self, critic, batch_input, returns):

        #print()

        values = critic.predict_step(batch_input)
        #print(tf.keras.losses.MSE(tf.reshape(returns, [-1,1]), values))
        #exit()


        #return tf.reduce_mean(tf.keras.losses.MSE(tf.reshape(returns, [-1,1]), values))
        return tf.keras.losses.MSE(tf.reshape(returns, [-1,1]), values)