import tensorflow as tf

class BaseActorLoss:

    def __init__(
            self,
            entropy_factor: float = None,
            loss_transform: str = None,
    ):
        if not entropy_factor is None:
            self.entropy_factor = tf.constant(entropy_factor, dtype=tf.float32)
        else:
            self.entropy_factor = None

        self.loss_transform = loss_transform

    def __call__(self, *args, **kwargs):
        pass

    def _add_entropy_loss(self, loss, act_probs, log_probs):
        if not self.entropy_factor is None:
            print(act_probs)
            print(log_probs)
            return loss + self.entropy_factor * tf.math.multiply(act_probs, log_probs)
        else:
            return loss

    def _return_loss(self, loss):

        loss = tf.squeeze(loss)

        if self.loss_transform == 'mean':
            return tf.math.reduce_mean(loss)

        if self.loss_transform == 'sum':
            return tf.math.reduce_sum(loss)

        return loss



class ActorLoss(BaseActorLoss):

    def __init__(
            self,
            entropy_factor: float = None,
            loss_transform: str = None,
    ):
        super().__init__(entropy_factor, loss_transform)

    def __call__(self, act_probs, log_probs, advantage):

        act_probs = tf.squeeze(act_probs)
        log_probs = tf.squeeze(log_probs)
        advantage = tf.squeeze(advantage)

        loss = tf.math.negative(tf.math.multiply(log_probs, advantage))
        loss = self._add_entropy_loss(loss, act_probs, log_probs)
        return self._return_loss(loss)


class SoftActorLoss(BaseActorLoss):

    def __init__(
            self,
            entropy_factor: float = None,
            loss_transform: str = None,
            train_with_argmax=False
    ):
        self.train_with_argmax = train_with_argmax
        super().__init__(entropy_factor, loss_transform)

    def __call__(self, batch_input, actor_model, q_model, dual_q_model=None):

        _, act_probs, log_probs, batch_action_as_input = actor_model.predict_step(batch_input, act_argmax=self.train_with_argmax)

        q = q_model.predict_step([batch_input, batch_action_as_input])

        if not dual_q_model is None:
            dual_q = dual_q_model.predict_step([batch_input, batch_action_as_input])
            q = tf.math.minimum(q, dual_q)


        loss = tf.squeeze(log_probs) - tf.squeeze(q)
        #loss = self._add_entropy_loss(loss, act_probs, log_probs)
        return self._return_loss(loss)


class PPOActorLoss(BaseActorLoss):

    def __init__(
            self,
            clip_param: float,
            entropy_factor: float = None,
            loss_transform: str = None,
            train_with_argmax=False
    ):
        self.clip_param = clip_param
        self.train_with_argmax = train_with_argmax
        super().__init__(entropy_factor, loss_transform)

    def __call__(self, probs, old_probs, advantage, critic_loss):

        entropy = tf.reduce_mean(tf.math.negative(tf.math.multiply(probability, tf.math.log(probability))))
        s_1, s_2 = self._calc_surrogates(probs, old_probs, advantage)

        loss = tf.math.negative(tf.reduce_mean(tf.math.minimum(s_1, s_2)) - critic_loss + 0.001 * entropy)
        return loss

    def _calc_surrogates(self, probs, old_probs, advantage):
        ratios = tf.math.divide(probs, old_probs)
        sur_1 = tf.math.multiply(ratios, advantage)
        sur_2 = tf.math.multiply(tf.clip_by_value(ratios, 1.0 - self.clip_param, 1.0 + self.clip_param), advantage)
        return sur_1, sur_2

    def _calc_surrogates_alternative(self, probs, old_probs, advantage):
        sur1 = []
        sur2 = []

        for pb, t, op in zip(probs, advantage, old_probs):
            t = tf.constant(t)
            op = tf.constant(op)
            # print(f"t{t}")
            # ratio = tf.math.exp(tf.math.log(pb + 1e-10) - tf.math.log(op + 1e-10))
            ratio = tf.math.divide(pb, op)
            # print(f"ratio{ratio}")
            s1 = tf.math.multiply(ratio, t)
            # print(f"s1{s1}")
            s2 = tf.math.multiply(tf.clip_by_value(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param), t)
            # print(f"s2{s2}")
            sur1.append(s1)
            sur2.append(s2)

        return tf.stack(sur1), tf.stack(sur2)

import numpy as np
class LamdaTransformReturns:

    def __init__(
            self,
            gamma: float = 0.99,
            lamda: float = 0.95,
    ):

        self.gamma = gamma
        self.lamda = lamda

    def __call__(self, rewards, dones, values):
        g = 0
        returns = []
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * values[i + 1] * dones[i] - values[i]
            g = delta + self.gamma * self.lamda * dones[i] * g
            returns.append(g + values[i])

        returns.reverse()
        adv = np.array(returns, dtype=np.float32) - values[:-1]
        adv = (adv - np.mean(adv)) / (np.std(adv) + 1e-10)

        return tf.stack(adv)


