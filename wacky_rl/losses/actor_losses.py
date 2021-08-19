import numpy as np
import tensorflow as tf
import wacky_rl
import wacky_rl.transform


class BaseActorLoss:
    _forward_required = False

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

    _forward_required = False

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

    _forward_required = True

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

    _forward_required = True

    def __init__(
            self,
            is_discrete: bool = True,
            clip_param: float = 0.2,
            entropy_factor: float = None,
            loss_transform: str = None,
            train_with_argmax=False
    ):
        self.is_discrete = is_discrete
        self.clip_param = clip_param
        self.train_with_argmax = train_with_argmax
        super().__init__(entropy_factor, loss_transform)

    #def __call__(self, actor, actions, batch_input, old_probs, advantage, critic_loss=None):
    def __call__(self, actor, actions, batch_input, old_probs, advantage, returns, critic):

        if self.is_discrete:
            _, _, _, dist = actor.predict_step(batch_input, act_argmax=False)
            indices = tf.cast(tf.stack([np.arange(len(actions)), tf.squeeze(actions)], axis=1), dtype=tf.int32)
            probs = tf.gather_nd(dist, indices)
            log_probs = tf.math.log(probs)

        else:
            dist = actor.predict_step(batch_input, act_argmax=False)[0]
            #_, _, _, dist = actor.predict_step(batch_input, act_argmax=False)
            #print(actions)
            probs = dist.calc_probs(tf.reshape(actions, [-1, len(actions)]))
            #probs = tf.squeeze(dist.prob(actions))
            #log_probs = tf.squeeze(dist.log_prob(actions))
            ##log_probs = log_probs - tf.math.log(1 - tf.math.pow(tf.math.tanh(actions), 2) + 1e-6)

        #entropy = tf.reduce_mean(tf.math.negative(tf.math.multiply(probs, log_probs)))
        #entropy = tf.math.negative(tf.math.multiply(probs, log_probs))
        #print(entropy)
        #print(old_probs)
        #print(probs)
        #print(np.mean(probs.numpy()))
        #exit()

        losses = []
        for i in range(dist.num_actions):
            s_1, s_2 = self._calc_surrogates_alternative(tf.stack(probs[i]), tf.stack(old_probs[i]), advantage)
            losses.append(tf.reduce_mean(tf.math.negative(tf.math.minimum(s_1, s_2))))

        losses = tf.reduce_mean(tf.stack(losses))
        #losses = tf.squeeze(tf.math.negative(tf.reduce_mean(tf.math.minimum(s_1, s_2)) + 0.001 * entropy))
        #losses = - tf.reduce_mean(tf.math.minimum(s_1, s_2))
        #losses = losses + self.entropy_factor * tf.reduce_mean(dist.entropy())
        #losses = self._add_entropy_loss(losses, probs, log_probs)

        critic_loss = critic.train_step(critic, batch_input, returns)
        losses = losses + 0.05 * tf.reduce_mean(critic_loss)

        #if not critic_loss is None:
            #losses = losses - tf.reduce_mean(self.entropy_factor * critic_loss)

        return losses

    def _calc_surrogates(self, log_probs, log_old_probs, advantage):

        log_probs = tf.squeeze(log_probs)
        log_old_probs = tf.squeeze(log_old_probs)
        advantage = tf.squeeze(advantage)
        #ratios = tf.math.divide_no_nan(log_probs, log_old_probs)

        ratios = tf.math.exp(log_probs - log_old_probs)

        sur_1 = tf.math.multiply_no_nan(ratios, advantage)
        sur_2 = tf.math.multiply_no_nan(tf.clip_by_value(ratios, 1.0 - self.clip_param, 1.0 + self.clip_param), advantage)
        return sur_1, sur_2

    def _calc_surrogates_alternative(self, probs, old_probs, advantage):
        sur1 = []
        sur2 = []

        probs = tf.squeeze(probs)
        old_probs = tf.squeeze(old_probs)
        advantage = tf.cast(tf.squeeze(advantage), dtype=tf.float32)

        for pb, t, op in zip(probs, advantage, old_probs):
            t = tf.constant(t)
            op = tf.constant(op)

            #ratio = tf.math.exp(tf.math.log(pb + 1e-10) - tf.math.log(op + 1e-10))
            ratio = tf.math.divide(pb, op)
            s1 = tf.math.multiply(ratio, t)
            s2 = tf.math.multiply(tf.clip_by_value(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param), t)

            sur1.append(s1)
            sur2.append(s2)

        return tf.stack(sur1), tf.stack(sur2)




