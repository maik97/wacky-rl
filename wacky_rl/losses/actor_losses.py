import tensorflow as tf
from wacky_rl import losses

class ActorLoss(losses.WackyLoss):

    def __init__(self, entropy_factor: float = 0.0):
        super().__init__()
        self.entropy_factor = entropy_factor

    def __call__(self, prediction, actions, advantage, *args, **kwargs):

        dist = prediction
        actions = tf.reshape(actions, [-1, len(actions)])
        log_probs = dist.calc_log_probs(actions)
        entropies = dist.calc_entropy(actions)

        if dist.num_actions > 1:
            losses = []
            for i in range(dist.num_actions):
                policy_loss = tf.reduce_mean(tf.math.negative(tf.math.multiply(log_probs[i], advantage)))
                entropy_loss = self.entropy_factor * entropies[i]
                losses.append(policy_loss + entropy_loss)

            loss = tf.reduce_mean(tf.stack(losses))

        else:
            policy_loss = tf.reduce_mean(tf.math.negative(tf.math.multiply(log_probs, advantage)))
            entropy_loss = self.entropy_factor * entropies
            loss = policy_loss + entropy_loss

        return loss


class SoftActorLoss(losses.WackyLoss):

    def __init__(self, entropy_factor: float = 0.0):
        super().__init__()
        self.entropy_factor = entropy_factor

    def __call__(self, prediction, actions, q, *args, **kwargs):

        dist = prediction
        actions = tf.reshape(actions, [-1, len(actions)])
        log_probs = dist.calc_log_probs(actions)
        entropies = dist.calc_entropy(actions)

        if dist.num_actions > 1:
            losses = []
            for i in range(dist.num_actions):
                policy_loss = tf.reduce_mean(tf.squeeze(log_probs[i]) - tf.squeeze(q))
                entropy_loss = self.entropy_factor * entropies[i]
                losses.append(policy_loss + entropy_loss)

            loss = tf.reduce_mean(tf.stack(losses))

        else:
            policy_loss = tf.reduce_mean(tf.squeeze(log_probs) - tf.squeeze(q))
            entropy_loss = self.entropy_factor * entropies
            loss = policy_loss + entropy_loss

        return loss


class PPOActorLoss(losses.WackyLoss):

    def __init__(self, clip_param: float = 0.2, entropy_factor: float = 0.0):
        super().__init__()
        self.clip_param = clip_param
        self.entropy_factor = entropy_factor

    def __call__(self, prediction, actions, old_probs, advantage, *args, **kwargs):

        dist = prediction
        actions = tf.stop_gradient(tf.reshape(actions, [-1, len(actions)]))
        probs = dist.calc_probs(actions)
        entropies = dist.calc_entropy(actions)
        old_probs = tf.stop_gradient(tf.reshape(old_probs, tf.shape(probs)))
        advantage = tf.stop_gradient(advantage)

        probs = tf.clip_by_value(probs, 1e-10, 1.0)
        old_probs = tf.clip_by_value(old_probs, 1e-10, 1.0)

        if dist.num_actions > 1:
            losses = []
            for i in range(dist.num_actions):
                s_1, s_2 = self._calc_surrogates(tf.stack(probs[i]), tf.stack(old_probs[i]), advantage)
                policy_loss = tf.reduce_mean(tf.math.negative(tf.math.minimum(s_1, s_2)))
                entropy_loss = self.entropy_factor * entropies[i]
                losses.append(policy_loss + entropy_loss)

            loss = tf.reduce_mean(tf.stack(losses))

        else:
            s_1, s_2 = self._calc_surrogates(tf.stack(probs), tf.stack(old_probs), advantage)
            policy_loss = tf.reduce_mean(tf.math.negative(tf.math.minimum(s_1, s_2)))
            entropy_loss = self.entropy_factor * entropies
            loss = policy_loss + entropy_loss

        return loss

    def _calc_surrogates(self, probs, old_probs, advantage):

        probs = tf.cast(tf.squeeze(probs), dtype=tf.float32)
        old_probs = tf.cast(tf.squeeze(old_probs), dtype=tf.float32)
        advantage = tf.cast(tf.squeeze(advantage), dtype=tf.float32)

        ratios = tf.math.exp(tf.math.log(probs + 1e-10) - tf.math.log(old_probs + 1e-10))

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

            #ratio = tf.math.divide(pb, op)
            ratio = tf.math.exp(tf.math.log(pb + 1e-10) - tf.math.log(op + 1e-10))
            s1 = tf.math.multiply(ratio, t)
            s2 = tf.math.multiply(tf.clip_by_value(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param), t)

            sur1.append(s1)
            sur2.append(s2)

        return tf.stack(sur1), tf.stack(sur2)




