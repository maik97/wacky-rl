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


class PPOActorLoss:

    def actor_loss(self, probs, actions, adv, old_probs, closs):
        probability = probs
        entropy = tf.reduce_mean(tf.math.negative(tf.math.multiply(probability, tf.math.log(probability))))
        # print(probability)
        # print(entropy)
        sur1 = []
        sur2 = []

        for pb, t, op in zip(probability, adv, old_probs):
            t = tf.constant(t)
            op = tf.constant(op)
            # print(f"t{t}")
            # ratio = tf.math.exp(tf.math.log(pb + 1e-10) - tf.math.log(op + 1e-10))
            ratio = tf.math.divide(pb, op)
            # print(f"ratio{ratio}")
            s1 = tf.math.multiply(ratio, t)
            # print(f"s1{s1}")
            s2 = tf.math.multiply(tf.clip_by_value(ratio, 1.0 - self.clip_pram, 1.0 + self.clip_pram), t)
            # print(f"s2{s2}")
            sur1.append(s1)
            sur2.append(s2)

        sr1 = tf.stack(sur1)
        sr2 = tf.stack(sur2)

        # closs = tf.reduce_mean(tf.math.square(td))
        loss = tf.math.negative(tf.reduce_mean(tf.math.minimum(sr1, sr2)) - closs + 0.001 * entropy)
        # print(loss)
        return loss

    def preprocess1(states, actions, rewards, done, values, gamma):
        g = 0
        lmbda = 0.95
        returns = []
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + gamma * values[i + 1] * done[i] - values[i]
            g = delta + gamma * lmbda * dones[i] * g
            returns.append(g + values[i])

        returns.reverse()
        adv = np.array(returns, dtype=np.float32) - values[:-1]
        adv = (adv - np.mean(adv)) / (np.std(adv) + 1e-10)
        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.int32)
        returns = np.array(returns, dtype=np.float32)
        return states, actions, returns, adv

    def test_reward(env):
        total_reward = 0
        state = env.reset()
        done = False
        while not done:
            action = np.argmax(agentoo7.actor(np.array([state])).numpy())
            next_state, reward, done, _ = env.step(action)
            state = next_state
            total_reward += reward

        return total_reward

