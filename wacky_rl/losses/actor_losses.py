import tensorflow as tf

class ActorLoss:

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

    def __call__(self, act_probs, log_probs, advantage):

        act_probs = tf.squeeze(act_probs)
        log_probs = tf.squeeze(log_probs)
        advantage = tf.squeeze(advantage)

        loss_actor = tf.math.negative(tf.math.multiply(log_probs, advantage))

        if not self.entropy_factor is None:
            entropy_loss = self.entropy_factor * tf.math.multiply(act_probs, log_probs)
            loss_actor = loss_actor + entropy_loss

        loss_actor = tf.squeeze(loss_actor)

        if self.loss_transform == 'mean':
            return tf.math.reduce_mean(loss_actor)

        if self.loss_transform == 'sum':
            return tf.math.reduce_sum(loss_actor)

        return tf.expand_dims(loss_actor, 1)

class SoftActorLoss:

    def __init__(self, train_with_argmax=True):
        self.train_with_argmax = train_with_argmax

    def __call__(self, batch_input, actor_model, q_models):

        _, _, log_probs, batch_action_as_input = actor_model(batch_input, {'act_argmax': self.train_with_argmax})

        if not isinstance(q_models, list):
            q_models = [q_models]

        q_list = [tf.squeeze(q_model([batch_input, batch_action_as_input])) for q_model in q_models]
        # q = tf.stack(q_list, axis=-1)
        # q = tf.math.reduce_min(q, axis=1)
        q = tf.math.minimum(tf.squeeze(q_list[0]), tf.squeeze(q_list[1]))

        loss = tf.squeeze(log_probs) - q

        return tf.reduce_mean(loss)


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

