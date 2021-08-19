import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

class DiscreteActorActionAlternative:

    def __init__(self):
        pass

    def __call__(self, x):

        dist = tfp.distributions.Categorical(probs=x.numpy(), dtype=tf.float32)

        action = dist.sample()
        act_prob = dist.prob(action)
        log_prob = dist.log_prob(action)

        return [action, act_prob, log_prob]


class DiscreteActorAction:

    def __init__(
            self,
            return_act_prob: bool = True,
            return_log_prob: bool = True,
            return_action_as_inputs: bool = False,
            return_dist=False,
    ):
        self.return_act_prob = return_act_prob
        self.return_log_prob = return_log_prob
        self.return_action_as_inputs = return_action_as_inputs
        self.return_dist = return_dist

    def __call__(self, x, act_argmax=False):

        if isinstance(x, list):
            x = x[0]

        if act_argmax:
            action = tf.math.argmax(x, axis=-1)
        else:
            action = tf.squeeze(tf.random.categorical(x, num_samples=1), axis=1)

        out_list = [action]

        if self.return_act_prob:
            act_prob = self.calc_act_prob(x, action)
            out_list.append(act_prob)

        if self.return_log_prob:
            if not self.return_act_prob:
                act_prob = self.calc_act_prob(x, action)
            log_prob = self.calc_log_prob(act_prob)
            out_list.append(log_prob)

        if self.return_action_as_inputs:
            action_as_input = self.calc_action_as_input(action, len(x))
            out_list.append(action_as_input)

        if self.return_dist:
            dist = tf.nn.softmax(x)
            out_list.append(dist)

        if len(out_list) > 1:
            return out_list
        else:
            return out_list[0]

    def calc_act_prob(self, x, action):
        return tf.gather_nd(tf.nn.softmax(x), tf.stack([np.arange(len(action)), action], axis=1))

    def calc_log_prob(self, act_prob):
        return tf.math.log(act_prob)

    def calc_action_as_input(self, action, n_actions):
        tf.one_hot(action, n_actions)


class SoftDiscreteActorAction(DiscreteActorAction):

    def __init__(self, n_actions):
        super().__init__()
        self.n_actions = n_actions

    def __call__(self, x, act_argmax=False):
        action, act_prob, log_prob = super().__call__(x, act_argmax)
        action_as_input = tf.one_hot(action, self.n_actions)
        return [action, act_prob, log_prob, action_as_input]


class ContinActionDistribution:

    def __init__(
            self,
            min_log_sigma=-20,
            max_log_sigma=0,
    ):
        self.min_log_sigma = min_log_sigma
        self.max_log_sigma = max_log_sigma

    def __call__(self, mu, log_sigma):
        mu = tf.squeeze(mu)
        log_sigma = tf.squeeze(log_sigma)
        log_sigma = self.min_log_sigma + 0.5 * (self.max_log_sigma - self.max_log_sigma) * (log_sigma + 1)
        sigma = tf.exp(log_sigma)
        return tfp.distributions.Normal(mu, sigma)


class ContinActorAction:

    def __init__(
            self,
            reparam: bool = False,
            rp=1e-6,
            log_prob_transform: str = 'sum',

            transform_action: bool = True,
            return_act_prob: bool = True,
            return_log_prob: bool = True,
            return_action_as_inputs: bool = False,
            return_dist = False,
    ):
        self.reparam = reparam
        self.rp =rp
        self.log_prob_transform = log_prob_transform

        self.transform_action = transform_action
        self.return_act_prob = return_act_prob
        self.return_log_prob = return_log_prob
        self.return_action_as_inputs = return_action_as_inputs
        self.return_dist = return_dist

        self.calc_dist = ContinActionDistribution()

    def __call__(self, x, act_argmax=False):

        mu, sigma = x
        print(mu)
        print(sigma)
        mu = tf.squeeze(mu)
        sigma = tf.clip_by_value(tf.squeeze(sigma), self.rp, 1)

        #act_probs_dist = tfp.distributions.Normal(mu, sigma)
        #act_probs_dist = self.calc_dist(mu, sigma)

        act_probs_dist = tfp.distributions.Normal(mu, sigma)
        print(act_probs_dist)
        exit()
        #act_probs_dist = self.calc_dist(mu, sigma)

        if act_argmax:
            actions = act_probs_dist.mean()
        else:
            actions = act_probs_dist.sample()

        #actions = tf.clip_by_value(actions, 0.0, 1.0)

        if self.reparam:
            actions += tf.random.normal(shape=tf.shape(actions), mean=0.0, stddev=0.1)


        #tanh_actions = tf.math.tanh(actions)

        #if self.transform_action:
            #out_list = [tanh_actions]
        #else:
            #out_list = [actions]

        out_list = [actions]

        if self.return_act_prob:
            act_probs = self.calc_act_prob(act_probs_dist, actions)
            out_list.append(act_probs)

        if self.return_log_prob:
            log_probs = self.calc_log_prob(act_probs_dist, actions)
            #log_probs = self.calc_log_prob_old(act_probs_dist, actions, tanh_actions)
            out_list.append(log_probs)

        if self.return_action_as_inputs:
            action_as_input = self.calc_action_as_input(out_list[0])
            out_list.append(action_as_input)

        if self.return_dist:
            out_list.append(act_probs_dist)

        #print(len(out_list))

        if len(out_list) > 1:
            return out_list
        else:
            return out_list[0]


    def calc_act_prob(self, act_probs_dist, actions):
        return tf.squeeze(act_probs_dist.prob(actions))

    def calc_log_prob(self, act_probs_dist, actions):
        return tf.squeeze(act_probs_dist.log_prob(actions))

    def calc_log_prob_old(self, act_probs_dist, actions, tanh_actions=None):
        log_probs = tf.squeeze(act_probs_dist.log_prob(actions))
        if not tanh_actions is None:
            log_probs = log_probs - tf.math.log(1 - tf.math.pow(tanh_actions, 2) + self.rp)

        if self.log_prob_transform == 'sum':
            return tf.math.reduce_sum(log_probs)
        if self.log_prob_transform == 'mean':
            return tf.math.reduce_sum(log_probs)
        return log_probs

    def calc_action_as_input(self, actions):
        return tf.reshape(actions, [-1, tf.shape(actions)[-1]])

class SoftContinActorAction(ContinActorAction):

    def __init__(
            self,
            reparam: bool = False,
            rp=1e-6,
            log_prob_transform: str = 'sum'
    ):
        super().__init__(reparam, rp, log_prob_transform)

    def __call__(self, x, act_argmax=False):
        action, act_prob, log_prob = super().__call__(x, act_argmax)
        action_as_input = tf.reshape(action, [-1,tf.shape(action)[-1]])
        #print(tf.shape(action_as_input))
        #action_as_input = tf.expand_dims(action, 0)
        #print(x)
        #print(action_as_input)
        #exit()
        return [action, act_prob, log_prob, action_as_input]
