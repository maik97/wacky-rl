import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Dense, Input

from wacky_rl.agents import AgentCore
from wacky_rl.models import WackyModel
from wacky_rl.memory import BufferMemory

from wacky_rl.losses import PPOActorLoss, MeanSquaredErrorLoss
from wacky_rl.transform import GAE
from wacky_rl.trainer import Trainer

from wacky_rl.transform import RunningMeanStd


class PPOSingleModel(AgentCore):


    def __init__(self, env, approximate_contin=False):
        super(PPOSingleModel, self).__init__()

        self.approximate_contin = approximate_contin
        self.memory = BufferMemory()
        self.advantage_and_returns = GAE()

        self.reward_rmstd = RunningMeanStd()

        input_layer = Input(env.observation_space.shape)
        hidden_layer = Dense(256, activation='relu')(input_layer)
        hidden_layer = Dense(256, activation='relu')(hidden_layer)

        action_layer = self.make_action_layer(env, approx_contin=approximate_contin)(hidden_layer)
        critic_layer = Dense(1)(hidden_layer)

        self.model = WackyModel(inputs=input_layer, outputs=[action_layer, critic_layer])

        self.actor_loss = PPOActorLoss(entropy_factor=0.0)
        self.critic_loss = MeanSquaredErrorLoss()
        self.optimizer = tf.keras.optimizers.Adam(3e-5, clipnorm=0.5)

    def act(self, inputs, act_argmax=False, save_memories=True):

        dist, val = self.model(tf.expand_dims(tf.squeeze(inputs), 0))

        if act_argmax:
            actions = dist.mean_actions()
        else:
            actions = dist.sample_actions()

        if save_memories:
            self.memory(actions, key='actions')
            self.memory(dist.calc_probs(actions), key='probs')
            self.memory(val, key='val')

        return self.transform_actions(dist, actions)

    def learn(self):

        action, old_probs, values, states, new_states, rewards, dones = self.memory.replay()

        self.reward_rmstd.update(rewards.numpy())
        rewards = rewards / np.sqrt(self.reward_rmstd.var + 1e-8)

        _ , next_value = self.model(tf.expand_dims(new_states[-1], 0))
        adv, ret = self.advantage_and_returns(rewards, dones, values, next_value)


        for i in range(len(adv)):
            self.memory(adv[i], key='adv')
            self.memory(ret[i], key='ret')

        a_loss_list = []
        c_loss_list = []

        for e in range(4):

            for mini_batch in self.memory.mini_batches(batch_size=64, num_batches=None, shuffle_batches=False):

                actions, old_probs, values, states, new_states, rewards, dones, adv, ret = mini_batch

                adv = tf.squeeze(adv)
                adv = (adv - tf.reduce_mean(adv)) / (tf.math.reduce_std(adv) + 1e-8)

                with tf.GradientTape() as tape:

                    pred_dist, pred_val = self.model(states)

                    a_loss = self.actor_loss(pred_dist, actions, old_probs, adv)
                    c_loss = self.critic_loss(pred_val, ret)

                    sum_loss = a_loss + 0.5 * c_loss

                self.optimizer.minimize(sum_loss, self.model.trainable_variables, tape=tape)

                a_loss_list.append(tf.reduce_mean(a_loss).numpy())
                c_loss_list.append(tf.reduce_mean(c_loss).numpy())

        self.memory.clear()
        return np.mean(a_loss_list), np.mean(c_loss_list)


def train_ppo():

    import gym
    #env = gym.make('CartPole-v0')
    env = gym.make("LunarLanderContinuous-v2")

    agent = PPOSingleModel(env)

    trainer = Trainer(env, agent)
    trainer.n_step_train(5_000_000, train_on_test=False)
    trainer.test(100)

if __name__ == "__main__":
    train_ppo()
