#import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
import random

from tensorflow.keras.layers import Dense, Input, LSTM
from tensorflow.keras import Model

from wacky_rl.agents import AgentCore
from wacky_rl.models import WackyModel
from wacky_rl.memory import BufferMemory
from wacky_rl.layers import DiscreteActionLayer, ContinActionLayer, RecurrentEncoder
from wacky_rl.losses import PPOActorLoss, SharedNetLoss, MeanSquaredErrorLoss
from wacky_rl.transform import GAE
from wacky_rl.trainer import Trainer

from wacky_rl.transform import RunningMeanStd


class SharedPPO(AgentCore):


    def __init__(self, env, approximate_contin=False):
        super(SharedPPO, self).__init__()

        self.approximate_contin = approximate_contin
        self.memory = BufferMemory()
        self.advantage_and_returns = GAE()

        self.reward_rmstd = RunningMeanStd()
        self.adv_rmstd = RunningMeanStd()

        # Actor:
        self.actor = WackyModel()
        self.actor.add(Dense(32, activation='relu'))
        self.actor.add(self.make_action_layer(env, approx_contin=approximate_contin))
        self.actor.compile(
            optimizer=tf.keras.optimizers.Adam(3e-5, clipnorm=0.5),
            loss=PPOActorLoss(entropy_factor=0.0),
        )

        # Critic:
        self.critic = WackyModel()
        self.critic.add(Dense(32, activation='relu'))
        self.critic.add(Dense(1))
        self.critic.compile(optimizer='adam', loss=MeanSquaredErrorLoss())

        # Shared Network for Actor and Critic:
        self.shared_model = WackyModel()
        self.shared_model.add(LSTM(32, stateful=False))
        self.shared_model.mlp_network(256, dropout_rate=0.0)
        self.shared_model.compile(
            optimizer=tf.keras.optimizers.Adam(3e-4, clipnorm=0.5),
            loss=SharedNetLoss(
                alphas=[1.0, 0.5],
                sub_models=[self.actor, self.critic]
            )
        )

    def act(self, inputs, act_argmax=False, save_memories=True):

        x = self.shared_model(tf.expand_dims(inputs, 0))
        dist = self.actor(x)

        if act_argmax:
            actions = dist.mean_actions()
        else:
            actions = dist.sample_actions()

        #print(actions)

        if save_memories:
            self.memory(actions, key='actions')
            self.memory(dist.calc_probs(actions), key='probs')

        if self.approximate_contin:
            return dist.discrete_to_contin(actions).numpy()

        return actions.numpy()

    def learn(self):

        action, old_probs, states, new_states, rewards, dones = self.memory.replay()

        #print(states)
        #exit()

        #self.reward_rmstd.update(rewards.numpy())
        #rewards = rewards / np.sqrt(self.reward_rmstd.var + 1e-8)
        values = self.critic.predict(self.shared_model.predict(tf.reshape(states, [len(states),6, -1,])))

        #print(values)

        next_value = self.critic.predict(self.shared_model.predict(tf.reshape(new_states[-1], [1,6,-1])))
        adv, ret = self.advantage_and_returns(rewards, dones, values, next_value)


        for i in range(len(adv)):
            self.memory(adv[i], key='adv')
            self.memory(ret[i], key='ret')

        a_loss_list = []
        c_loss_list = [0]

        for e in range(10):
            for mini_batch in self.memory.mini_batches(batch_size=64, num_batches=None, shuffle_batches=False):

                action, old_probs, states, new_states, rewards, dones, adv, ret = mini_batch

                adv = tf.squeeze(adv)
                adv = (adv - tf.reduce_mean(adv)) / (tf.math.reduce_std(adv) + 1e-8)

                a_loss = self.shared_model.train_step(
                    tf.reshape(states, [len(states), 6, -1]),
                    loss_args=[[action, old_probs, adv], [ret]]
                )

                a_loss_list.append(tf.reduce_mean(a_loss).numpy())

        self.memory.clear()
        return np.mean(a_loss_list), np.mean(c_loss_list)


def train_ppo():

    import gym
    #env = gym.make('CartPole-v0')
    env = gym.make("LunarLanderContinuous-v2")

    agent = SharedPPO(env)

    trainer = Trainer(env, agent)
    trainer.n_step_train(5_000_000, train_on_test=False)
    trainer.test(100)

if __name__ == "__main__":
    train_ppo()
