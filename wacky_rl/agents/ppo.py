import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import Model

from wacky_rl.agents import AgentCore
from wacky_rl.models import WackyModel
from wacky_rl.memory import BufferMemory
from wacky_rl.layers import DiscreteActionLayer, ContinActionLayer
from wacky_rl.losses import PPOActorLoss
from wacky_rl.transform import GAE
from wacky_rl.trainer import Trainer


class PPO(AgentCore):

    def __init__(self, env, approximate_contin=False):
        super(PPO, self).__init__()

        self.approximate_contin = approximate_contin
        self.memory = BufferMemory()
        self.advantage_and_returns = GAE()

        # Actor:
        model_layer = [
            Dense(256, activation='relu'),
            Dense(256, activation='relu'),
        ]

        num_actions = int(self.decode_space(env.action_space))

        if self.space_is_discrete(env.action_space):
            model_layer.append(DiscreteActionLayer(num_bins=num_actions))
        elif self.approximate_contin:
            model_layer.append(DiscreteActionLayer(num_bins=21, num_actions=num_actions))
        else:
            model_layer.append(ContinActionLayer(num_actions=num_actions))

        self.actor = WackyModel(
            model_layer=model_layer,
            optimizer=tf.keras.optimizers.RMSprop(3e-4),
            loss=PPOActorLoss(entropy_factor=0.001),
            grad_clip=True,
        )

        # Critic:
        critic_input = Input(shape=env.observation_space.shape)
        critic_dense = Dense(256, activation='relu')(critic_input)
        critic_dense = Dense(256, activation='relu')(critic_dense)
        critic_out = Dense(1)(critic_dense)
        self.critic = Model(critic_input, critic_out)
        self.critic.compile(optimizer='adam', loss='mse')

    def act(self, inputs, act_argmax=False, save_memories=True):

        inputs = tf.expand_dims(tf.squeeze(inputs), 0)
        dist = self.actor(inputs, act_argmax=True)

        if act_argmax:
            actions = dist.mean_actions()
        else:
            actions = dist.sample_actions()

        if save_memories:
            self.memory(actions, key='actions')
            self.memory(dist.calc_probs(actions), key='probs')

        if self.approximate_contin:
            return dist.discrete_to_contin(actions).numpy()
        return actions.numpy()

    def learn(self):

        action, old_probs, states, new_states, rewards, dones = self.memory.replay()
        values = self.critic.predict(states)
        next_value = self.critic.predict(tf.expand_dims(new_states[-1], 0))
        adv, ret = self.advantage_and_returns(rewards, dones, values, next_value)

        for i in range(len(adv)):
            self.memory(adv[i], key='adv')
            self.memory(ret[i], key='ret')

        a_loss_list = []
        c_loss_list = []

        for e in range(3):
            for mini_batch in self.memory.mini_batches(batch_size=32, num_batches=None, shuffle_batches=True):

                action, old_probs, states, new_states, rewards, dones, adv, ret = mini_batch

                adv = tf.squeeze(adv)
                adv = (adv - tf.reduce_mean(adv)) / (tf.math.reduce_std(adv) + 1e-8)

                c_loss = self.critic.train_on_batch(states, ret)
                a_loss = self.actor.train_step(states, action, old_probs, adv)

                a_loss_list.append(tf.reduce_mean(a_loss).numpy())
                c_loss_list.append(tf.reduce_mean(c_loss).numpy())

        self.memory.clear()
        return np.mean(a_loss_list), np.mean(c_loss_list)


def train_ppo():

    import gym
    # env = gym.make('CartPole-v0')
    env = gym.make("LunarLanderContinuous-v2")
    agent = PPO(env, approximate_contin=False)

    trainer = Trainer(env, agent)
    trainer.n_step_train(5_000_000)
    trainer.test(100)

if __name__ == "__main__":
    train_ppo()