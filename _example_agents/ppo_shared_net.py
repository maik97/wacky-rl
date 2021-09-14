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

from wacky_rl.logger import StatusPrinter


class SharedPPO(AgentCore):


    def __init__(self, env, approximate_contin=True, logger=None):
        super(SharedPPO, self).__init__()

        self.logger = logger

        self.approximate_contin = approximate_contin
        self.memory = BufferMemory()
        self.advantage_and_returns = GAE()

        self.reward_rmstd = RunningMeanStd()
        self.adv_rmstd = RunningMeanStd()

        initializer = tf.keras.initializers.Orthogonal()


        # Actor:
        self.actor = WackyModel(model_name='actor', logger=logger)
        self.actor.add(Dense(64, activation='tanh', kernel_initializer=initializer))
        self.actor.add(self.make_action_layer(env, approx_contin=approximate_contin, kernel_initializer=initializer))
        self.actor.compile(
            optimizer=tf.keras.optimizers.Adam(3e-4, clipnorm=0.5),
            loss=PPOActorLoss(entropy_factor=0.0),
        )

        # Critic:
        self.critic = WackyModel(model_name='critic', logger=logger)
        self.critic.add(Dense(64, activation='tanh'))
        self.critic.add(Dense(1))
        self.critic.compile(
            optimizer=tf.keras.optimizers.Adam(3e-4, clipnorm=0.5),
            loss=MeanSquaredErrorLoss()
        )

        # Shared Network for Actor and Critic:
        self.shared_model = WackyModel(model_name='shared_network', logger=logger)
        self.shared_model.add(LSTM(32, stateful=False))
        self.shared_model.mlp_network(256, dropout_rate=0.0)
        self.shared_model.compile(
            optimizer=tf.keras.optimizers.Adam(3e-5, clipnorm=0.5),
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

        probs = dist.calc_probs(actions)

        if act_argmax:
            self.logger.log_mean('argmax probs', probs)
        else:
            self.logger.log_mean('probs', probs)

        if save_memories:
            self.memory(actions, key='actions')
            self.memory(probs, key='probs')

        if self.approximate_contin:
            return dist.discrete_to_contin(actions).numpy()

        return actions.numpy()

    def learn(self):

        action, old_probs, states, new_states, rewards, dones = self.memory.replay()

        self.reward_rmstd.update(rewards.numpy())
        rewards = rewards / np.sqrt(self.reward_rmstd.var + 1e-8)
        values = self.critic.predict(self.shared_model.predict(tf.reshape(states, [len(states),6, -1,])))
        next_value = self.critic.predict(self.shared_model.predict(tf.reshape(new_states[-1], [1,6,-1])))

        adv, ret = self.advantage_and_returns(rewards, dones, values, next_value)

        #self.adv_rmstd.update(adv.numpy())
        #adv = adv / np.sqrt(self.adv_rmstd.var + 1e-8)
        #adv = (adv - self.adv_rmstd.mean) / np.sqrt(self.adv_rmstd.var + 1e-8)

        self.logger.log_mean('values', np.mean(np.append(values, next_value)))
        self.logger.log_mean('adv', np.mean(adv.numpy()))
        self.logger.log_mean('ret', np.mean(ret.numpy()))


        for i in range(len(adv)):
            self.memory(adv[i], key='adv')
            self.memory(ret[i], key='ret')

        losses = []

        for e in range(10):
            for mini_batch in self.memory.mini_batches(batch_size=64, num_batches=None, shuffle_batches=False):

                action, old_probs, states, new_states, rewards, dones, adv, ret = mini_batch

                adv = tf.squeeze(adv)
                adv = (adv - tf.reduce_mean(adv)) / (tf.math.reduce_std(adv) + 1e-8)

                loss = self.shared_model.train_step(
                    tf.reshape(states, [len(states), 6, -1]),
                    loss_args=[[action, old_probs, adv], [ret]]
                )

                losses.append(tf.reduce_mean(loss).numpy())

        self.memory.clear()
        return np.mean(losses)


def train_ppo():

    import gym
    #env = gym.make('CartPole-v0')
    env = gym.make("LunarLanderContinuous-v2")


    agent = SharedPPO(env, logger=StatusPrinter('test'))

    trainer = Trainer(env, agent)
    trainer.n_step_train(5_000_000, train_on_test=False)
    trainer.test(100)

if __name__ == "__main__":
    train_ppo()
