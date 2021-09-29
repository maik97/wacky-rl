#import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import Model

from wacky_rl.agents import AgentCore
from wacky_rl.models import WackyModel
from wacky_rl.memory import BufferMemory
from wacky_rl.layers import DiscreteActionLayer, ContinActionLayer
from wacky_rl.losses import PPOActorLoss, MeanSquaredErrorLoss
from wacky_rl.transform import GAE
from wacky_rl.trainer import Trainer
from wacky_rl.logger import StatusPrinter
from wacky_rl.transform import RunningMeanStd


class PPO(AgentCore):

    def __init__(
            self,
            env,
            learning_rate=3e-5,
            entropy_factor=0.0,
            clipnorm=0.5,
            batch_size=64,
            epochs=10,
            approximate_contin=False,
            logger=None,
            standardize_rewards=False
    ):
        super(PPO, self).__init__()

        self.batch_size = batch_size
        self.epochs = epochs

        self.approximate_contin = approximate_contin
        self.memory = BufferMemory()
        self.advantage_and_returns = GAE()

        if logger is None:
            logger = StatusPrinter('ppo')
        self.logger = logger

        self.standardize_rewards = standardize_rewards
        self.reward_rmstd = RunningMeanStd()

        # Actor:
        num_actions = int(self.decode_space(env.action_space))

        initializer = tf.keras.initializers.Orthogonal()

        if self.space_is_discrete(env.action_space):
            out_layer = DiscreteActionLayer(num_bins=num_actions, kernel_initializer=initializer)
            self.disct_actor = True
        elif self.approximate_contin:
            out_layer= DiscreteActionLayer(num_bins=21, num_actions=num_actions, kernel_initializer=initializer)
            self.disct_actor = True
        else:
            out_layer= ContinActionLayer(num_actions=num_actions, kernel_initializer=initializer)


        self.actor = WackyModel(model_name='actor', logger=logger)
        self.actor.mlp_network(64, kernel_initializer=initializer)
        self.actor.add(out_layer)

        if clipnorm is None:
            self.actor.compile(
                optimizer=tf.keras.optimizers.RMSprop(learning_rate),
                loss=PPOActorLoss(entropy_factor=entropy_factor),
            )

        else:
            self.actor.compile(
                optimizer=tf.keras.optimizers.RMSprop(learning_rate, clipnorm=clipnorm),
                loss=PPOActorLoss(entropy_factor=entropy_factor),
            )

        # Critic:
        critic_input = Input(shape=env.observation_space.shape)
        critic_dense = Dense(64, activation='relu')(critic_input)
        critic_dense = Dense(64, activation='relu')(critic_dense)
        critic_out = Dense(1)(critic_dense)
        self.critic = Model(inputs=critic_input, outputs=critic_out)#, model_name='critic', logger=logger)
        self.critic.compile(optimizer='adam', loss='mse')

        self.old_test_reward = None

    def act(self, inputs, act_argmax=False, save_memories=True):

        inputs = tf.expand_dims(tf.squeeze(inputs), 0)
        dist = self.actor(inputs, act_argmax=True)

        greedy = np.random.random(1)
        #print(greedy)
        if act_argmax or greedy>0.5:
            actions = dist.mean_actions()
        else:
            actions = dist.sample_actions()

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

        if self.standardize_rewards:
            self.reward_rmstd.update(rewards.numpy())
            rewards = rewards / np.sqrt(self.reward_rmstd.var + 1e-8)

        values = self.critic.predict(states)
        next_value = self.critic.predict(tf.expand_dims(new_states[-1], 0))
        adv, ret = self.advantage_and_returns(rewards, dones, values, next_value)

        self.logger.log_mean('values', np.mean(np.append(values, next_value)))
        self.logger.log_mean('adv', np.mean(adv.numpy()))
        self.logger.log_mean('ret', np.mean(ret.numpy()))

        for i in range(len(adv)):
            self.memory(adv[i], key='adv')
            self.memory(ret[i], key='ret')


        hist = self.critic.fit(states, ret, verbose=0, epochs=self.epochs, batch_size=self.batch_size)
        c_loss = hist.history['loss']
        self.logger.log_mean('critic loss', np.mean(c_loss))

        losses = []
        for e in range(self.epochs):
            for mini_batch in self.memory.mini_batches(batch_size=self.batch_size, num_batches=None, shuffle_batches=True):

                action, old_probs, states, new_states, rewards, dones, adv, ret = mini_batch

                adv = tf.squeeze(adv)
                adv = (adv - tf.reduce_mean(adv)) / (tf.math.reduce_std(adv) + 1e-8)


                a_loss = self.actor.train_step(states, action, old_probs, adv)

                #losses.append(tf.reduce_mean(a_loss).numpy()+tf.reduce_mean(c_loss).numpy())
                losses.append(tf.reduce_mean(a_loss).numpy())

        self.memory.clear()
        #self.memory.pop_array('ret')
        #self.memory.pop_array('adv')
        return np.mean(losses)

    def test_compare_with_old_policy(self, test_reward):
        if self.old_test_reward is None:
            self.old_test_reward = test_reward
            self.old_weights = self.actor.get_weights()
            return
        '''
        if self.old_test_reward > test_reward:
            weights = self.actor.get_weights()

            for i in range(len(self.old_weights)):
                weights[i] = self.old_weights[i] * 0.75 + weights[i] * (1 - 0.75 )

            self.actor.set_weights(weights)
            self.old_weights = weights
            #self.old_test_reward = self.old_test_reward * 0.95 + test_reward * (1 - 0.95)

        else:
            #self.old_test_reward = self.old_test_reward * (1 - 0.95) + test_reward * 0.95
            weights = self.actor.get_weights()

            for i in range(len(self.old_weights)):
                weights[i] = self.old_weights[i] * (1 - 0.75) + weights[i] * 0.55

            self.actor.set_weights(weights)
            self.old_weights = weights
        

        reward_momentum = 0.5#(self.old_test_reward) / (self.old_test_reward + test_reward)
        weights = self.actor.get_weights()

        for i in range(len(self.old_weights)):
            weights[i] = self.old_weights[i] * reward_momentum + weights[i] * (1 - reward_momentum)

        self.actor.set_weights(weights)
        self.old_weights = weights

        self.logger.log_mean('sum reward old', np.round(self.old_test_reward, 1))
        self.logger.log_mean('sum reward momentum', np.round(reward_momentum, 4))
        self.old_test_reward = self.old_test_reward * 0.9 + test_reward * (1 - 0.9)
        '''


def train_ppo():

    import gym
    #env = gym.make('CartPole-v0')
    env = gym.make("LunarLanderContinuous-v2")
    agent = PPO(env)

    trainer = Trainer(env, agent)
    trainer.n_step_train(5_000_000, train_on_test=False)
    trainer.test(100)

if __name__ == "__main__":
    train_ppo()
