import abc
import numpy as np
import torch as th
from torch import nn

import wacky.functional as funky
from wacky.functional.get_optimizer import get_optim
from wacky.networks import OffPolicyNetworkWrapper

class ReinforcementLearnerArchitecture(funky.WackyBase):

    """def __init__(self, network: nn.Module, optimizer: str, lr: float, *args, **kwargs):
        super(ReinforcementLearnerArchitecture, self).__init__()

        self.network = network
        if isinstance(self.network, OffPolicyNetworkWrapper):
            self.optimizer = get_optim(optimizer, self.network.behavior.parameters(), lr, *args, **kwargs)
        else:
            self.optimizer = get_optim(optimizer, self.network.parameters(), lr, *args, **kwargs)"""

    def __init__(self, *args, **kwargs):
        super(ReinforcementLearnerArchitecture, self).__init__()

    def reset(self):
        pass

    @abc.abstractmethod
    def call(self, state, deterministic=False, remember=True) -> th.Tensor:
        pass

    @abc.abstractmethod
    def learn(self):
        pass

    def warm_up(self):
        pass

    def train(self):
        pass

    def test(self, env, num_episodes, render=True):

        for e in range(num_episodes):

            self.reset()
            done = False
            state = env.reset()
            sum_rewards = 0
            while not done:
                state = th.FloatTensor(state).unsqueeze(0)
                action = self.call(state, deterministic=True, remember=False)
                if isinstance(action, th.Tensor):
                    action = action.detach()[0].numpy()
                state, reward, done, _ = env.step(action)
                sum_rewards += reward

                if render:
                    env.render()

            print('rewards:', sum_rewards)

        env.close()


class MonteCarloLearner(ReinforcementLearnerArchitecture):

    def __init__(self, network, optimizer: str, lr: float, *args, **kwargs):
        super(MonteCarloLearner, self).__init__(network, optimizer, lr, *args, **kwargs)

    def train(self, env, num_episodes, render=False):

        for e in range(num_episodes):

            self.reset()
            done = False
            state = env.reset()

            while not done:

                state = th.FloatTensor(state).unsqueeze(0)
                action = self.call(state, deterministic=False).detach()[0]
                state, reward, done, _ = env.step(action.numpy())
                self.next_state(state)
                self.reward_signal(reward)
                self.done_signal(done)

                if render:
                    env.render()

            self.learn()
            print('episode:', e,
                  'rewards:', self.memory['rewards'].sum().numpy(),
                  'probs:', th.exp(self.memory['log_prob'].detach()).mean().numpy()
            )


class BootstrappingLearner(ReinforcementLearnerArchitecture):

    def __init__(self, network, optimizer: str, lr: float, *args, **kwargs):
        super(BootstrappingLearner, self).__init__(network, optimizer, lr, *args, **kwargs)

    def train(self, env, num_steps, train_interval, render=False):

        done = True
        train_interval_counter = funky.ThresholdCounter(train_interval)
        episode_rewards = funky.ValueTracer()
        for t in range(num_steps):

            if done:
                state = env.reset()
                episode_rewards.sum()

            state = th.FloatTensor(state).unsqueeze(0)
            action = self.call(state, deterministic=False)
            if isinstance(action, th.Tensor):
                action = action.detach()[0].numpy()
            state, reward, done, _ = env.step(action)
            self.next_state(state)
            self.reward_signal(reward)
            self.done_signal(done)
            episode_rewards(reward)

            if render:
                env.render()

            if train_interval_counter():
                self.learn()
                print('steps:', t,
                      'rewards:', episode_rewards.reduce_mean(decimals=3),
                      'actions:', self.memory.numpy('actions', reduce='mean', decimals=3),
                      'epsilon:', np.round(self.epsilon_greedy.eps, 3),
                )
                self.reset()
                #self.test(env, 1)

