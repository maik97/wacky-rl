import torch as th
import numpy as np

from wacky.agents import ReinforcementLearnerArchitecture
from wacky.losses import NoBaselineLoss, WithBaselineLoss, BaseWackyLoss
from wacky.scores import MonteCarloReturns, BaseReturnCalculator
from wacky.memory import MemoryDict

from wacky.networks import ActorNetwork, WackyNetwork
from wacky.optimizer import TorchOptimizer, WackyOptimizer

from wacky.backend import WackyTypeError


def make_REINFORCE(
        env,
        network=None,
        optimizer: str = 'Adam',
        lr: float = 0.001,
        gamma: float = 0.99,
        standardize_returns: bool = False,
        loss_scale_factor: float = 1.0,
        baseline: str = None,
        *args, **kwargs
):

    if network is not None and not isinstance(network, (list, int)):
        raise WackyTypeError(network, (list, int), parameter='network', optional=True)

    network = ActorNetwork(
        action_space=env.action_space,
        in_features=env.observation_space,
        network=[64, 64] if network is None else network,
    )

    print(network)

    optimizer = TorchOptimizer(
        optimizer=optimizer,
        network_parameter=network,
        lr=lr,
    )

    memory = MemoryDict()

    returns_calc = MonteCarloReturns(
        gamma=gamma,
        standardize=standardize_returns,
    )

    if baseline is None:
        loss_fn = NoBaselineLoss(loss_scale_factor)
    else:
        loss_fn = WithBaselineLoss(loss_scale_factor, baseline)

    return REINFORCE(network, optimizer, memory, returns_calc, loss_fn, *args, **kwargs)


class REINFORCE(ReinforcementLearnerArchitecture):

    def __init__(
            self,
            network: WackyNetwork,
            optimizer: (TorchOptimizer, WackyOptimizer),
            memory: MemoryDict = None,
            returns_calc: BaseReturnCalculator = None,
            loss_fn: BaseWackyLoss = None,
            *args, **kwargs
    ):
        super(REINFORCE, self).__init__(*args, **kwargs)

        self.network = network
        self.optimizer = optimizer
        self.memory = MemoryDict() if memory is None else memory
        self.returns_calc = MonteCarloReturns() if returns_calc is None else returns_calc
        self.loss_fn = NoBaselineLoss() if loss_fn is None else loss_fn

    def reset(self):
        self.memory.clear()

    def call(self, state, deterministic=False, remember=True):
        action, log_prob = self.network(state)
        if remember:
            self.memory['log_prob'].append(th.squeeze(log_prob))
        return action

    def learn(self):
        self.memory.stack()
        self.memory['returns'] = self.returns_calc(self.memory)
        loss = self.loss_fn(self.memory)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach()

    def train(self, env, num_episodes, render=False):

        for e in range(num_episodes):

            self.reset()
            done = False
            state = env.reset()

            while not done:

                state = th.FloatTensor(state).unsqueeze(0)
                action = self.call(state, deterministic=False).detach()[0]
                state, reward, done, _ = env.step(action.numpy())
                self.memory['rewards'].append(reward)

                if render:
                    env.render()

            self.network.reset()

            loss = self.learn()
            print('episode:', e,
                  'rewards:', self.memory['rewards'].sum().numpy(),
                  'probs:', np.round(th.exp(self.memory['log_prob'].detach()).mean().numpy(),4),
                  'loss:', np.round(loss.numpy(), 2),
                  )


def main():
    import gym
    env = gym.make('CartPole-v0')
    agent = make_REINFORCE(env, network=[16, 16])
    agent.train(env, 10000)
    agent.test(env, 100)


if __name__ == '__main__':
    main()
