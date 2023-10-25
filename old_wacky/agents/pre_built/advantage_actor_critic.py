import numpy as np
import torch as th

from wacky.agents import ReinforcementLearnerArchitecture
from wacky.losses import AdvantageLoss, ValueLossWrapper, BaseWackyLoss
from wacky.scores import MonteCarloReturns, CalcAdvantages
from wacky.memory import MemoryDict

from wacky.networks import ActorCriticNetworkConstructor, WackyNetwork
from wacky.optimizer import TorchOptimizer, WackyOptimizer

from wacky.backend import WackyTypeError


def make_adv_ac(
        env,
        network=None,
        optimizer: str = 'Adam',
        lr: float = 0.0007,
        gamma: float = 0.99,
        actor_loss_scale_factor: float = 1.0,
        critic_loss_scale_factor: float = 0.5,
        *args, **kwargs
):

    if network is not None and not isinstance(network, (list, int)):
        raise WackyTypeError(network, (list, int), parameter='network', optional=True)


    network = ActorCriticNetworkConstructor(
        observation_space=env.observation_space,
        action_space=env.action_space,
        shared_network=[64, 64] if network is None else network,
    ).build()

    print(network)

    optimizer = TorchOptimizer(
        optimizer=optimizer,
        network_parameter=network,
        lr=lr,
    )

    memory = MemoryDict()

    returns_calc = MonteCarloReturns(gamma)
    advantages_calc = CalcAdvantages()

    actor_loss_fn = AdvantageLoss(scale_factor=actor_loss_scale_factor)
    critic_loss_fn = ValueLossWrapper(th.nn.SmoothL1Loss(), scale_factor=critic_loss_scale_factor)

    return AdvantageActorCritic(network, optimizer, memory, returns_calc, advantages_calc, actor_loss_fn, critic_loss_fn, *args, **kwargs)


class AdvantageActorCritic(ReinforcementLearnerArchitecture):

    def __init__(
            self,
            network: WackyNetwork,
            optimizer: (TorchOptimizer, WackyOptimizer),
            memory: MemoryDict = None,
            returns_calc: MonteCarloReturns = None,
            advantages_calc: CalcAdvantages = None,
            actor_loss_fn: BaseWackyLoss = None,
            critic_loss_fn: BaseWackyLoss = None,
            *args, **kwargs
    ):
        super(AdvantageActorCritic, self).__init__(*args, **kwargs)

        self.network = network
        self.optimizer = optimizer
        self.memory = MemoryDict() if memory is None else memory

        self.returns = MonteCarloReturns() if returns_calc is None else returns_calc
        self.advantages = CalcAdvantages() if advantages_calc is None else advantages_calc
        self.actor_loss_fn = AdvantageLoss() if actor_loss_fn is None else actor_loss_fn
        self.critic_loss_fn = ValueLossWrapper(th.nn.SmoothL1Loss()) if critic_loss_fn is None else critic_loss_fn

    def call(self, state, deterministic=False, remember=True):
        (action, log_prob), value = self.network(state)
        if remember:
            self.memory['log_prob'].append(log_prob[0])
            self.memory['values'].append(value[0])
        return action

    def learn(self):
        self.memory.stack()
        self.memory['returns'] = self.returns(self.memory)
        self.memory['advantage'] = self.advantages(self.memory).detach()

        loss_actor = self.actor_loss_fn(self.memory)
        loss_critic = self.critic_loss_fn(self.memory)

        loss = loss_actor + loss_critic
        self.optimizer.apply_loss(loss)
        return loss_actor, loss_critic

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

            loss_actor,  loss_critic = self.learn()
            loss_actor, loss_critic = loss_actor.detach().numpy(), loss_critic.detach().numpy()
            print('episode:', e,
                  'rewards:', self.memory['rewards'].sum().numpy(),
                  'probs:', np.round(th.exp(self.memory['log_prob'].detach()).mean().numpy(),4),
                  'loss:', np.round(loss_actor + loss_critic, 2),
                  'loss_actor:', np.round(loss_actor, 2),
                  'loss_critic:', np.round(loss_critic, 2),
                  )

    def reset(self):
        self.memory.clear()


def main():
    import gym
    env = gym.make('CartPole-v0')
    # env = gym.make('LunarLanderContinuous-v2')

    agent = make_adv_ac(env)
    agent.train(env, 1000)
    agent.test(env, 100)


if __name__ == '__main__':
    main()
