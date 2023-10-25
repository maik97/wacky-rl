import torch as th
import numpy as np

from wacky.agents import ReinforcementLearnerArchitecture
from wacky.losses import ClippedSurrogateLoss, ValueLossWrapper, BaseWackyLoss
from wacky.scores import GeneralizedAdvantageEstimation, BaseReturnCalculator
from wacky.memory import MemoryDict

from wacky import functional as funky

from wacky.networks import ActorCriticNetworkConstructor, WackyNetwork
from wacky.optimizer import TorchOptimizer, WackyOptimizer

from wacky.backend import WackyTypeError


def make_PPO(
        env,
        network=None,
        optimizer: str = 'Adam',
        lr: float = 0.0003,
        gamma: float = 0.99,
        lamda:float = 0.95,
        actor_loss_scale_factor: float = 1.0,
        critic_loss_scale_factor: float = 0.5,
        epochs: int = 10,
        batch_size: int = 64,
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

    returns_calc = GeneralizedAdvantageEstimation(gamma, lamda)

    actor_loss_fn = ClippedSurrogateLoss(scale_factor=actor_loss_scale_factor)
    critic_loss_fn = ValueLossWrapper(th.nn.SmoothL1Loss(), scale_factor=critic_loss_scale_factor)


    return PPO(network, optimizer, memory, returns_calc, actor_loss_fn, critic_loss_fn, epochs, batch_size, *args, **kwargs)


class PPO(ReinforcementLearnerArchitecture):

    def __init__(
            self,
            network: WackyNetwork,
            optimizer: (TorchOptimizer, WackyOptimizer),
            memory: MemoryDict = None,
            returns_calc: GeneralizedAdvantageEstimation = None,
            actor_loss_fn: BaseWackyLoss = None,
            critic_loss_fn: BaseWackyLoss = None,
            epochs: int = 10,
            batch_size: int = 64,
            *args, **kwargs
    ):
        super(PPO, self).__init__(*args, **kwargs)

        self.network = network
        self.optimizer = optimizer
        self.memory = MemoryDict() if memory is None else memory
        self.returns_and_advantages = GeneralizedAdvantageEstimation() if returns_calc is None else returns_calc
        self.actor_loss_fn = ClippedSurrogateLoss() if actor_loss_fn is None else actor_loss_fn
        self.critic_loss_fn = ValueLossWrapper(th.nn.SmoothL1Loss()) if critic_loss_fn is None else critic_loss_fn

        self.remember_rewards = True
        self.reset_memory = True

        self.epochs = epochs
        self.batch_size = batch_size

    def call(self, state, deterministic=False, remember=True):
        action, log_prob = self.network.actor(state)
        if remember:
            self.memory['old_log_prob'].append(log_prob[0].detach())
            self.memory['states'].append(np.squeeze(state))
            self.memory['actions'].append(action[0].detach())
        return action

    def learn(self):
        loss_a = []
        loss_c = []
        for e in range(self.epochs):
            for batch in self.memory.batch(self.batch_size, shuffle=True):

                log_prob, value = self.network.eval_action(batch['states'], batch['actions'])
                next_values = self.network.critic(batch['next_states'])

                batch['log_prob'] = log_prob
                batch['values'] = value.reshape(-1,1)
                batch['next_values'] = next_values.reshape(-1,1)

                ret, adv = self.returns_and_advantages(batch)
                batch['returns'] = ret.reshape(-1,1)
                batch['advantage'] = adv.detach()

                loss_actor = self.actor_loss_fn(batch)
                loss_critic = self.critic_loss_fn(batch)

                loss = loss_actor + loss_critic
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                loss_a.append(loss_actor.detach().numpy())
                loss_c.append(loss_critic.detach().numpy())
        return np.mean(loss_a), np.mean(loss_c)

    def reset(self):
        self.memory.clear()

    def step(self, env, state, deterministic=False):
        state = th.FloatTensor(state).unsqueeze(0)
        action = self.call(state, deterministic=deterministic)
        if isinstance(action, th.Tensor):
            action = action.detach()[0].numpy()
        state, reward, done, _ = env.step(action)
        #reward -= int(done)
        return state, reward, done

    def train(self, env, num_steps=None, train_interval=2048, render=False):

        done = True
        #train_interval_counter = funky.ThresholdCounter(train_interval)
        #episode_rewards = [] #funky.ValueTracer()
        episode_reward_list = []
        episode_reward = None
        for t in range(num_steps):

            if done:
                state = env.reset()
                if episode_reward is not None:
                    episode_reward_list.append(episode_reward)
                episode_reward = 0

            state, reward, done = self.step(env, state, deterministic=False)

            self.memory['rewards'].append(reward)
            self.memory['dones'].append(done)
            self.memory['next_states'].append(np.squeeze(state))
            episode_reward += reward
            #episode_rewards.append(reward)

            if render:
                env.render()

            if t % train_interval == 0:
                loss_a, loss_c, = self.learn()
                print('steps:', t + 1,
                      'rewards:', np.round(np.mean(episode_reward_list), decimals=3),
                      'prob:', np.round(np.exp(self.memory.numpy('old_log_prob', reduce='mean')), 3),
                      'actor_loss:', np.round(loss_a, 4),
                      'critic_loss:', np.round(loss_c, 4),
                      )
                self.reset()

def main():
    import gym
    from wacky import functional as funky
    env = gym.make('CartPole-v0')
    #env = gym.make('LunarLanderContinuous-v2')
    agent = make_PPO(env)
    agent.train(env, 100_000)
    wait = input('press enter')
    agent.test(env, 100)


if __name__ == '__main__':
    main()
