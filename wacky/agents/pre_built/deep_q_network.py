import numpy as np
import torch as th

from wacky.agents import ReinforcementLearnerArchitecture
from wacky.losses import ValueLossWrapper
from wacky.scores import NStepReturns, TemporalDifferenceReturns, GeneralizedReturns
from wacky.memory import NumpyMemoryDict, PrioritizedExperienceReplay
from wacky.exploration import DiscountingEpsilonGreedy, InterpolationEpsilonGreedy
from wacky.networks import OffPolicyNetworkWrapper
from wacky.optimizer import TorchOptimizer

from wacky import functional as funky


class DQN(ReinforcementLearnerArchitecture):

    def __init__(
            self,
            action_space,
            observations_space,
            network=None,
            polyak=1.0,
            optimizer: str = 'Adam',
            lr: float = 0.0001,
            buffer_size=1_000_000,
            greedy_explorer=None,
            n_steps: int = 1,
            gamma: float = 0.99,
            batch_size=32,
            epochs=1,
            double=True,
            duelling=True,
            per=False,
            *args, **kwargs
    ):
        super(DQN, self).__init__(*args, **kwargs)

        if duelling:
            make_net_func = funky.make_duelling_q_net
        else:
            make_net_func = funky.make_q_net

        self.network = OffPolicyNetworkWrapper(
            make_net_func=make_net_func,
            polyak=polyak,
            in_features=observations_space,
            out_features=action_space,
            net=network
        )

        self.optimizer = TorchOptimizer(
            optimizer=optimizer,
            network_parameter=self.network,
            lr=lr,
        )

        self.network.override_target()

        if per:
            self.experience_replay = PrioritizedExperienceReplay().set_maxlen(buffer_size)
        else:
            self.experience_replay = NumpyMemoryDict().set_maxlen(buffer_size)
        self.per = per

        if greedy_explorer is None:
            self.greedy_explorer = DiscountingEpsilonGreedy(
                action_space,
                eps_init=1.0,
                eps_discount=0.999995,
                eps_min=0.1,
            )
        else:
            self.greedy_explorer = greedy_explorer

        if n_steps == 1:
            self.calc_returns = TemporalDifferenceReturns(gamma=gamma)
        else:
            self.calc_returns = NStepReturns(gamma=gamma, n=n_steps)

        self.loss_fn = th.nn.SmoothL1Loss(reduction='none')

        self.batch_size = batch_size
        self.epochs = epochs
        self.double = double

    def call(self, state, deterministic=False, remember=True):
        action = self.greedy_explorer(self.network.behavior, state, deterministic)
        if remember:
            self.experience_replay['states'].append(np.squeeze(state))
            self.experience_replay['actions'].append(action)
        return action

    def current_value_from_behavior_network(self, batch):
        actions = batch['actions'].type(th.int64).reshape(-1, 1)
        return self.network.behavior(batch['states']).gather(1, actions)

    def next_value_from_target_network(self, batch):
        next_states = batch['next_states']
        if self.double:
            selected_action = self.network.behavior(next_states).argmax(dim=1, keepdim=True)
            return self.network.target(next_states).gather(1, selected_action)
        else:
            return self.network.target(next_states).max(dim=1, keepdim=True)[0].detach()

    def learn(self):

        for epoch in range(self.epochs):

            if self.per:
                pass
            for batch in self.experience_replay.generate_batches(self.batch_size, num_batches=1):

                if batch is None:
                    break  # batch is None if replay buffer memory doesn't have enough samples

                if self.per:
                    batch, weights, batch_indices = batch

                batch['values'] = self.current_value_from_behavior_network(batch)
                batch['next_values'] = self.next_value_from_target_network(batch)

                batch['returns'] = self.calc_returns(batch)

                elem_wise_loss = self.loss_fn(batch['returns'], batch['values'])

                # PER: importance sampling before average
                if self.per:
                    loss = (elem_wise_loss * weights).mean()
                else:
                    loss = elem_wise_loss.mean()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if self.per:
                    new_priorities = elem_wise_loss.detach().numpy() + self.experience_replay.eps
                    self.experience_replay.update_priorities(batch_indices, new_priorities)

    def step(self, env, state, deterministic=False):
        self.greedy_explorer.step()
        state = th.FloatTensor(state).unsqueeze(0)
        action = self.call(state, deterministic=deterministic)
        if isinstance(action, th.Tensor):
            action = action.detach()[0].numpy()
        state, reward, done, _ = env.step(action)
        reward -= int(done)
        return state, reward, done

    def warm_up(self, env, num_steps=50_000):

        done = True
        episode_rewards = funky.ValueTracer()
        for t in range(num_steps):

            if done:
                state = env.reset()
                episode_rewards.sum()

            state, reward, done = self.step(env, state, deterministic=False)

            self.experience_replay['next_states'].append(state)
            self.experience_replay['rewards'].append(reward)
            self.experience_replay['dones'].append(done)

            if self.per:
                self.experience_replay.tree_pointer_step()

            episode_rewards(reward)

            if self.per:
                fraction = min(t / num_steps, 1.0)
                self.experience_replay.beta = self.experience_replay.beta + fraction * (1.0 - self.experience_replay.beta)


        print('warm-up:', t+1,
              'rewards:', episode_rewards.reduce_mean(decimals=3),
              'actions:', self.experience_replay.read('actions', reduce='mean', decimals=3),
              'epsilon:', np.round(self.greedy_explorer.eps, 3),
              )

    def train(self, env, num_steps=None, train_interval=1_000, update_interval=1_000, log_interval=1_000, render=False):

        if num_steps is None:
            num_steps = self.num_steps

        done = True
        #train_interval_counter = funky.ThresholdCounter(train_interval)
        #update_interval_counter = funky.ThresholdCounter(update_interval)
        episode_rewards = funky.ValueTracer()
        for t in range(num_steps):

            if done:
                state = env.reset()
                episode_rewards.sum()

            state, reward, done = self.step(env, state, deterministic=False)

            self.experience_replay['next_states'].append(np.squeeze(state))
            self.experience_replay['rewards'].append(reward)
            self.experience_replay['dones'].append(done)

            if self.per:
                self.experience_replay.tree_pointer_step()

            episode_rewards(reward)

            if render:
                env.render()

            if t % train_interval == 0:
                self.learn()

            if t % update_interval == 0:
                self.network.update_target_weights()

            if t % log_interval == 0:
                print('steps:', t + 1,
                      'rewards:', episode_rewards.reduce_mean(decimals=3),
                      'actions:', self.experience_replay.read('actions', reduce='mean', decimals=3),
                      'epsilon:', np.round(self.greedy_explorer.eps, 3),
                      #'tree_pointer:', self.experience_replay.tree_pointer,
                      'pointer:', self.experience_replay.pointer,
                      'max_index:', self.experience_replay.max_index,
                      )

            if self.per:
                fraction = min(t / num_steps, 1.0)
                self.experience_replay.beta = self.experience_replay.beta + fraction * (1.0 - self.experience_replay.beta)


def compare_sb3(env):
    from stable_baselines3 import DQN
    model = DQN("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=100_000, log_interval=4)


def main():
    import gym
    import time
    env = gym.make('CartPole-v0')

    #tic_sb3 = time.perf_counter()
    #compare_sb3(env)
    #toc_sb3 = time.perf_counter()

    warm_up_steps = 10_000
    num_steps = 200_000

    greedy_explorer = InterpolationEpsilonGreedy(
        env.action_space,
        eps_interpolation='linear',
        eps_init=1.0,
        eps_min=0.1,
        ramp_point_a=(warm_up_steps / (num_steps + warm_up_steps)),
        ramp_point_b=0.5,
        num_steps=(num_steps + warm_up_steps)
    )

    agent = DQN(
        action_space=env.action_space,
        observations_space=env.observation_space,
        greedy_explorer=greedy_explorer,
        buffer_size=100_000,
        per=True
    )

    tic_wacky = time.perf_counter()
    #agent.warm_up(env, warm_up_steps)
    agent.train(env, num_steps, train_interval=4, update_interval=1000)
    toc_wacky = time.perf_counter()

    #print(f"sb3_time: {toc_sb3 - tic_sb3:0.4f}")
    print(f"wacky_time: {toc_wacky - tic_wacky:0.4f}")

    agent.test(env, 100)


if __name__ == '__main__':
    main()
