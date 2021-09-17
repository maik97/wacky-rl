import numpy as np
from wacky_rl import layers
from collections import deque

class AgentCore:

    def __init__(self, obs_seq_lenght = 6):
        self.approx_contin = False
        self.obs_seq_lenght = obs_seq_lenght
        self.obs_mem = deque(maxlen=obs_seq_lenght)

        if not hasattr(self, 'logger'):
            self.logger = None

        if not hasattr(self, 'approx_contin'):
            self.approx_contin = False

        if not hasattr(self, 'env'):
            self.env = None

    def __call__(self, *args, **kwargs):
        return self.act( *args, **kwargs)

    def act(self, inputs, act_argmax=False):
        raise NotImplementedError('When subclassing the `AgentCore` class, you should '
                                  'implement a `act` method.')

    def learn(self, *args, **kwargs):
        raise NotImplementedError('When subclassing the `AgentCore` class, you should '
                                  'implement a `train` method.')

    def decode_space(self, gym_space):

        from gym import spaces

        if isinstance(gym_space, spaces.Box):
            import numpy as np
            return int(np.squeeze(gym_space.shape))

        elif isinstance(gym_space, spaces.Discrete):
            return int(gym_space.n)

        else:
            raise AttributeError('gym_space not understood: {}, use space.Box or space.Discrete'.format(gym_space))

    def space_is_contin(self, gym_space):
        from gym import spaces
        return isinstance(gym_space, spaces.Box)

    def space_is_discrete(self, gym_space):
        from gym import spaces
        return isinstance(gym_space, spaces.Discrete)

    def make_action_layer(self, env, num_bins=21, num_actions=None, approx_contin=False, *args, **kwargs):

        if num_actions is None:
            num_actions = int(self.decode_space(env.action_space))

        if self.space_is_discrete(env.action_space):
            self.out_format = int
            return layers.DiscreteActionLayer(num_bins=num_actions, *args, **kwargs)
        elif approx_contin:
            self.out_format = float
            self.approx_contin = True
            return layers.DiscreteActionLayer(num_bins=num_bins, num_actions=num_actions, *args, **kwargs)
        else:
            self.out_format = float
            return layers.ContinActionLayer(num_actions=num_actions, *args, **kwargs)

    def compare_with_old_policy(self, test_reward):
        pass

    def transform_actions(self, dist, actions, lows=None, highs=None, scale=1.0):

        if self.approx_contin:
            actions = dist.discrete_to_contin(actions)

        actions = np.squeeze(actions.numpy()) * scale

        if not lows is None or not highs is None:
            actions = np.clip(actions, a_min=lows, a_max=highs)

        return actions.astype(self.out_format)

    def take_step(self, obs, save_memories=True, render_env=False, act_argmax=False):

        self.obs_mem.append(obs)
        # print(np.squeeze(np.array(self.obs_mem)))
        action = self.act(np.squeeze(np.array(self.obs_mem)), act_argmax=act_argmax, save_memories=save_memories)
        # action = self.agent.act(np.ravel(np.squeeze(np.array(self.obs_mem))), act_argmax=act_argmax, save_memories=save_memories)
        new_obs, r, done, _ = self.env.step(np.squeeze(action))

        if save_memories:
            self.memory({
                'obs': np.squeeze(np.array(self.obs_mem))
                # 'obs': np.ravel(np.squeeze(np.array(self.obs_mem)))
            }
            )
            self.obs_mem.append(new_obs)
            self.memory({
                # 'obs': np.array(self.obs_mem),
                'new_obs': np.squeeze(np.array(self.obs_mem)),
                # 'new_obs': np.ravel(np.squeeze(np.array(self.obs_mem))),
                'rewards': r,
                'dones': float(1 - int(done)),
            }
            )

        if render_env:
            self.env.render()

        return done, new_obs, r

    def sample_warmup(self, num_episodes, render_env=False):
        for e in range(num_episodes):

            done = False
            obs = self.env.reset()

            while not done:
                done, obs, _ = self.take_step(obs, save_memories=True, render_env=render_env)
        self.env.close()

    def episode_train(self, num_episodes, render_env=False):
        for e in range(num_episodes):

            done = False
            obs = self.env.reset()
            reward_list = []
            for i in range(self.obs_seq_lenght):
                self.obs_mem.append(np.zeros(np.shape(obs)))

            while not done:
                done, obs, r = self.take_step(obs, save_memories=True, render_env=render_env)
                # self.env.render()
                reward_list.append(r)

                if done:
                    a_loss, c_loss = self.learn()
                    if self.logger is None:
                        print()
                        print('# Episode', e)
                        print('# Sum R:', np.round(np.sum(reward_list), 1))
                        print('# Loss A:', np.round(np.mean(a_loss), 4))
                        print('# Loss C:', np.round(np.mean(c_loss), 4))
                    else:
                        self.logger.log_mean('reward', np.round(np.sum(reward_list)))
                        self.logger.print_status(e)

        self.env.close()

    def n_step_train(
            self,
            num_steps,
            n_steps=2048,
            render_env=False,
            train_on_test=True,
            render_test=True,
    ):

        train_after = n_steps
        episode_reward_list = []
        s = 0
        while s < num_steps:

            obs = self.env.reset()
            done = False
            reward_list = []
            for i in range(self.obs_seq_lenght):
                self.obs_mem.append(np.zeros(np.shape(obs)))

            while not done:
                done, obs, r = self.take_step(obs, save_memories=True, render_env=render_env)
                reward_list.append(r)
                s += 1

                if done:
                    obs = self.env.reset()
                    episode_reward_list.append(np.sum(reward_list))
                    reward_list = []

            if s >= train_after:
                train_after += n_steps
                a_loss, c_loss = self.learn()
                if self.logger is None:
                    print()
                    print('# steps', s)
                    print('# Sum R:', np.round(np.sum(episode_reward_list), 1))
                    print('# Loss A:', np.round(np.mean(a_loss), 4))
                    print('# Loss C:', np.round(np.mean(c_loss), 4))
                else:
                    self.logger.log_mean('sum reward', np.round(np.mean(episode_reward_list)))
                    # print('sum reward:', np.round(np.sum(episode_reward_list), 1))
                    self.logger.print_status(s)
                episode_reward_list = []

                # Test:
                if train_on_test or render_test:
                    done = False
                    while not done:
                        done, obs, r = self.take_step(obs, save_memories=True, render_env=render_test, act_argmax=True)
                        reward_list.append(r)

                        if done:
                            print('test reward:', np.round(sum(reward_list), 1))
                            obs = self.env.reset()
                            reward_list = []

        self.env.close()
