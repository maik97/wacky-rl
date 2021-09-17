import numpy as np
from collections import deque

class Trainer:

    def __init__(
            self,
            env,
            agent,
            obs_mem_lenght = 1,
    ):

        self.env = env
        self.agent = agent
        self.obs_mem = deque(maxlen=obs_mem_lenght)
        self.obs_seq_lenght = obs_mem_lenght

        if hasattr(self.agent, 'logger'):
            self.logger = self.agent.logger
        else:
            self.logger = None

        self.old_sum_r = None


    def take_step(self, obs, save_memories=True, render_env=False, act_argmax=False):

        self.obs_mem.append(obs)
        #print(np.squeeze(np.array(self.obs_mem)))
        #action = self.agent.act(np.squeeze(np.array(self.obs_mem)), act_argmax=act_argmax, save_memories=save_memories)
        action = self.agent.act(np.ravel(np.squeeze(np.array(self.obs_mem))), act_argmax=act_argmax, save_memories=save_memories)
        new_obs, r, done, _ = self.env.step(np.squeeze(action))

        if save_memories:
            self.agent.memory({
                    'obs': np.squeeze(np.array(self.obs_mem))
                    #'obs': np.ravel(np.squeeze(np.array(self.obs_mem)))
                }
            )
            self.obs_mem.append(new_obs)
            self.agent.memory({
                    #'obs': np.array(self.obs_mem),
                    'new_obs': np.squeeze(np.array(self.obs_mem)),
                    #'new_obs': np.ravel(np.squeeze(np.array(self.obs_mem))),
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
                #self.env.render()
                reward_list.append(r)

                if done:
                    a_loss, c_loss = self.agent.learn()
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
            n_steps = 2048,
            render_env = False,
            train_on_test = True,
            render_test = True,
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
                self.agent.learn()
                if not self.logger is None:
                    self.logger.log_mean('sum reward', np.round(np.mean(episode_reward_list)))
                self.agent.compare_with_old_policy(np.mean(episode_reward_list))
                episode_reward_list = []

                # Test:
                if train_on_test or render_test:
                    done = False
                    while not done:
                        done, obs, r = self.take_step(obs, save_memories=True, render_env=render_test, act_argmax=True)
                        reward_list.append(r)

                        if done:
                            if not self.logger is None:
                                self.logger.log_mean('test reward', np.round(sum(reward_list), 1))
                            obs = self.env.reset()
                            reward_list = []

                if not self.logger is None:
                    self.logger.print_status(s)

        self.env.close()

    def test(self, num_steps):

        for e in range(num_steps):

            done = False
            obs = self.env.reset()
            reward_list = []

            while not done:
                actions = self.agent.act(obs, act_argmax=True, save_memories=False)
                obs, r, done, _ = self.env.step(np.squeeze(actions))
                self.env.render()
                reward_list.append(r)

                if done:
                    print('Epoch', e, '- Sum R:', sum(reward_list))
                    obs = self.env.reset()
                    reward_list = []

        self.env.close()
