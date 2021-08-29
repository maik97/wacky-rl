import numpy as np

class Trainer:

    def __init__(
            self,
            env,
            agent,
    ):

        self.env = env
        self.agent = agent

    def sample_warmup(self, num_episodes):
        for e in range(num_episodes):

            done = False
            obs = self.env.reset()

            while not done:

                action = self.agent.act(obs)
                new_obs, r, done, _ = self.env.step(np.squeeze(action))
                self.agent.memory({
                    'obs': obs,
                    'new_obs': new_obs,
                    'rewards': r,
                    'dones': float(1 - int(done)),
                }
                )
                obs = new_obs

    def episode_train(self, num_episodes):
        for e in range(num_episodes):

            done = False
            obs = self.env.reset()
            reward_list = []

            while not done:

                action = self.agent.act(obs)
                new_obs, r, done, _ = self.env.step(np.squeeze(action))
                reward_list.append(r)
                self.agent.memory({
                    'obs': obs,
                    'new_obs': new_obs,
                    'rewards': r,
                    'dones': float(1 - int(done)),
                }
                )
                obs = new_obs
                self.env.render()

                if done:
                    a_loss, c_loss = self.agent.learn()
                    print()
                    print('# Episode', e)
                    print('# Sum R:', np.round(np.sum(reward_list), 1))
                    print('# Loss A:', np.round(np.mean(a_loss), 4))
                    print('# Loss C:', np.round(np.mean(c_loss), 4))

    def n_step_train(
            self,
            num_steps,
            n_steps = 2048,
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

            while not done:

                actions = self.agent.act(obs)
                new_obs, r, done, _ = self.env.step(np.squeeze(actions))

                self.agent.memory({
                    'obs': obs,
                    'new_obs': new_obs,
                    'rewards': r,
                    'dones': float(1 - int(done)),
                }
                )
                reward_list.append(r)
                obs = new_obs
                s += 1

                if done:
                    obs = self.env.reset()
                    episode_reward_list.append(np.sum(reward_list))
                    reward_list = []

            if s >= train_after:
                train_after += n_steps
                a_loss, c_loss = self.agent.learn()
                print()
                print('# Steps', s)
                print('# Mean R:', np.round(np.mean(episode_reward_list), 1))
                print('# Loss A:', np.round(np.mean(a_loss), 4))
                print('# Loss C:', np.round(np.mean(c_loss), 4))
                episode_reward_list = []

                # Test:
                if train_on_test or render_test:
                    done = False
                    while not done:
                        actions = self.agent.act(obs, act_argmax=True, save_memories=train_on_test)
                        new_obs, r, done, _ = self.env.step(np.squeeze(actions))

                        if train_on_test:
                            self.agent.memory({
                                'obs': obs,
                                'new_obs': new_obs,
                                'rewards': r,
                                'dones': float(1 - int(done)),
                            })

                        if render_test:
                            self.env.render()

                        reward_list.append(r)
                        obs = new_obs

                        if done:
                            print('# Test R:', np.round(sum(reward_list), 1))
                            obs = self.env.reset()
                            reward_list = []

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
