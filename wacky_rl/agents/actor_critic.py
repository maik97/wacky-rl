from wacky_rl.cores.actor_critic_core import ActorCriticCore

class A2C(ActorCriticCore):

    def __init__(self, env):

        self.env = env
        super().__init__(n_actions = self.decode_action_space(self.env.action_space))

    def training(self, steps):

        state = self.env.reset()
        rewards = []

        for t in range(steps):
            state, reward, done, _ = self.env.step(self.act(state))
            rewards.append(reward)

            if done:
                loss = self.train(rewards)
                print('sum_reward:', sum(rewards), 'loss:', loss)
                state = self.env.reset()
                rewards = []

    def testing(self, steps):

        state = self.env.reset()

        rewards = []
        for t in range(steps):
            state, reward, done, _ = self.env.step(self.act(state))
            rewards.append(reward)
            self.env.render()

            if done:
                print('sum_reward:', sum(rewards))
                state = self.env.reset()
                rewards = []
