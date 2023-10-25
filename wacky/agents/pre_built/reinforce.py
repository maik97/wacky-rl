import torch
from torch import nn
import numpy as np

from wacky.modules import ModuleConstructor
from wacky.modules.actor_critic import Actor
from wacky.modules.feature_extraction_constructor import FeatureExtractionConstructor
from wacky.optimizer._optimizer_constructor import OptimizerConstructor
from wacky.memory import MemoryDict

from wacky.returns.episode_returns import EpisodeReturns
from wacky.losses.policy_gradient_losses import PolicyGradientLoss


class REINFORCE:

    @classmethod
    def from_env(
            cls,
            env,
            feature_extractor='Flatten',
            policy='SimpleMLP',
            optimizer='Adam',
            optimizer_kwargs=None,
            memory=None,
            returns_fn=None,
            loss_fn=None,
            *args, **kwargs

    ):

        feature_extractor = FeatureExtractionConstructor.from_space(
            feature_extractor=feature_extractor,
            space=env.observation_space,
        )

        policy = ModuleConstructor.construct(policy)

        actor = Actor(
            action_space=env.action_space,
            module=nn.Sequential(feature_extractor, policy)
        )

        return cls(
            actor=actor,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            memory=memory,
            returns_fn=returns_fn,
            loss_fn=loss_fn,
            *args, ** kwargs
        )

    def __init__(
            self,
            actor,
            optimizer='Adam',
            optimizer_kwargs=None,
            memory=None,
            returns_fn=None,
            loss_fn=None,
            *args, **kwargs
    ):
        super(REINFORCE, self).__init__(*args, **kwargs)

        self.actor = actor
        self.optimizer = OptimizerConstructor.construct(optimizer, self.actor.parameters(), optimizer_kwargs)
        self.memory = memory or MemoryDict()
        self.returns_fn = returns_fn or EpisodeReturns()
        self.loss_fn = loss_fn or PolicyGradientLoss()

    def __call__(self, state, deterministic=False):
        return self.actor(state, deterministic=deterministic)

    def learn(self):
        returns = self.returns_fn(
            rewards=self.memory['rewards'],
            done_flags=self.memory['dones']
        )
        loss = self.loss_fn(
            log_probs=self.memory['log_prob'],
            policy_gradient_term=returns,
        )
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.detach()

    def step(self, state, env, remember=True, deterministic=False, render=False):
        state = torch.FloatTensor(state).unsqueeze(0)
        action, log_prob = self.actor(state, deterministic=deterministic)
        state, reward, done, truncated, _ = env.step(action.item())
        done = done or truncated
        if remember:
            self.memory['rewards'].append(reward)
            self.memory['dones'].append(int(done))
            self.memory['log_prob'].append(log_prob.squeeze())
        if render:
            env.render()
        return state, done

    def train(self, env, num_episodes, render=False):

        for e in range(num_episodes):

            self.memory.clear()
            done = False
            state, _ = env.reset()

            while not done:
                state, done = self.step(state, env, remember=True, deterministic=False, render=render)

            self.memory.stack()
            loss = self.learn()
            print(
                'episode:', e,
                'rewards:', self.memory['rewards'].sum().numpy(),
                'probs:', np.round(torch.exp(self.memory['log_prob'].detach()).mean().numpy(), 4),
                'loss:', np.round(loss.numpy(), 2),
            )

    def test(self, env, num_episodes, render=True):

        for e in range(num_episodes):

            self.memory.clear()
            done = False
            state = env.reset()

            while not done:
                state, done = self.step(state, env, remember=True, deterministic=True, render=render)

            self.memory.stack()
            print(
                'episode:', e,
                'rewards:', self.memory['rewards'].sum().numpy(),
                'probs:', np.round(torch.exp(self.memory['log_prob'].detach()).mean().numpy(), 4),
            )


def main():
    import gym
    env = gym.make('CartPole-v0')
    agent = REINFORCE.from_env(env)
    agent.train(env, 1_000)
    agent.test(env, 100, render=False)
    print(agent.actor)


if __name__ == '__main__':
    main()
