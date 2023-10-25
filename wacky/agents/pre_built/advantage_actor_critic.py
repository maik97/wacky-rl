import numpy as np
import torch
import torch as th
from torch import nn

from wacky.losses import PolicyGradientLoss
from wacky.modules import ModuleConstructor
from wacky.memory import MemoryDict
from wacky.modules.actor_critic import Actor, Critic, ActorCritic
from wacky.modules.feature_extraction_constructor import FeatureExtractionConstructor

from wacky.optimizer import OptimizerConstructor
from wacky.returns import GAE


class AdvantageActorCritic:

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
            actor_loss_fn=None,
            critic_loss_fn=None,
            *args, **kwargs

    ):

        feature_extractor = FeatureExtractionConstructor.from_space(
            feature_extractor=feature_extractor,
            space=env.observation_space,
        )

        policy = ModuleConstructor.construct(policy)

        actor_critic = ActorCritic(
            actor=Actor(action_space=env.action_space),
            critic=Critic(critic_layer=nn.LazyLinear, critic_layer_kwargs=dict(out_features=1)),
            shared_module=nn.Sequential(feature_extractor, policy),
        )

        return cls(
            actor_critic=actor_critic,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            memory=memory,
            returns_fn=returns_fn,
            actor_loss_fn=actor_loss_fn,
            critic_loss_fn=critic_loss_fn,
            *args, **kwargs
        )

    def __init__(
            self,
            actor_critic,
            optimizer,
            optimizer_kwargs=None,
            memory=None,
            returns_fn=None,
            actor_loss_fn=None,
            critic_loss_fn=None,
            *args, **kwargs
    ):
        super(AdvantageActorCritic, self).__init__()

        self.actor_critic = actor_critic
        self.optimizer = OptimizerConstructor.construct(optimizer, self.actor_critic.parameters(), optimizer_kwargs)
        self.memory = MemoryDict() if memory is None else memory

        self.returns_fn = GAE(
            discount_factor=0.99,
            smoothing_factor=0.95,
            normalize_advantage=False,
            normalize_returns=False
        ) if returns_fn is None else returns_fn

        self.actor_loss_fn = PolicyGradientLoss() if actor_loss_fn is None else actor_loss_fn
        self.critic_loss_fn = nn.HuberLoss() if critic_loss_fn is None else critic_loss_fn

    def __call__(self, state, deterministic=False):
        return self.actor_critic(state, deterministic=deterministic)

    def learn(self):
        self.memory.stack()

        advantage, returns = self.returns_fn(
            rewards=self.memory['rewards'],
            values=self.memory['values'],
            done_flags=self.memory['done_flags'],
        )

        loss_actor = self.actor_loss_fn(
            log_probs=self.memory['log_prob'],
            policy_gradient_term=advantage
        )

        loss_critic = self.critic_loss_fn(
            input=self.memory['values'],
            target=returns
        )

        loss = loss_actor + loss_critic

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss_actor, loss_critic, loss

    def step(self, state, env, remember=True, deterministic=False, render=False):
        state = th.FloatTensor(state).unsqueeze(0)
        (action, log_prob), value = self.actor_critic(state, deterministic=deterministic)
        state, reward, done, truncated, _ = env.step(action.item())
        done = done or truncated
        if remember:
            self.memory['log_prob'].append(log_prob.squeeze())
            self.memory['values'].append(value.squeeze())
            self.memory['rewards'].append(reward)
            self.memory['done_flags'].append(int(done))
        if render:
            env.render()
        return state, reward, done

    def train(self, env, num_episodes, mini_batch_size=32, render=False):

        step_counter = 0
        for e in range(num_episodes):

            done = False
            state, _ = env.reset()

            episode_rewards = []
            losses = []

            while not done:
                step_counter += 1
                state, reward, done = self.step(state, env, remember=True, deterministic=False, render=render)
                episode_rewards.append(reward)

                if step_counter >= mini_batch_size:
                    self.memory.stack()
                    loss_actor, loss_critic, total_loss = self.learn()
                    step_counter = 0
                    self.memory.clear()
                    losses.append(total_loss.item())

            print(
                'episode:', e,
                'rewards:', np.sum(episode_rewards),
                'loss:', np.round(np.mean(losses), 2),
            )

    def test(self, env, num_episodes, render=False):

        for e in range(num_episodes):

            done = False
            state, _ = env.reset()

            while not done:
                state, reward, done = self.step(state, env, remember=True, deterministic=True, render=render)

            self.memory.stack()
            loss_actor, loss_critic, total_loss = self.learn()
            print('episode:', e,
                  'rewards:', self.memory['rewards'].sum().numpy(),
                  'probs:', np.round(torch.exp(self.memory['log_prob'].detach()).mean().numpy(), 4),
                  'loss:', np.round(total_loss.item(), 2),
                  'loss_actor:', np.round(loss_actor.item(), 2),
                  'loss_critic:', np.round(loss_critic.item(), 2),
                  )
            self.memory.clear()


def main():
    import gym
    env = gym.make('CartPole-v0')
    # env = gym.make('LunarLanderContinuous-v2')

    agent = AdvantageActorCritic.from_env(env)
    agent.train(env, 2000)
    agent.test(env, 100, render=True)

    print(agent.actor_critic)


if __name__ == '__main__':
    main()
