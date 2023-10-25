from collections import defaultdict

import torch
from torch import nn, optim
import numpy as np

from wacky.logging.pretty_print import PrettyPrinter
from wacky.losses import ClippedSurrogateLoss
from wacky.modules import ModuleConstructor
from wacky.memory import MemoryDict

from wacky.modules.actor_critic import ActorCritic, Actor, Critic
from wacky.modules.feature_extraction_constructor import FeatureExtractionConstructor

from wacky.optimizer import OptimizerConstructor
from wacky.returns import GAE


# From stable baselines
def explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """
    Computes fraction of variance that ypred explains about y.
    Returns 1 - Var[y-ypred] / Var[y]

    interpretation:
        ev=0  =>  might as well have predicted zero
        ev=1  =>  perfect prediction
        ev<0  =>  worse than just predicting zero

    :param y_pred: the prediction
    :param y_true: the expected value
    :return: explained variance of ypred and y
    """
    assert y_true.ndim == 1 and y_pred.ndim == 1
    var_y = np.var(y_true)
    return np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y


class PPO:

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

        module_actor = ModuleConstructor.construct(policy)
        module_critic = ModuleConstructor.construct(policy)

        actor_critic = ActorCritic(
            shared_module=feature_extractor,
            actor=Actor(
                module=module_actor,
                action_space=env.action_space
            ),
            critic=Critic(
                module=module_critic,
                critic_layer=nn.LazyLinear,
                critic_layer_kwargs=dict(out_features=1)
            ),
        )

        def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
            if hasattr(layer, 'weight') and layer.weight is not None:
                nn.init.orthogonal_(layer.weight, std)
            if hasattr(layer, 'bias') and layer.bias is not None:
                nn.init.constant_(layer.bias, bias_const)
            layer.initialized = True  # Mark the layer as initialized

        def recursive_init(module, std=np.sqrt(2), bias_const=0.0, skip_initialized=True):
            if skip_initialized and hasattr(module, 'initialized') and module.initialized:
                return
            layer_init(module, std=std, bias_const=bias_const)
            for child in module.children():
                recursive_init(child, std=std, bias_const=bias_const, skip_initialized=skip_initialized)

        # Dummy Forward (Creates LazyModule Parameters):
        dummy_state, _ = env.reset()
        actor_critic.forward(torch.FloatTensor(dummy_state).unsqueeze(0))

        # Init Parameters:
        recursive_init(actor_critic.actor.action_layer, std=0.01)  # policy output weights init with scale 0.01
        recursive_init(actor_critic.critic.critic_layer, std=1)  # critic output weights init with scale 1
        recursive_init(actor_critic, std=np.sqrt(2))  # hidden weights init with scale np.sqrt(2)

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
            loss_conf=None,
    ):
        super(PPO, self).__init__()

        self.actor_critic = actor_critic

        self.optimizer = OptimizerConstructor.construct(
            optimizer=optimizer,
            params=self.actor_critic.parameters(),
            optimizer_kwargs=dict(lr=3e-4) if optimizer_kwargs is None else optimizer_kwargs
        )

        self.memory = MemoryDict() if memory is None else memory

        self.returns_fn = GAE(
            discount_factor=0.99,
            smoothing_factor=0.95,
            normalize_advantage=False,
            normalize_returns=False
        ) if returns_fn is None else returns_fn

        self.actor_loss_fn = ClippedSurrogateLoss() if actor_loss_fn is None else actor_loss_fn
        self.critic_loss_fn = nn.MSELoss() if critic_loss_fn is None else critic_loss_fn

        self.loss_conf = {
            'max_grad_norm': 0.5,
            'actor_coef': 1.0,
            'critic_coef': 0.5,
            'entropy_coef': 0.5,
        } if loss_conf is None else loss_conf

    def __call__(self, state, deterministic=False):
        return self.actor_critic(state, deterministic=deterministic)

    def step(self, state, env, remember=True, deterministic=False, render=False):
        if remember:
            self.memory['states'].append(state.squeeze())
        state = torch.FloatTensor(state).unsqueeze(0)

        (action, log_prob), value = self.actor_critic(state, deterministic=deterministic)
        state, reward, done, truncated, _ = env.step(action.item())
        done = done or truncated
        if remember:
            self.memory['old_log_prob'].append(log_prob.squeeze().detach())
            self.memory['values'].append(value.squeeze().detach())
            self.memory['actions'].append(action.squeeze().detach())
            self.memory['rewards'].append(reward)
            self.memory['done_flags'].append(int(done))
        if render:
            env.render()
        return state, reward, done

    def learn(self, next_state, epochs, batch_size):

        next_state = torch.FloatTensor(next_state).unsqueeze(0)
        next_value = self.actor_critic.critic_forward(next_state)

        self.memory.stack()
        self.memory['adv'], self.memory['returns'] = self.returns_fn(
            rewards=self.memory['rewards'].squeeze(),
            values=self.memory['values'].squeeze(),
            done_flags=self.memory['done_flags'].squeeze(),
            next_value=next_value.detach()
        )

        metrics = defaultdict(list)
        for e in range(epochs):
            for batch in self.memory.batch(batch_size, shuffle=False):

                #if len(batch['states']) < 2:
                #    continue

                (log_probs, entropy), values = self.actor_critic.eval_action(batch['states'], batch['actions'])

                advantages = batch['adv'].unsqueeze(-1)
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                loss_actor, ratio, clip_fraction, *_ = self.actor_loss_fn.forward(
                    log_probs=log_probs.unsqueeze(-1),
                    old_log_probs=batch['old_log_prob'].unsqueeze(-1),
                    policy_gradient_term=advantages
                )

                loss_critic = self.critic_loss_fn(
                    input=values,
                    target=batch['returns'].unsqueeze(-1)
                )

                if entropy is None:
                    loss_entropy = -torch.mean(-log_probs)
                else:
                    loss_entropy = -torch.mean(entropy)

                loss = (
                    self.loss_conf['actor_coef'] * loss_actor
                    + self.loss_conf['critic_coef'] * loss_critic
                    + self.loss_conf['entropy_coef'] * loss_entropy
                )

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.loss_conf['max_grad_norm'])
                self.optimizer.step()

                with torch.no_grad():
                    log_ratio = log_probs - batch['old_log_prob'].unsqueeze(-1)
                    approx_kl_div = ((log_ratio - 1).exp() - log_ratio)

                metrics['loss_actor'].append(loss_actor.item())
                metrics['loss_critic'].append(loss_critic.item())
                metrics['loss_entropy'].append(loss_entropy.item())
                metrics['ratio'].append(ratio.mean().item())
                metrics['clip_fraction'].append(clip_fraction.mean().item())
                metrics['approx_kl_div'].append(approx_kl_div.mean().item())

        expl_variance = explained_variance(
            y_pred=self.memory['values'].squeeze().numpy(),
            y_true=self.memory['returns'].squeeze().numpy()
        )
        metrics['expl_variance'].append(expl_variance.mean())
        return metrics

    def train(self, env, num_steps, epochs=1, batch_size=64, train_interval=2048, render=False):

        pretty_printer = PrettyPrinter()

        done = True
        episode_counter = 0
        episode_reward_list = []
        episode_reward = 0
        step_count = 0
        for t in range(num_steps):

            if done:
                state, _ = env.reset()
                episode_reward_list.append(episode_reward)
                episode_reward = 0
                episode_counter += 1

            if step_count + 1 >= train_interval:
                # Update:
                metrics = self.learn(next_state=state, epochs=epochs, batch_size=batch_size)

                # Print Progress:
                pretty_printer.add_section('progress/', {
                    'steps': t + 1,
                    'episodes': episode_counter,
                })
                pretty_printer.add_section('window/', {
                    'mean_ep_rew': np.round(np.mean(episode_reward_list), 4),
                    'mean_probs': np.round(np.exp(self.memory['old_log_prob'].mean().item()), 4),
                })
                pretty_printer.add_section('estimates/', {
                    'mean_advantages': np.round(self.memory['adv'].mean().item(), 4),
                    'mean_returns': np.round(self.memory['returns'].mean().item(), 4),
                })
                pretty_printer.add_section('losses/', {
                    'actor': np.round(np.mean(metrics['loss_actor']), 4),
                    'critic': np.round(np.mean(metrics['loss_critic']), 4),
                    'entropy': np.round(np.mean(metrics['loss_entropy']), 4),
                })
                pretty_printer.add_section('other/', {
                    'ratio': np.round(np.mean(metrics['ratio']), 4),
                    'clip_fraction': np.round(np.mean(metrics['clip_fraction']), 4),
                    'approx_kl_div': np.round(np.mean(metrics['approx_kl_div']), 4),
                    'expl_variance': np.round(np.mean(metrics['expl_variance']), 4),
                })
                pretty_printer.dump()

                # Clear:
                episode_reward_list = []
                self.memory.clear()
                step_count = 0

            # Step:
            state, reward, done = self.step(state, env, deterministic=False, render=render)
            episode_reward += reward
            step_count += 1

    def test(self, env, num_episodes, render=True):

        for e in range(num_episodes):

            state, _ = env.reset()
            episode_reward = 0
            done = False
            while not done:
                state, reward, done = self.step(state, env, deterministic=True, render=render)
                episode_reward += reward

                if render:
                    env.render()

            print(
                'episode:', e,
                'rewards:', episode_reward,
                'prob:', np.round(np.exp(self.memory.numpy('old_log_prob', reduce='mean')), 3),
            )
            self.memory.clear()


def main():
    import gym
    env = gym.make('LunarLander-v2', max_episode_steps=500)

    #from stable_baselines3 import PPO as SB3PPO
    #sb3_agent = SB3PPO(policy='MlpPolicy', env=env, verbose=1)
    #sb3_agent.learn(total_timesteps=2_000_000)

    agent = PPO.from_env(env)
    print(agent.actor_critic)

    agent.train(env, 20_000_000, epochs=10, batch_size=64)
    wait = input('press enter')

    env = gym.make('LunarLander-v2', render_mode="human", max_episode_steps=500)
    agent.test(env, 100)


if __name__ == '__main__':
    main()
