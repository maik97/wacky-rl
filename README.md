# Wacky-RL

Create custom reinforcement learning agents with `wacky-rl`.

## Installation

- Install Wacky-RL with pip:

```
pip install wacky-rl
```

- Install with git:

```
git clone https://github.com/maik97/wacky-rl.git
cd wacky-rl
python setup.py install
```

## Dependencies

- tensorflow >= 2.5
- tensorflow-probability >= 0.12.2
- gym >= 0.17.3

## Documentation 

See the [documentation](https://wacky-rl.rtfd.io) for a detailed explanation on creating your own agents with `wacky-rl`.
For some examples check out the [agents](https://github.com/maik97/wacky-rl/tree/master/wacky_rl/agents).

## Examples

Alternatively you can use the prebuilt agents.

####A2C:
```python
import gym
from wacky_rl.agents import A2C
from wacky_rl.trainer import Trainer

env = gym.make('CartPole-v0')
# env = gym.make("LunarLanderContinuous-v2")
agent = A2C(env)

trainer = Trainer(env, agent)
trainer.episode_train(300)
trainer.test(100)
```

####PPO:
```python
import gym
from wacky_rl.agents import PPO
from wacky_rl.trainer import Trainer

# env = gym.make('CartPole-v0')
env = gym.make("LunarLanderContinuous-v2")
agent = PPO(env)

trainer = Trainer(env, agent)
trainer.n_step_train(5_000_000)
trainer.test(100)
```

## Prebuilt Agents

- [ ] DQN
- [x] A2C 
- [ ] SAC
- [x] PPO


## Citing

If you use `wacky-rl` in your research, you can cite it as follows:

```bibtex
@misc{schürmann2021wackyrl,
    author = {Maik Schürmann},
    title = {wacky-rl},
    year = {2021},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/maik97/wacky-rl}},
}
```
