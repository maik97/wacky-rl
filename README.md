# Wacky-RL

Create custom reinforcement learning agents with wacky-rl.

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
- gym >= 0.17.3

## Documentation 

See the documentation for a detailed explanation on creating your own agents with Wacky-RL.

## Example

Additionally to creating your own agents, you can use the prebuilt agents:

```python
import gym

from wacky_rl import MultiAgentCompiler
from wacky_rl.agents import DiscreteActorCriticCore

agent = MultiAgentCompiler(gym.make('CartPole-v0'), log_dir='_logs')
agent.assign_agent(DiscreteActorCriticCore(), at_index=0
agent = agent.build(max_steps_per_episode=None)
agent.train_agent(500)
```

## Prebuilt Agents

- [ ] DQN
- [x] Discrete Actor Critic 
- [ ] Continuous Actor Critc
- [ ] SAC
- [ ] PPO


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