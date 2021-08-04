# Wacky-RL

Create custom reinforcement learning agents with wacky-rl.

## Installation

- Install Wacky-RL from Pypi (recommended):

```
pip install wacky-rl
```

- Install from Github source:

```
git clone https://github.com/maik97/wacky-rl.git
cd wacky-rl
python setup.py install
```

## Example

```python
import gym

from wacky_rl import MultiAgentCompiler
from wacky_rl.agents import DiscreteActorCriticCore

agent = MultiAgentCompiler(gym.make('CartPole-v0'), log_dir='_logs')
agent.assign_agent(DiscreteActorCriticCore())
agent = agent.build(max_steps_per_episode=None)
agent.train_agent(500)
```

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