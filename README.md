# Custom Reinforcement Learning Environments

Create your own custom reinforcement learning agents (or use the prebuilt agents).
With it's modular approach `wacky-rl` makes the implementation of reinforcement learning easy and flexible - without restricting you!

See the [documentation](https://wacky-rl.rtfd.io) for a detailed explanation on creating your own agents with `wacky-rl`.
For some examples check out the prebuilt [agents](https://github.com/maik97/wacky-rl/tree/master/wacky/agents/pre_built).

## Prebuilt Agents

- [X] DQN to RAINBOW 
  [[code]](https://github.com/maik97/wacky-rl/blob/master/wacky/agents/pre_built/deep_q_network.py),
  [[1]](http://arxiv.org/abs/1312.5602),
  [[2]](https://www.nature.com/articles/nature14236),
  [[3]](http://arxiv.org/abs/1509.06461),
  [[4]](https://arxiv.org/abs/1710.02298)
- [x] REINFORCE 
  [[code]](https://github.com/maik97/wacky-rl/blob/master/wacky/agents/pre_built/reinforce.py),
  [[5]](https://people.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf)
- [x] A2C 
  [[code]](https://github.com/maik97/wacky-rl/blob/master/wacky/agents/pre_built/advantage_actor_critic.py),
  [[6]](https://arxiv.org/abs/1602.01783)
- [ ] SAC
  [code],
  [[7]](https://arxiv.org/pdf/1801.01290.pdf),
  [[8]](https://arxiv.org/pdf/1812.05905.pdf)
- [x] PPO
  [[code]](https://github.com/maik97/wacky-rl/blob/master/wacky/agents/pre_built/proximal_policy_optimization.py),
  [[9]](https://arxiv.org/abs/1707.06347),
  [[10]](http://proceedings.mlr.press/v37/schulman15.pdf)

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

- torch
- gym >= 0.17.3
- numpy

