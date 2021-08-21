# Wacky-RL

Create your own custom reinforcement learning agents (or use the prebuilt agents).
With it's modular approach `wacky-rl` makes the implementation of reinforcement learning easy and flexible - without restricting you!

Wacky-RL uses [Tensorflow 2](https://www.tensorflow.org/install) and you can create WackyModel's based on [Keras](https://keras.io/).

See the [documentation](https://wacky-rl.rtfd.io) for a detailed explanation on creating your own agents with `wacky-rl`.
For some examples check out the prebuilt [agents](https://github.com/maik97/wacky-rl/tree/master/wacky_rl/agents).

## Prebuilt Agents

- [ ] DQN
- [x] A2C 
- [ ] SAC
- [x] PPO

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

## Examples

#### Prebuilt A2C:
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

#### Prebuilt PPO:
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

#### Create your own Model:
```python
import wacky_rl
import tensorflow as tf

N_ACTIONS = 2

# Create your model like a Keras Sequential Model, https://keras.io/guides/sequential_model/
model = wacky_rl.models.WackyModel() # A list of layers can also be passed directly
model.add(tf.keras.layers.Dense(64))
model.add(tf.keras.layers.Dense(64))
model.add(wacky_rl.layers.DiscreteActionLayer(num_bins=N_ACTIONS))

# Alternatively create your model with the Keras Functional API, https://keras.io/guides/functional_api/
input_layer = tf.keras.layers.Input(shape=env.observation_space.shape)
hidden_dense = tf.keras.layers.Dense(256, activation='relu')(input_layer)
hidden_dense = tf.keras.layers.Dense(256, activation='relu')(hidden_dense)
output_layer = wacky_rl.layers.DiscreteActionLayer(num_bins=N_ACTIONS)
model = wacky_rl.models.WackyModel(inputs=input_layer, outputs=output_layer)

# If you choose to create your model by subclassing WackyModel instead,
# make sure to create a call() method, see: https://keras.io/guides/making_new_layers_and_models_via_subclassing/

# Compile the model:
model.compile(
    optimizer=tf.keras.optimizers.RMSprop(3e-4, clipnorm=0.5),
    loss=wacky_rl.losses.ActorLoss(entropy_factor=0.001)
)

# Call the model:
model(...)

# Train the model (don't use fit() or train_on_batch() when using a WackyLoss):
model.train_step(...)
```

#### Create your own Agent by subclassing:

```python
import tensorflow as tf
import wacky_rl
from wacky_rl.agents import AgentCore


class CustomAgent(AgentCore):

    def __init__(self, env):
        super().__init__()
        
        # Use the space_is_discrete() for flexible actions
        if self.space_is_discrete(env.action_space):
            out_layer = wacky_rl.layers.DiscreteActionLayer(...)
        else:
            out_layer = wacky_rl.layers.ContinActionLayer(...)
        
        # Create your models:
        self.actor = ...
        self.critic = ...
        
        # Add a memory:
        self.memory = wacky_rl.memory.BufferMemory()
        
        # In most cases you want to calculate returns or rewards, check out to the transform folder more Classes.
        self.calc_returns = wacky_rl.transform.ExpectedReturnsCalculator()
        
    def act(self, inputs, act_argmax=False, save_memories=True):
        inputs = tf.expand_dims(tf.squeeze(inputs), 0)
        dist = self.actor(inputs, act_argmax=True)
        
        if act_argmax: # if testing
            actions = dist.mean_actions()
        else:
            actions = dist.sample_actions()

        if save_memories:
            self.memory(actions, key='actions')

        return actions.numpy()

    def learn(self): # Example from a2c
        
        actions, states, new_states, rewards, dones = self.memory.replay()

        values = self.critic.predict(states)
        returns = self.calc_returns(rewards)

        adv = returns - values
        adv = tf.squeeze(adv)
        adv = (adv - tf.reduce_mean(adv)) / (tf.math.reduce_std(adv) + 1e-8)

        c_loss = self.critic.train_step(states, returns)
        a_loss = self.actor.train_step(states, actions=actions, advantage=adv)

        self.memory.clear()

        return a_loss.numpy(), c_loss.numpy()
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
