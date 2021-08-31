import numpy as np
from wacky_rl import layers

class AgentCore:

    def __init__(self):
        self.approx_contin = False

    def __call__(self, *args, **kwargs):
        return self.act( *args, **kwargs)

    def act(self, inputs, act_argmax=False):
        raise NotImplementedError('When subclassing the `AgentCore` class, you should '
                                  'implement a `act` method.')

    def learn(self, *args, **kwargs):
        raise NotImplementedError('When subclassing the `AgentCore` class, you should '
                                  'implement a `train` method.')

    def decode_space(self, gym_space):

        from gym import spaces

        if isinstance(gym_space, spaces.Box):
            import numpy as np
            return int(np.squeeze(gym_space.shape))

        elif isinstance(gym_space, spaces.Discrete):
            return int(gym_space.n)

        else:
            raise AttributeError('gym_space not understood: {}, use space.Box or space.Discrete'.format(gym_space))

    def space_is_contin(self, gym_space):
        from gym import spaces
        return isinstance(gym_space, spaces.Box)

    def space_is_discrete(self, gym_space):
        from gym import spaces
        return isinstance(gym_space, spaces.Discrete)

    def make_action_layer(self, env, num_bins=21, num_actions=None, approx_contin=False):

        if num_actions is None:
            num_actions = int(self.decode_space(env.action_space))

        if self.space_is_discrete(env.action_space):
            return layers.DiscreteActionLayer(num_bins=num_actions)
        elif approx_contin:
            self.approx_contin = True
            return layers.DiscreteActionLayer(num_bins=num_bins, num_actions=num_actions)
        else:
            return layers.ContinActionLayer(num_actions=num_actions)


    def transform_actions(self, dist, actions, lows=-1.0, highs=1.0, scale=1.0):

        if self.approx_contin:
            actions = dist.discrete_to_contin(actions)

        actions = np.squeeze(actions.numpy()) * scale
        return np.clip(actions, a_min=lows, a_max=highs)
