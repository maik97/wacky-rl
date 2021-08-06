

class AgentCore:

    def __init__(self):
        pass

    def __call__(self, inputs):
        return self.act(inputs)

    def act(self, inputs):
        raise NotImplementedError('Method "act" must be implemented!')


    def train(self, rewards):

        raise NotImplementedError('Method "train" must be implemented!')

    def decode_space(self, gym_space):

        from gym import spaces

        if isinstance(gym_space, spaces.Box):
            import numpy as np
            return int(np.squeeze(gym_space.shape))

        elif isinstance(gym_space, spaces.Discrete):
            return int(gym_space.n)

        else:
            raise AttributeError('gym_space not understood: {}, use space.Box or space.Discrete'.format(gym_space))