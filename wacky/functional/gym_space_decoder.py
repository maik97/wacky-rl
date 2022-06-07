from gym import spaces

def decode_gym_space(space, allowed_spaces=None):

    if not isinstance(space, spaces.Space):
        raise TypeError("space must be of type gym.space.Spaces, not", type(space))

    if allowed_spaces is None:
        allowed_spaces = [
            spaces.Box, spaces.Discrete, spaces.MultiDiscrete, spaces.MultiBinary, spaces.Tuple, spaces.Dict
        ]

    if isinstance(space, spaces.Box):
        if not spaces.Box in allowed_spaces:
            raise TypeError('spaces.Box was not allowed')
        else:
            return space.shape

    elif isinstance(space, spaces.Discrete):
        if not spaces.Discrete in allowed_spaces:
            raise TypeError('spaces.Discrete was not allowed')
        else:
            return space.n

    elif isinstance(space, spaces.MultiDiscrete):
        if not spaces.MultiDiscrete in allowed_spaces:
            raise TypeError('spaces.MultiDiscrete was not allowed')
        else:
            return space.nvec

    elif isinstance(space, spaces.MultiBinary):
        if not spaces.MultiBinary in allowed_spaces:
            raise TypeError('spaces.MultiBinary was not allowed')
        else:
            return space.n

    elif isinstance(space, spaces.Tuple):
        if not spaces.Tuple in allowed_spaces:
            raise TypeError('spaces.Tuple was not allowed')
        else:
            return [decode_gym_space(subspace) for subspace in space]

    elif isinstance(space, spaces.Dict):
        if not spaces.Dict in allowed_spaces:
            raise TypeError('spaces.Dict was not allowed')
        else:
            return [decode_gym_space(subspace) for subspace in space]
