from gym import spaces
import torch
import torch.nn as nn

from wacky.modules import ModuleConstructor


class FlattenModule(nn.Module):
    def __init__(self, flatten_instruction):
        super(FlattenModule, self).__init__()
        self.flatten_instruction = flatten_instruction

    def forward(self, x):
        x = self._flatten_obs(x, self.flatten_instruction)
        return x

    def _flatten_obs(self, obs, instruction):

        if instruction == "same":
            return obs

        elif instruction[0] == "flatten":
            orig_shape = obs.shape  # Original shape
            target_shape = instruction[1]  # Target shape, e.g., (2, 2)

            if len(orig_shape) == len(target_shape):
                return torch.flatten(obs, start_dim=0, end_dim=-1)
            else:
                return torch.flatten(obs, start_dim=-len(target_shape), end_dim=-1)
                # leading_shape = orig_shape[:-len(target_shape)]  # if orig_shape (64, 3, 3), leading_shape (64,)
                # trailing_shape = orig_shape[-len(target_shape):]  # if orig_shape is (64, 3, 3), trailing_shape (3, 3)
                # Make sure the trailing_shape matches target_shape
                # assert trailing_shape == target_shape, (
                #    f"Trailing shape {trailing_shape} does not match target shape {target_shape}"
                # )
                # return obs.reshape(*leading_shape, -1)  # -1 computes the remaining dimensions

        elif instruction[0] == "tuple":
            flat_list = []
            for o, inst in zip(obs, instruction[1]):
                flat_list.append(self._flatten_obs(o, inst))
            return torch.cat(flat_list, dim=-1)

        elif instruction[0] == "dict":
            flat_list = []
            for key in instruction[1].keys():
                flat_list.append(self._flatten_obs(obs[key], instruction[1][key]))
            return torch.cat(flat_list, dim=-1)


class FeatureExtractionConstructor:

    @staticmethod
    def flatten_instructions(space):

        if isinstance(space, (spaces.Discrete, spaces.MultiBinary, spaces.MultiDiscrete)):
            return "same"

        elif isinstance(space, spaces.Box):
            return ("flatten", space.shape)

        elif isinstance(space, spaces.Tuple):
            components = [FeatureExtractionConstructor.flatten_instructions(subspace) for subspace in space.spaces]
            return ("tuple", components)

        elif isinstance(space, spaces.Dict):
            components = {key: FeatureExtractionConstructor.flatten_instructions(subspace) for key, subspace in
                          space.spaces.items()}
            return ("dict", components)

        else:
            raise ValueError(f"Unknown space {space}")

    @staticmethod
    def from_space(feature_extractor, space):
        if feature_extractor == 'Flatten':
            flatten_instruction = FeatureExtractionConstructor.flatten_instructions(space)
            return FlattenModule(flatten_instruction)
        else:
            raise NotImplementedError(f'feature_extractor: {feature_extractor}')

    @staticmethod
    def from_module(feature_extractor):
        return ModuleConstructor.construct(feature_extractor)


if __name__ == "__main__":
    nested_dict_space = spaces.Dict({
        "first_level": spaces.Dict({
            "discrete": spaces.Discrete(4),
            "box": spaces.Box(low=0, high=1, shape=(2, 2))
        }),
        "second_level": spaces.Dict({
            "discrete": spaces.Discrete(5),
            "box": spaces.Box(low=0, high=1, shape=(3, 3))
        })
    })

    flatten_module = FeatureExtractionConstructor.from_space('Flatten', nested_dict_space)

    nested_dict_obs = {
        "first_level": {"discrete": torch.tensor([[1, 0]]), "box": torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])},
        "second_level": {"discrete": torch.tensor([[0, 1]]),
                         "box": torch.tensor([[[5.0, 6.0, 7.0], [8.0, 9.0, 10.0], [11.0, 12.0, 13.0]]])},
    }

    print("Nested dict obs:", nested_dict_obs)
    flat_output = flatten_module(nested_dict_obs)
    print("Flat output:", flat_output)
    print("Flat output shape:", flat_output.shape)


