import unittest

import numpy as np
import torch
from gym import spaces
from torch import nn
from torch.distributions import Normal, Categorical

from wacky.modules.distributions import DistributionLayer, DistributionConstructor, DiscreteDistribution, \
    ContinuousDistribution


class TestContinuousDistribution(unittest.TestCase):

    def test_make_dist(self):
        # Create a ContinuousDistribution instance
        cont_dist = ContinuousDistribution(distribution_type=Normal, min_sigma=0.5)

        # Create dummy mu and sigma
        mu = torch.Tensor([0.0])
        sigma = torch.Tensor([0.2])

        # Make the distribution
        dist = cont_dist.make_dist((mu, sigma))

        # Test if min_sigma is applied correctly
        self.assertEqual(dist.stddev.item(), 0.5)

        # Test if the distribution is of correct type
        self.assertIsInstance(dist, Normal)

    def test_forward(self):
        cont_dist = ContinuousDistribution(distribution_type=Normal)

        mu = torch.Tensor([0.0])
        sigma = torch.Tensor([0.2])

        action, log_prob = cont_dist.forward((mu, sigma))

        # Verify the action and log_prob are tensors
        self.assertIsInstance(action, torch.Tensor)
        self.assertIsInstance(log_prob, torch.Tensor)

    def test_eval_action(self):
        cont_dist = ContinuousDistribution(distribution_type=Normal)

        mu = torch.Tensor([0.0])
        sigma = torch.Tensor([0.2])

        log_prob = cont_dist.eval_action((mu, sigma), torch.Tensor([0.0]))

        # Verify log_prob is a tensor
        self.assertIsInstance(log_prob, torch.Tensor)


# Similar tests for DiscreteDistribution
class TestDiscreteDistribution(unittest.TestCase):

    def test_make_dist(self):
        dist = DiscreteDistribution(activation="Softmax", distribution_type=Categorical)

        logits = torch.Tensor([[0.2, 0.8]])

        # Make distribution
        distribution = dist.make_dist(logits)

        # Check if it returns a Categorical Distribution object
        self.assertIsInstance(distribution, Categorical)

        # Check if probabilities sum to 1
        self.assertAlmostEqual(torch.sum(torch.exp(distribution.logits)).item(), 1.0, places=6)

    def test_forward(self):
        dist = DiscreteDistribution(activation="Softmax", distribution_type=Categorical)

        logits = torch.Tensor([[0.2, 0.8]])

        action, log_prob = dist.forward(logits)

        self.assertTrue(torch.is_tensor(action))
        self.assertTrue(torch.is_tensor(log_prob))

    def test_eval_action(self):
        dist = DiscreteDistribution(activation="Softmax", distribution_type=Categorical)

        logits = torch.Tensor([[0.2, 0.8]])
        action = torch.Tensor([1])

        log_prob = dist.eval_action(logits, action)

        self.assertTrue(torch.is_tensor(log_prob))


class TestDistributionConstructor(unittest.TestCase):

    def test_get_total_output_dimension(self):
        # Test with Discrete space
        space = spaces.Discrete(5)
        self.assertEqual(DistributionConstructor.get_total_output_dimension(space), 5)

        # Test with Box space
        space = spaces.Box(low=0, high=1, shape=(3,), dtype=float)
        self.assertEqual(DistributionConstructor.get_total_output_dimension(space), 6)

        # Test with MultiBinary space
        space = spaces.MultiBinary(4)
        self.assertEqual(DistributionConstructor.get_total_output_dimension(space), 4)

        # Test with MultiDiscrete space
        space = spaces.MultiDiscrete([2, 3])
        self.assertEqual(DistributionConstructor.get_total_output_dimension(space), 5)

        # Test with Tuple space
        space = spaces.Tuple((spaces.Discrete(2), spaces.Discrete(3)))
        self.assertEqual(DistributionConstructor.get_total_output_dimension(space), 5)

        # Test with Dict space
        space = spaces.Dict({"a": spaces.Discrete(2), "b": spaces.Discrete(3)})
        self.assertEqual(DistributionConstructor.get_total_output_dimension(space), 5)

    def test_construct_layer(self):
        space = spaces.Discrete(5)
        in_features = 4
        layer = nn.Linear
        layer_args = {'bias': False}
        constructed_layer = DistributionConstructor.construct_layer(in_features, space, layer, layer_args)
        self.assertIsInstance(constructed_layer, nn.Linear)
        self.assertEqual(constructed_layer.in_features, 4)
        self.assertEqual(constructed_layer.out_features, 5)
        self.assertFalse(constructed_layer.bias)

        # ... Add more checks for different spaces

    def test_construct_distribution(self):
        # Test with Discrete space
        in_features = 4
        space = spaces.Discrete(5)
        discrete_dist_kwargs = {'activation': nn.Softmax}
        dist = DistributionConstructor.construct_distribution(in_features, space, discrete_dist_kwargs)
        self.assertIsInstance(dist, DiscreteDistribution)

        # ... Add more checks for different spaces and kwargs

        # Test with Continuous space
        space = spaces.Box(low=0, high=1, shape=(3,), dtype=float)
        continuous_dist_kwargs = {'activation_mu': nn.Tanh, 'activation_sigma': nn.Tanh}
        dist = DistributionConstructor.construct_distribution(in_features, space,
                                                              continuous_dist_kwargs=continuous_dist_kwargs)
        self.assertIsInstance(dist, ContinuousDistribution)

        # ... Add more checks for different spaces and kwargs

        # Test with unsupported space
        space = "UnsupportedSpace"
        with self.assertRaises(ValueError):
            DistributionConstructor.construct_distribution(in_features, space)


class TestDistributionLayer(unittest.TestCase):

    def test_layer(self):
        # Test with Discrete space
        in_features = 10
        out_features = 5
        space = spaces.Discrete(5)

        distribution_layer = DistributionLayer(in_features, space, nn.Linear)
        x = torch.randn((1, in_features))
        output = distribution_layer.layer(x)

        self.assertEqual(output.shape, (1, out_features))

        # Test with Box space
        in_features = 10
        dim = 5
        space = spaces.Box(low=-1, high=1, shape=(dim,), dtype=np.float32)

        distribution_layer = DistributionLayer(in_features, space, nn.Linear)
        x = torch.randn((1, in_features))
        output = distribution_layer.layer(x)

        self.assertEqual(output[0].shape, (1, dim))  # mu
        self.assertEqual(output[1].shape, (1, dim))  # sigma

        # Test with MultiDiscrete space
        in_features = 10
        nvec = [2, 3, 4]  # 3 subspaces, with 2, 3, and 4 discrete actions each
        space = spaces.MultiDiscrete(nvec)

        distribution_layer = DistributionLayer(
            in_features,
            space,
            nn.Linear,
        )

        x = torch.randn((1, in_features))
        output = distribution_layer.layer(x)

        self.assertEqual(output[0].shape, (1, 2))  # first
        self.assertEqual(output[1].shape, (1, 3))  # second
        self.assertEqual(output[2].shape, (1, 4))  # third

        # Test with Tuple space containing a Discrete and a Box space
        in_features = 10
        space = spaces.Tuple((spaces.Discrete(2), spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)))

        distribution_layer = DistributionLayer(
            in_features,
            space,
            nn.Linear,
        )

        x = torch.randn((1, in_features))
        output = distribution_layer.layer(x)

        self.assertEqual(output[0].shape, (1, 2))  # discrete
        self.assertEqual(output[1][0].shape, (1, 3))  # contin mu
        self.assertEqual(output[1][1].shape, (1, 3))  # contin sigma

        # Test with Dict space containing a Discrete and a Box space
        in_features = 10
        space = spaces.Dict({
            'action': spaces.Discrete(2),
            'params': spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)
        })

        distribution_layer = DistributionLayer(
            in_features,
            space,
            nn.Linear,
        )

        x = torch.randn((1, in_features))
        output = distribution_layer.layer(x)

        self.assertEqual(output['action'].shape, (1, 2))  # discrete
        self.assertEqual(output['params'][0].shape, (1, 3))  # discrete
        self.assertEqual(output['params'][1].shape, (1, 3))  # discrete

    def test_forward(self):
        # Test with Discrete space
        in_features = 10
        out_features = 5
        space = spaces.Discrete(5)

        distribution_layer = DistributionLayer(in_features, space, nn.Linear)
        x = torch.randn((1, in_features))
        output = distribution_layer(x)

        # Assume that the output of DistributionLayer's forward method is a tuple (action, log_prob)
        self.assertIsInstance(output, tuple)

        # Test with Box space
        in_features = 10
        dim = 5
        space = spaces.Box(low=-1, high=1, shape=(dim,), dtype=np.float32)

        distribution_layer = DistributionLayer(in_features, space, nn.Linear)
        x = torch.randn((1, in_features))
        output = distribution_layer(x)

        # Assume that the output of DistributionLayer's forward method is a tuple (action, log_prob)
        self.assertIsInstance(output, tuple)

        # Test with MultiDiscrete space
        in_features = 10
        nvec = [2, 3, 4]
        space = spaces.MultiDiscrete(nvec)

        distribution_layer = DistributionLayer(
            in_features,
            space,
            nn.Linear,
        )

        x = torch.randn((1, in_features))
        output = distribution_layer(x)
        for o in output:
            # Assume that the output of DistributionLayer's forward method is a tuple (action, log_prob)
            self.assertIsInstance(o, tuple)

        # Test with Tuple space
        in_features = 10
        space = spaces.Tuple((spaces.Discrete(2), spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)))

        distribution_layer = DistributionLayer(
            in_features,
            space,
            nn.Linear,
        )

        x = torch.randn((1, in_features))
        output = distribution_layer(x)

        for o in output:
            # Assume that the output of DistributionLayer's forward method is a tuple (action, log_prob)
            self.assertIsInstance(o, tuple)

        # Test with Dict space
        in_features = 10
        space = spaces.Dict({
            'action': spaces.Discrete(2),
            'params': spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)
        })

        distribution_layer = DistributionLayer(
            in_features,
            space,
            nn.Linear,
        )

        x = torch.randn((1, in_features))
        output = distribution_layer(x)

        for o in output.items():
            # Assume that the output of DistributionLayer's forward method is a tuple (action, log_prob)
            self.assertIsInstance(o, tuple)

    def test_split_and_route_output(self):
        # Test with Discrete space
        in_features = 10
        out_features = 5
        space = spaces.Discrete(5)

        distribution_layer = DistributionLayer(in_features, space, nn.Linear)
        x = torch.randn((1, out_features))
        output = distribution_layer.split_and_route_output(x, space)

        self.assertEqual(output.shape, (1, out_features))

        # Test with Box space
        in_features = 10
        dim = 5
        space = spaces.Box(low=-1, high=1, shape=(dim,), dtype=np.float32)

        distribution_layer = DistributionLayer(in_features, space, nn.Linear)
        x = torch.randn((1, dim * 2))
        output = distribution_layer.split_and_route_output(x, space)

        self.assertEqual(len(output), 2)  # Should be split into mu and sigma
        self.assertEqual(output[0].shape, (1, dim))
        self.assertEqual(output[1].shape, (1, dim))

        # Test with MultiDiscrete space
        in_features = 10
        nvec = [2, 3, 4]
        space = spaces.MultiDiscrete(nvec)

        distribution_layer = DistributionLayer(
            in_features,
            space,
            nn.Linear,
        )

        x = torch.randn((1, sum(nvec)))
        output = distribution_layer.split_and_route_output(x, space)

        self.assertEqual(len(output), len(nvec))  # Should be the same as the length of nvec
        for out, n in zip(output, nvec):
            self.assertEqual(out.shape, (1, n))

            # Test with Tuple space
            in_features = 10
            space = spaces.Tuple((spaces.Discrete(2), spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)))

            distribution_layer = DistributionLayer(
                in_features,
                space,
                nn.Linear,
            )

            x = torch.randn((1, 2 + 3 * 2))
            output = distribution_layer.split_and_route_output(x, space)

            self.assertEqual(len(output), 2)  # Should be the same as the number of spaces in the Tuple
            self.assertEqual(output[0].shape, (1, 2))  # For the Discrete space
            self.assertEqual(len(output[1]), 2)  # For the Box space (mu and sigma)
            self.assertEqual(output[1][0].shape, (1, 3))  # mu should have shape (1, 3)
            self.assertEqual(output[1][1].shape, (1, 3))  # sigma should have shape (1, 3)

            # Test with Dict space
            in_features = 10
            space = spaces.Dict({
                'action': spaces.Discrete(2),
                'params': spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)
            })

            distribution_layer = DistributionLayer(
                in_features,
                space,
                nn.Linear,
            )

            x = torch.randn((1, 2 + 3 * 2))
            output = distribution_layer.split_and_route_output(x, space)

            self.assertEqual(len(output), 2)  # Should be the same as the number of keys in the Dict
            self.assertEqual(output['action'].shape, (1, 2))  # For the Discrete space
            self.assertEqual(len(output['params']), 2)  # For the Box space (mu and sigma)
            self.assertEqual(output['params'][0].shape, (1, 3))  # mu should have shape (1, 3)
            self.assertEqual(output['params'][1].shape, (1, 3))  # sigma should have shape (1, 3)


if __name__ == '__main__':
    unittest.main()
