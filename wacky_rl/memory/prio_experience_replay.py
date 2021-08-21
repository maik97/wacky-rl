'''
absolute value of the magnitude of TD error:
p_t = |\delta_t| + e

stochastic prioritization (probability of being chosen):
P(i) = \frac{p_i^a}{\sum_k p_k^a}

importance sampling weights:
(\frac{1}{N} \times \frac{1}{P(i)})^b

'''
import numpy as np
import tensorflow as tf
import random
from wacky_rl.memory.segment_tree import MinSegmentTree, SumSegmentTree
from wacky_rl.memory import BufferMemory


class PrioritizedReplayBuffer:
    '''
    Based on:
    https://github.com/Curt-Park/rainbow-is-all-you-need/blob/master/03.per.ipynb
    '''

    def __init__(
            self,
            maxlen: int = 10_000,
            alpha: float = 0.6
    ):
        """Initialization."""
        assert alpha >= 0


        self.maxlen = maxlen
        self.max_priority, self.tree_ptr = 1.0, 0
        self.alpha = alpha

        self.memory = BufferMemory(maxlen)

        # capacity must be positive and a power of 2.
        tree_capacity = 1
        while tree_capacity < self.maxlen:
            tree_capacity *= 2

        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)

    def remember(self, tensor_list):
        """Store experience and priority."""

        self.memory(tensor_list)

        self.sum_tree[self.tree_ptr] = self.max_priority ** self.alpha
        self.min_tree[self.tree_ptr] = self.max_priority ** self.alpha
        self.tree_ptr = (self.tree_ptr + 1) % self.maxlen

    def sample(self, beta: float = 0.4, batch_size: int = 32):# -> Dict[str, np.ndarray]:
        """Sample a batch of experiences."""
        assert beta > 0

        batch_size = min(batch_size, len(self.memory))

        indices = self._sample_proportional(batch_size)

        mem_list = self.memory.replay(indices=indices)
        weights = tf.cast(tf.stack([self._calculate_weight(i, beta) for i in indices]), dtype=tf.float32)
        return mem_list+[weights,indices]

    def update_priorities(self, indices: list, priorities: np.ndarray):
        """Update priorities of sampled transitions."""
        priorities = priorities.numpy().reshape(-1)
        #print('indices',indices)
        #print('prioritie',priorities)
        assert len(indices) == len(priorities)

        for idx, priority in zip(indices, priorities):
            assert priority > 0
            assert 0 <= idx < len(self.memory)

            self.sum_tree[idx] = priority ** self.alpha
            self.min_tree[idx] = priority ** self.alpha

            self.max_priority = max(self.max_priority, priority)

    def _sample_proportional(self, batch_size):# -> List[int]:
        """Sample indices based on proportions."""
        indices = []
        p_total = self.sum_tree.sum(0, len(self.memory) - 1)
        segment = p_total / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            upperbound = random.uniform(a, b)
            idx = self.sum_tree.retrieve(upperbound)
            indices.append(idx)

        return indices

    def _calculate_weight(self, idx: int, beta: float):
        """Calculate the weight of the experience at idx."""
        # get max weight
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * len(self.memory)) ** (-beta)

        # calculate weights
        p_sample = self.sum_tree[idx] / self.sum_tree.sum()
        weight = (p_sample * len(self.memory)) ** (-beta)
        weight = weight / max_weight

        return weight