import random
from typing import List

import torch as th
import numpy as np

from wacky.memory import NumpyMemoryDict, MaxLenMemoryArray
from wacky.memory.segment_tree import SumSegmentTree, MinSegmentTree


class PrioritizedExperienceReplay(NumpyMemoryDict):

    def __init__(self, alpha=0.2, beta=0.6, eps=1e-6, max_priority=1.0):
        super(PrioritizedExperienceReplay, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.tree_pointer = 0
        self.max_priority = max_priority

    def set_maxlen(self, maxlen):
        super(PrioritizedExperienceReplay, self).set_maxlen(maxlen)
        # capacity must be positive and a power of 2.
        tree_capacity = 1
        while tree_capacity < self.maxlen:
            tree_capacity *= 2

        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)
        return self

    def tree_pointer_step(self):
        self.tree_pointer += 1
        if self.tree_pointer >= self.maxlen:
            self.tree_pointer = 0

        #print(self.tree_pointer, self.pointer)

        self.sum_tree[self.tree_pointer] = self.max_priority ** self.alpha
        self.min_tree[self.tree_pointer] = self.max_priority ** self.alpha

    def __getitem__(self, y):
        if y not in self.keys():
            self.__setitem__(key=y, value=MaxLenMemoryArray(maxlen=self.maxlen, key=y))
        return super(NumpyMemoryDict, self).__getitem__(y)

    def __setitem__(self, key, value):
        super(NumpyMemoryDict, self).__setitem__(key, value)

    def _sample_proportional(self, num_samples, max_index):
        """Sample indices based on proportions."""
        indices = []
        p_total = self.sum_tree.sum(0, max_index - 1)
        segment = p_total / num_samples

        for i in range(num_samples):
            a = segment * i
            b = segment * (i + 1)
            upperbound = random.uniform(a, b)
            idx = self.sum_tree.retrieve(upperbound)
            indices.append(idx)

        return indices

    def _calculate_weight(self, idx: int, beta: float, max_index):
        """Calculate the weight of the experience at idx."""
        # get max weight
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * max_index) ** (-beta)

        # calculate weights
        p_sample = self.sum_tree[idx] / self.sum_tree.sum()
        weight = (p_sample * max_index) ** (-beta)
        weight = weight / max_weight
        return weight

    def update_priorities(self, indices: List[int], priorities: np.ndarray):
        """Update priorities of sampled transitions."""

        assert len(indices) == len(priorities)

        for idx, priority in zip(indices, priorities):
            assert priority > 0
            assert 0 <= idx < self.max_index

            self.sum_tree[idx] = priority ** self.alpha
            self.min_tree[idx] = priority ** self.alpha

            self.max_priority = max(self.max_priority, priority)

    def generate_batches(
            self,
            batch_size,
            num_batches=None,
            shuffle_mode='step',
            max_index=None,
            as_tensors=True,
    ):

        max_index = self.max_index if max_index is None else max_index
        if batch_size > max_index:
            print('Warning: Not enough warm up, batch size > memory length')
            return None

        if num_batches is None:
            raise AttributeError('num_batches can not be None')

        if shuffle_mode != 'step':
            raise AttributeError("shuffle_mode must be 'step'")

        prio_indices = self._sample_proportional(num_batches * batch_size, max_index)
        weights = np.array([self._calculate_weight(i, self.beta, max_index) for i in prio_indices])

        prio_indices = np.array(prio_indices).reshape(num_batches, batch_size)
        weights = weights.reshape(num_batches, batch_size, 1)
        for batch_indices, w in zip(prio_indices, weights):
            yield self.make_batch_dict(batch_indices, as_tensors=as_tensors), th.Tensor(w), batch_indices

