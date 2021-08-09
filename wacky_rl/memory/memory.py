import numpy as np
import tensorflow as tf
from collections import deque
import random


class BasicMemory:

    def __init__(self, maxlen=None):
        self.mem = deque(maxlen=maxlen)

    def remember(self, tensor_list):
        self.mem.append([tf.cast(elem, dtype=tf.float32) for elem in tensor_list])

    def recall(self, *args, **kwargs):
        mem_list = list(zip(*list(self.mem)))
        self.mem.clear()
        return [tf.stack(elem) for elem in mem_list]


class ReplayBuffer(BasicMemory):

    def __init__(self, maxlen=10000):
        super().__init__(maxlen)

    def remember(self, tensor_list):
        super().remember(tensor_list)

    def sample(self, batch_size=32):
        batch_size = min(batch_size, len(self.mem))
        mem_list = random.sample(self.mem, batch_size)
        mem_list = list(zip(*list(mem_list)))
        return [tf.stack(elem) for elem in mem_list]

    def _sample_by_indices(self, indices):
        mem_list = self.mem[indices]
        mem_list = list(zip(*list(mem_list)))
        return [tf.stack(elem) for elem in mem_list]


class ShortTermLongTermBuffer:

    def __init__(self, maxlen_shortterm=None, maxlen_longterm=1_000_000):
        self.short_term_mem = BasicMemory(maxlen=maxlen_shortterm)
        self.long_term_mem = ReplayBuffer(maxlen=maxlen_longterm)

    def remember(self, tensor_list):
        self.short_term_mem.remember(tensor_list)
        self.long_term_mem.remember(tensor_list)

    def recall_short_term(self):
        return self.short_term_mem.recall()

    def sample_long_term(self, batch_size=64):
        return self.long_term_mem.recall(batch_size)


class PriorityBuffer:

    def __init__(self, maxlen, initital_prio=1.0):
        self.mem = ReplayBuffer(maxlen=maxlen)
        self.prio_mem = deque(maxlen=maxlen)
        self.initital_prio = initital_prio

    def remember(self, tensor_list):
        self.mem.remember(tensor_list)
        [self.prio_mem.append(self.initital_prio) for i in range(len(tensor_list[0]))]

    def sample(self):
        self.cur_indices = None

    def update_priority(self, priorities):
        self.prio_mem[self.cur_indices] = priorities


