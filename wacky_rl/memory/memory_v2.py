import numpy as np
import tensorflow as tf
from collections import deque
import random

class TensorMemory:

    _pointer = 0
    _lenght = 0

    def __init__(self, maxlen=None):
        self._memory = None
        self.maxlen = maxlen

    def __call__(self, list_of_items):

        if self._memory is None:
            self._memory = [tf.TensorArray(size=self.maxlen, dtype=tf.float32) for i in range(len(list_of_items))]

        for i in range(len(list_of_items)):
            self._memory[i] = self._add_item_to_tensor_array(self._memory[i], list_of_items[i])

        self._lenght += 1
        if not self.maxlen is None:
            self._lenght = min(self._lenght, self.maxlen)

    @property
    def length(self):
        return self._lenght

    @property
    def number_of_types(self):
        if self._memory is None:
            return 0
        else:
            return len(self._memory)

    def _add_item_to_tensor_array(self, tensor_array, item):
        if not self.maxlen is None:
            if self._pointer >= self.maxlen:
                self._pointer = 0
        tensor_array = tensor_array.write(self._pointer, tf.cast(item, dtype=tf.float32))
        self._pointer += 1
        return tensor_array

    def gather_memories(self, indices):
        return [mem.gather(indices) for mem in self._memory]

    def read_memories(self):
        return [mem.stack() for mem in self._memory]

    def clear(self):
        self._memory = None
        self._pointer = 0

class BasicMemory:

    memory = TensorMemory()

    def __init__(self):
        pass

    def remember(self, list_of_items):
        self.memory(list_of_items)

    def recall(self):
        mem_list = self.memory.read_memories()
        self.memory.clear()
        return mem_list

class ReplayBuffer(BasicMemory):

    def __init__(self, maxlen=None):
        super().__init__()
        self.memory = TensorMemory(maxlen)

    def sample(self, batch_size):
        batch_size = min(batch_size, self.memory.length)
        indices = np.random.choice(self.memory.length, batch_size, replace=False)
        return self.memory.gather_memories(indices)

    def recall(self, clear_after_read=True):
        mem_list = self.memory.read_memories()
        if clear_after_read:
            self.memory.clear()
        return mem_list

'''
class PrioReplayBuffer(ReplayBuffer):

    def __init__(self, maxlen=100_000):
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
'''

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


