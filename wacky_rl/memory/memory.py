import numpy as np
import tensorflow as tf

class TensorMemory:

    _pointer = 0
    _lenght = 0

    def __init__(self, maxlen=None):
        self._memory = None
        self.maxlen = maxlen

    def __call__(self, list_of_items):

        if self._memory is None:
            if self.maxlen is None:
                self._memory = [
                    tf.TensorArray(
                        size=0,
                        dtype=tf.float32,
                        dynamic_size=True,
                        name='items_at_{}'.format(i)
                    ) for i in range(len(list_of_items))
                ]
            else:
                self._memory = [
                    tf.TensorArray(
                        size=self.maxlen,
                        dtype=tf.float32,
                        name='items_at_{}'.format(i)
                    ) for i in range(len(list_of_items))
                ]

        for i in range(len(list_of_items)):
            self._memory[i] = self._add_item_to_tensor_array(self._memory[i], list_of_items[i])

        self._pointer += 1
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

    @property
    def tensors(self):
        return self._memory

    def _add_item_to_tensor_array(self, tensor_array, item):
        if not self.maxlen is None:
            if self._pointer >= self.maxlen:
                self._pointer = 0
        tensor_array = tensor_array.write(self._pointer, tf.cast(item, dtype=tf.float32))
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

    def get(self, indices):
        return self.memory.gather_memories(indices)

    def recall(self):
        mem_list = self.memory.read_memories()
        self.memory.clear()
        return mem_list

    def sample(self, *args, **kwargs):
        return self.recall()

    def clear(self):
        self.memory.clear()
        print(self.memory._memory)


class ReplayBuffer(BasicMemory):

    def __init__(self, maxlen=10_000):
        super().__init__()
        self.memory = TensorMemory(maxlen)

    def sample_mini_batch(self,mini_batch_size, num_mini_batches=None, clear_memory=False):
        if not num_mini_batches is None:
            num_mini_batches = min(num_mini_batches, int(self.memory.length/mini_batch_size))
            num_mini_batches = max(num_mini_batches, 1)
        else:
            num_mini_batches = max(int(self.memory.length/mini_batch_size), 1)

        mini_batches = []
        for _ in range(num_mini_batches):
            if self.memory.length < mini_batch_size:
                mini_batches.append(
                self.memory.gather_memories(np.arange(0, self.memory.length))
            )

            else:
                start_index = np.random.randint(0, self.memory.length - mini_batch_size)
                mini_batches.append(
                    self.memory.gather_memories(np.arange(start_index, start_index+mini_batch_size))
                )

        if clear_memory:
            self.memory.clear()
        return mini_batches

    def sample(self, batch_size):
        batch_size = min(batch_size, self.memory.length)
        indices = np.random.choice(self.memory.length, batch_size, replace=False)
        return self.memory.gather_memories(indices)

    def recall(self, clear_after_read=True):
        mem_list = self.memory.read_memories()
        if clear_after_read:
            self.memory.clear()
        return mem_list


class ShortTermLongTermBuffer:

    def __init__(self, short_term_mem=None, long_term_mem=None):

        if short_term_mem is None:
            self.short_term_mem = BasicMemory()
        else:
            self.short_term_mem = short_term_mem

        if long_term_mem is None:
            self.long_term_mem = ReplayBuffer()
        else:
            self.long_term_mem = long_term_mem

    def remember(self, tensor_list):
        self.short_term_mem.remember(tensor_list)
        self.long_term_mem.remember(tensor_list)

    def recall_short_term(self):
        return self.short_term_mem.recall()

    def sample_long_term(self, batch_size=64):
        return self.long_term_mem.sample(batch_size)

