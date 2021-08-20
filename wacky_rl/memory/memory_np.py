import random
import numpy as np
import tensorflow as tf

class BufferMemory:

    def __init__(self, maxlen=None):

        self._memory = []
        self.maxlen = maxlen
        self.index_dict = {}
        self._lenght = 0

    @property
    def num_arrays(self):
        return len(self._memory)

    def __len__(self):
        return self._lenght

    def _check_maxlen(self):
        if not self.maxlen is None:
            if self._lenght > self.maxlen:
                self.pop(0)
                self._lenght -= 1

    def _get_index(self, index_or_key):
        if isinstance(index_or_key, str):
            return self.index_dict[index_or_key]
        return index_or_key

    def keys(self):
        return self.index_dict.keys()

    def pop(self, index):
        for elem in self._memory: np.delete(elem, index)

    def pop_array(self, index_or_key):

        index = self._get_index(index_or_key)
        self._memory.pop(index)

        if isinstance(index_or_key, str):
            self.index_dict.pop(index_or_key, None)

        for key in self.index_dict.keys():
            if self.index_dict[key] > index:
                self.index_dict[key] = self.index_dict[key] - 1

    def clear(self):
        self._memory = []
        self.index_dict = {}
        self._lenght = 0

    def __call__(self, to_remember, *arg, **kwargs):

        if isinstance(to_remember, list):
            for i in range(len(to_remember)):
                self._add_item_to_memory(to_remember[i], index=i)

        elif isinstance(to_remember, dict):
            for key in to_remember.keys():
                self._add_item_to_memory(to_remember[key], index=self.index_dict[key])

        else:
            self._add_item_to_memory(to_remember, *arg, **kwargs)

        self._lenght += 1
        self._check_maxlen()

    def _add_item_to_memory(self, items_to_remember, key=None, index=None):

        if isinstance(items_to_remember, tf.Tensor):
            items_to_remember = items_to_remember.numpy()

        if not key is None:
            if not key in self.keys():
                self.index_dict[key] = self.num_arrays
            index = self.index_dict[key]

        if not index is None:
            try:
                self._memory[index] = np.concatenate([self._memory[index], np.array([items_to_remember])], axis=0)
            except:
                self._memory.append(np.array([items_to_remember]))

    def __getitem__(self, key_or_index):
        if isinstance(key_or_index, str):
            key_or_index = self._memory[self.index_dict[key_or_index]]
        return self._memory[key_or_index]

    def replay(self, to_tensor=True, indices=None):
        if indices is None:
            mem_list = self._memory
        else:
            mem_list = [elem[indices] for elem in self._memory]
        if to_tensor:
            return [tf.stack(elem) for elem in mem_list]

        return mem_list

    def mini_batches(self, batch_size, num_batches=None, shuffle_batches=False):

        batches = []
        for elem in self._memory:
            batches.append(np.array_split(np.copy(elem), self._lenght // batch_size))

        batches = list(zip(*batches))

        if shuffle_batches:
            random.shuffle(batches)

        if num_batches is None:
            return batches
        return batches[-num_batches:]
