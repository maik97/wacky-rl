from collections import UserDict
import torch as th
import numpy as np
from wacky.backend import WackyTypeError


class MemoryArray:

    def __init__(self, key):
        self.maxlen = None
        self.data = None
        self.pointer = 0
        self.is_full = False
        self.key = key

    @property
    def shape(self):
        if self.data is None:
            return None
        else:
            return self.data.shape

    def append(self, item: th.Tensor) -> None:

        if isinstance(item, th.Tensor):
            item = item.detach().numpy()
        elif not isinstance(item, np.ndarray):
            item = np.array(item)

        if item.ndim == 1:
            item = np.expand_dims(item, 0)

        if self.data is None:
            self.data = item
        else:
            self.data = np.append(self.data, item, axis=-1)
        self.pointer += 1

    def change_maxlen(self, maxlen):
        new_memory_array = MaxLenMemoryArray(maxlen)
        new_memory_array.append(self.data)

        if maxlen > self.pointer:
            new_memory_array.is_full = True
            new_memory_array.pointer = maxlen % self.pointer
        else:
            new_memory_array.is_full = False
            new_memory_array.pointer = self.pointer



class MaxLenMemoryArray(MemoryArray):

    def __init__(self, maxlen, key):
        super(MaxLenMemoryArray, self).__init__(key=key)
        self.maxlen = maxlen
        self.data = None
        self.pointer = 0
        self.is_full = False

    def append(self, item: th.Tensor) -> None:

        if isinstance(item, th.Tensor):
            item = item.detach().numpy()
        elif not isinstance(item, np.ndarray):
            item = np.array(item)

        if item.ndim == 0:
            item = item.reshape(1, 1)
        if item.ndim == 1:
            item = np.expand_dims(item, 0)


        if self.data is None:
            print(self.key)
            self.data = np.empty(np.append(self.maxlen, item.shape[1:]))
            print(self.data.shape)

        batch_size = item.shape[0]
        if batch_size + self.pointer > self.maxlen:
            self.data[self.pointer:] = item[:self.maxlen - self.pointer]
            self.data[:batch_size + self.pointer - self.maxlen] = item[self.maxlen - self.pointer:]
            self.pointer = batch_size + self.pointer - self.maxlen
            self.is_full = True
        else:
            self.data[self.pointer:self.pointer+batch_size] = item
            self.pointer = batch_size + self.pointer


    def change_maxlen(self, maxlen):
        if self.maxlen < maxlen:
            self.data = self.data[maxlen - self.maxlen:]
        elif self.maxlen > maxlen:
            data = np.empty((maxlen, self.shape))
            data[:self.maxlen] = self.data
            self.data = data
            self.is_full = False
        self.maxlen = maxlen


class NumpyMemoryDict(UserDict):

    def __init__(self, *args, **kwargs):
        self.maxlen = None
        super(NumpyMemoryDict, self).__init__(*args, **kwargs)

    def set_maxlen(self, maxlen):
        self.maxlen = maxlen
        if self.keys():
            for key in self.keys():
                self[key].change_maxlen(self.maxlen)
        return self

    def __getitem__(self, y):
        if y not in self.keys():
            if self.maxlen is None:
                self.__setitem__(key=y, value=MemoryArray(key=y))
            else:
                self.__setitem__(key=y, value=MaxLenMemoryArray(maxlen=self.maxlen, key=y))
        return super(NumpyMemoryDict, self).__getitem__(y)

    def __setitem__(self, key, value):
        if isinstance(value, MemoryArray):
            super(NumpyMemoryDict, self).__setitem__(key, value)
        else:
            memory_array = self[key]
            memory_array.append(value)
            super(NumpyMemoryDict, self).__setitem__(key, memory_array)

    def read(self, key:str, reduce:str=None, decimals:int=None):

        vals = self[key].data[:self.max_index]

        if reduce is not None:
            if reduce == 'mean':
                vals =  np.mean(vals)
            elif reduce == 'sum':
                vals = np.sum(vals)
            else:
                raise WackyTypeError(reduce, ('mean', 'sum'), parameter='reduce', optional=True)

        if decimals is None:
            return vals
        else:
            return np.round(vals, decimals)

    def clear(self) -> None:
        super(NumpyMemoryDict, self).clear()

    @property
    def pointer_array(self):
        return np.array([memory_array.pointer for memory_array in self.values()])

    @property
    def pointer(self):
        if not self.is_synchronized:
            raise Exception()
        else:
            return next(iter(self)).pointer

    @property
    def is_full(self):
        return np.all([memory_array.is_full for memory_array in self.values()])

    @property
    def is_synchronized(self):
        pointer_array = self.pointer_array
        return np.all(pointer_array == pointer_array[0])

    @property
    def max_index(self):
        pointer_array = np.array([memory_array.pointer for memory_array in self.values()])
        # check synch:
        if not np.all(pointer_array == pointer_array[0]):
            raise Exception()
        if self.is_full:
            return self.maxlen
        else:
            return self[next(iter(self))].pointer

    def random_sample_indices(self, size, max_index=None):
        if max_index is None:
            max_index = self.max_index
        return np.random.randint(0, max_index, size=size)

    def make_batch_dict(self, indices, as_tensors):
        batch = {}
        for k, v in self.items():
            if as_tensors:
                batch[k] = th.Tensor(v.data[indices])
            else:
                batch[k] = v.data[indices]
        return batch

    def generate_batches(
            self,
            batch_size,
            num_batches=None,
            shuffle_mode='step',
            max_index=None,
            as_tensors=True,
    ):

        max_index = self.max_index if max_index is None else max_index
        num_batches = max_index // batch_size if num_batches is None else num_batches

        if batch_size > max_index:
            print('Warning: Not enough warm up, batch size > memory length')
            return None

        if shuffle_mode is None:
            indices = np.arange(self.pointer - num_batches * batch_size, self.pointer).reshape(num_batches, batch_size)
        elif shuffle_mode == 'step':
            indices = self.random_sample_indices(size=(num_batches, batch_size), max_index=max_index)
        elif shuffle_mode == 'batch':
            indices = self.random_sample_indices(size=num_batches, max_index=max_index-batch_size)
            indices = np.array(
                [np.arange(ind_start, ind_end) for ind_start, ind_end in zip(indices, indices+batch_size)]
            )
        else:
            raise KeyError()

        for batch_indices in indices:
            #print(batch_indices)

            yield self.make_batch_dict(batch_indices, as_tensors=as_tensors)
