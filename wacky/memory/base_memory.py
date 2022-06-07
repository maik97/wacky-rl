from collections import UserDict, UserList
from collections.abc import Iterable
import random
import torch as th
import numpy as np
from wacky.backend import WackyTypeError


def all_equal(iterable):
    iterator = iter(iterable)

    try:
        first_item = next(iterator)
    except StopIteration:
        return True

    for x in iterator:
        if x != first_item:
            return False
    return True


class TensorList(UserList):

    def __init__(self, initlist=None, maxlen=None):
        self.maxlen = maxlen
        super(TensorList, self).__init__(initlist)

        if initlist is not None:
            for idx in range(len(self)):
                if not isinstance(self[idx], th.Tensor):
                    self[idx] = th.tensor(self[idx], dtype=th.float)

    def append(self, item: th.Tensor) -> None:
        if not isinstance(item, th.Tensor):
            item = th.tensor(item, dtype=th.float)
        super(TensorList, self).append(item)
        self.check_maxlen()

    def check_maxlen(self):
        if self.maxlen is not None:
            while len(self) > self.maxlen:
                self.pop(0)


class MemoryDict(UserDict):

    def __init__(self, *args, **kwargs):
        self.stacked = False
        self.maxlen = None
        super(MemoryDict, self).__init__(*args, **kwargs)

    def set_maxlen(self, maxlen):
        self.maxlen = maxlen
        if not self.stacked:
            for key in self.keys():
                self[key].maxlen = self.maxlen
                self[key].check_maxlen()
        return self

    def __getitem__(self, y):
        if y not in self.keys() and not self.stacked:
            self.__setitem__(key=y, value=TensorList(maxlen=self.maxlen))
        return super(MemoryDict, self).__getitem__(y)

    def __setitem__(self, key, value):

        if not self.stacked:
            e_1 = None
            e_2 = None

            if not isinstance(value, Iterable):
                try:
                    value = TensorList([value], maxlen=self.maxlen)
                except Exception as e:
                    e_1 = f"\n Converting value at key '{key}' to Iterable failed:\n {e}"

            if not isinstance(value, (TensorList, th.Tensor)):
                try:
                    value = TensorList(value, maxlen=self.maxlen)
                except Exception as e:
                    e_2 = f"\n Converting value at key '{key}' to TensorList failed:\n {e}"

            if not isinstance(value, TensorList):
                e_3 = (f"\n Setting value at key '{key}' failed: Invalid type {type(value)}"
                       f"\n While not stacked, assigned values to keys of {MemoryDict} must be convertable to {list},"
                       f"\n Call the stack_tensors() method first, if you are trying to store a tensor"
                       f" or using some of the function wrapper based on memory.")
                if e_2 is not None:
                    e_3 = e_2 + "\n" + e_3
                if e_1 is not None:
                    e_3 = e_1 + "\n" + e_3
                raise TypeError(e_3)

        elif not isinstance(value, th.Tensor):
            raise TypeError(f"\n Setting value at key '{key}' failed:"
                            f" Expected type {th.Tensor} got {type(value)} instead."
                            f"\n While stacked, assigned values to keys of {MemoryDict} must be type {th.Tensor}."
                            f"\n Call the clear() method first, if you are trying to store a list of tensors.")

        super(MemoryDict, self).__setitem__(key, value)

    def stacked_key(self, key):
        return th.stack(tuple(self[key]))

    def numpy(self, key:str, reduce:str=None, decimals:int=None):
        if self.stacked:
            vals = self[key].detach().numpy()
        else:
            vals = self.stacked_key(key).detach().numpy()

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

    def stack(self):
        self.stacked = True
        for key in self.keys():
            self[key] = self.stacked_key(key)

    def clear(self) -> None:
        self.stacked = False
        super(MemoryDict, self).clear()

    def sample(self, batch_size=32, num_batches=1, copy_dict=True):
        mem = self.copy() if copy_dict else self
        mem.stack()

        if mem.global_keys_len is None:
            raise Exception("Lengths of keys not equal:", str(mem.keys_len_dict))

        samples = []
        for i in range(num_batches):
            rand_idx = np.random.randint(0, int(mem.global_keys_len - batch_size))
            sub_mem = MemoryDict()
            sub_mem.stacked = True
            for key in mem.keys():
                sub_mem[key] = mem[key][rand_idx:rand_idx+batch_size]
            samples.append(sub_mem)
        return samples

    def batch(self, batch_size, shuffle=False):
        split_mem = self.split(batch_size, copy_dict=True)

        if split_mem.global_keys_len is not None:
            sub_memories = []
            for idx in range(split_mem.global_keys_len):
                sub_mem = MemoryDict()
                sub_mem.stacked = True
                for key in split_mem.keys():
                    sub_mem[key] = split_mem[key][idx]
                sub_memories.append(sub_mem)
            if shuffle:
                random.shuffle(sub_memories)
            return sub_memories

        else:
            raise Exception("Lengths of keys not equal:", str(split_mem.keys_len_dict))

    def split(self, split_size_or_sections, copy_dict=True):
        split_mem = self.copy() if copy_dict else self

        if not split_mem.stacked:
            split_mem.stack()
        split_mem.stacked = False

        for key in split_mem.keys():
            split_mem[key] = th.split(split_mem[key], split_size_or_sections)

        return split_mem

    def compare_len(self, keys: list):
        return all_equal(len(self[key]) for key in keys)

    @property
    def keys_len_dict(self):
        return {key: len(self[key]) for key in self.keys()}

    @property
    def keys_len_list(self):
        return [len(self[key]) for key in self.keys()]

    @property
    def global_keys_len(self):
        if all_equal(self.keys_len_list):
            return self.keys_len_list[0]
        else:
            return None
