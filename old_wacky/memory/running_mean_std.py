import numpy as np
import torch as th
from copy import deepcopy


class RunningMeanStd(object):
    def __init__(self, count_eps: float = 1e-4, shape=(), norm_eps=1e-8):
        """
        Taken and modified from stable-baselines3:
        https://github.com/DLR-RM/stable-baselines3/blob/ddbe0e93f9fe55152f2354afd058b28e6ccc3345/stable_baselines3/common/running_mean_std.py
        Calulates the running mean and std of a data stream
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
        :param count_eps: helps with arithmetic issues
        :param shape: the shape of the data stream's output
        """
        self.mean = np.zeros(shape, np.float64)
        self.var = np.ones(shape, np.float64)
        self.count = count_eps
        self.eps = norm_eps

    def normalize(self, arr, update=False, subtract_mean=False, a_min=None, a_max=None):

        if isinstance(arr, th.Tensor):
            arr = arr.numpy()

        arr = deepcopy(arr)

        if update:
            self.update(arr)

        if subtract_mean:
            arr = arr - self.mean

        arr = arr / np.sqrt(self.var + self.eps)

        if a_min is not None or a_max is not None:
            arr = np.clip(arr, a_min, a_max)

        return arr

    def unnormalize(self, arr, add_mean=False):

        if isinstance(arr, th.Tensor):
            arr = arr.numpy()

        arr = deepcopy(arr)

        arr = arr * np.sqrt(self.var + self.eps)

        if add_mean:
            arr = arr + self.mean

        return arr

    def update(self, arr: np.ndarray) -> None:
        if isinstance(arr, th.Tensor):
            arr = arr.numpy()

        batch_mean = np.mean(arr, axis=0)
        batch_var = np.var(arr, axis=0)
        batch_count = arr.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: int) -> None:
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = m_2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count