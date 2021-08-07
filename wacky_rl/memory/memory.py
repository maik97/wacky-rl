import tensorflow as tf
from collections import deque
import random


class BasicMemory:

    def __init__(self):
        self.mem = deque()

    def remember(self, tensor_list):
        self.mem.append([tf.squeeze(elem) for elem in tensor_list])

    def recall(self, *args, **kwargs):
        mem_list = list(zip(*list(self.mem)))
        self.mem.clear()
        return [tf.expand_dims(tf.stack(elem),1) for elem in mem_list]


class ReplayBuffer(BasicMemory):

    def __init__(self):
        super().__init__()

    def recall(self, batch_size):
        batch_size = min(batch_size, len(self.mem))
        mem_list = random.sample(self.mem, batch_size)
        mem_list = list(zip(*list(mem_list)))
        self.mem.clear()
        return [tf.expand_dims(tf.stack(elem), 1) for elem in mem_list]
