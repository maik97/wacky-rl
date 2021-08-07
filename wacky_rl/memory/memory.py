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

    def __init__(self, maxlen=32):
        super().__init__(maxlen)

    def recall(self, batch_size=32):
        batch_size = min(batch_size, len(self.mem))
        mem_list = random.sample(self.mem, batch_size)
        mem_list = list(zip(*list(mem_list)))
        #self.mem.clear()
        return [tf.stack(elem) for elem in mem_list]
