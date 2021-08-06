import tensorflow as tf


class BasicMemory:

    def __init__(self):
        self.mem = None
        self.t = 0


    def remember(self, tensor_list):

        if self.mem is None:
            self.mem = [
                tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True, clear_after_read=True) for e in tensor_list
            ]


        self.write_to_mem(self.t, [tf.cast(elem,tf.float32) for elem in tensor_list] )
        self.t = self.t + 1

    def write_to_mem(self, t, tensor_list):
        for i in tf.range(len(tensor_list)):
            self.mem[i] = self.mem[i].write(tf.cast(t,tf.int32), tf.cast(tensor_list[i],tf.float32))

    def recall(self):
        mem_list = [elem.stack() for elem in self.mem]
        self.mem = None
        self.t = 0
        return mem_list
