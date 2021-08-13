import tensorflow as tf

class ActionScaling:

    def __init__(self, action_space):

        self.low = tf.constant(action_space.low, dtype=tf.float32)
        self.high = tf.constant(action_space.high, dtype=tf.float32)
        self.scale = tf.constant((self.high - self.low) / 2, dtype=tf.float32)
        self.reloc = tf.constant(self.high - self.scale, dtype=tf.float32)

    def scale_to_space(self, action):
        action = action*self.scale + self.reloc
        return tf.clip_by_value(action, self.low, self.high)

    def reverse_scale(self, action):
         action = (action - self.reloc) / self.scale
         return tf.clip_by_value(action, -1.0, 1.0)