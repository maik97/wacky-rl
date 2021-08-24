import tensorflow as tf
from wacky_rl import losses


class SharedNetLoss(losses.WackyLoss):

    def __init__(self, alphas: list =None):
        super().__init__()
        self.alphas = alphas

    def __call__(self, inputs, *args, **kwargs):


        return loss