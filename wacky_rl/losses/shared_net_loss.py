import tensorflow as tf
from wacky_rl import losses


class SharedNetLoss(losses.WackyLoss):

    def __init__(self, alphas: list =None, sub_models=None):
        super().__init__()
        self.alphas = alphas
        self.sub_models = sub_models

    def __call__(self, batch_inputs, *args, **kwargs):

        loss = 0.0
        for model in self.sub_models:
            loss = loss + self.alphas * model.train_step(batch_inputs, *args, **kwargs)

        return loss