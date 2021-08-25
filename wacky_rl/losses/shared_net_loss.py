import tensorflow as tf
from wacky_rl import losses
from wacky_rl import models


class SharedNetLoss(losses.WackyLoss):

    def __init__(self, alphas: list =None, sub_models=None):
        super().__init__()
        self.alphas = alphas
        self.sub_models = sub_models

    def __call__(self, prediction, loss_args, *args, **kwargs):

        #print(prediction)

        loss = 0.0

        for i in range(len(self.sub_models)):
            print(i)
            loss = loss + self.alphas[i] * self.sub_models[i].train_step(prediction, *loss_args[i])

        return loss