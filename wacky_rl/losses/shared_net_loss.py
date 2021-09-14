import tensorflow as tf
from wacky_rl import losses
from wacky_rl import models
from itertools import count



class SharedNetLoss(losses.WackyLoss):
    _ids = count(0)

    def __init__(self, alphas: list =None, sub_models=None, sub_agents=None, logger=None):
        super().__init__()
        self.alphas = alphas
        self.sub_models = sub_models
        self.sub_agents = sub_agents
        self.logger = logger
        self.id = next(self._ids)

        if self.sub_models is None == self.sub_agents is None:
            raise Exception('sub_models and sub_agents cant be both (un)defined at the same time')

    def __call__(self, prediction, loss_args, *args, **kwargs):

        #print(prediction)

        loss = 0.0

        if not self.sub_models is None:
            for i in range(len(self.sub_models)):
                #print(i)
                loss = loss + self.alphas[i] * self.sub_models[i].train_step(prediction, *loss_args[i])

        if not self.sub_agents is None:
            for i in range(len(self.sub_agents)):
                loss = loss + self.alphas[i] * self.sub_models[i].learn()

        if not self.logger is None:
            self.logger.log_mean('shared_net_loss_' + str(self.id), loss)

        return loss