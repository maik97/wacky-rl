import numpy as np
import tensorflow as tf
import random

from tensorflow.keras.layers import Dense, Input, LSTM
from tensorflow.keras import Model

from wacky_rl.agents import AgentCore
from wacky_rl.models import WackyModel
from wacky_rl.memory import BufferMemory
from wacky_rl.layers import DiscreteActionLayer, ContinActionLayer, RecurrentEncoder
from wacky_rl.losses import PPOActorLoss, SharedNetLoss, MeanSquaredErrorLoss
from wacky_rl.transform import GAE
from wacky_rl.trainer import Trainer

from wacky_rl.transform import RunningMeanStd

from wacky_rl.logger import StatusPrinter

from wacky_rl.wacky_tape import WackyGradientTape

class MultiAgent(AgentCore):


    def __init__(self, env, approximate_contin=False, logger=None):
        super(MultiAgent, self).__init__()

        self.env = env
        self.logger = logger
        self.approximate_contin = approximate_contin

        self._shared_model = None
        self._agents = []
        self.alphas = []


    def add_agent(self, action_id, agent, alpha=1.0, *args, **kwargs):
        self._agents.append(agent(action_id=action_id, *args, **kwargs))
        self.alphas.append(alpha)

    def add_shared_model(self, model=None, optimizer=None):

        if model is None:
            self.multi_agent_shared_model = WackyModel(model_name='MultiAgentSharedModel', logger=self.logger)
            self.multi_agent_shared_model.mlp_network(256)

        self._wacky_tape = WackyGradientTape(optimizer=optimizer)
        self._tape = self._wacky_tape.reset_tape(self._shared_model.trainable_variables)


    def act(self, inputs, act_argmax=False, save_memories=True):

        actions = []
        for agent in self._agents:
            if self._shared_model is None:
                actions.append(agent.act(inputs))
            else:
                actions.append(agent.act(tf.squeeze(self._shared_model(tf.expand_dims(inputs, 0)))))

        return actions

    def learn(self):

        losses = []
        for i in range(len(self._agents)):
            losses.append(self.alphas[i] * self._agents[i].learn())

        loss = np.mean(losses)

        if not self._shared_model is None:
            self._wacky_tape.apply_tape(self._tape, loss, self._shared_model.trainable_variables)
            self._tape = self._wacky_tape.reset_tape(self._shared_model.trainable_variables)

        return np.mean(losses)

