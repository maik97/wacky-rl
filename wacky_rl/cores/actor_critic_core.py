import wacky_rl
from wacky_rl.cores._agent_core import AgentCore
from wacky_rl.models.wacky_model import WackyModel
from tensorflow.keras import layers


class ActorCriticCore(AgentCore):

    def __init__(
            self,
            common_model: (str, WackyModel) = None,
            actor_model: WackyModel = None,
            critic_model: WackyModel = None,

            common_loss = None,
            actor_loss = None,
            critic_loss = None,

            actor_out_func = None,
            n_actions: int = None,

            memory = None,

            calc_returns = None,
    ):

        super().__init__()

        if critic_model is None and n_actions is None:
            raise AttributeError('Either critic_model or n_actions must be specified.')

        # Common Model:
        if common_model == 'auto':

            if common_loss is None:
                common_loss = wacky_rl.losses.SumMultipleLosses(alpha_list=[0.5, 0.5])

            self.common_model = WackyModel(
                model_layer = [layers.Dense(64, activation='relu')],
                loss = common_loss,
            )

        else:
            self.common_model = common_model

        # Actor Model:
        if actor_model is None:

            if actor_loss is None:
                actor_loss = wacky_rl.losses.DiscreteActorLoss()

            if actor_out_func is None:
                actor_out_func = wacky_rl.actions.DiscreteActorAction()

            self.actor_model = WackyModel(
                model_layer = [layers.Dense(64, activation='relu')],
                out_units = n_actions,
                loss = actor_loss,
                out_function = actor_out_func
            )

        else:
            self.actor_model = actor_model

        # Critic Model:
        if critic_model is None:

            self.critic_model = WackyModel(
                model_layer = [layers.Dense(64, activation='relu')],
                out_units = n_actions,
                loss = critic_loss,
            )

        else:
            self.critic_model = critic_model

        if memory is None:
            self.memory = wacky_rl.memory.BasicMemory()
        else:
            self.memory = memory

        if calc_returns is None:
            self.calc_returns = wacky_rl.transform.ExpectedReturnsCalculator()
        else:
            self.calc_returns = calc_returns


    def act(self, inputs):

        if not self.common_model is None:
            inputs = self.common_model(inputs)

        state_value = self.critic_model(inputs)
        action, act_prob, log_prob = self.actor_model(inputs)

        self.memory.remember([state_value, action, act_prob, log_prob])

        return action

    def train(self, rewards):

        state_value, action, act_prob, log_prob = self.memory.recall()

        returns = self.calc_returns(state_value, rewards)

        loss_critic = self.critic_model.train_step(state_value, returns)
        loss_actor = self.actor_model.train_step(returns, log_prob)

        loss_sum = loss_actor + loss_critic

        if not self.common_model is None:
            return self.common_model.train_step(loss_sum)
        return loss_sum

    def wacky_train(self, rewards):

        state_value, action, act_prob, log_prob = self.memory.recall()

        returns = self.calc_returns(state_value, rewards)

        if not self.common_model is None:
            return self.common_model.train_step(
                (self.critic_model.train_step, state_value, returns),
                (self.actor_model.train_step, returns, log_prob)
            )

        loss_critic = self.critic_model.train_step(state_value, returns)
        loss_actor = self.actor_model.train_step(returns, log_prob)
        return loss_actor + loss_critic






