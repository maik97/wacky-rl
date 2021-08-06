import tensorflow as tf

from wacky_rl import BaseAgentCore
from wacky_rl.wacky_tape.wacky_tape import ParallelGradientTape
from wacky_rl.losses import discrete_actor_loss_with_tfp
from wacky_rl.models import BaseLayers
from wacky_rl.memory import LoggingTensorArray
from wacky_rl.actions import act_discrete_actor_critic, act_discrete_actor_critic_with_tfp


class DiscreteActorCriticCore(BaseAgentCore):

    def __init__(
            self,

            gamma: float = 1.0,
            standardize: str = None,
            g_mean: float = 0.0,
            g_rate: float = 0.15,
            discount_rewards: str = None,
            normalize_rewards: str = None,
            reward_range: tuple = None,
            dynamic_rate: float = 0.15,
            auto_transform_actions: bool = True,

            alpha_common: float = 0.9,
            alpha_actor: float = 0.005,
            alpha_critic: float = 0.000001,

            calc_action = None,

            loss_func_critic: tf.keras.losses.Loss = None,
            loss_func_actor = None,

            tape_common: ParallelGradientTape = None,
            tape_actor: ParallelGradientTape = None,
            tape_critic: ParallelGradientTape = None,

            common_layer = None,
            actor_layer = None,
            critic_layer = None,

            common_hidden = 1,
            actor_hidden = 1,
            critic_hidden = 1,

            common_units = 512,
            actor_units = 256,
            critic_units = 256,

            actor_activation = 'softmax',
            critic_activation = None,
    ):

        super().__init__(
            gamma, standardize, g_mean, g_rate, discount_rewards, normalize_rewards,
            reward_range, dynamic_rate, auto_transform_actions
        )

        self.agent_type = 'discrete_a2c'
        self.agent_act_space = 'discrete'
        self.uses_q_future = False
        self.use_target_model = False

        self.alpha_common = alpha_common
        self.alpha_actor = alpha_actor
        self.alpha_critic = alpha_critic

        if calc_action is None:
            self.calc_action = act_discrete_actor_critic_with_tfp
            self.calc_action = act_discrete_actor_critic
        else:
            self.calc_action = calc_action

        if loss_func_critic is None:
            #self.loss_func_critic = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)
            self.loss_func_critic = tf.keras.losses.mean_squared_error
        else:
            self.loss_func_critic = loss_func_critic

        if loss_func_actor is None:
            #self.loss_func_actor = discrete_actor_loss
            self.loss_func_actor = discrete_actor_loss_with_tfp
        else:
            self.loss_func_actor = loss_func_actor

        if tape_common is None:
            self.tapes['common'] = ParallelGradientTape(optimizer=tf.keras.optimizers.RMSprop(learning_rate=7e-5))
        else:
            self.tapes['common'] = tape_common

        if tape_actor is None:
            self.tapes['actor'] = ParallelGradientTape(optimizer=tf.keras.optimizers.RMSprop(learning_rate=7e-3))
        else:
            self.tapes['actor'] = tape_actor

        if tape_critic is None:
            self.tapes['critic'] = ParallelGradientTape(grad_alpha=0.5, optimizer=tf.keras.optimizers.RMSprop(learning_rate=7e-3))
        else:
            self.tapes['critic'] = tape_critic

        if common_layer is None:
            self.common_layer = BaseLayers(None, n_hidden=common_hidden, units_hidden=common_units)
        else:
            self.common_layer = common_layer

        if actor_layer is None:
            self.actor_layer = BaseLayers(None, n_hidden=actor_hidden, units_hidden=actor_units)
        else:
            self.actor_layer = actor_layer

        if critic_layer is None:
            self.critic_layer = BaseLayers(None, n_hidden=critic_hidden, units_hidden=critic_units)
        else:
            self.critic_layer = critic_layer

        self.actor_activation = actor_activation
        self.critic_activation = critic_activation

    def create_output_layers(self, n_actions):

        self.actor_layer.add_output_layer(n_actions, activation_out=self.actor_activation)
        self.critic_layer.add_output_layer(1, activation_out=self.critic_activation)

        self.values = LoggingTensorArray(self.name + '_values', self.logger)
        self.act_probs = LoggingTensorArray(self.name + '_act_probs', self.logger)
        self.log_probs = LoggingTensorArray(self.name + '_log_probs', self.logger)

    def start_gradient_recording(self):

        self.tape_com = self.tapes['common'].reset_tape(self.common_layer.trainable_variables)
        self.tape_cri = self.tapes['critic'].reset_tape(self.critic_layer.trainable_variables)
        self.tape_act = self.tapes['actor'].reset_tape(self.actor_layer.trainable_variables)

        self.values.reset_tensor(dtype=tf.float32, size=0, dynamic_size=True, clear_after_read=True)
        self.act_probs.reset_tensor(dtype=tf.float32, size=0, dynamic_size=True, clear_after_read=True)
        self.log_probs.reset_tensor(dtype=tf.float32, size=0, dynamic_size=True, clear_after_read=True)
        self.rewards.reset_tensor(dtype=tf.float32, size=0, dynamic_size=True, clear_after_read=True)

    def act(self, inputs, t, info_dict):

        if isinstance(inputs, list):
            common = self.common_layer([tf.expand_dims(tf.convert_to_tensor(elem), 0) for elem in inputs])
        else:
            common = self.common_layer(tf.expand_dims(tf.convert_to_tensor(inputs), 0))

        val = self.critic_layer(common)
        actions = self.actor_layer(common)

        action, act_prob, log_prob = self.calc_action(actions)

        self.values.write_to_tensor(t, tf.squeeze(val))
        self.act_probs.write_to_tensor(t, act_prob)
        self.log_probs.write_to_tensor(t, tf.squeeze(log_prob))

        return action

    def calc_loss_and_update_weights(self):

        vals = self.values.stack_tensor()
        act_probs = self.act_probs.stack_tensor()

        rewards = self.transform_rewards()
        ret = self.get_expected_return(rewards)

        loss_critic = self.loss_func_critic(tf.expand_dims(vals, 1), tf.expand_dims(ret, 1))
        loss_critic = self.tapes['critic'].apply_tape(self.tape_cri, loss_critic, self.critic_layer.trainable_variables)

        loss_actor = self.loss_func_actor(
            self.log_probs.stack_tensor(), tf.expand_dims(act_probs, 1), tf.expand_dims(ret - vals, 1), loss_critic
        )
        loss_actor = self.tapes['actor'].apply_tape(self.tape_act, loss_actor, self.actor_layer.trainable_variables)


        loss_common = (self.alpha_actor * loss_actor) + (self.alpha_critic * loss_critic)
        loss_common = self.tapes['common'].apply_tape(self.tape_com, loss_common, self.common_layer.trainable_variables)

        return self.alpha_common * loss_common
