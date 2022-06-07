import abc
from wacky import functional as funky
from wacky.memory.running_mean_std import RunningMeanStd


class BaseReturnCalculator(funky.MemoryBasedFunctional):

    def __init__(
            self,
            reward_calc_rms=False,
            reward_eps_rms=1e-4,
            reward_norm_eps=1e-8,
            reward_subtract_mean=False,
            reward_min=None,
            reward_max=None,
            return_calc_rms=False,
            return_eps_rms=1e-4,
            return_norm_eps=1e-8,
            return_subtract_mean=False,
            return_min=None,
            return_max=None,
    ):
        super(BaseReturnCalculator, self).__init__()

        self.reward_calc_rms = reward_calc_rms
        if self.reward_calc_rms:
            self.rms_reward = RunningMeanStd(reward_eps_rms, shape=(), norm_eps=reward_norm_eps)
            self.reward_subtract_mean = reward_subtract_mean
            self.reward_min = reward_min
            self.reward_max = reward_max

        self.return_calc_rms = return_calc_rms
        if self.reward_calc_rms:
            self.rms_return = RunningMeanStd(return_eps_rms, shape=(), norm_eps=return_norm_eps)
            self.return_subtract_mean = return_subtract_mean
            self.return_min = return_min
            self.return_max = return_max

    @abc.abstractmethod
    def call(self, memory):
        raise NotImplementedError()

    def rms_normalize_rewards(self, rewards):
        if self.reward_calc_rms:
            return self.rms_reward.normalize(
                rewards, True, self.reward_subtract_mean, self.reward_min, self.reward_max
            )
        else:
            return rewards

    def rms_normalize_returns(self, returns):
        if self.return_calc_rms:
            return self.rms_return.normalize(
                returns, True, self.return_subtract_mean, self.return_min, self.return_max
            )
        else:
            return returns


class MonteCarloReturns(BaseReturnCalculator):

    def __init__(self, gamma=0.99, eps=1e-07, standardize=False, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.gamma = gamma
        self.eps = eps
        self.standardize = standardize

    def call(self, memory):
        returns = funky.monte_carlo_returns(
            rewards=self.rms_normalize_rewards(memory['rewards']),
            gamma=self.gamma,
            eps=self.eps,
            standardize=self.standardize,
        )
        return self.rms_normalize_returns(returns)


class TemporalDifferenceReturns(BaseReturnCalculator):

    def __init__(self, gamma=0.99, eps=1e-07, standardize=False, *args, **kwargs):
        super(TemporalDifferenceReturns, self).__init__(*args, **kwargs)

        self.gamma = gamma
        self.eps = eps
        self.standardize = standardize

    def call(self, memory):
        returns = funky.temporal_difference_returns(
            rewards=self.rms_normalize_rewards(memory['rewards']),
            dones=memory['dones'],
            next_values=memory['next_values'],
            gamma=self.gamma,
            eps=self.eps,
            standardize=self.standardize,
        )
        return self.rms_normalize_returns(returns)


class NStepReturns(BaseReturnCalculator):

    def __init__(self, n=16, gamma=0.99, eps=1e-07, standardize=False, *args, **kwargs):
        super(NStepReturns, self).__init__(*args, **kwargs)

        self.n = n
        self.gamma = gamma
        self.eps = eps
        self.standardize = standardize

    def call(self, memory):
        returns = funky.n_step_returns(
            rewards=self.rms_normalize_rewards(memory['rewards']),
            dones=memory['dones'],
            values=memory['values'],
            next_values=memory['next_values'],
            n=self.n,
            gamma=self.gamma,
            eps=self.eps,
            standardize=self.standardize,
        )
        return self.rms_normalize_returns(returns)


class GeneralizedReturns(BaseReturnCalculator):

    def __init__(self, gamma=0.99, lamda=0.95, eps=1e-07, standardize=False, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.gamma = gamma
        self.lamda = lamda
        self.eps = eps
        self.standardize = standardize

    def call(self, memory):
        returns = funky.generalized_returns(
            rewards=self.rms_normalize_rewards(memory['rewards']),
            dones=memory['dones'],
            values=memory['values'],
            next_values=memory['next_values'],
            gamma=self.gamma,
            lamda=self.lamda,
            eps=self.eps,
            standardize=self.standardize)
        return self.rms_normalize_returns(returns)


class CalcAdvantages(funky.MemoryBasedFunctional):

    def __init__(self, eps=1e-07, standardize=False):
        super().__init__()

        self.eps = eps
        self.standardize = standardize

    def call(self, memory):
        return funky.calc_advantages(
            returns=memory['returns'],
            values=memory['values'],
            eps=self.eps,
            standardize=self.standardize,
        )


class GeneralizedAdvantageEstimation(funky.MemoryBasedFunctional):

    def __init__(self, gamma=0.99, lamda=0.95, eps=1e-07,
                 standardize_returns=False, standardize_advantages=False, *args, **kwargs):
        super().__init__()

        self.returns_calculator = GeneralizedReturns(gamma, lamda, eps, standardize_returns, *args, **kwargs)
        self.advantage_calculator = CalcAdvantages(eps, standardize_advantages)

    def call(self, memory):
        returns = self.returns_calculator(memory)
        memory['returns'] = returns
        advantages = self.advantage_calculator(memory)
        return returns, advantages
