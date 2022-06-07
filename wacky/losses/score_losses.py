import torch as th
from wacky import functional as funky
from wacky import memory as mem
from wacky.losses import BaseWackyLoss


class NoBaselineLoss(BaseWackyLoss):
    def __init__(self, scale_factor=1.0, wacky_reduce='mean', *args, **kwargs):
        super(NoBaselineLoss, self).__init__(scale_factor, wacky_reduce, *args, **kwargs)

    def call(self, memory: [dict, mem.MemoryDict]) -> th.Tensor:
        return funky.basic_score_loss(
            score=memory['returns'],
            log_prob=memory['log_prob'],
        )


class WithBaselineLoss(BaseWackyLoss):
    def __init__(self, baseline=None, scale_factor=1.0, wacky_reduce='mean', *args, **kwargs):
        super(WithBaselineLoss, self).__init__(scale_factor, wacky_reduce, *args, **kwargs)
        self.baseline_calc = baseline  # TODO: Implement baseline functions

    def call(self, memory: [dict, mem.MemoryDict]) -> th.Tensor:
        return funky.basic_score_loss(
            score=memory['baseline_returns'],
            log_prob=memory['log_prob'],
        )


class AdvantageLoss(BaseWackyLoss):

    def __init__(self, scale_factor=1.0, wacky_reduce='mean', *args, **kwargs):
        super(AdvantageLoss, self).__init__(scale_factor, wacky_reduce, *args, **kwargs)

    def call(self, memory: [dict, mem.MemoryDict]) -> th.Tensor:
        return funky.basic_score_loss(
            score=memory['advantage'],
            log_prob=memory['log_prob'],
        )


class ClippedSurrogateLoss(BaseWackyLoss):

    def __init__(self, clip_range: float = 0.2, scale_factor=1.0, wacky_reduce='mean', *args, **kwargs):
        """
        Wrapper for wacky.functional.clipped_surrogate_loss() that uses a Dict or MemoryDict to look up arguments.
        Initializes all necessary function hyperparameters.

        :param clip_range: Hyperparameter for the clipped surrogate loss
        """
        super(ClippedSurrogateLoss, self).__init__(scale_factor, wacky_reduce, *args, **kwargs)
        self.clip_range = clip_range

    def call(self, memory: [dict, mem.MemoryDict]) -> th.Tensor:
        """
        Calls wacky.functional.clipped_surrogate_loss()

        :param memory: Must have following keys: ['advantage', 'old_log_prob', 'log_prob']
        :return: policy_loss
        """
        return funky.clipped_surrogate_loss(
            advantage=memory['advantage'],
            old_log_prob=memory['old_log_prob'],
            log_prob=memory['log_prob'],
            clip_range=self.clip_range
        )
