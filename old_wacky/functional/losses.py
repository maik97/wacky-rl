import torch as th


def basic_score_loss(score, log_prob):
    score = score.detach()
    return - (score * log_prob).sum()


def clipped_surrogate_loss(advantage, old_log_prob, log_prob, clip_range):

    advantage = advantage.detach()
    #advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

    # ratio between old and new policy, should be one at the first iteration
    old_log_prob = old_log_prob.detach()
    ratio = th.exp(log_prob - old_log_prob)

    # clipped surrogate loss
    policy_loss_1 = advantage * ratio
    policy_loss_2 = advantage * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
    policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

    return policy_loss
