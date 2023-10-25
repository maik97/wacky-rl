import torch as th
from wacky import functional as funky


def monte_carlo_returns(rewards, gamma=0.99, eps=1e-07, standardize=False):

    future_return = 0.0
    returns = []
    for r in reversed(rewards):
        future_return = r + gamma * future_return
        returns.insert(0, future_return)

    returns = th.tensor(returns)

    if standardize:
        returns = funky.standardize_tensor(returns, eps)

    return th.unsqueeze(returns, dim=-1)


def temporal_difference_returns(rewards, dones, next_values, gamma=0.99, eps=1e-07, standardize=False):
    returns = rewards + gamma * next_values * (1-dones)
    #returns = th.tensor(returns)
    # batch['rewards'] + 0.99 * batch['next_values'] * (1 - batch['dones'])

    if standardize:
        returns = funky.standardize_tensor(returns, eps)

    return returns#.reshape(-1, 1)


def n_step_returns(rewards, dones, values, next_values, n=16, gamma=0.99, eps=1e-07, standardize=False):

    returns = []
    for i in range(len(rewards)):
        if len(rewards) - 1 > i + n:
            future_rewards = rewards[i:int(i + n)]
            exp_return = (gamma ** int(n + 1)) * next_values[i + n + 1] * (1-any(dones[i:i + n + 1]))
        else:
            future_rewards = rewards[i:]
            exp_return = 0
        for j in range(len(future_rewards)):
            exp_return += (gamma ** j) * future_rewards[j]
            if dones[i+j]:
                break
        returns.append(exp_return - values[i])

    returns = th.tensor(returns)

    if standardize:
        returns = funky.standardize_tensor(returns, eps)

    return th.unsqueeze(returns, dim=-1)

#
def n_step_returns_old(rewards, dones, values, next_values, n=16, gamma=0.99, lamda=1.0, eps=1e-07, standardize=False):

    returns = []
    for i in range(len(rewards)):
        g = 0.0
        future_rewards = rewards[i:int(i+n)] if len(rewards) > i+n else rewards[i:]

        for j in reversed(range(len(future_rewards))):
            delta = future_rewards[j] + gamma * next_values[i+j] * (1-int(dones[i+j])) - values[i+j]
            g = delta + gamma * lamda * (1-int(dones[i+j])) * g
        returns.append(g + values[i])

    returns = th.tensor(returns)
    print(returns)
    print(th.squeeze(generalized_returns(rewards, dones, values, next_values, lamda=lamda)))
    exit()

    if standardize:
        returns = funky.standardize_tensor(returns, eps)

    return th.unsqueeze(returns, dim=-1)


def generalized_returns(rewards, dones, values, next_values, gamma=0.99, lamda=0.95, eps=1e-07, standardize=False):

    g = 0.0
    returns = []
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * next_values[i] * (1-int(dones[i])) - values[i]
        g = delta + gamma * lamda * (1-int(dones[i])) * g
        future_return = g + values[i]
        returns.insert(0, future_return)

    returns = th.tensor(returns)

    if standardize:
        returns = funky.standardize_tensor(returns, eps)

    return th.unsqueeze(returns, dim=-1)


def calc_advantages(returns, values, eps=1e-07, standardize=False):

    advantages = th.sub(returns, values)
    if standardize:
        advantages = funky.standardize_tensor(advantages, eps)
    return advantages


def generalized_advantage_estimation(
        rewards, dones, values, next_values, gamma=0.99, lamda=0.95, eps=1e-07,
        standardize_returns=False, standardize_advantages=False
):

    returns = generalized_returns(rewards, dones, values, next_values, gamma, lamda, eps, standardize_returns)
    advantages = calc_advantages(returns, values, eps, standardize_advantages)

    return returns, advantages
