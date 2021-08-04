import gym

from wacky_rl import MultiAgentCompiler

def test_discrete_actor_critic():
    #from wacky_rl.agents import DiscreteActorCriticCore
    from test_ac import DiscreteActorCriticCore

    agent = MultiAgentCompiler(gym.make('CartPole-v0'), log_dir='_logs')
    [agent.assign_agent(DiscreteActorCriticCore(),at_index=i) for i in range(len(agent.action_outputs))]
    agent = agent.build(max_steps_per_episode=None)
    agent.train_agent(60000)


test_discrete_actor_critic()
