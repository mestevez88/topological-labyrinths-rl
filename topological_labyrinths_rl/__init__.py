from gym.envs.registration import register

register(
    id='topological-labyrinths-2D-v0',
    entry_point='topological_labyrinths_rl.envs.graph_mdp:GraphEnv2D',
    max_episode_steps=500
)

register(
    id='meta-topological-labyrinths-2D-v0',
    entry_point='topological_labyrinths_rl.envs.graph_mdp:MetaGraphEnv2D',
)
