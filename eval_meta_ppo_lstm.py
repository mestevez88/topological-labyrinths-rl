import gym
import numpy as np

from sb3_contrib import RecurrentPPO

import topological_labyrinths_rl

env = gym.make("meta-topological-labyrinths-2D-v0")
model = RecurrentPPO.load("ppo_recurrent_meta_topological_labyrinth", env=env)

obs = env.reset()
# cell and hidden state of the LSTM
lstm_states = None
num_envs = 1
# Episode start signals are used to reset the lstm states
episode_starts = np.ones((num_envs,), dtype=bool)

dones = False
while not dones:
    action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
    obs, rewards, dones, info = env.step(action)
    episode_starts = dones
    env.render()
