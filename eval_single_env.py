import gym
import numpy as np

from stable_baselines3.a2c import A2C

import topological_labyrinths_rl

env = gym.make("topological-labyrinths-2D-v0", draw_deterministic=True)
model = A2C.load("a2c_topological_labyrinth", env=env)

obs = env.reset()

done = False
while not done:
    action, lstm_states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()
