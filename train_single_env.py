from datetime import datetime

from stable_baselines3.common.logger import configure
from stable_baselines3.common.env_checker import check_env
from sb3_contrib import RecurrentPPO
from stable_baselines3.a2c import A2C
import gym

import topological_labyrinths_rl


tmp_path = f"tmp/single_env_log/{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}"
# set up logger
new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])

env = gym.make("topological-labyrinths-2D-v0", draw_deterministic=True)

model = A2C("MlpPolicy", env, verbose=1)
model.set_logger(new_logger)

model.learn(10000)

model.save("a2c_topological_labyrinth")
