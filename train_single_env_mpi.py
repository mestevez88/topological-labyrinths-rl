from datetime import datetime
import random

from gym.wrappers import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.logger import configure as configure_logger
from stable_baselines3.common.env_checker import check_env
from sb3_contrib import RecurrentPPO
from stable_baselines3 import A2C
from stable_baselines3 import DQN
from stable_baselines3 import PPO
import gym
from tqdm import tqdm

from topological_labyrinths_rl.envs.library_3x3 import LIBRARY_3X3_LABYRINTHS

models = {
    # "A2C": A2C,
    # "DQN": lambda policy, envir: DQN(policy, envir, learning_starts=100)
    "PPO": lambda policy, envir: PPO(policy, envir, n_steps=192)
}

n_trials_per_env = 5
n_env_samples_per_pi = 5
n_pis = len(LIBRARY_3X3_LABYRINTHS.pi_sorted_indexes)

for pi in range(n_pis):
    n_envs_avail = len(LIBRARY_3X3_LABYRINTHS.pi_sorted_indexes[pi])
    for env_idx in random.sample(range(n_envs_avail), min(n_envs_avail, n_env_samples_per_pi)):
        env = gym.make("topological-labyrinths-2D-v0",
                       envs_linrary=LIBRARY_3X3_LABYRINTHS, pi=pi, draw_deterministic=env_idx)
        check_env(env)
        eval_env = gym.make("topological-labyrinths-2D-v0",
                            envs_linrary=LIBRARY_3X3_LABYRINTHS, pi=pi, draw_deterministic=env_idx)
        check_env(eval_env)

        for model_name, model_class in models.items():
            print(f"Training {model_name} with {n_trials_per_env} trials in env "
                  f"<pi={pi}, env_nr={env_idx + 1}/{n_envs_avail}>")
            for i in range(n_trials_per_env):
                model = model_class("MlpPolicy", env)
                log_path = f"results/single_env_3x3_log/{model_name}/pi_{pi}/env_{env_idx}/run_{i}"
                new_logger = configure_logger(log_path, ["csv"])
                model.set_logger(new_logger)
                eval_callback = EvalCallback(eval_env, log_path=log_path, eval_freq=5,
                                             deterministic=True, render=False, n_eval_episodes=5,
                                             warn=False, verbose=False)
                model.learn(total_timesteps=10_000, log_interval=5, callback=eval_callback)
