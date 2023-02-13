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
from mpi4py import MPI
import numpy as np
from stable_baselines3.common.utils import set_random_seed

from topological_labyrinths_rl.envs.library_3x3 import LIBRARY_3X3_LABYRINTHS

models = {
    "A2C": A2C,
    "DQN": lambda policy, envir: DQN(policy, envir, learning_starts=100),
    "PPO": lambda policy, envir: PPO(policy, envir, n_steps=192)
}

n_trials_per_env = 5
n_env_samples_per_pi = 5
n_pis = len(LIBRARY_3X3_LABYRINTHS.pi_sorted_indexes)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
    experiment_key = f"{random.getrandbits(32):X}"
    print(f"Starting experiments {experiment_key} for 3x3 env")
    env_dict = {pi: [] for pi in range(n_pis)}
    for pi in env_dict:
        n_envs_avail = len(LIBRARY_3X3_LABYRINTHS.pi_sorted_indexes[pi])
        for env_idx in random.sample(range(n_envs_avail), min(n_envs_avail, n_env_samples_per_pi)):
            env_dict[pi].append(env_idx)
else:
    experiment_key = None
    env_dict = None

# broadcasts
experiment_key = comm.bcast(experiment_key, root=0)
env_dict = comm.bcast(env_dict, root=0)

set_random_seed(seed=rank)  # set fixed random seed for each worker

for pi, envs_list in env_dict.items():
    n_envs_avail = len(LIBRARY_3X3_LABYRINTHS.pi_sorted_indexes[pi])
    for env_idx in envs_list:
        env = gym.make("topological-labyrinths-2D-v0",
                       envs_library=LIBRARY_3X3_LABYRINTHS, pi=pi, draw_deterministic=env_idx)
        eval_env = gym.make("topological-labyrinths-2D-v0",
                            envs_library=LIBRARY_3X3_LABYRINTHS, pi=pi, draw_deterministic=env_idx)

        # make sure train and eval envs conform to gym/stable-baselines api
        check_env(env)
        check_env(eval_env)

        for model_name, model_class in models.items():
            model = model_class("MlpPolicy", env)
            print(f"Training {model_name} <pi={pi}, env_nr={env_idx + 1}/{n_envs_avail}, trial={rank}/{n_trials_per_env}>")
            log_path = f"results/single_env_3x3_log/{experiment_key}/{model_name}/pi_{pi}/env_{env_idx}/run_{rank}"
            eval_callback = EvalCallback(eval_env, log_path=log_path, eval_freq=5, deterministic=False, render=False,
                                         n_eval_episodes=5, warn=False, verbose=False)
            model.learn(total_timesteps=10_000, log_interval=5, callback=eval_callback)
