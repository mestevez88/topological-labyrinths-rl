import os
import pickle
import random

from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import A2C
from stable_baselines3 import DQN
from stable_baselines3 import PPO
import gym
from mpi4py import MPI
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
    mazes_file = os.path.join("mazes", "mazes_3x3_54AB0B86.p")
    experiment_key = f"{random.getrandbits(32):X}"
    print(f"Starting experiments {experiment_key} with mazes {mazes_file} env")
    mazes = pickle.load(open(mazes_file, "rb"))
else:
    experiment_key = None
    mazes = None

# broadcasts
experiment_key = comm.bcast(experiment_key, root=0)
mazes = comm.bcast(mazes, root=0)

set_random_seed(seed=rank)  # set fixed random seed for each worker

for pi, envs_list in mazes["mazes"].items():
    for env_idx, a_env in enumerate(envs_list):
        env = gym.make("topological-labyrinths-2D-v0", a_env=a_env, maze_scale=mazes["info"]["lx"])
        eval_env = gym.make("topological-labyrinths-2D-v0", a_env=a_env, maze_scale=mazes["info"]["lx"])

        # make sure train and eval envs conform to gym/stable-baselines api
        check_env(env)
        check_env(eval_env)

        for model_name, model_class in models.items():
            model = model_class("MlpPolicy", env)
            print(f"Training {model_name} <pi={pi}, env_nr={env_idx + 1}/{len(envs_list)}, trial={rank}/{n_trials_per_env}>")
            log_path = f"results/single_env_3x3_log/{experiment_key}/{model_name}/pi_{pi}/env_{env_idx}/run_{rank}"
            eval_callback = EvalCallback(eval_env, log_path=log_path, eval_freq=5, deterministic=False, render=False,
                                         n_eval_episodes=5, warn=False, verbose=False)
            model.learn(total_timesteps=10_000, log_interval=5, callback=eval_callback)
