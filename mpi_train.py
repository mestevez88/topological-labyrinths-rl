import argparse
import os
import pickle
import random
import shutil

from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import A2C
from stable_baselines3 import DQN
from stable_baselines3 import PPO
import gym
from mpi4py import MPI
from stable_baselines3.common.utils import set_random_seed

from topological_labyrinths_rl.envs.library_3x3 import LIBRARY_3X3_LABYRINTHS
from callback_state_trajectory import StateTrajectoryCallback

models = {
    "A2C": A2C,
    "DQN": lambda policy, envir: DQN(policy, envir, learning_starts=100),
    "PPO": lambda policy, envir: PPO(policy, envir, n_steps=192)
}

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
n_trials_per_env = comm.Get_size()

parser = argparse.ArgumentParser(prog='viz_trajectory.py', description='Draw learning curves for each experiment')
parser.add_argument('mazes_path', help="path to mazes pickle file")
parser.add_argument('--n_eval_episodes', help="how many episodes to average for evaluation", type=int, default=5)
parser.add_argument('--eval_freq', help="how often to evaluate during training", type=int, default=20)
parser.add_argument('--n_samples_pi', help="not implemented", type=int, default=None)
args = parser.parse_args()

n_eval_episodes = args.n_eval_episodes
eval_freq = args.eval_freq

if rank == 0:
    mazes_file = args.mazes_path
    experiment_key = f"{random.getrandbits(32):X}"
    print(f"Starting experiments {experiment_key} with mazes {mazes_file} env")
    mazes = pickle.load(open(mazes_file, "rb"))
    experiment_root = os.path.join("results", f"experiments_{experiment_key}")
    os.makedirs(experiment_root, exist_ok=True)
    shutil.copy(mazes_file, os.path.join(experiment_root, "mazes.p"))
else:
    experiment_root = None
    mazes = None

# broadcasts
experiment_root = comm.bcast(experiment_root, root=0)
mazes = comm.bcast(mazes, root=0)

set_random_seed(seed=rank)  # set fixed random seed for each worker

for pi, envs_list in mazes["mazes"].items():
    for env_idx, a_env in enumerate(envs_list):
        env = gym.make("topological-labyrinths-2D-v0", a_env=a_env, lx=mazes["info"]["lx"], lz=mazes["info"]["lz"])
        eval_env = gym.make("topological-labyrinths-2D-v0", a_env=a_env, lx=mazes["info"]["lx"], lz=mazes["info"]["lz"])

        # make sure train and eval envs conform to gym/stable-baselines api
        check_env(env)
        check_env(eval_env)

        for model_name, model_class in models.items():
            model = model_class("MlpPolicy", env)
            print(f"Training {model_name} <pi={pi}, env_nr={env_idx + 1}/{len(envs_list)}, trial={rank}/{n_trials_per_env}>")
            log_path = os.path.join(experiment_root, model_name, f"pi_{pi}", f"env_{env_idx}", f"run_{rank}")
            eval_callback = EvalCallback(eval_env, log_path=log_path, eval_freq=eval_freq, deterministic=False,
                                         render=False, n_eval_episodes=n_eval_episodes, warn=False, verbose=False)
            state_trajectory_callback = StateTrajectoryCallback(log_path=log_path)
            model.learn(total_timesteps=10_000, callback=[eval_callback, state_trajectory_callback])
