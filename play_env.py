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

parser = argparse.ArgumentParser(prog='viz_trajectory.py', description='Draw learning curves for each experiment')
parser.add_argument('mazes_path', help="path to mazes pickle file")
parser.add_argument('--pi', help="which pi to choose from", type=int, default=0)
parser.add_argument('--env', help="choose env index in pi", type=int, default=0)
args = parser.parse_args()

mazes_file = args.mazes_path
experiment_key = f"{random.getrandbits(32):X}"
print(f"Starting random walk with mazes {mazes_file} env")
mazes = pickle.load(open(mazes_file, "rb"))

envs_list = mazes["mazes"][args.pi]
a_env = envs_list[args.env]
env = gym.make("topological-labyrinths-2D-v0", a_env=a_env, lx=mazes["info"]["lx"], lz=mazes["info"]["lz"])

# make sure train env conforms to gym/stable-baselines api
check_env(env)

state = env.reset()
done = False
actions = []
states = []

sum_reward = 0
while not done:
    action = env.action_space.sample()
    state, reward, done, _ = env.step(action)
    sum_reward += reward
    actions.append(action)
    states.append(state)

print(f"Episode is over! You got {round(sum_reward, 2)} score.")
print(states)
print(actions)
