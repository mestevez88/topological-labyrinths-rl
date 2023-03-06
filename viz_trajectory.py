import argparse
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

from viz_maze import draw_trajectory


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='viz_trajectory.py', description='Draw learning curves for each experiment')
    parser.add_argument('path', help="path to experiment folder")
    parser.add_argument('--algo', help="collect data until pi", type=str, default="A2C")
    parser.add_argument('--pi', help="collect data until pi", type=int, default=1)
    parser.add_argument('--env', help="collect data until pi", type=int, default=1)
    parser.add_argument('--run', help="collect data until pi", type=int, default=1)
    args = parser.parse_args()

    mazes = pickle.load(open(os.path.join(args.path, "mazes.p"), "rb"))
    maze = mazes["mazes"][args.pi][args.env]

    trajectory = np.load(os.path.join(args.path, args.algo,
                                      f"pi_{args.pi}", f"env_{args.env}", f"run_{args.run}", "trajectory.npz"))['arr_0']

    _, ax = plt.subplots(1, 1, figsize=(10, 10))
    draw_trajectory(mazes["info"]["lx"], mazes["info"]["lz"], maze, trajectory, ax=ax)

    plt.show()
