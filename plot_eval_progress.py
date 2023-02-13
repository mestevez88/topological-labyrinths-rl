import os
import sys

import numpy as np
from matplotlib import pyplot as plt

plt.rcParams["figure.autolayout"] = True

_, ax = plt.subplots(2, 2, sharex=True, sharey=True)

for i, algo in enumerate(["A2C", "DQN", "PP0"]):
    algo_root = os.path.join(sys.argv[1], f'{algo}')
    for j, mode in enumerate(["single env", "multiple envs"]):
        for pi in range(1):
            pi_dir = os.path.join(algo_root, f"pi_{pi}")
            env_dirs = [os.path.join(pi_dir, env_dir) for env_dir in os.listdir(pi_dir)]
            list_aggregate = []

            for env_dir in env_dirs:
                for run_dir in os.listdir(env_dir):
                    npz_path = os.path.join(env_dir, run_dir, 'evaluations.npz')
                    results = np.load(npz_path)['results']
                    list_aggregate.append(results)
                # wasn't sure if I should average out all the runs from the different environments
                # (or how to aggregate in general), so also compute statistics from just a single environment to compare
                # visually at least
                if mode == "single env":
                    break

            # axis 1 will be of length n_eval_episodes + n_training trials
            agg_results = np.concatenate(list_aggregate, axis=1)
            mean = agg_results.mean(axis=1)
            std = agg_results.std(axis=1)

            ax[i, j].plot(mean, label=r"$\pi$="+str(pi))
            ax[i, j].fill_between(range(len(mean)), mean - std, mean + std, alpha=.3)
        ax[i, j].legend(loc='lower right')
        ax[i, j].set(xlabel="iterations", ylabel="avg reward", title=f"{algo} ({mode})")
        ax[i, j].label_outer()

plt.show()
