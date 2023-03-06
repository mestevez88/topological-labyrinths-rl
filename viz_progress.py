import argparse
import os

import numpy as np
from matplotlib import pyplot as plt


def draw_progress(exp_path: str, max_pi: int):
    algos = list(filter(lambda x: os.path.isdir(os.path.join(exp_path, x)), os.listdir(exp_path)))
    range_max_pi = max_pi + 1 if max_pi is not None else max_pi

    _, ax = plt.subplots(len(algos), 2, figsize=(4*len(algos), 4))

    for i, algo in enumerate(algos):
        algo_root = os.path.join(exp_path, f'{algo}')
        pis = os.listdir(algo_root)
        pis.sort()
        pis = pis[:range_max_pi]

        for j, mode in enumerate(["single env", "multiple envs"]):
            for pi in pis:
                pi_dir = os.path.join(algo_root, pi)
                env_dirs = [os.path.join(pi_dir, env_dir) for env_dir in os.listdir(pi_dir)]
                list_aggregate = []

                for env_dir in env_dirs:

                    for run_dir in os.listdir(env_dir):
                        npz_path = os.path.join(env_dir, run_dir, 'evaluations.npz')
                        results = np.load(npz_path)['results']
                        list_aggregate.append(results)

                    # wasn't sure if I should average out all the runs from the different environments
                    # (or how to aggregate in general), so also compute statistics from just a single environment to
                    # compare visually at least
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='viz_progress.py', description='Draw learning curves for each experiment')
    parser.add_argument('path', help="path to experiment folder")
    parser.add_argument('--max_pi', help="collect data until pi", type=int, default=None)

    args = parser.parse_args()

    draw_progress(args.path, max_pi=args.max_pi)
