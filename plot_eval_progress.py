import os

import pandas as pd
from matplotlib import pyplot as plt
import argparse

plt.rcParams["figure.autolayout"] = True

_, ax = plt.subplots(2, 2, sharex=True, sharey=True)

for i, algo in enumerate(["A2C", "PPO"]):
    algo_root = os.path.join('results', 'single_env_3x3_log', f'{algo}')
    for j, mode in enumerate(["single env", "multiple envs"]):
        for pi in range(2):
            pi_dir = os.path.join(algo_root, f"pi_{pi}")
            env_dirs = [os.path.join(pi_dir, env_dir) for env_dir in os.listdir(pi_dir)]
            if mode == "single env":
                env_dir = env_dirs[0]
                list_dfs = [pd.read_csv(os.path.join(env_dir, run_dir, "progress.csv"))
                            for run_dir in os.listdir(env_dir)]
            else:
                list_dfs = [pd.read_csv(os.path.join(env_dir, run_dir, "progress.csv"))
                            for env_dir in env_dirs
                            for run_dir in os.listdir(env_dir)]

            df_concat = pd.concat(list_dfs)
            print(len(df_concat))

            by_row_index = df_concat.groupby(df_concat.index)
            df_means = by_row_index.mean()
            df_stds = by_row_index.std()

            ax[i, j].plot(df_means.index, df_means['rollout/ep_rew_mean'], label=r"$\pi$="+str(pi))
            ax[i, j].fill_between(df_means.index,
                                  df_means['rollout/ep_rew_mean'] - df_stds['rollout/ep_rew_mean'],
                                  df_means['rollout/ep_rew_mean'] + df_stds['rollout/ep_rew_mean'], alpha=.1)
        ax[i, j].legend(loc='lower right')
        ax[i, j].set(xlabel="iterations", ylabel="avg reward", title=f"{algo} ({mode})")
        ax[i, j].label_outer()


plt.show()
