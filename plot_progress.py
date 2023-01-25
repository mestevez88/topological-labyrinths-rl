import os

import pandas as pd
from matplotlib import pyplot as plt
import argparse

plt.rcParams["figure.autolayout"] = True

_, ax = plt.subplots(3, 2, sharey=True)

for i, algo in enumerate(["A2C", "DQN", "PPO"]):
    algo_root = os.path.join('results', 'single_env_3x3_log', f'{algo}')
    if algo == "DQN":
        index_column_name = 'time/episodes'
    # if algo == "A2C":
    # if algo == "PPO":
    else:
        index_column_name = 'time/iterations'
    for j, mode in enumerate(["single env", "multiple envs"]):
        for pi in range(5):
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
            by_row_index = df_concat.groupby(df_concat[index_column_name])
            df_means = by_row_index.mean()
            df_stds = by_row_index.std()

            df_means = df_means[df_means.index <= 5000]
            df_stds = df_stds[df_stds.index <= 5000]

            ax[i, j].plot(df_means.index, df_means['rollout/ep_rew_mean'], label=r"$\pi_1$="+str(pi))
            ax[i, j].fill_between(df_means.index,
                                  df_means['rollout/ep_rew_mean'] - df_stds['rollout/ep_rew_mean'],
                                  df_means['rollout/ep_rew_mean'] + df_stds['rollout/ep_rew_mean'], alpha=.3)
        ax[i, j].legend(loc='lower right')
        ax[i, j].set(xlabel=index_column_name, ylabel="avg reward", title=f"{algo} ({mode})")
        ax[i, j]._label_outer_yaxis(check_patch=False)


plt.show()
