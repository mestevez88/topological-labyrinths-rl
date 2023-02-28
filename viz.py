import os
import pickle

import matplotlib.pyplot as plt
import numpy as np


def plot_state_function(state_function: np.ndarray, adjacency: np.ndarray, n_tiles_per_state: int = 15, ax: plt.Axes = None, cmap: str = "Reds"):
    n = n_tiles_per_state
    single_state_tiles = np.arange(n * n).reshape(n, n)
    i, j = state_function.shape

    a_shape = adjacency.shape
    assert len(a_shape) == 2
    assert a_shape[0] == a_shape[1]

    assert i*j == a_shape[0]

    total_mask = np.pad(np.zeros([n * i, n * j]) != 0, 1, constant_values=True)

    for state in range(i * j):
        mask = np.zeros([n, n]) != 0
        if not ((state % j != 0) and adjacency[state, state - 1]):
            mask = mask | (single_state_tiles % n == 0)

        if not ((state <= (i * j - j - 1)) and adjacency[state, state + j]):
            mask = mask | (single_state_tiles > (n * n - n - 1))

        if not ((state % j < (j - 1)) and adjacency[state, state + 1]):
            mask = mask | (single_state_tiles % n == (n - 1))

        if not ((state >= j) and adjacency[state, state - j]):
            mask = mask | (single_state_tiles < n)

        x = state // j
        y = state % j

        total_mask[1 + n * x:1 + n * x + n, 1 + n * y:1 + n * y + n] = mask

    func_enlarged = np.pad(np.repeat(np.repeat(state_function, n, axis=0), n, axis=1), 1, mode="edge")

    masked_func = np.ma.masked_where(total_mask, func_enlarged)

    pcm = ax.imshow(masked_func, cmap=cmap)

    ax.tick_params(
        axis='both',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        left=False,  # ticks along the bottom edge are off
        right=False,  # ticks along the top edge are off
        labelbottom=False,
        labelleft=False)  # labels along the bottom edge are off

    ax.set_facecolor('grey')

    return pcm


def draw_maze(i, j, adjacency, ax):
    return plot_state_function(np.ones([i, j]), adjacency, ax=ax)


def draw_maze_collection(maze_collection):
    n_mazes = maze_collection["info"]["n_mazes"]
    lx = maze_collection["info"]["lx"]
    lz = maze_collection["info"]["lz"]
    pis = maze_collection["info"]["pis"]
    _, axs = plt.subplots(len(pis), n_mazes)
    for j, pi in enumerate(pis):
        for i, (ax, maze) in enumerate(zip(axs[j], maze_collection["mazes"][pi])):
            draw_maze(lx, lz, maze, ax=ax)
            ax.set_title(f"Maze(pi={pi}, sample={i})")
    plt.show()


if __name__ == "__main__":
    mazes = pickle.load(open(os.path.join("mazes", "mazes_5x5_392FD851.p"), "rb"))

    draw_maze_collection(mazes)
