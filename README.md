# topological-labyrinths-rl
Reinforcement Learning on Topological Labyrinth Environments

# Setup
Set up python environment for this project (anaconda, venv, etc.) and install requirements:

```shell
pip install -r requirements.txt
```
please let me know if requirements are missing.

# Usage
## Running the training scripts
For now, we can train an agent with an MLP-Policy using the A2C, PPO and DQN algorithms.
The implementations are provided by the stable-baselines3 library. 

But first, we  need to generate some mazes 
(it also comes with a help page, so just pass the -h flag to see how to customize):
```shell
python maze_sampler.py
```
It will generate a maze dictionary file at `mazes/mazes_<lx>x<lz>_<maze_dict_key>.p`.
You can visualize the maze dictionary by executing

```shell
python viz_maze.py <path/to/mazes.p>
```

To run training with multiprocessing execute

```shell
mpiexec -n 5 python mpi_train.py <path/to/mazes.p>
```

This will train the agent and drop logs at `results/experiments_<experiment_key>`.

## Visualizing results
The `viz_` scripts will help you visualize the results.

So, for example execute:
```shell
python viz_progress.py <path/to/experiment/results>
```

or

```shell
python viz_progress.py <path/to/experiment/results>
```
