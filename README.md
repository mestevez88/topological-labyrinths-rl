# topological-labyrinths-rl
Reinforcement Learning on Topological Labyrinth Environments

# Setup
Set up python environment for this project (anaconda, venv, etc.) and install requirements:

```shell
pip install -r requirements.txt
```

# Usage
## Running the training scripts
For now, we can train a meta-env-agent (LSTM-Policy trained with PPO) or a single-env-agent 
(MLP-Policy trained with A2C) using the stable-baselines3 implementations. 

To run training e.g. execute

```shell
python train_single_env.py
```

This will train the agent and drop logs at `tmp/single_env_log` (or `tmp/meta_env_log` respectively).
The agent's parameters will be saved as `a2c_topological_labyrinth.zip` (or `ppo_recurrent_meta_topological_labyrinth.zip` respectively).

## Visualizing results
The `eval_` scripts will render the agent on the grid environment as a black circle on a graph with blue nodes. 
Its goal is to navigate to the green node. 

![](img/graph_3x3.png)

So, for example execute:
```shell
python eval_single_env.py
```

You can also start a tensorboard session and visualize the training process:
```shell
tensorboard --log_dir temp/single_env_log
```
