from datetime import datetime

from stable_baselines3.common.logger import configure
from stable_baselines3.common.env_checker import check_env
from sb3_contrib import RecurrentPPO

import topological_labyrinths_rl


tmp_path = f"tmp/meta_graph_log/{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}"
# set up logger
new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])

model = RecurrentPPO(
    "MlpLstmPolicy",
    "meta-topological-labyrinths-2D-v0",
    verbose=1,
    policy_kwargs={"shared_lstm": True, "enable_critic_lstm": False}
)
model.set_logger(new_logger)

model.learn(10000)

model.save("ppo_recurrent_meta_topological_labyrinth")
