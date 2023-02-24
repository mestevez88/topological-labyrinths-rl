import gym
import numpy as np

from graph_mdp import GraphEnv2D


class MetaGraphEnv2D(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, mazes, n_episodes_per_trial=10):
        super(MetaGraphEnv2D, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.n_episodes_per_trial = n_episodes_per_trial
        self.meta_env = GraphEnv2D()
        self.meta_env.reset()
        self.action_space = self.meta_env.action_space
        self.obs_size = self.meta_env.n_states + 4 + 1 + 1
        # Example for using image as input:
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(self.obs_size,))
        self.inner_episode_counter = 0

    def step(self, action: int):
        observation, reward, done, info = self.meta_env.step(action)

        d = np.array([1], dtype=np.float32) if done else np.array([0], dtype=np.float32)  # inner loop done-variable

        # check if a neu trial (=meta-episode) with a new env needs to start or
        if done:
            if self.inner_episode_counter == self.n_episodes_per_trial:
                self.inner_episode_counter = 0
            else:
                done = False
                self.inner_episode_counter += 1
                observation = self.meta_env.reset()

        obs_one_hot = np.zeros(self.meta_env.n_states, dtype=np.float32)
        obs_one_hot[observation] = 1

        a_one_hot = np.zeros(4, dtype=np.float32)
        a_one_hot[action] = 1

        r = np.array([reward], dtype=np.float32)

        meta_obs = np.concatenate([obs_one_hot, a_one_hot, r, d])

        return meta_obs, reward, done, info

    def reset(self):
        obs = self.meta_env.reset_to_new_env()

        # just encode the start state, leave everything else at zero
        dummy_meta_obs = np.zeros(self.obs_size, dtype=np.float32)
        dummy_meta_obs[obs] = 1

        return dummy_meta_obs

    def render(self, mode='human'):
        self.meta_env.render()

    def close(self):
        self.meta_env.close()
