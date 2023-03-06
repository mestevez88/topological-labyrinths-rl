import random
from enum import Enum
from typing import Union, Optional

import gym
from gym import spaces
import numpy as np
import itertools


class GridAction(Enum):
    LEFT = 0
    UP = 1
    RIGHT = 2
    DOWN = 3


class GraphEnv2D(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, a_env, lx, lz, enable_render=False, sparse_reward=True, max_reward=10):
        super(GraphEnv2D, self).__init__()

        self.lx = lx
        self.lz = lz

        self.max_reward = max_reward

        if sparse_reward:
            self.reward_func = self.sparse_reward_func
        else:
            self.reward_func = self.shaped_reward_func

        self.n_actions = 4
        self.action_space = spaces.Discrete(self.n_actions)
        self.n_states = self.lx * self.lz
        self.observation_space = spaces.Discrete(self.n_states)

        self.a_env = a_env

        self.start_state = 0
        self.goal_state = self.n_states - 1

        self.state = self.start_state

    def next_state(self, action: Union[GridAction, int]) -> int:
        # x, y = tuple(self.state_coords[self.state])
        # x_new, y_new = x, y
        #
        # if (action is GridAction.LEFT) and (x > 0):
        #     x_new, y_new = x - 1, y
        # if (action is GridAction.UP) and (y < (self.maze_scale - 1)):
        #     x_new, y_new = x, y + 1
        # if (action is GridAction.RIGHT) and (x < (self.maze_scale - 1)):
        #     x_new, y_new = x + 1, y
        # if (action is GridAction.DOWN) and (y > 0):
        #     x_new, y_new = x, y - 1
        #
        # target_state = int(x_new * self.maze_scale + y_new)

        if action is GridAction.LEFT and (self.state % self.lz != 0) and self.a_env[self.state, self.state - 1]:
            self.state = self.state - 1

        if action is GridAction.DOWN and (self.state <= (self.lx * self.lz - self.lz - 1)) and self.a_env[self.state, self.state + self.lz]:
            self.state = self.state + self.lz

        if action is GridAction.RIGHT and (self.state % self.lz < (self.lz - 1)) and self.a_env[self.state, self.state + 1]:
            self.state = self.state + 1

        if action is GridAction.UP and (self.state >= self.lz) and self.a_env[self.state, self.state - self.lz]:
            self.state, self.state - self.lz

        return self.state

    def sparse_reward_func(self, state):
        if state == self.goal_state:
            return self.max_reward
        else:
            return -1

    def shaped_reward_func(self, state):
        pass

    def step(self, action: int):
        state = self.next_state(GridAction(action))

        if state == self.goal_state:
            done = True
        else:
            done = False

        reward = self.reward_func(state)

        info = {"state": state, "reward": reward, "done": done}
        return state, reward, done, info

    def reset(self):
        self.state = self.start_state
        return self.state

    def render(self, mode='human'):
        pass

    def close(self):
        pass
