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

    def __init__(self, a_env, maze_scale, enable_render=False):
        super(GraphEnv2D, self).__init__()

        self.maze_scale = maze_scale

        self.n_actions = 4
        self.action_space = spaces.Discrete(self.n_actions)
        self.n_states = self.maze_scale * self.maze_scale
        self.observation_space = spaces.Discrete(self.n_states)

        self.state_coords = np.array([[a, b] for a, b in itertools.product(range(self.maze_scale), range(self.maze_scale))])
        self.pos = {state: self.state_coords[state] for state in range(self.n_states)}

        self.graph_adjacency = a_env

        self.start_state = 0
        self.goal_state = self.n_states - 1

        self.state = self.start_state

    def next_state(self, action: Union[GridAction, int]) -> int:
        x, y = tuple(self.state_coords[self.state])
        x_new, y_new = x, y

        if (action is GridAction.LEFT) and (x > 0):
            x_new, y_new = x - 1, y
        if (action is GridAction.UP) and (y < (self.maze_scale - 1)):
            x_new, y_new = x, y + 1
        if (action is GridAction.RIGHT) and (x < (self.maze_scale - 1)):
            x_new, y_new = x + 1, y
        if (action is GridAction.DOWN) and (y > 0):
            x_new, y_new = x, y - 1

        target_state = int(x_new * self.maze_scale + y_new)

        if self.graph_adjacency[self.state, target_state]:
            self.state = target_state

        return self.state

    def step(self, action: int):
        state = self.next_state(GridAction(action))

        if state == self.goal_state:
            reward = 1
            done = True
        else:
            reward = -1
            done = False

        info = {}
        return state, reward, done, info

    def reset(self):
        self.state = self.start_state
        return self.state

    def render(self, mode='human'):
        pass

    def close(self):
        pass
