import random
from enum import Enum
from typing import Union, Optional

import gym
from gym import spaces
import numpy as np
import itertools

from matplotlib import pyplot as plt
import networkx as nx


class GridAction(Enum):
    LEFT = 0
    UP = 1
    RIGHT = 2
    DOWN = 3


class GraphEnv2D(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, envs_library, pi, enable_render=True, draw_deterministic: Optional[int] = None):
        super(GraphEnv2D, self).__init__()

        self.envs_library = envs_library
        self.maze_scale = self.envs_library.maze_scale
        self.pi = pi

        self.n_actions = 4
        self.action_space = spaces.Discrete(self.n_actions)
        self.n_states = self.maze_scale * self.maze_scale
        self.observation_space = spaces.Discrete(self.n_states)

        self.state_coords = np.array([[a, b] for a, b in itertools.product(range(self.maze_scale), range(self.maze_scale))])
        self.pos = {state: self.state_coords[state] for state in range(self.n_states)}

        if draw_deterministic:
            index = self.envs_library.pi_sorted_indexes.get(self.pi, 0)[draw_deterministic]
            self.current_graph_adjacency = self.envs_library.Ap[index]

            self.start_state = 0
            self.goal_state = self.n_states - 1

        else:
            self.current_graph_adjacency = self.envs_library.random_choice(pi=self.pi)

            states_list = list(range(self.n_states))
            self.start_state = states_list.pop(random.choice(states_list))
            self.goal_state = random.choice(states_list)

        self.state = self.start_state

        self.enable_render = enable_render

        if self.enable_render:
            plt.ion()

    def nx_draw_graph(self):
        node_color = ['#1f78b4']*self.n_states
        node_size = [300] * self.n_states

        node_color[self.goal_state] = '#1f7554'

        node_color[self.state] = '#1f3224'
        node_size[self.state] = 500

        nx_graph = nx.from_numpy_matrix(self.current_graph_adjacency)

        plt.cla()
        nx.draw(nx_graph, pos=self.pos, node_color=node_color, node_size=node_size)
        plt.pause(0.5)

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

        if self.current_graph_adjacency[self.state, target_state]:
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

    def reset_to_new_env(self):
        self.current_graph_adjacency = self.envs_library.random_choice(pi=self.pi)
        states_list = list(range(self.n_states))
        self.start_state = states_list.pop(random.choice(states_list))
        self.goal_state = random.choice(states_list)
        return self.reset()

    def reset(self):
        self.state = self.start_state
        return self.state

    def render(self, mode='human'):
        if self.enable_render:
            self.nx_draw_graph()

    def close(self):
        pass
