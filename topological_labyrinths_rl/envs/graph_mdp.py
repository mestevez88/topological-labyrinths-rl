import random
import time
from enum import Enum
from typing import Union, Optional

import gym
from gym import spaces
import numpy as np
import itertools

from matplotlib import pyplot as plt
from scipy import linalg
import networkx as nx


class GridAction(Enum):
    LEFT = 0
    UP = 1
    RIGHT = 2
    DOWN = 3


class Library3x3Labyrinths:
    def __init__(self):
        lx = 3
        lz = 3
        A = np.zeros((lx * lz, lx * lz))
        D = np.zeros((lx * lz, lx * lz))

        for i in range(lx * lz):
            for j in range(lx * lz):
                if j == i:
                    A[i, j] = 0
                    if j < lx or j >= lx * (lz - 1):
                        if j % lx == 0 or j % lx == lx - 1:
                            D[i, j] = 2
                        else:
                            D[i, j] = 3
                    else:
                        if j % lx == 0 or j % lx == lx - 1:
                            D[i, j] = 3
                        else:
                            D[i, j] = 4
                elif j > i:
                    if j - i == 1 and j % lx != 0:
                        A[i, j] = 1
                        A[j, i] = 1
                    elif j - i == lx:
                        A[i, j] = 1
                        A[j, i] = 1
                    else:
                        continue
                else:
                    continue

        prod = itertools.product([0, 1], repeat=(lx - 1) * lz + (lz - 1) * lx)
        self.Ap = []
        self.Dp = []
        for p in prod:
            Adp = np.zeros((lx * lz, lx * lz))
            Degp = np.zeros((lx * lz, lx * lz))
            c = 0
            for i in range(lx * lz):
                for j in range(lx * lz):
                    if j > i:
                        if A[i, j] == 1:
                            Adp[i, j] = p[c]
                            Adp[j, i] = p[c]
                            c = c + 1
                        else:
                            Adp[i, j] = A[i, j]
                            Adp[j, i] = A[i, j]
                    else:
                        continue
                Degp[i, i] = sum(Adp[i])
            self.Ap.append(Adp)
            self.Dp.append(Degp)

        # Laplacian
        Lp = []
        nullp = []
        comp = []
        cnx = []
        n = 0
        for p in range(len(self.Ap)):
            Lap = np.zeros((lx * lz, lx * lz))
            for i in range(lx * lz):
                for j in range(lx * lz):
                    Lap[i, j] = self.Dp[p][i, j] - self.Ap[p][i, j]
            nullp.append(linalg.null_space(Lap))
            comp.append(nullp[p].shape[1])
            Lp.append(Lap)
            if nullp[p].shape[1] == 1:
                cnx.append(p)
                n = n + 1
            else:
                continue
        print(f'Connected subgraphs: {n}')

        # classification by homotopy
        self.pi0 = []
        self.pi1 = []
        self.pi2 = []
        self.pi3 = []
        self.pi4 = []
        for i in cnx:
            nst = sum(sum(self.Ap[i])) / 2
            if nst == (lx * lz) - 1:
                self.pi0.append(i)
            elif nst == (lx * lz):
                self.pi1.append(i)
            elif nst == (lx * lz) + 1:
                self.pi2.append(i)
            elif nst == (lx * lz) + 2:
                self.pi3.append(i)
            else:
                self.pi4.append(i)

        self.pi_sorted_indexes = {
            0: self.pi0,
            1: self.pi1,
            2: self.pi2,
            3: self.pi3,
            4: self.pi4,
        }

    def random_choice(self, pi: int = 0):
        default = 0
        indexes = self.pi_sorted_indexes.get(pi, default)
        return self.Ap[random.choice(indexes)]

    def print_summary(self):
        print('Generate all 3x3 mazes')
        print(f'{len(self.pi0)} mazes with pi_1=0')
        print(f'{len(self.pi1)} mazes with pi_1=1')
        print(f'{len(self.pi2)} mazes with pi_1=2')
        print(f'{len(self.pi3)} mazes with pi_1=3')
        print(f'{len(self.pi4)} mazes with pi_1=4')
        print(f'Total: {len(self.Ap)}')


LIBRARY_3X3_LABYRINTHS = Library3x3Labyrinths()


class GraphEnv2D(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, maze_scale=3, pi=1, enable_render=True, draw_deterministic: Optional[int] = None):
        super(GraphEnv2D, self).__init__()
        self.maze_scale = maze_scale

        # for now
        assert maze_scale == 3

        self.n_actions = 4
        self.action_space = spaces.Discrete(self.n_actions)
        self.n_states = self.maze_scale * self.maze_scale
        self.observation_space = spaces.Discrete(self.n_states)
        self.envs_library = LIBRARY_3X3_LABYRINTHS
        self.state_coords = np.array([[a, b] for a, b in itertools.product(range(self.maze_scale), range(self.maze_scale))])
        self.pos = {state: self.state_coords[state] for state in range(self.n_states)}
        self.pi = pi

        if draw_deterministic:
            index = self.envs_library.pi_sorted_indexes.get(self.pi, 0)[draw_deterministic]
            self.current_graph_adjacency = self.envs_library.Ap[index]

            self.start_state = 0
            self.goal_state = 8

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

        target_state = int(x_new * 3 + y_new)

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


class MetaGraphEnv2D(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, n_episodes_per_trial=10):
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

    def step(self, action):
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
