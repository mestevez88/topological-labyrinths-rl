#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 11:22:15 2023

@author: manuelestevez
"""
import argparse
import logging
import os
import pickle
import random
from collections import defaultdict
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from viz_maze import draw_maze, draw_maze_collection


def n_edges_grid(lx, lz):
    return (lx - 1) * lz + (lz - 1) * lx


def max_pis(lx, lz):
    v = lx * lz
    e_grid = n_edges_grid(lx, lz)
    e_spanning_tree = v - 1
    return e_grid - e_spanning_tree


def dfs(adj_list, visited, vertex, result, key):
    visited.add(vertex)
    result[key].append(vertex)
    for neighbor in adj_list[vertex]:
        if neighbor not in visited:
            dfs(adj_list, visited, neighbor, result, key)


def generate_random_maze_adjacency(lx: int, lz: int, pi: int = 0):
    nb = {}
    # neighbors dictionary
    for i in range(lx * lz):
        if i % lx == 0:
            if np.floor(i / lx) == 0:
                nb[i] = [i + 1, i + lx]
            elif np.floor(i / lx) == lz - 1:
                nb[i] = [i - lx, i + 1]
            else:
                nb[i] = [i - lx, i + 1, i + lx]
        elif i % lx == lx - 1:
            if np.floor(i / lx) == 0:
                nb[i] = [i - 1, i + lx]
            elif np.floor(i / lx) == lz - 1:
                nb[i] = [i - lx, i - 1]
            else:
                nb[i] = [i - lx, i - 1, i + lx]
        else:
            if np.floor(i / lx) == 0:
                nb[i] = [i - 1, i + 1, i + lx]
            elif np.floor(i / lx) == lz - 1:
                nb[i] = [i - lx, i - 1, i + 1]
            else:
                nb[i] = [i - lx, i - 1, i + 1, i + lx]

    A = set()  # initial configuration (snake)
    edges = set()  # all possible edges
    for i in range(lx * lz):
        for j in range(lx * lz):
            if j in nb[i]:
                edges.add(frozenset({i, j}))
                if np.floor(i / lx) % 2 == 0:
                    if j != i + 1:
                        if j % lx == lx - 1 and j > i:
                            A.add(frozenset({i, j}))
                        else:
                            continue
                    else:
                        A.add(frozenset({i, j}))
                else:
                    if j != i + 1:
                        if j % lx == 0 and j > i:
                            A.add(frozenset({i, j}))
                        else:
                            continue
                    else:
                        A.add(frozenset({i, j}))
            else:
                continue

    for _ in range(pi):
        A.add(random.sample(tuple(edges - A), 1)[0])

    if edges - A:
        repeat = 100000

        for i in range(repeat):
            NA = edges - A
            A1 = set()
            RA = random.choice(list(A))
            RNA = random.choice(list(NA))
            for r in A:
                A1.add(r)
            A1.remove(RA)
            A1.add(RNA)
            adj_list = defaultdict(list)
            for x, y in A1:
                adj_list[x].append(y)
                adj_list[y].append(x)
            result = defaultdict(list)
            visited = set()
            for vertex in adj_list:
                if vertex not in visited:
                    dfs(adj_list, visited, vertex, result, vertex)
            logging.debug(result.values())
            if len(result.values()) == 1 and len(visited) == lx * lz:
                A = A1
                logging.debug(("cnx"))
                logging.debug((len(result.values())))
            else:
                logging.debug(('discnx'))
                logging.debug((len(result.values())))

    # Adjacency matrix
    Ap = np.zeros((lx * lz, lx * lz))
    for i in nb:
        for j in nb:
            if frozenset({i, j}) in A:
                Ap[i, j] = 1
            else:
                continue

    return Ap


def sample_random_mazes(lx: int, lz: int, nm: int, save_pickle: bool = False, max_pi: Optional[int] = None):
    """
    randomly generate collection of lx x lz mazes
    @param max_pi: maximum fundamental cycle to sample from
    @param save_pickle: if True, write pickled object to disc
    @param lx: width of the maze
    @param lz: height of te maze
    @param nm: number of independently sampled mazes per pi
    @return: collection of mazes
    """
    run_name = f"mazes_{lx}x{lz}_{random.getrandbits(32):X}"

    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(filename=f"logs/maze_sampler_{run_name}.txt",
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.DEBUG)

    os.makedirs("mazes", exist_ok=True)

    if max_pi:
        pis = list(range(min(max_pis(lx, lz), max_pi) + 1))
    else:
        pis = list(range(max_pis(lx, lz) + 1))

    print(f"Generate {args.lx}x{args.lz} mazes with pi_1={set(pis)}")
    logging.info(f"Generate {args.lx}x{args.lz} mazes with pi_1={set(pis)}")

    mazes = {"mazes": {pi: [generate_random_maze_adjacency(lx, lz, pi=pi) for _ in tqdm(range(nm))] for pi in pis},
             "info": {"lx": lx, "lz": lz, "n_mazes": nm, "pis": pis}}

    if save_pickle:
        path = os.path.join("mazes", f"{run_name}.p")
        print(f"Save at {path}")
        logging.info(f"Save at {path}")
        pickle.dump(mazes, open(path, "wb"))

    return mazes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Maze Sampler', description='Generate collection of rectangular mazes of a '
                                                                      'given size sorted by their fundamental group',
                                     epilog='Have fun torturing your agent ;)')

    parser.add_argument('-x', '--lx', help="maze width", default=4)  # option that takes a value
    parser.add_argument('-z', '--lz', help="maze height", default=4)
    parser.add_argument('-n', '--n_mazes', default=10)  # on/off flag

    args = parser.parse_args()

    mazes = sample_random_mazes(args.lx, args.lz, args.n_mazes, save_pickle=True)

    draw_maze_collection(mazes)
