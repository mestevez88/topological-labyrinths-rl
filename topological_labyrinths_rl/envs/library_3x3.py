import random
import itertools

from scipy import linalg
import numpy as np


class Library3x3Labyrinths:
    def __init__(self):
        lx = 3
        lz = 3
        A = np.zeros((lx * lz, lx * lz))
        D = np.zeros((lx * lz, lx * lz))
        self.maze_scale = 3

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
