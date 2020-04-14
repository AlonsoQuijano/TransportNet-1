import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from scipy.stats import entropy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
np.set_printoptions(suppress=True)


class Sinkhorn:

    def __init__(self, n, L, W, people_num, iter_num, eps):
        self.n = n
        self.L = L
        self.W = W
        self.people_num = people_num
        self.num_iter = iter_num
        self.eps = eps

    def sinkhorn(self, k, cost_matrix, lambda_W_prev, lambda_L_prev):

        lambda_L = np.full((self.n,), 0.0, dtype=np.double)
        lambda_W = np.full((self.n,), 0.0, dtype=np.double)

        if k % 2 == 0:
            lambda_W = lambda_W_prev
            lambda_L = np.log(np.nansum(
                (np.exp(-lambda_W_prev - 1 - cost_matrix)).T
                / self.L, axis=0
            ))
        else:
            lambda_L = lambda_L_prev
            lambda_W = np.log(np.nansum(
                (np.exp(-lambda_L - 1 - cost_matrix.T)).T
                / self.W, axis=0
            ))

        return lambda_W, lambda_L

    def iterate(self, cost_matrix):

        lambda_L = np.full((self.n,), 0.0, dtype=np.double)
        lambda_W = np.full((self.n,), 0.0, dtype=np.double)

        lambda_Ln = np.full((self.n,), 0.0, dtype=np.double)
        lambda_Wn = np.full((self.n,), 0.0, dtype=np.double)

        for k in range(self.num_iter):

            lambda_Wn, lambda_Ln = self.sinkhorn(k, cost_matrix, lambda_W, lambda_L)

            delta = np.linalg.norm(np.concatenate((lambda_Ln - lambda_L,
                                                   lambda_Wn - lambda_W)))

            lambda_L, lambda_W = lambda_Ln, lambda_Wn

            if delta < self.eps:
                break

        r = self.rec_d_i_j(lambda_Ln, lambda_Wn, cost_matrix)
        return r

    def rec_d_i_j(self, lambda_L, lambda_W, cost_matrix):
        er = np.exp(-1 - cost_matrix - (np.reshape(lambda_L, (self.n, 1)) + lambda_W))
        return er * self.people_num