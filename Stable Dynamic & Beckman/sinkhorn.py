import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np

np.set_printoptions(suppress=True)


class Sinkhorn:

    def __init__(self, L, W, people_num, iter_num, eps):
        self.L = L
        self.W = W
        assert (len(L) == len(W))
        self.n = len(L)
        self.people_num = people_num
        self.num_iter = iter_num
        self.eps = eps
        self.multistage_i = 0

    def iterate(self, cost_matrix):

        lambda_L = np.zeros(self.n)
        lambda_W = np.zeros(self.n)
        # r = None

        for k in range(self.num_iter):

            lambda_Wn, lambda_Ln = self.sinkhorn(k, cost_matrix, lambda_W, lambda_L)

            delta = np.linalg.norm(np.concatenate((lambda_Ln - lambda_L,
                                                   lambda_Wn - lambda_W)))

            lambda_L, lambda_W = lambda_Ln, lambda_Wn

            if delta < self.eps:
                break

        r = self.rec_d_i_j(lambda_Ln, lambda_Wn, cost_matrix)
        # self.multistage_i += 1
        return r, lambda_L, lambda_W

    def sinkhorn(self, k, cost_matrix, lambda_W_prev, lambda_L_prev):
        lambda_L = np.zeros((self.n,), dtype=np.double)
        lambda_W = np.zeros((self.n,), dtype=np.double)

        if k % 2 == 0:
            lambda_L = lambda_L_prev + np.log(self.L) - np.log(
                np.dot(self.B(cost_matrix, lambda_L, lambda_W), np.ones(self.n))
            )
        else:
            lambda_W = lambda_W_prev + np.log(self.W) - np.log(
                np.dot(np.transpose(self.B(cost_matrix, lambda_L, lambda_W)), np.ones(self.n))
            )

        lambda_W = np.reshape(lambda_W, (self.n))
        lambda_L = np.reshape(lambda_L, (self.n))
        return lambda_W, lambda_L

    def rec_d_i_j(self, lambda_L, lambda_W, cost_matrix):
        er = np.zeros((self.n, self.n))
        bottom = self.B(cost_matrix, lambda_L, lambda_W)

        for i, lambda_L_i in enumerate(lambda_L, 0):
            for j, lambda_W_j in enumerate(lambda_W, 0):
                cost_matrix_i_j = cost_matrix[i][j]
                top = self.B_i_j(cost_matrix_i_j, lambda_L_i, lambda_W_j)

                error = top / bottom
                er[i][j] = error

        return er * self.people_num

    def B_i_j(self, cost_matrix_i_j, lambda_L_i, lambda_W_j):
        return np.exp(-cost_matrix_i_j + lambda_L_i + lambda_W_j)

    def B(self, cost_matrix, lambda_L, lambda_W):
        bottom = 0
        for i, lambda_L_i in enumerate(lambda_L, 0):
            for j, lambda_W_j in enumerate(lambda_W, 0):
                cost_matrix_i_j = cost_matrix[i][j]
                bottom += self.B_i_j(cost_matrix_i_j, lambda_L_i, lambda_W_j)
        return bottom

    # def sinkhorn(self, k, cost_matrix, lambda_W_prev, lambda_L_prev):
    #
    #     if k % 2 == 0:
    #         lambda_W = lambda_W_prev
    #         lambda_L = np.log(np.nansum(
    #             (np.exp(-lambda_W_prev - 1 - cost_matrix)).T
    #             / self.L, axis=0
    #         ))
    #     else:
    #         lambda_L = lambda_L_prev
    #         lambda_W = np.log(np.nansum(
    #             (np.exp(-lambda_L - 1 - cost_matrix.T)).T
    #             / self.W, axis=0
    #         ))
    #
    #     return lambda_W, lambda_L

    # def rec_d_i_j(self, lambda_L, lambda_W, cost_matrix):
    #     er = np.exp(-1 - cost_matrix - (np.reshape(lambda_L, (self.n, 1)) + lambda_W))
    #     return er * self.people_num
