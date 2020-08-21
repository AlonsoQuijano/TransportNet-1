import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import tensorflow as tf
import tensorflow.distributions
# from tensorflow.distributions import Dirichlet, Multinomial
from scipy.stats import entropy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
np.set_printoptions(suppress=True)

import data_handler as dh
import sinkhorn as skh
import oracles as oracle
import model as md
import time
import pickle
import transport_graph as tg

net_name = 'Custom_net.tntp'
trips_name = 'Custom_trips.tntp'

# best_sink_beta = 2.5402
sink_num_iter, sink_eps = 5000, 10**(-8)

if __name__ == '__main__':

    handler = dh.DataHandler()
    graph_data = handler.GetGraphData(net_name, columns=['init_node', 'term_node', 'capacity', 'free_flow_time'])
    graph_correspondences, total_od_flow = handler.GetGraphCorrespondences(trips_name)

    handler = dh.DataHandler()

    max_iter = 2000
    alpha = 0.9
    graph_table = graph_data['graph_table']

    n = np.max(graph_data['graph_table']['init_node'].as_matrix())
    print('n: ', n)
    correspondence_matrix = handler.from_dict_to_cor_matr(graph_correspondences, n)

    T = np.array([[0.0, 0.5], [0.5, 0.0]]) #handler.get_T_from_shortest_distances(n, graph_data)
    paycheck = np.array([1000, 2000])
    T = handler.get_T_new(n, T, paycheck)
    best_sink_beta = n / np.nansum(T)

    print('T_0: ', T)
    print('correspondence_matrix_0: ', correspondence_matrix)

    for ms_i in range(3):

        lambda_L = np.full((n,), 0.0, dtype=np.double)
        lambda_W = np.full((n,), 0.0, dtype=np.double)

        L = np.nansum(correspondence_matrix, axis=1)
        W = np.nansum(correspondence_matrix, axis=0)

        people_num = np.nansum(L)

        L = handler.distributor_L_W(L)
        W = handler.distributor_L_W(W)

        L = L / np.nansum(L)
        W = W / np.nansum(W)

        s = skh.Sinkhorn(n, L, W, people_num, sink_num_iter, sink_eps)
        T =  handler.get_T_new(n, T, paycheck)
        print('T: ', T)
        cost_matrix = np.nan_to_num(T * best_sink_beta, nan=100)
        print('cost_matrix: ', cost_matrix)

        rec, lambda_L, lambda_W = s.iterate(cost_matrix, lambda_L, lambda_W)
        # rec = np.flip(rec, 1)
        print('rec: ', rec)

        sink_correcpondences_dict = handler.from_cor_matrix_to_dict(rec)

        model = md.Model(graph_data, sink_correcpondences_dict,
                         total_od_flow, mu=0)


        for i, eps_abs in enumerate(np.logspace(1, 3, 1)):
            print(i, eps_abs)
            solver_kwargs = {'eps_abs': eps_abs,
                             'max_iter': max_iter}

            result = model.find_equilibrium(solver_name='ustm', composite=True,
                                            solver_kwargs=solver_kwargs,
                                            base_flows=alpha * graph_data['graph_table']['capacity'])

        T_dict = result['zone travel times']
        T_matrix = np.full((n, n), np.inf)
        for key in T_dict.keys():
            T_matrix[key[0] - 1][key[1] - 1] = T_dict[key]
        T = T_matrix
        # T = np.array([[0, 1], [1, 0]])

    print('s dict: ', sink_correcpondences_dict)
    # print(result.keys())
    print('eps_abs =', eps_abs)

    print('flows: ', result['flows'])
    print('times: ', result['times'])
    print('zone travel times', result['zone travel times'])

