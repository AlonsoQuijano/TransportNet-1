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

import data_handler as dh
import sinkhorn as skh
import oracles as oracle
import model as md
import time
import pickle
import transport_graph as tg
import csv

net_name = 'vl_links.txt'
trips_name = 'vl_trips.txt'
parsers = 'vladik' 
# net_name = 'SiouxFalls_net.tntp' 
# trips_name =  'SiouxFalls_trips.tntp'
# parsers = 'tntp' 

best_sink_beta = 0.005
sink_num_iter, sink_eps = 2500, 10**(-8)
DEFAULT_COST = 100

def get_times_inverse_func(graph_table, times, rho = 0.15, mu=0.25):

    capacities = graph_table['capacity'].to_numpy()
    freeflowtimes = graph_table['free_flow_time'].to_numpy()
    # print('hm: ', np.power(times / freeflowtimes, mu))
    return np.transpose( (capacities / rho) * (np.power(times / freeflowtimes, mu) - 1.0))


def init_LW(corr_matrix, n):
    L = np.nansum(corr_matrix, axis=1)
    W = np.nansum(corr_matrix, axis=0)
    people_num = np.nansum(L)
    L = handler.distributor_L_W(L)
    W = handler.distributor_L_W(W)
    L = L / np.nansum(L)
    W = W / np.nansum(W)
    return L, W, people_num


if __name__ == '__main__':

    handler = dh.DataHandler()
    graph_data = handler.GetGraphData(net_name, eval(f'handler.{parsers}_net_parser'), columns=['init_node', 'term_node', 'capacity', 'free_flow_time'])
    graph_correspondences, total_od_flow = handler.GetGraphCorrespondences(trips_name, eval(f'handler.{parsers}_corr_parser'))

    handler = dh.DataHandler()

    max_iter = 2
    alpha = 0.9
    graph_table = graph_data['graph_table']

    n = np.max(graph_table['init_node'].to_numpy())
    print('n: ', n)
    correspondence_matrix = handler.from_dict_to_cor_matr(graph_correspondences, n)

    T_dict = handler.get_T_from_t(graph_data['graph_table']['free_flow_time'],  graph_data, graph_correspondences)
    # np.array([[0.0, 0.5], [0.5, 0.0]]) #
    T = np.full((n, n), np.inf)
    for key in T_dict['zone travel times'].keys():
        T[key[0] - 1][key[1] - 1] = T_dict['zone travel times'][key]

    # print(T)

    # best_sink_beta = n / np.nansum(T)
    T_0 = np.zeros(np.shape(T))

    L, W, people_num = init_LW(correspondence_matrix, n)
    # print('T after changes: ', T)

    for ms_i in range(5000):

        print('iteration: ', ms_i)

        s = skh.Sinkhorn(n, L, W, people_num, sink_num_iter, sink_eps)
        T = np.nan_to_num(T, nan=0, posinf=0, neginf=0)

        # best_sink_beta = n / np.nansum(T)

        # зачем тут nan_to_num если Т уже через него пропущена
        cost_matrix = np.nan_to_num(T * best_sink_beta, nan=DEFAULT_COST)
        rec, _, _ = s.iterate(cost_matrix)

        sink_correcpondences_dict = handler.from_cor_matrix_to_dict(rec)

        L, W, people_num = init_LW(rec, n)

        model = md.Model(graph_data, sink_correcpondences_dict,
                         total_od_flow, mu=0.25)

        for i, eps_abs in enumerate(np.logspace(1, 3, 1)):
            solver_kwargs = {'eps_abs': eps_abs,
                             'max_iter': max_iter}

            result = model.find_equilibrium(solver_name='ustm', composite=True,
                                            solver_kwargs=solver_kwargs,
                                            base_flows=alpha * graph_data['graph_table']['capacity'])

        graph_data['graph_table']['free_flow_time'] = result['times']
        T_dict = result['zone travel times']
        T_matrix = np.full((n, n), np.inf)

        for key in T_dict.keys():
            T_matrix[key[0] - 1][key[1] - 1] = T_dict[key]
        T = T_matrix
        T_0 = T

        flows_inverse_func = get_times_inverse_func(graph_table, result['times'], rho=0.15, mu=0.25)

        subg = result['subg']

        # for key in result['subg'].keys():
        #     subg[key[0] - 1][key[1] - 1] = result['subg'][key]

        print('subg: ', np.shape(subg), np.shape(flows_inverse_func))

        if max_iter == 2:

            np.savetxt('KEV_res/multi/flows/' + str(ms_i) + '_flows.txt', result['flows'], delimiter=' ')
            np.savetxt('KEV_res/multi/times/' + str(ms_i) + '_time.txt', result['times'], delimiter=' ')
            np.savetxt('KEV_res/multi/corr_matrix/' + str(ms_i) + '_corr_matrix.txt', rec, delimiter=' ')
            np.savetxt('KEV_res/multi/inverse_func/' + str(ms_i) + '_inverse_func.txt', flows_inverse_func,
                       delimiter=' ')
            np.savetxt('KEV_res/multi/subg/' + str(ms_i) + '_nabla_func.txt', subg, delimiter=' ')

        elif max_iter == 1:
            print('Mistake, should be 2! Counter in ustm starts from 1!')

        else:
            np.savetxt('KEV_res/iter/flows/' + str(ms_i) + '_flows.txt', result['flows'], delimiter=' ')
            np.savetxt('KEV_res/iter/times/' + str(ms_i) + '_time.txt', result['times'], delimiter=' ')
            np.savetxt('KEV_res/iter/corr_matrix/' + str(ms_i) + '_corr_matrix.txt', rec, delimiter=' ')
            np.savetxt('KEV_res/multi/inverse_func/' + str(ms_i) + '_inverse_func.txt', flows_inverse_func,
                       delimiter=' ')
            np.savetxt('KEV_res/multi/subg/' + str(ms_i) + '_nabla_func.txt', subg, delimiter=' ')

    with open('KEV_res/' + 'result.csv', 'w') as f:
        w = csv.DictWriter(f, result.keys())
        w.writeheader()
        w.writerow(result)

