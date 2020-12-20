import warnings
# warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
np.set_printoptions(suppress=True)

import data_handler as dh
import sinkhorn as skh
import model as md
import csv

# net_name = 'vl_links.txt'
# trips_name = 'vl_trips.txt'
# parsers = 'vladik'
net_name = 'SiouxFalls_net.tntp'
trips_name = 'SiouxFalls_trips.tntp'
parsers = 'tntp'

best_sink_beta = 0.005
sink_num_iter, sink_eps = 2500, 10**(-8)
INF_COST = 100
INF_TIME = 1e10



def get_times_inverse_func(graph_table, times, rho = 0.15, mu=0.25):
    capacities = graph_table['capacity'].to_numpy()
    freeflowtimes = graph_table['free_flow_time'].to_numpy()
    # print('hm: ', np.power(times / freeflowtimes, mu))
    return np.transpose( (capacities / rho) * (np.power(times / freeflowtimes, mu) - 1.0))


def init_LW(corr_matrix):
    L = np.array(corr_matrix.sum(axis=1))
    W = np.array(corr_matrix.sum(axis=0))
    print('summarized')
    people_num = np.nansum(L)
    L = handler.distributor_L_W(L)
    W = handler.distributor_L_W(W)
    L = L / np.nansum(L)
    W = W / np.nansum(W)
    return L, W, people_num

def T_matrix_from_dict(T_dict, shape, old_to_new):
    print('fill T')
    T = np.zeros(shape)
    for key in T_dict.keys():
        source, target = old_to_new[key[0]], old_to_new[key[1]]
        T[source][target] = T_dict[key]
    return T

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
    correspondence_matrix, old_to_new, new_to_old = handler.reindexed_corr_matrix(graph_correspondences)
    print('fill correspondence_matrix')

    T_dict = handler.get_T_from_t(graph_data['graph_table']['free_flow_time'],
                                             graph_data, graph_correspondences)
    T = T_matrix_from_dict(T_dict, correspondence_matrix.shape, old_to_new)

    # best_sink_beta = n / np.nansum(T)
    T_0 = np.zeros(T.shape)

    print('init LW')
    L, W, people_num = init_LW(correspondence_matrix)
    print('L, W', L, W)
    # print('T after changes: ', T)

    for ms_i in range(5000):

        print('iteration: ', ms_i)

        s = skh.Sinkhorn(L, W, people_num, sink_num_iter, sink_eps)
        T = np.nan_to_num(T, nan=0, posinf=0, neginf=0)

        # best_sink_beta = T.shape[0] / np.nansum(T)

        # зачем тут nan_to_num если Т уже через него пропущена с другим nan=
        cost_matrix = np.nan_to_num(T * best_sink_beta, nan=INF_COST)
        rec, _, _ = s.iterate(cost_matrix)

        sink_correspondences_dict = handler.corr_matrix_to_dict(rec, new_to_old)

        L, W, people_num = init_LW(rec)

        model = md.Model(graph_data, sink_correspondences_dict,
                         total_od_flow, mu=0.25)

        for i, eps_abs in enumerate(np.logspace(1, 3, 1)):
            solver_kwargs = {'eps_abs': eps_abs,
                             'max_iter': max_iter}

            result = model.find_equilibrium(solver_name='ustm', composite=True,
                                            solver_kwargs=solver_kwargs,
                                            base_flows=alpha * graph_data['graph_table']['capacity'])

        graph_data['graph_table']['free_flow_time'] = result['times']
        T_dict = result['zone travel times']
        T = T_matrix_from_dict(T_dict, rec.shape, old_to_new)
        T_0 = T

        flows_inverse_func = get_times_inverse_func(graph_table, result['times'], rho=0.15, mu=0.25)

        subg = result['subg']

        # for key in result['subg'].keys():
        #     subg[key[0] - 1][key[1] - 1] = result['subg'][key]

        print('subg shape: ', np.shape(subg), 'flows_inv shape: ',  np.shape(flows_inverse_func))

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

