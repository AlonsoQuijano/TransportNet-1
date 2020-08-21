from math import sqrt
import numpy as np
import sinkhorn as skh
import data_handler as dh
import time
import pandas as pd

best_sink_beta = 2.5402

net_name = 'Custom_net.tntp'
trips_name = 'Custom_trips.tntp'

sink_num_iter, sink_eps = 10000, 10**(-10)
from history import History


def universal_similar_triangles_method(oracle, prox, primal_dual_oracle,
                                       t_start, L_init=None, max_iter=1000,
                                       eps=1e-5, eps_abs=None, stop_crit='dual_gap_rel',
                                       verbose_step=100, verbose=False, save_history=False):

    print('multi-stage! universal_similar_triangles_method')

    if stop_crit == 'dual_gap_rel':
        def crit():
            return duality_gap <= eps * duality_gap_init
    elif stop_crit == 'dual_gap':
        def crit():
            return duality_gap <= eps_abs
    elif stop_crit == 'max_iter':
        def crit():
            return it_counter == max_iter
    elif callable(stop_crit):
        crit = stop_crit
    else:
        raise ValueError("stop_crit should be callable or one of the following names: \
                         'dual_gap', 'dual_gap_rel', 'max iter'")

    L_value = L_init if L_init is not None else np.linalg.norm(oracle.grad(t_start))

    A_prev = 0.0
    y_start = u_prev = t_prev = np.copy(t_start)
    A = u = t = y = None

    grad_sum = None
    grad_sum_prev = np.zeros(len(t_start))

    flows_weighted = primal_dual_oracle.get_flows(y_start)
    primal, dual, duality_gap_init, state_msg = primal_dual_oracle(flows_weighted, y_start)
    if save_history:
        history = History('iter', 'primal_func', 'dual_func', 'dual_gap', 'inner_iters')
        history.update(0, primal, dual, duality_gap_init, 0)
    if verbose:
        print(state_msg)
    if eps_abs is None:
        eps_abs = eps * duality_gap_init

    success = False
    inner_iters_num = 0

    # multi-stage

    handler = dh.DataHandler()
    graph_data = handler.GetGraphData(net_name, columns=['init_node', 'term_node', 'capacity', 'free_flow_time'])
    graph_correspondences, total_od_flow = handler.GetGraphCorrespondences(trips_name)

    n = np.max(graph_data['graph_table']['init_node'].as_matrix())
    print('n: ', n)
    correspondence_matrix = handler.from_dict_to_cor_matr(graph_correspondences, n)

    T = handler.get_T_from_shortest_distances(n, graph_data)
    print('T_0: ', T)

    lambda_L = np.full((n,), 0.0, dtype=np.double)
    lambda_W = np.full((n,), 0.0, dtype=np.double)

    L = np.nansum(correspondence_matrix, axis=1)
    W = np.nansum(correspondence_matrix, axis=0)

    people_num = np.nansum(L)

    L = handler.distributor_L_W(L)
    W = handler.distributor_L_W(W)

    L = L / np.nansum(L)
    W = W / np.nansum(W)

    # print("L, W:", L, W)

    # multi-stage ends

    for it_counter in range(1, max_iter + 1):

        # corr reconstruction
        s = skh.Sinkhorn(n, L, W, people_num, sink_num_iter, sink_eps)
        cost_matrix = np.nan_to_num(T * best_sink_beta, nan=100)
        print(cost_matrix)

        rec, lambda_L, lambda_W = s.iterate(cost_matrix, lambda_L, lambda_W)

        # print('rec: ', rec)

        sink_correcpondences_dict = handler.from_cor_matrix_to_dict(rec)
        oracle.correspondences = sink_correcpondences_dict

        # print('or corr: ', oracle.correspondences)

        while True:
            inner_iters_num += 1

            alpha = 0.5 / L_value + sqrt(0.25 / L_value ** 2 + A_prev / L_value)
            A = A_prev + alpha

            y = (alpha * u_prev + A_prev * t_prev) / A
            grad_y = oracle.grad(y)
            flows = primal_dual_oracle.get_flows(y)  # grad() is called here
            grad_sum = grad_sum_prev + alpha * grad_y
            u = prox(grad_sum / A, y_start, 1.0 / A)
            t = (alpha * u + A_prev * t_prev) / A

            left_value = (oracle.func(y) + np.dot(grad_y, t - y) +
                          0.5 * alpha / A * eps_abs) - oracle.func(t)
            right_value = - 0.5 * L_value * np.sum((t - y) ** 2)
            if left_value >= right_value:
                break
            else:
                L_value *= 2

        A_prev = A
        L_value /= 2

        t_prev = t
        u_prev = u
        grad_sum_prev = grad_sum
        flows_weighted = (flows_weighted * (A - alpha) + flows * alpha) / A

        # print('t: ', t)
        # T = np.array(t)
        # T = T.reshape((2, 2))
        # print('T: ', T)

        graph_data['graph_table']['free_flow_time'] = t
        T = handler.get_T_from_shortest_distances(n, graph_data)
        print('T: ', T)

        primal, dual, duality_gap, state_msg = primal_dual_oracle(flows_weighted, t)
        if save_history:
            history.update(it_counter, primal, dual, duality_gap, inner_iters_num)
        if verbose and (it_counter % verbose_step == 0):
            print('\nIterations number: {:d}'.format(it_counter))
            print('Inner iterations number: {:d}'.format(inner_iters_num))
            print(state_msg, flush=True)
        if crit():
            success = True
            break

    result = {'times': t, 'flows': flows_weighted,
              'iter_num': it_counter,
              'res_msg': 'success' if success else 'iterations number exceeded'}
    if save_history:
        result['history'] = history.dict
    if verbose:
        print('\nResult: ' + result['res_msg'])
        print('Total iters: ' + str(it_counter))
        print(state_msg)
        print('Oracle elapsed time: {:.0f} sec'.format(oracle.time))
    return result