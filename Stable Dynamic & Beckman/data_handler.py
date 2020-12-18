from scanf import scanf
import re
import numpy as np
import pandas as pd
import transport_graph as tg
import copy
import graph_tool.topology as gtt

#TODO: DOCUMENTATION!!!
class DataHandler:
    @staticmethod
    def vladik_net_parser(file_name):
        graph_data = {}
        links = pd.read_csv(file_name, sep='\t',skiprows=0)
        links_ab = links[['ANODE', 'BNODE', 'cap_ab', 'LENGTH']]
        links_ab.columns = ['init_node', 'term_node', 'capacity', 'free_flow_time']
        links_ba = links[['ANODE', 'BNODE', 'cap_ba', 'LENGTH']]
        links_ba.columns = ['init_node', 'term_node', 'capacity', 'free_flow_time']
        df = links_ab.append(links_ba, ignore_index=True)
        # graph_data['nodes number'] = scanf('<NUMBER OF NODES> %d', metadata)[0]

        return df, 1

    @staticmethod
    def tntp_net_parser(file_name):
        metadata = ''
        with open(file_name, 'r') as myfile:
            for index, line in enumerate(myfile):
                if re.search(r'^~', line) is not None:
                    skip_lines = index + 1
                    headlist = re.findall(r'[\w]+', line)
                    break
                else:
                    metadata += line

        graph_data = {}
        nn = scanf('<NUMBER OF NODES> %d', metadata)[0]
        nl = scanf('<NUMBER OF LINKS> %d', metadata)[0]
        nz = scanf('<NUMBER OF ZONES> %d', metadata)[0]
        print('NUMBER OF NODES, LINKS, ZONES: ', nn, nl, nz)
        first_thru_node = scanf('<FIRST THRU NODE> %d', metadata)[0]

        dtypes = {'init_node' : np.int32, 'term_node' : np.int32, 'capacity' : np.float64, 'length': np.float64,
                  'free_flow_time': np.float64, 'b': np.float64, 'power': np.float64, 'speed': np.float64,'toll': np.float64,
                  'link_type' : np.int32}
        df = pd.read_csv(file_name, names = headlist, dtype = dtypes, skiprows = skip_lines, sep = r'[\s;]+', engine='python',
                         index_col = False)
        return df, first_thru_node

    def GetGraphData(self, file_name, parser, columns):
        
        graph_data = {}
        df, first_thru_node = parser(file_name)
        df = df[columns]
        df.insert(loc = list(df).index('init_node') + 1, column = 'init_node_thru', value = (df['init_node'] >= first_thru_node))
        df.insert(loc = list(df).index('term_node') + 1, column = 'term_node_thru', value = (df['term_node'] >= first_thru_node))
        graph_data['graph_table'] = df
        graph_data['nodes number'] = len(set(df.init_node.values) | set(df.term_node.values))
        graph_data['links number'] = df.shape[0]
        graph_data['zones number'] = graph_data['nodes number'] # nodes in corr graph actually
        print('nUMBER OF NODES, LINKS, ZONES: ', graph_data['nodes number'], 
            graph_data['links number'], graph_data['zones number'])
        return graph_data 
        
    @staticmethod
    def vladik_corr_parser(file_name):
        graph_correspondences = {}
        od = 0
        with open(file_name, 'r') as fin:
            fin = list(fin)[1:]
            for line in fin:
                frm, to, load = [float(x) for x in line.split(',')]

                if frm not in graph_correspondences:
                    graph_correspondences[frm] = {'targets':[], 'corrs': []}

                graph_correspondences[frm]['targets'].append(to)
                graph_correspondences[frm]['corrs'].append(load)
                od += load

            return graph_correspondences#, od


        
    @staticmethod
    def tntp_corr_parser(file_name):
        with open(file_name, 'r') as myfile:
            trips_data = myfile.read()

        total_od_flow = scanf('<TOTAL OD FLOW> %f', trips_data)[0]
        print('total_od_flow scanned', total_od_flow)
        #zones_number = scanf('<NUMBER OF ZONES> %d', trips_data)[0]

        origins_data = re.findall(r'Origin[\s\d.:;]+', trips_data)

        graph_correspondences = {}
        for data in origins_data:
            origin_index = scanf('Origin %d', data)[0]
            origin_correspondences = re.findall(r'[\d]+\s+:[\d.\s]+;', data)
            targets = []
            corrs_vals = []
            for line in origin_correspondences:
                target, corrs = scanf('%d : %f', line)
                targets.append(target)
                corrs_vals.append(corrs)
            graph_correspondences[origin_index] = {'targets' : targets, 'corrs' : corrs_vals}

        return graph_correspondences

        


    @staticmethod
    def GetGraphCorrespondences(file_name, parser):
        
        graph_corrs = parser(file_name)
        total_od_flow = sum(np.nansum(val['corrs']) for key, val in graph_corrs.items())
        print('total_od_flow calculated', total_od_flow)
        
        return graph_corrs, total_od_flow


    def ReadAnswer(self, filename):
        with open(filename) as myfile:
            lines = myfile.readlines()
        lines = lines[1 :]
        flows = []
        times = []
        for line in lines:
            _, _, flow, time = scanf('%d %d %f %f', line)
            flows.append(flow)
            times.append(time)
        return {'flows' : flows, 'times' : times}

    #### Katya multi-stage methods

    def create_C(self, df, n, column_name):
        C = np.full((n, n), np.nan, dtype=np.double)
        column_ind = df.columns.get_loc(column_name)

        for index, raw_data_line in df.iterrows():
            i, j = int(raw_data_line[0]) - 1, int(raw_data_line[1]) - 1

            C[i, j] = raw_data_line[column_ind]
        return C

    def from_dict_to_cor_matr(self, dictnr, n):
        correspondence_matrix = np.full((n, n), np.nan, dtype=np.double)
        i = 1
        if n > 1:
            for key in dictnr.keys(): #dictnr[i].keys()
                for k, v in zip(dictnr[i]['targets'], dictnr[i]['corrs']):
                    correspondence_matrix[key - 1][k - 1] = v # костыль!
                i += 1
        else:
            for key in dictnr.keys():
                for k, v in zip(dictnr[i][key].keys(), dictnr[i][key].values()):
                    correspondence_matrix[int(key) - 1][int(k)] = v
        # print('corr mtrx: ', correspondence_matrix)
        return correspondence_matrix

    def from_cor_matrix_to_dict(self, corr_matrix):
        d = {}
        # print(np.shape(corr_matrix))
        n = np.shape(corr_matrix)[0]

        for i in range(1, n + 1):
            d[i] = {}
            l = list(range(1, n + 1))
            l.remove(i)
            d[i]['targets'] = [i] + l
            l_2 = []

            for j, t in zip(range(1, n + 1), d[i]['targets']):
                value = corr_matrix[i - 1][t - 1]
                l_2.append(value)
                d[i]['corrs'] = l_2
        return d

    def distributor_L_W(self, array):
        max_value = np.max(array)
        max_value_index = np.where(array == np.max(array))

        unique, counts = np.unique(array, return_counts=True)
        array_dict = dict(zip(unique, counts))
        # TODO remove exception
        try:
            zero_num = array_dict[0]
        except KeyError:
            print('this array without 0')
            return array
        array[max_value_index] = max_value - zero_num

        array[np.where(array == 0)[0]] = 1.0

        return array

    def get_t_from_shortest_distances(self, n, graph_data):

        targets = []
        df = graph_data['graph_table']
        T = None
        i = 0
        for source in range(n):

            targets = [source]
            targets += range(0, n)

            graph_table = graph_data['graph_table']

            graph = tg.TransportGraph(graph_table,
                                      graph_data['links number'],
                                      graph_data['nodes number'])


            t_exp = np.array(df['free_flow_time'], dtype='float64').flatten()
            distances, pred_map = graph.shortest_distances(source=source,
                                                           targets=targets,
                                                           times=t_exp)

            if i == 0:
                T = distances[1:]
            else:
                T = np.vstack([T, distances[1:]])
            targets = []
            i += 1

        return T

    def _index_nodes(self, graph_table, graph_correspondences):
        table = graph_table.copy()
        inits = np.unique(table['init_node'][table['init_node_thru'] == False])
        terms = np.unique(table['term_node'][table['term_node_thru'] == False])
        through_nodes = np.unique([table['init_node'][table['init_node_thru'] == True],
                                   table['term_node'][table['term_node_thru'] == True]])

        nodes = np.concatenate((inits, through_nodes, terms))
        nodes_inds = list(zip(nodes, np.arange(len(nodes))))
        init_to_ind = dict(nodes_inds[: len(inits) + len(through_nodes)])
        term_to_ind = dict(nodes_inds[len(inits):])

        table['init_node'] = table['init_node'].map(init_to_ind)
        table['term_node'] = table['term_node'].map(term_to_ind)
        correspondences = {}
        for origin, dests in graph_correspondences.items():
            dests = copy.deepcopy(dests)
            correspondences[init_to_ind[origin]] = {'targets': list(map(term_to_ind.get, dests['targets'])),
                                                    'corrs': dests['corrs']}

        inds_to_nodes = dict(zip(range(len(nodes)), nodes))
        return inds_to_nodes, correspondences, table

    def get_T_from_t(self, t, graph_data, graph_correspondences):

        result = {}
        result['zone travel times'] = {}

        import transport_graph as tg

        inds_to_nodes, graph_correspondences_, graph_table_ = self._index_nodes(graph_data['graph_table'],
                                                                                   graph_correspondences)

        # print('data handler: ', len(inds_to_nodes), graph_data['links number'])
        graph_dh = tg.TransportGraph(graph_table_, len(inds_to_nodes), graph_data['links number'])

        for source in graph_correspondences_:

            targets = graph_correspondences_[source]['targets']
            travel_times, _ = graph_dh.shortest_distances(source, targets, t)
            # print('in model.py, travel_times: ', travel_times)
            # mapping nodes' indices to initial nodes' names:
            source_nodes = [inds_to_nodes[source]] * len(targets)
            target_nodes = list(map(inds_to_nodes.get, targets))
            result['zone travel times'].update(zip(zip(source_nodes, target_nodes), travel_times))

        return result

    def get_T_new(self, n, T, paycheck):

        T_new = np.full((n, n), np.nan)
        for i in range(n):
            for j in range(n):
                T_new[i][j] = T[i][j] - paycheck[j]

        return T_new
