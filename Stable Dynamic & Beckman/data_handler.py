from scanf import scanf
import re
import numpy as np
import pandas as pd
import transport_graph as tg

#TODO: DOCUMENTATION!!!
class DataHandler:
    def GetGraphData(self, file_name, columns):
        graph_data = {}

        metadata = ''
        with open(file_name, 'r') as myfile:
            for index, line in enumerate(myfile):
                if re.search(r'^~', line) is not None:
                    skip_lines = index + 1
                    headlist = re.findall(r'[\w]+', line)
                    break
                else:
                    metadata += line
        graph_data['nodes number'] = scanf('<NUMBER OF NODES> %d', metadata)[0]
        graph_data['links number'] = scanf('<NUMBER OF LINKS> %d', metadata)[0]
        graph_data['zones number'] = scanf('<NUMBER OF ZONES> %d', metadata)[0]
        first_thru_node = scanf('<FIRST THRU NODE> %d', metadata)[0]

        dtypes = {'init_node' : np.int32, 'term_node' : np.int32, 'capacity' : np.float64, 'length': np.float64,
                  'free_flow_time': np.float64, 'b': np.float64, 'power': np.float64, 'speed': np.float64,'toll': np.float64,
                  'link_type' : np.int32}
        df = pd.read_csv(file_name, names = headlist, dtype = dtypes, skiprows = skip_lines, sep = r'[\s;]+', engine='python',
                         index_col = False)
        df = df[columns]

        df.insert(loc = list(df).index('init_node') + 1, column = 'init_node_thru', value = (df['init_node'] >= first_thru_node))
        df.insert(loc = list(df).index('term_node') + 1, column = 'term_node_thru', value = (df['term_node'] >= first_thru_node))
        graph_data['graph_table'] = df
        return graph_data


    def GetGraphCorrespondences(self, file_name):
        with open(file_name, 'r') as myfile:
            trips_data = myfile.read()

        total_od_flow = scanf('<TOTAL OD FLOW> %f', trips_data)[0]
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
        return graph_correspondences, total_od_flow


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
        try:
            zero_num = array_dict[0]
        except KeyError:
            print('this array without 0')
            return array
        array[max_value_index] = max_value - zero_num
        for index in np.where(array == 0)[0]:
            array[index] = 1.0

        return array

    def get_T_from_shortest_distances(self, n, graph_data):

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

    def get_T_new(self, n, T, paycheck):

        T_new = np.full((n, n), np.nan)
        for i in range(n):
            for j in range(n):
                T_new[i][j] = T[i][j] - paycheck[j]

        return T_new