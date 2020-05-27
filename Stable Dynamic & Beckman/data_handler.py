from scanf import scanf
import re
import numpy as np
import pandas as pd
import transport_graph as tg

# TODO: DOCUMENTATION!!!
class DataHandler:
    def GetGraphData(self, file_name, columns_order):
        graph_data = {}

        with open(file_name, 'r') as myfile:
            data = myfile.read()

        graph_data['kNodesNumber'] = scanf('<NUMBER OF NODES> %d', data)[0]
        graph_data['kLinksNumber'] = scanf('<NUMBER OF LINKS> %d', data)[0]

        headlist = re.compile("\t"
                              "[a-zA-Z ]+"
                              "[\(\)\/\w]*").findall(data)

        my_headlist = ['Init node', 'Term node', 'Capacity', 'length', 'Free Flow Time']

        datalist = re.compile("[\t0-9.]+\t;").findall(data)

        datalist = [line.strip('[\t;]') for line in datalist]
        datalist = [line.split('\t') for line in datalist]

        df = pd.DataFrame(np.asarray(datalist)[:, columns_order], columns=my_headlist)
        # df = pd.DataFrame(np.asarray(datalist)[:, range(0, len(headlist))], columns = headlist)
        # df = df[list(np.array(headlist)[columns_order])]
        # print(list(np.array(headlist)[[0, 1, 2, 4]]))

        # init nodes
        df['Init node'] = pd.to_numeric(df['Init node'], downcast='integer')
        # final nodes
        df['Term node'] = pd.to_numeric(df['Term node'], downcast='integer')

        # capacities
        df['Capacity'] = pd.to_numeric(df['Capacity'], downcast='float')
        # length
        df['length'] = pd.to_numeric(df['length'], downcast='float')
        # free flow times
        df['Free Flow Time'] = pd.to_numeric(df['Free Flow Time'], downcast='float')

        # Table for graph ready!
        graph_data['graph_table'] = df

        return graph_data

    def GetGraphCorrespondences(self, file_name):
        with open(file_name, 'r') as myfile:
            trips_data = myfile.read()

        total_od_flow = scanf('<TOTAL OD FLOW> %f', trips_data)[0]

        # kZonesNumber = scanf('<NUMBER OF ZONES> %d', trips_data)[0]
        p = re.compile("Origin[ \t]+[\d]+")
        origins_list = p.findall(trips_data)
        origins = np.array([int(re.sub('[a-zA-Z ]', '', line)) for line in origins_list])

        p = re.compile("\n"
                       "[0-9.:; \n]+"
                       "\n\n")
        res_list = p.findall(trips_data)
        res_list = [re.sub('[\n \t]', '', line) for line in res_list]

        graph_correspondences = {}
        for origin_index in range(0, len(origins)):
            origin_correspondences = res_list[origin_index].strip('[\n;]').split(';')
            graph_correspondences[origins[origin_index]] = dict([scanf("%d:%f", line)
                                                                 for line in origin_correspondences])
        return graph_correspondences, total_od_flow

    def ReadAnswer(self, filename):
        with open(filename) as myfile:
            lines = myfile.readlines()
        lines = np.array(lines)[range(1, len(lines))]
        values_dict = {'flow': [], 'time': []}
        for line in lines:
            line = line.strip('[ \n]')
            nums = line.split(' \t')
            values_dict['flow'].append(float(nums[2]))
            values_dict['time'].append(float(nums[3]))
        return values_dict

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

        for key in dictnr[i].keys():
            for k, v in zip(dictnr[key].keys(),
                            dictnr[key].values()):
                if v != 0:
                    correspondence_matrix[key - 1][k - 1] = v
            i += 1

        return correspondence_matrix

    def from_cor_matrix_to_dict(self, corr_matrix):
        d = {}
        print(np.shape(corr_matrix))
        n = np.shape(corr_matrix)[0]
        buf_l = []
        buf_ind = []

        for i in range(1, n + 1):
            for j in range(1, n + 1):
                buf_ind.append(int(j))
                buf_l.append(float(corr_matrix[i - 1][j - 1]))
            if i in d:
                d[i].append(dict(zip(buf_ind, buf_l)))
            else:
                d[i] = dict(zip(buf_ind, buf_l))

            buf_ind = []
            buf_l = []
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

            graph = tg.TransportGraph(graph_data)
            t_exp = np.array(df['Free Flow Time'], dtype='float64').flatten()
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