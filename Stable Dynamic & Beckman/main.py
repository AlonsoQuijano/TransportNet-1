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

if __name__ == '__main__':

    net_name = 'data/EMA_net.tntp'
    trips_name = 'data/EMA_trips.tntp'

    handler = dh.DataHandler()
    graph_data = handler.GetGraphData(net_name, columns_order = np.array([0, 1, 2, 3, 4]))
    graph_correspondences, total_od_flow = handler.GetGraphCorrespondences(trips_name)

    model = md.Model(graph_data, graph_correspondences,
                     total_od_flow, mu=0.25, rho=0.15)

    max_iter = 100

    for i, eps_abs in enumerate(np.logspace(1, 3, 1)):
        print('eps_abs =', eps_abs)
        solver_kwargs = {'eps_abs': eps_abs,
                         'max_iter': max_iter}
        tic = time.time()
        result = model.find_equilibrium(solver_name='multi-stage',
                                        solver_kwargs=solver_kwargs,
                                        verbose=False)