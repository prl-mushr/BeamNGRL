import time 
import math as m
import numpy as np
from scipy.stats import mannwhitneyu as utest
import traceback
import matplotlib.pyplot as plt

data = np.load('compare_CE_MPPI.npy')

# results = np.hstack((temp, noise_, NS, NO, avg_cost, avg_dt)) # this is the data format. avg_cost is a 1D array of length 50, avg_dt is a 1D array of length 50
# get temp, noise_, NS, NO, avg_cost, avg_dt
temp = data[:,0]
noise_ = data[:,1]
NS = data[:,2]
NO = data[:,3]
avg_cost = data[:,4:54]
avg_dt = data[:,54:104]

plt.figure()
for t in temp:
    for n in noise_:
        for ns in NS:
            for no in NO:
                # get the index of the data
                idx = np.where((temp==t) & (noise_==n) & (NS==ns) & (NO==no))
                # get the data
                avg_cost_ = avg_cost[idx]
                avg_dt_ = avg_dt[idx]
                # plot
                plt.plot(avg_dt_, avg_cost_, label='temp = {}, noise = {}, NS = {}, NO = {}'.format(t, n, ns, no))
                plt.xlabel('dt')
                plt.ylabel('cost')
plt.show()
