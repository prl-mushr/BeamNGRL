import time 
import math as m
import numpy as np
from scipy.stats import mannwhitneyu as utest
import traceback
import matplotlib.pyplot as plt
# import for ttest that takes mean and std:
from scipy.stats import ttest_ind_from_stats as ttest

data = np.load('compare_CE_MPPI_obs_2_samples_64_temp_1_noise_1_opt_4_soft_cost_no.npy')

# results = np.hstack((temp, noise_, NS, NO, avg_cost, avg_dt)) # this is the data format. avg_cost is a 1D array of length 50, avg_dt is a 1D array of length 50
# get temp, noise_, NS, NO, avg_cost, avg_dt
temp = data[:,0]
noise_ = data[:,1]
NS = data[:,2]
NO = data[:,3]
avg_cost = data[:,4:54]
avg_dt = data[:,54:104]

mean_cost = np.mean(avg_cost, axis=1)
std_cost = np.std(avg_cost, axis=1)

# print mean_cost where NO = 1, 2, 4, temp = 0.1, noise_ = 0.1 and NS = 128
noise_scale = np.array([5.0])#, 2.0, 3.0, 3.0])  # noise scale factor. scale factor of 5 results in throttle variance = 1
temperatures = np.arange(0.0, 0.1, 0.1)
for t in temperatures:
    for noise in noise_scale:
        if(t == 0.0):
            t = 0.01
        MPPI_1_mean, MPPI_1_std = mean_cost[(NO == 1) & (temp == t) & (noise_ == noise) & (NS ==64)], std_cost[(NO == 1) & (temp == t) & (noise_ == noise) & (NS ==64)]
        MPPI_2_mean, MPPI_2_std = mean_cost[(NO == 2) & (temp == t) & (noise_ == noise) & (NS == 32)], std_cost[(NO == 2) & (temp == t) & (noise_ == noise) & (NS == 32)]
        MPPI_4_mean, MPPI_4_std = mean_cost[(NO == 4) & (temp == t) & (noise_ == noise) & (NS == 16)], std_cost[(NO == 4) & (temp == t) & (noise_ == noise) & (NS == 16)]

        # perform ttest between MPPI_1_mean and MPPI_2_mean, where MPPI_1_mean is the mean of the avg_cost for NO = 1, temp = t, noise_ = noise, NS = 128
        # MPPI_1_mean is a single number, MPPI_2_mean is a single number:
        _, p1 = ttest(MPPI_1_mean, MPPI_1_std,50, MPPI_2_mean, MPPI_2_std, 50)
        _, p2 = ttest(MPPI_2_mean, MPPI_1_std,50, MPPI_4_mean, MPPI_4_std, 50)


        print("temp: {}, noise: {}, num_opts:{}, effective samples: {}, mean cost: {}, p {}".format(t, noise, 1, 256, MPPI_1_mean, 1) )
        print("temp: {}, noise: {}, num_opts:{}, effective samples: {}, mean cost: {}, p {}".format(t, noise, 2, 256, MPPI_2_mean, p1) )
        print("temp: {}, noise: {}, num_opts:{}, effective samples: {}, mean cost: {}, p {}".format(t, noise, 4, 256, MPPI_4_mean, p2) )