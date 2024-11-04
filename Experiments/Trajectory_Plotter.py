# this python file loads a trajectory (.npy file) and plots the 3d coordinates from it
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import os
import sys

waypoints = np.load('Waypoints/ditch-0.npy')

def Plot_trajectory(waypoints):
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # ax.plot(waypoints[:idx,0], waypoints[:idx,1], waypoints[:idx,2], label='Trajectory')
    diff = waypoints - waypoints[0]
    waypoints = waypoints[0] + 1.2*diff
    np.save('Waypoints/ditch-4.npy', waypoints)
    exit()
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('z')
    plt.show()

Plot_trajectory(waypoints)