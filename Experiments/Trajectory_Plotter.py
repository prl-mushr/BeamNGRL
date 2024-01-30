# this python file loads a trajectory (.npy file) and plots the 3d coordinates from it
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import os
import sys

waypoints = np.load('Waypoints/race-1.npy')

def Plot_trajectory(waypoints):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    idx = np.min(np.where(waypoints[:,2] > 65)[0])
    ax.plot(waypoints[:idx,0], waypoints[:idx,1], waypoints[:idx,2], label='Trajectory')
    np.save('Waypoints/race-2.npy', waypoints[:idx,:])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()

Plot_trajectory(waypoints)