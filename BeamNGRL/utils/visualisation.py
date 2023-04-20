import cv2
import numpy as np


def costmap_vis(states, pos, goal, costmap, resolution_inv):
    goal -= pos
    map_size = costmap.shape[0]//2
    goal_X = int((goal[0]*resolution_inv) + map_size)
    goal_Y = int((goal[1]*resolution_inv) + map_size)
    cv2.line(costmap, (map_size, map_size), (goal_X, goal_Y), (0,1,0), 1)
    cv2.circle(costmap, (goal_X, goal_Y), int(resolution_inv*0.2), (1,0,0), -1)
    if(states is not None):
        print_states = states
        x = print_states[...,0].flatten()
        y = print_states[...,1].flatten()
        X = np.array((x * resolution_inv) + map_size, dtype=np.int32)
        Y = np.array((y * resolution_inv) + map_size, dtype=np.int32)
        costmap[Y,X] = np.array([0,0,1])
    costmap = cv2.resize(costmap, (500,500), interpolation= cv2.INTER_AREA)
    costmap = cv2.flip(costmap, 0)  # this is just for visualization
    cv2.imshow("map", costmap)
    cv2.waitKey(1)
