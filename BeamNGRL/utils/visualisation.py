import numpy as np
import cv2

def costmap_vis(states, pos, goal, costmap, resolution_inv):
    goal -= pos
    map_size = costmap.shape[0] // 2
    goal_X = np.clip( int((goal[0] * resolution_inv) + map_size), 0, map_size*2 - 1)
    goal_Y = np.clip( int((goal[1] * resolution_inv) + map_size), 0, map_size*2 - 1)
    cv2.line(costmap, (map_size, map_size), (goal_X, goal_Y), (0, 1, 0), 1)
    cv2.circle(costmap, (goal_X, goal_Y), int(resolution_inv * 0.2), (1, 0, 0), -1)
    if states is not None:
        if(len(costmap.shape)<3):
            print_states = states
            x = print_states[:, :, :, 0].flatten()
            y = print_states[:, :, :, 1].flatten()
            X = np.clip( np.array((x * resolution_inv) + map_size, dtype=np.int32), 0, map_size*2)
            Y = np.clip( np.array((y * resolution_inv) + map_size, dtype=np.int32), 0, map_size*2)
            costmap[Y, X] = 0
        else:
            print_states = states
            x = print_states[:, :, :, 0].flatten()
            y = print_states[:, :, :, 1].flatten()
            X = np.clip( np.array((x * resolution_inv) + map_size, dtype=np.int32), 0, map_size*2)
            Y = np.clip( np.array((y * resolution_inv) + map_size, dtype=np.int32), 0, map_size*2)
            costmap[Y, X] = np.array([0, 0, 0])
    costmap = cv2.resize(costmap, (500, 500), interpolation=cv2.INTER_AREA)
    costmap = cv2.flip(costmap, 0)  # this is just for visualization
    cv2.imshow("map", costmap)
    cv2.waitKey(1)