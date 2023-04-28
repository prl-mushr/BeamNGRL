import numpy as np
import cv2
import matplotlib.pyplot as plt

fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
ax.set_box_aspect([1,1,0.5]) # set the aspect ratio of the plot
ax.view_init(elev=60, azim=-45) # set the viewpoint of the plot
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Elevation map')

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

def elevation_map_vis(elev_map):
    global ax
    x, y = np.meshgrid(range(elev_map.shape[1]), range(elev_map.shape[0]))
    ax.clear()
    ax.plot_surface(x, y, elev_map, cmap='viridis', alpha=0.8, linewidth=0)
    # Update the plot
    plt.show(block=False)
    plt.pause(0.001)
