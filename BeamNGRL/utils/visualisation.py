import numpy as np
import cv2
import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf
from scipy.spatial import Delaunay


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
            X = np.clip( np.array((x * resolution_inv) + map_size, dtype=np.int32), 0, map_size*2 - 1)
            Y = np.clip( np.array((y * resolution_inv) + map_size, dtype=np.int32), 0, map_size*2 - 1)
            costmap[Y, X] = 0
        else:
            print_states = states
            x = print_states[:, :, :, 0].flatten()
            y = print_states[:, :, :, 1].flatten()
            X = np.clip( np.array((x * resolution_inv) + map_size, dtype=np.int32), 0, map_size*2 - 1)
            Y = np.clip( np.array((y * resolution_inv) + map_size, dtype=np.int32), 0, map_size*2 - 1)
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

class Vis:
    def __init__(self):
    # x y z ? (m)
        self.car = g.Box((1.0, 0.5, 0.3))
        # radius (m)
        self.goal = g.Sphere(0.2)

        self.vis = meshcat.Visualizer().open()

        self.vis['car'].set_object(self.car)
        self.vis['goal'].set_object(self.goal, g.MeshBasicMaterial(color=0xaaffaa))

    def setcar(self, pos, rpy):
        self.vis['car'].set_transform(
            tf.compose_matrix(
                angles=(rpy[0], rpy[1], rpy[2]), 
                translate=(pos[0], pos[1], pos[2]))
            )

    def setgoal(self, pos):
        self.vis['goal'].set_transform(tf.translation_matrix((pos[0], pos[1], 0.0)))

    def set_terrain(self, height_map, map_res, map_size):
        # Compute the x and y coordinates of the vertices
        x_coords, y_coords = np.meshgrid(
                                        np.arange(0, height_map.shape[1]*map_res, map_res) - map_size,
                                        np.arange(0, height_map.shape[0]*map_res, map_res) - map_size
                                        )
        vertices = np.column_stack((
            x_coords.flatten(),
            y_coords.flatten(),
            height_map.flatten()
        ))
        points = np.column_stack((
            x_coords.flatten(),
            y_coords.flatten()
        ))
        tri = Delaunay(points)
        self.vis["terrain"].set_object(meshcat.geometry.TriangularMeshGeometry(
            vertices=vertices,
            faces=tri.simplices,
        ))