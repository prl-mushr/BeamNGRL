import numpy as np
import cv2
import time
import math as m
import traceback
import numba
from numba import jit

positions = np.load('map_data_binary_50/positions.npy')
color = np.load('map_data_binary_50/color.npy')
depth = np.load('map_data_binary_50/depth.npy')
segmt = np.load('map_data_binary_50/segment.npy')

## photogrammetry. Yes I'm old school.
@jit(nopython=True, parallel=True)
def photogrammetry(depth_image, elevation_map, pos, color_image, color_map, segment_image, segment_map):
    CX_DEPTH = depth_image.shape[0]/2
    CY_DEPTH = depth_image.shape[1]/2
    global resolution
    ## camera position in BEV image space(meters -> pixels):
    position = np.array([pos[0]*resolution + elevation_map.shape[0]//2, pos[1]*resolution + elevation_map.shape[1]//2], dtype=np.int32)
    width = depth_image.shape[0]
    height = depth_image.shape[1]
    for u in range(0, depth_image.shape[0], 1):  # change the "1" to 5 to produce a quick preview before you go for full resolution
        for v in range(0, depth_image.shape[1], 1):
            z = depth_image[u,v]
            y = ((v - CY_DEPTH)*z*F_inv)
            x = ((u - CX_DEPTH)*z*F_inv)
            U = -int(y*resolution) + position[1]
            V = -int(x*resolution) + position[0]
            elevation_map[U,V] = pos[2] - z
            color_map[U,V] = color_image[u,v]
            segment_map[U,V] = segment_image[u,v]

    return elevation_map, color_map, segment_map


fov = 90.0/57.3
A = 1000.0  # input images have 1000x1000 pixel resolution
tile_size = 100 # we consider a resolution of 0.1 meters or 10 pixels per meter
F_inv = m.tan(fov*0.5)/(A/2)

elevation_map_size = A*8  # this is kinda hacky. There is a way to do this more "programmatically"
resolution = A/tile_size

elevation_map = np.zeros((int(elevation_map_size + A) , int(elevation_map_size + A)))  
color_map = np.zeros((int(elevation_map_size + A) , int(elevation_map_size + A), 3), dtype=np.uint8)
segment_map = np.zeros((int(elevation_map_size + A) , int(elevation_map_size + A), 4), dtype=np.uint8)
for i in range(len(positions)):
    now = time.time()
    elevation_map, color_map, segment_map = photogrammetry(depth[i], elevation_map, positions[i], color[i], color_map, segmt[i], segment_map)
    dt = time.time() - now
    print('Frame: %d/%d, Time: %f' % (i, len(positions), dt))

## flip image before saving because of north-south flipping
np.save('map_data/elevation_map.npy', elevation_map)
cv2.imwrite('map_data/elevation_map.png', (cv2.flip(elevation_map, 0)*65535/100).astype(np.uint16))
cv2.imwrite('map_data/color_map.png', cv2.flip(color_map, 0))
cv2.imwrite('map_data/segmt_map.png', cv2.flip(segment_map, 0))
print("saved image")
