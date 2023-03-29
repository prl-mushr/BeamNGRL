import time
from time import sleep
import math as m
import cv2
import numpy as np
import traceback
import json
import os

def main():
    # color_map = cv2.imread('map_data/elevation_map.png')
    color_map = cv2.imread('map_data/segmt_map.png')
    color_map = cv2.flip(color_map, 0)
    image_shape = color_map.shape
    resolution = 10 # 5 pixels per meter
    map_size = 20  # 20 x 20 m grid around the car
    color_map_orig = np.copy(color_map)
    trail_data = json.load(open('meta_data/paths.json', encoding='utf-8'))
    path_map = np.ones_like(color_map)*255
    for line in trail_data:
        pos = np.array(line["nodes"])[:,:2]
        pos[:,1] *= -1
        img_pos = np.array(pos*resolution + (image_shape[0]//2), dtype= np.int32).reshape((-1, 1, 2))
        path_map = cv2.polylines(path_map, [img_pos], False, (0,0,0), resolution*4)
    cv2.blur(path_map, (5, 5))

    cv2.imwrite('map_data/paths.png', path_map)

    ## this is useful only as a template for "how to" plot meta-data
    # tree_map = np.ones_like(color_map)*0
    # for root, dirs, files in os.walk(r'meta_data/'):
    #     for file in files:
    #         if file.endswith('.json') and file != 'paths.json':
    #             print(file)
    #             tree_data = json.load(open('meta_data/'+file, encoding='utf-8'))
    #             for line in tree_data:
    #                 pos = np.array(line["pos"])[:2]
    #                 pos[1] *= -1
    #                 img_pos = np.array(pos*resolution + (image_shape[0]//2), dtype= np.int32)
    #                 scale = line["scale"]
    #                 cv2.circle(tree_map, (img_pos[0], img_pos[1]), int(3*scale), (255,255,255), -1)
    # # cv2.imwrite()
    color_map = (path_map*0.5 + 0.5*color_map)/255
    cv2.imshow('window', cv2.resize(color_map, (2000,2000), cv2.INTER_AREA))
    cv2.imshow('window2', cv2.resize(color_map_orig, (2000,2000), cv2.INTER_AREA))
    cv2.waitKey(0)

if __name__ == '__main__':
    main()

