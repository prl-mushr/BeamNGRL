import time
import math as m
import cv2
import numpy as np
import traceback



def main():
    print("starting")
    elevation_map = cv2.imread('map_data/elevation_map.png')
    color_map = cv2.imread('map_data/color_map.png')
    segmt_map = cv2.imread('map_data/segmt_map.png')
    path_map  = cv2.imread('map_data/paths.png')

    image_shape = color_map.shape
    resolution = 10 # 10 pixels per meter
    map_size = 50  # 50 x 50 m grid around the car

    mask = np.zeros((2*map_size*resolution, 2*map_size*resolution), np.uint8)
    mask = cv2.circle(mask,(map_size*resolution, map_size*resolution), map_size*resolution, 255,thickness=-1)
    mask_center = (map_size*resolution, map_size*resolution)
    mask_size = (2 * map_size*resolution, 2 * map_size*resolution)

    ## this "nested" function uses variables from the intrinsic data, be careful if you move this function out
    def get_map_bf_no_rp(map_img):
        BEV = map_img[Y_min:Y_max, X_min:X_max,:]
        BEV = cv2.bitwise_and(BEV, BEV, mask=mask)
        # get rotation matrix using yaw:
        rotate_matrix = cv2.getRotationMatrix2D(center=mask_center, angle= -rpy[i,2]*57.3, scale=1)
        # rotate the image using cv2.warpAffine
        BEV = cv2.warpAffine(src=BEV, M=rotate_matrix, dsize=mask_size)
        return BEV

    def transform_world_to_bodyframe(x, y, xw, yw, th):
        x -= xw
        y -= yw
        R = np.zeros((2,2))
        ct, st = m.cos(-th), m.sin(-th)
        R[0,0], R[0,1], R[1,0], R[1,1] = ct, -st, st, ct
        X = np.array(x)
        Y = np.array(y)
        V = np.array([X,Y])
        O = np.matmul(R, V)
        x, y = O[0,:], O[1,:]
        return x, y

    for j in range(5):
        intrinsic_data = np.load('map_data/IRL_trial_intrinsic{}.npy'.format(j))
        
        pos    = intrinsic_data[:, :3]
        rpy    = intrinsic_data[:, 3:6]
        vel_bf = intrinsic_data[:, 6:9]
        vel_wf = intrinsic_data[:, 9:12]
        A      = intrinsic_data[:, 12:15]
        G      = intrinsic_data[:, 15:18]
        steer  = intrinsic_data[:, 18]
        gas    = intrinsic_data[:, 19]
        brake  = intrinsic_data[:, 20]
        U      = intrinsic_data[:, 18:21]  # control inputs

        wdwnf  = intrinsic_data[:, 21:25]
        whrzf  = intrinsic_data[:, 25:29]
        lonslp = intrinsic_data[:, 29:33]
        latslp = intrinsic_data[:, 33:37]
        whspd  = intrinsic_data[:, 37:41]
        tmstmp = intrinsic_data[:, 41]
        dt     = intrinsic_data[:, 42]

        now = time.time()
        for i in range(len(intrinsic_data) - 15):

            img_X = int(pos[i, 0]*resolution + image_shape[0]//2)
            img_Y = int(-pos[i, 1]*resolution + image_shape[1]//2)
            
            Y_min = int(img_Y - map_size*resolution)
            Y_max = int(img_Y + map_size*resolution)

            X_min = int(img_X - map_size*resolution)
            X_max = int(img_X + map_size*resolution)

            ## inputs:
            BEV_color = get_map_bf_no_rp(color_map)  # crops circle, rotates into body frame
            BEV_heght = get_map_bf_no_rp(elevation_map)
            BEV_segmt = get_map_bf_no_rp(segmt_map)

            start_state = np.hstack((np.zeros(3), vel_bf[i], A[i], G[i]))
            controls = U[i:i+15,:].flatten()
            U_X = np.hstack((start_state, controls))  # stack state and controls

            pos_x, pos_y = transform_world_to_bodyframe(pos[i:i+15,0].copy(), pos[i:i+15, 1].copy(), pos[i,0].copy(), pos[i,1].copy(), rpy[i,2])
            pos_z = pos[i:i+15,2] - pos[i,2]
            pos_state = np.vstack((pos_x, pos_y, pos_z)).T
            state = np.hstack((pos_state, vel_bf[i:i+15,:], A[i:i+15,:], G[i:i+15,:]))
            # visualization:
            BEV = cv2.resize(BEV_color, (500,500), interpolation= cv2.INTER_AREA)
            cv2.imshow('color', BEV)
            BEV = cv2.resize(BEV_heght, (500,500), interpolation= cv2.INTER_AREA)
            cv2.imshow('height', BEV)
            BEV = cv2.resize(BEV_segmt, (500,500), interpolation= cv2.INTER_AREA)
            cv2.imshow('segment', BEV)
            cv2.waitKey(1)
            time.sleep(0.03)
    print(time.time() - now)

if __name__ == '__main__':
    main()

