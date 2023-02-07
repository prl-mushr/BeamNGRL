#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from point_warping import *
import math as m
import time
from Bezier import *
import argparse

plt.ion()

cam_height = 0.845 # found from 3d model of car (urdf)
X_max = 320 #image pixels in horizontal direction
Y_max = 240 #image pixels in vertical direction
X_Center = X_max//2 #assuming image center is at geometric center (camera matrix should be used, but this will do for now)
Y_Center = Y_max//2
pitch = 0
focalLength = 50 #hd1080
fov_h = 90/57.3 #fov 
K_h = m.tan(fov_h/2)
fov_v = 58/57.3
K_v = m.tan(fov_v/2)
K = np.array([[focalLength,   0.        , X_Center],
       [  0.        , focalLength,      Y_Center],
       [  0.        ,   0.        ,   1.        ]])

x_scale = 30
y_scale = 30

def rotate(x,y,theta):
    X_new = m.cos(theta)*x - m.sin(theta)*y
    Y_new = m.sin(theta)*x + m.cos(theta)*y
    return X_new, Y_new

def xy_to_img(x,y,speed):
    # R = int(255*(max(0,-speed)))
    # G = int(255*(max(0, speed)))
    x1 = x+0.1
    x2 = x-0.1
    x3 = x+0.3
    x4 = x-0.3
    _y = y
    x,y = XY2img(x,_y,cam_height,K_v,K_h,X_max,Y_max,X_Center,Y_Center,pitch,480)
    x1,y1 = XY2img(x1,_y,cam_height,K_v,K_h,X_max,Y_max,X_Center,Y_Center,pitch,480)
    x2,y2 = XY2img(x2,_y,cam_height,K_v,K_h,X_max,Y_max,X_Center,Y_Center,pitch,480)
    x3,y3 = XY2img(x3,_y,cam_height,K_v,K_h,X_max,Y_max,X_Center,Y_Center,pitch,480)
    x4,y4 = XY2img(x4,_y,cam_height,K_v,K_h,X_max,Y_max,X_Center,Y_Center,pitch,480)
    # x,y = x*x_scale + X_Center, Y_max - (y-1.8)*y_scale
    image = np.zeros((240,320),dtype=np.uint8)
    # for i in range(5):
        # pts = np.column_stack((np.int32(x-30+15*i),np.int32(y)))
    pts = np.column_stack((np.int32(x),np.int32(y)))
    cv2.polylines(image, [pts], False, 255, 20)
    pts = np.column_stack((np.int32(x1),np.int32(y1)))
    cv2.polylines(image, [pts], False, 255, 20)
    pts = np.column_stack((np.int32(x2),np.int32(y2)))
    cv2.polylines(image, [pts], False, 255, 20)
    pts = np.column_stack((np.int32(x3),np.int32(y3)))
    cv2.polylines(image, [pts], False, 255, 20)
    pts = np.column_stack((np.int32(x4),np.int32(y4)))
    cv2.polylines(image, [pts], False, 255, 20)
    return image,x,y

def get_lr(s):
    s /= 30
    l = 0.5*(1-s)
    r = 0.5*(1+s)
    return l,r

def create_image_data(folder,index):
    filename_intrinsic = folder + 'training_data_intrinsic_' + str(index) + '.npy'
    filename_depth = folder + 'training_data_depth_' + str(index) + '.npy'
    filename_color = folder + 'training_data_color_' + str(index) + '.npy'
    filename_segmt = folder + 'training_data_segmt_' + str(index) + '.npy'

    data_intrinsic = np.load(filename_intrinsic)
    data_depth = np.load(filename_depth)
    data_color = np.load(filename_color)
    data_segmt = np.load(filename_segmt)

    
    N = 15
    offset_y = 2
    cam_pos = 1.8
    side_cam_tilt = 15/57.3
    cam_sep = 0.1
    front_off = offset_y - cam_pos
    r = front_off + (cam_sep/m.tan(side_cam_tilt))
    output_c = []
    output_l = []
    output_r = []
    points = data[:,3]
    inputs = data[:-N,0:3]
    print(data.shape)
    # cv2.namedWindow('visualize',cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('visualize',800,600)
    for j in range(0,len(points)-N):
        x = np.zeros(N)
        y = np.zeros(N)
        for i in range(N):
            x[i] = points[j+i][0]
            y[i] = points[j+i][1]
        x -= x[0]
        y -= y[0]
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        distance = distancecalcy(x[0],x[-1],y[0],y[-1])
        steer = 7.5*points[j][4]/57.3
        speed = points[j][3]
        head = points[j][2] - steer - m.pi/2
        xc,yc = rotate(x,y,-head)
        yc += offset_y
        xc *= 0.75
        xl,yl = rotate(x,y,-head - side_cam_tilt)
        x0,y0 = rotate(0,r,-side_cam_tilt)
        yl += offset_y + (y0 - front_off)
        xl += x0
        xl *= 0.75
        xr,yr = rotate(x,y,-head + side_cam_tilt)
        x0,y0 = rotate(0,r,side_cam_tilt)
        yr += offset_y + (y0 - front_off)
        xr += x0
        xr *= 0.75
        out_img_c,xc,yc = xy_to_img(xc,yc,speed/10)
        out_img_l,xl,yl = xy_to_img(xl,yl,speed/10)
        out_img_r,xr,yr = xy_to_img(xr,yr,speed/10)

        if(distance>6):
            output_c.append([inputs[j,0], out_img_c])
            output_l.append([inputs[j,1], out_img_l])
            output_r.append([inputs[j,2], out_img_r])
    # print(output_c.shape)
    output_c = np.array(output_c)
    output_l = np.array(output_l)
    output_r = np.array(output_r)
    final_out = np.concatenate((output_c,output_l,output_r),axis=0)
    np.random.shuffle(final_out)
    print(final_out.shape)
    np.save('MUSHR_320x240_shuffled_Image_{}.npy'.format(args.dataset_name),final_out)
    del data
    del inputs
    del output_c
    del output_l
    del output_r

def create_steering_data(args):
    N = 15
    data = np.load('MUSHR_320x240_{}.npy'.format(args.dataset_name),allow_pickle=True)
    output_c = []
    output_l = []
    output_r = []
    points = data[:,3]
    inputs = data[:-N,0:3]
    print(data.shape)

    for j in range(0,len(points)-N):
        s_c = points[j][4]*15
        l_c,r_c = get_lr(s_c)
        s_l = s_c + 15
        l_l,r_l = get_lr(s_l)
        s_r = s_c - 15
        l_r,r_r = get_lr(s_r)
        output_c.append([inputs[j,0], np.array([l_c,r_c]) ])
        output_l.append([inputs[j,1], np.array([l_l,r_l]) ])
        output_r.append([inputs[j,2], np.array([l_r,r_r]) ])
    # print(output_c.shape)
    output_c = np.array(output_c)
    output_l = np.array(output_l)
    output_r = np.array(output_r)
    final_out = np.concatenate((output_c,output_l,output_r),axis=0)
    np.random.shuffle(final_out)
    print(final_out.shape)
    np.save('MUSHR_320x240_shuffled_Steering_{}.npy'.format(args.dataset_name),final_out)
    del data
    del inputs
    del output_c
    del output_l
    del output_r

def create_bezier_data(args):
    data = np.load('MUSHR_320x240_{}.npy'.format(args.dataset_name),allow_pickle=True)

    N = 15
    offset_y = 3
    cam_pos = 1.8
    side_cam_tilt = 15/57.3
    cam_sep = 0.1
    front_off = offset_y - cam_pos
    r = front_off + (cam_sep/m.tan(side_cam_tilt))
    output_c = []
    output_l = []
    output_r = []
    points = data[:,3]
    inputs = data[:-N,0:3]
    print(data.shape)

    for j in range(0,len(points)-N):
        x = np.zeros(N)
        y = np.zeros(N)
        for i in range(N):
            x[i] = points[j+i][0]
            y[i] = points[j+i][1]
        x -= x[0]
        y -= y[0]
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        distance = distancecalcy(x[0],x[-1],y[0],y[-1])

        steer = 8*points[j][4]/57.3
        speed = points[j][3]
        head = points[j][2] - steer - m.pi/2
        xc,yc = rotate(x,y,-head)

        xl,yl = rotate(x,y,-head - side_cam_tilt)

        xr,yr = rotate(x,y,-head + side_cam_tilt)

        B_c,Cx,Cy = find_bezier(xc,yc)
        B_l,Lx,Ly = find_bezier(xl,yl)
        B_r,Rx,Ry = find_bezier(xr,yr) # may use it in the future. maybe

        s_c = points[j][4]*15
        s_l = s_c + 15
        s_r = s_c - 15

        if(distance>6.0):
            output_c.append([inputs[j,0], B_c/distance,s_c])
            output_l.append([inputs[j,1], B_l/distance,s_l])
            output_r.append([inputs[j,2], B_r/distance,s_r]) 
    # print(output_c.shape)
    output_c = np.array(output_c)
    output_l = np.array(output_l)
    output_r = np.array(output_r)
    final_out = np.concatenate((output_c,output_l,output_r),axis=0)
    np.random.shuffle(final_out)
    print(final_out.shape)
    np.save('MUSHR_320x240_shuffled_Bezier_{}.npy'.format(args.dataset_name),final_out)
    del data
    del inputs
    del output_c
    del output_l
    del output_r

if __name__ == '__main__':
    model_list = [
    "steering",
    "bezier",
    "image-image"
    ]
    parser = argparse.ArgumentParser(description='ddqn')
    parser.add_argument('--dataset_name', type=str, default="4", help='suffix for the dataset name')
    parser.add_argument('--model', type=str, default="steering", help='type of model', choices=model_list)
    args = parser.parse_args()

    if(args.model=="steering"):
        create_steering_data(args)
    elif(args.model=="bezier"):
        create_bezier_data(args)
    elif(args.model=="image-image"):
        create_image_data(args)
    else:
        print("no matching dataset type")
