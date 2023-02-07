#!/usr/bin/env python
import numpy as np
import cv2
import math as m
def eulerAnglesToRotationMatrix(theta) :
	R_x = np.array([[1,         0,                  0                   ],
					[0,         m.cos(theta[0]), -m.sin(theta[0]) ],
					[0,         m.sin(theta[0]), m.cos(theta[0])  ]
					])



	R_y = np.array([[m.cos(theta[1]),    0,      m.sin(theta[1])  ],
					[0,                     1,      0                   ],
					[-m.sin(theta[1]),   0,      m.cos(theta[1])  ]
					])

	R_z = np.array([[m.cos(theta[2]),    -m.sin(theta[2]),    0],
					[m.sin(theta[2]),    m.cos(theta[2]),     0],
					[0,                     0,                      1]
					])
	                 
	                 
	R = np.dot(R_z, np.dot( R_y, R_x ))

	return R



def rotateImage(inimg,pitch,roll,yaw,fov_h,fov_v):
	R = eulerAnglesToRotationMatrix(np.array([0,yaw,roll]))
	fov_h = fov_h/57.3
	K_h = m.tan(fov_h/2)
	fov_v = fov_v/57.3
	K_v = m.tan(fov_v/2)
	Ymax = inimg.shape[0]//2
	# print(fov_v)
	delta_y = Ymax*m.tan(pitch)/K_v

	T = np.array([[1, 0, 0], [0, 1, delta_y],[0,0,1]])
	w = inimg.shape[1]
	h = inimg.shape[0]
	inimg = cv2.warpPerspective(inimg, R, (w,h))
	return cv2.warpPerspective(inimg, T, (w,h))

def rotate_points(X,Y,theta):
	X_new = X*m.cos(theta) - Y*m.sin(theta)
	Y_new = X*m.sin(theta) + Y*m.cos(theta)
	return X_new,Y_new


def img2XY(X_pix, Y_pix,height,K_v,K_h,Xmax,Ymax,X_Center,Y_Center_default,pitch):
	Y_Center = Y_Center_default - Ymax*m.tan(pitch)/K_v
	Y = height*Ymax/(K_v*(Y_pix - Y_Center))
	X = Y*K_h*(X_pix - X_Center)/Xmax
	return X,Y

def img2XY_3Lanes(X_pix_L, X_pix_C, X_pix_R, Y_pix, height, K_v, K_h, Xmax, Ymax,X_Center,Y_Center_default,pitch):
	Y_Center = Y_Center_default - Ymax*m.tan(pitch)/K_v
	Y = height*Ymax/(K_v*(Y_pix - Y_Center))
	X_L = Y*K_h*(X_pix_L - X_Center)/Xmax
	X_C = Y*K_h*(X_pix_C - X_Center)/Xmax
	X_R = Y*K_h*(X_pix_R - X_Center)/Xmax
	return X_L,X_C,X_R,Y


def warped_img2XY_3Lanes(X_pix_L, X_pix_C, X_pix_R, Y_pix, height, K_v, K_h, Xmax, Ymax,X_Center,Y_Center_default,pitch,src_X,src_Y,dst_X,dst_Y,Y_crop):
	Y_pix *= float(src_Y)/float(dst_Y)
	Y_pix += Y_crop # the lower Y cutoff
	X_pix_L *= src_X/dst_X #convert from 600->2048 scale
	X_pix_C *= src_X/dst_X
	X_pix_R *= src_X/dst_X

	Y_Center = Y_Center_default - Ymax*m.tan(pitch)/K_v
	Y = height*Ymax/(K_v*(Y_pix - Y_Center))
	X_L = Y[-1]*K_h*(X_pix_L - X_Center)/Xmax #replace the flipud with flip for later versions of numpy uwu
	X_C = Y[-1]*K_h*(X_pix_C - X_Center)/Xmax
	X_R = Y[-1]*K_h*(X_pix_R - X_Center)/Xmax
	return X_L,X_C,X_R,Y

#X: image space X, Y: image space Y, height: height of camera
#K_v, K_h: constants related to the camera's fov, X_Center/Y_Center: point where horizon meets lateral center of image,
# pitch: downwards pitch of the camera in radians
# cutoff: maximum number of pixels in vertical direction
def XY2img(X,Y,height,K_v,K_h,Xmax,Ymax,X_Center,Y_Center_default,pitch,cutoff):
	Y_Center = Y_Center_default - Ymax*m.tan(pitch)/K_v
	Y_pix = (Y_Center + height*Ymax/(Y*K_v))
	X_pix = (X_Center + X*Xmax/(Y*K_h))
	if(cutoff<Ymax):
		index = np.where((Y_pix>0)&(Y_pix<cutoff))
	else: 
		index = np.where((Y_pix>0)&(Y_pix<Ymax))
	Y_pix = Y_pix[index]
	X_pix = X_pix[index]
	return X_pix,Y_pix

def XY2img_3Lanes(X_L,X_C,X_R,Y,height,K_v,K_h,Xmax,Ymax,X_Center,Y_Center_default,pitch,cutoff):
	Y_Center = Y_Center_default - Ymax*m.tan(pitch)/K_v
	Y_pix = (Y_Center + height*Ymax/(Y*K_v))
	X_pix_L = (X_Center + X_L*Xmax/(Y*K_h))
	X_pix_C = (X_Center + X_C*Xmax/(Y*K_h))
	X_pix_R = (X_Center + X_R*Xmax/(Y*K_h))
	if(cutoff<Ymax):
		index = np.where((Y_pix>0)&(Y_pix<cutoff))
	else: 
		index = np.where((Y_pix>0)&(Y_pix<Ymax))
	Y_pix = Y_pix[index]
	X_pix_L = X_pix_L[index]
	X_pix_C = X_pix_C[index]
	X_pix_R = X_pix_R[index]
	return X_pix_L,X_pix_C,X_pix_R,Y_pix


































