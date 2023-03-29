import numpy as np

# read IRL data from folder
folder_name = "IRL_data"
intrinsic_data = np.load(folder_name + "/IRL_trial_intrinsic0.npy")
color_data = np.load(folder_name + "/IRL_trial_color0.npy")
depth_data = np.load(folder_name + "/IRL_trial_depth0.npy")
segmt_data = np.load(folder_name + "/IRL_trial_segmt0.npy")

# print data shape
print("intrinsic_data.shape: " + str(intrinsic_data.shape))
# intrinsic data is np.hstack((pos(3), rpy(3), vel_bf(3), vel_wf(3), A(3), G(3), steering(1), throttle(1), brake(1), wheeldownforce(4), wheelhorizontalforce(4), wheelslip(4), wheelsideslip(4), wheelspeed(4), timestamp(1), delta_t(1)))
print("color_data.shape: " + str(color_data.shape))
print("depth_data.shape: " + str(depth_data.shape))
print("segmt_data.shape: " + str(segmt_data.shape))

