import numpy as np
import cv2
from beamng_interface import *
import traceback

def main(map_name, start_pos, start_quat):
    map_res = 0.25
    map_size = 16 # 16 x 16 map

    bng_interface = get_beamng_default(
        car_model='RACER',
        start_pos=start_pos,
        start_quat=start_quat,
        map_name=map_name,
        car_make='sunburst',
        beamng_path=BNG_HOME,
        map_res=map_res,
        map_size=map_size
        )
    # set lock-step to true if you want the simulator to pause while you calculate the controls:
    # this will make the overall simulation slower since it takes some time to communicate the pause/resume command + whatever time you take to compute controls

    # bng_interface.set_lockstep(True)

    while True:
        try:
            bng_interface.state_poll()
            # state is np.hstack((pos, rpy, vel, A, G, st, th/br)) ## note that velocity is in the body-frame
            # state information follows ROS REP103 standards (so basically ROS standards): world refernce frame for (x,y,z) is east-north-up(ENU). Body frame ref is front-left-up(FLU)
            state =  bng_interface.state
            pos = state[:3]  # example of how to get car position in world frame. All data points except for dt are 3 dimensional.
            ## if you just want position, you can also do pos = bng_interface.pos

            ## camera and depth currently unavailable on Ubuntu!
            # color, depth, segmt = bng_interface.color, bng_interface.depth, bng_interface.segmt
            # lidar_pts = bng_interface.lidar_pts

            ## get robot_centric BEV (not rotated into robot frame)
            BEV_color = bng_interface.BEV_color
            BEV_heght = (bng_interface.BEV_heght + 2.0)/4.0  # note that BEV_heght (elevation) has a range of +/- 2 meters around the center of the elevation.
            BEV_segmt = bng_interface.BEV_segmt
            BEV_path  = bng_interface.BEV_path  # trail/roads
            BEV_normal = bng_interface.BEV_normal
            ## displaying BEV for visualization:
            BEV = cv2.resize(BEV_color, (500,500), interpolation= cv2.INTER_AREA)
            cv2.imshow('color', BEV)
            BEV = cv2.resize(BEV_heght, (500,500), interpolation= cv2.INTER_AREA)
            cv2.imshow('height', BEV)
            BEV = cv2.resize(BEV_normal[:,:,1], (500,500), interpolation= cv2.INTER_AREA)
            cv2.imshow('segment', BEV)
            cv2.waitKey(1)
            ## you can now "encapsulate the BEV and state into whatever form of "observation" you want.
            ## example of how to send controls:
            ## steering is 0th index, throttle/brake/reverse is 1st index. steering is +ve when turning left (following ROS REP103 convention)
            action = np.ones(2, dtype=np.float64)  # has to be numpy array. The inputs are always between (-1.0, 1.0) (for both throttle and steering)
            # bng_interface.send_ctrl(action)

            ## check if the car has flipped over. This can be replaced by whatever your reset condition is
            if(bng_interface.flipped_over):
                bng_interface.reset()

        except Exception:
            print(traceback.format_exc())
    bng_interface.bng.close()



if __name__ == '__main__':
    # position of the vehicle for tripped_flat on grimap_v2
    start_point = np.array([-67, 336, 34.5])
    start_quat = np.array([0, 0, 0.3826834, 0.9238795])
    map_name = "small_island"
    main(map_name,start_point, start_quat)
