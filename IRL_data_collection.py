import numpy as np
import cv2
from beamng_interface import *
import traceback

def main(map_name, start_point, start_quat, BeamNG_dir='/home/stark/BeamNG/BeamNG'):
    bng_interface = beamng_interface(homedir= BeamNG_dir, userfolder=BeamNG_dir+'/userfolder')
    bng_interface.load_scenario(scenario_name=map_name, car_make='RG_RC', car_model='Short_Course_Truck',
                                start_pos=start_point, start_rot=start_quat)
    ## set the BEV map attributes:
    bng_interface.set_map_attributes(map_size = 16, resolution=0.25) # 16x16 meter grid around the car with 0.25 m resolution

    # set lock-step to true if you want the simulator to pause while you calculate the controls:
    # this will make the overall simulation slower since it takes some time to communicate the pause/resume command + whatever time you take to compute controls
    # bng_interface.set_lockstep(True)

    start = time.time()
    intrinsic_data = []
    color_data = []
    elevt_data = []
    segmt_data = []
    path_data = []
    folder_name = "IRL_data"
    episode_time = 10 ## 100 second episode: change this to you liking

    while True:
        try:
            bng_interface.state_poll()
            # state is np.hstack((pos, rpy, vel, A, G, st, th/br)) ## note that velocity is in the body-frame
            # state information follows ROS REP103 standards (so basically ROS standards): world refernce frame for (x,y,z) is east-north-up(ENU). Body frame ref is front-left-up(FLU)
            state =  bng_interface.state
            ## get robot_centric BEV (not rotated into robot frame)
            BEV_color = bng_interface.BEV_color
            BEV_heght = (bng_interface.BEV_heght + 2.0)/4.0  # note that BEV_heght (elevation) has a range of +/- 2 meters around the center of the elevation.
            BEV_segmt = bng_interface.BEV_segmt
            BEV_path  = bng_interface.BEV_path  # trail/roads


            intrinsic_data.append(state)
            color_data.append(BEV_color)
            elevt_data.append(BEV_heght)
            segmt_data.append(BEV_segmt)
            path_data.append(BEV_path)

            if(time.time() - start > episode_time):
                print("saving data")
                intrinsic_data = np.array(intrinsic_data)
                color_data = np.array(color_data)
                elevt_data = np.array(elevt_data)
                segmt_data = np.array(segmt_data)
                path_data  = np.array(path_data)
                np.save(folder_name + "/IRL_trial_intrinsic.npy", intrinsic_data)
                np.save(folder_name + "/IRL_trial_color.npy", color_data)
                np.save(folder_name + "/IRL_trial_elevt.npy", elevt_data)
                np.save(folder_name + "/IRL_trial_segmt.npy", segmt_data)
                np.save(folder_name + "/IRL_trial_path.npy", path_data)
                break

        except Exception:
            print(traceback.format_exc())
    bng_interface.bng.close()


if __name__ == '__main__':
    # position of the vehicle for tripped_flat on grimap_v2
    start_point = np.array([-67, 336, 34.5])
    start_quat = np.array([0, 0, 0.3826834, 0.9238795])
    map_name = "small_island"
    main(map_name,start_point, start_quat)