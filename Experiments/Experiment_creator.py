import numpy as np
from BeamNGRL.BeamNG.beamng_interface import get_beamng_default
from BeamNGRL.utils.visualisation import costmap_vis
import traceback
import yaml
import os
from pathlib import Path
import time
import cv2
from BeamNGRL.utils.keygrabber import KeyGrabber

'''
drive to the start point, press w to indicate start pose (we start recording positions), press s to record end pose. 
All poses in between will be recorded and saved to a file called "experiment.npy"
'''

def main(map_name, start_pos, start_quat, config_path, BeamNG_dir="/home/stark/", target_WP=None):
    with open(config_path + 'Map_config.yaml') as f:
        Map_config = yaml.safe_load(f)

    map_res = Map_config["map_res"]
    
    bng_interface = get_beamng_default(
        car_model='offroad',
        start_pos=start_pos,
        start_quat=start_quat,
        map_name=map_name,
        car_make='sunburst',
        map_res=Map_config["map_res"],
        map_size=Map_config["map_size"],
        elevation_range=4.0
    )
    kg = KeyGrabber()
    recording = False
    wp_list = []

    while True:
        try:
            bng_interface.state_poll()
            state = np.copy(bng_interface.state)
            quat = bng_interface.vehicle.state['rotation']
            pos = np.copy(state[:3])  # example of how to get car position in world frame. All data points except for dt are 3 dimensional.
            wp = np.hstack((pos, quat))
            
            for c in kg.read():
                if c in 'wW':
                    if(not recording):
                        print("started recording!")
                        wp_list = [] # clear the list.
                    recording = True
                elif c in 'sS':
                    if(recording):
                        print("stopped recording!")
                        wp_list_np = np.array(wp_list)
                        x = int(wp_list_np[0,0])
                        y = int(wp_list_np[0,1])
                        filepath = str(Path(os.getcwd()).parent.absolute()) + "/Experiments/Waypoints"
                        filepath += "waypoints-{}-{}-{}.npy".format(map_name, str(x), str(y))
                        np.save(filepath, wp_list_np)
                    recording = False

                else:
                    pass
            
            if(recording):
                wp_list.append(wp)


        except Exception:
            print(traceback.format_exc())

    # bng_interface.bng.close()


if __name__ == "__main__":
    # position of the vehicle for tripped_flat on grimap_v2
    start_point = np.array([-67, 336, 34.5])
    start_quat = np.array([0, 0, 0.3826834, 0.9238795])
    map_name = "small_island"
    config_path = str(Path(os.getcwd()).parent.absolute()) + "/BeamNGRL/control/UW_mppi/Configs/"
    main(map_name, start_point, start_quat, config_path)