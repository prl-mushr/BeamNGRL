import numpy as np
from BeamNGRL.BeamNG.beamng_interface import get_beamng_default
from BeamNGRL.utils.visualisation import costmap_vis
import traceback
import yaml
import os
from pathlib import Path
from BeamNGRL.utils.keygrabber import KeyGrabber
import argparse

'''
drive to the start point, press w to indicate start pose (we start recording positions), press s to record end pose. 
All poses in between will be recorded and saved to a file called "experiment.npy"
'''

def main(config_path=None, hal_config_path=None, args=None):
    if config_path is None:
        print("no config file provided!")
        exit()
    if hal_config_path is None:
        print("no hal config file provided!")
        exit()
    with open(config_path) as f:
        Config = yaml.safe_load(f)
    with open(hal_config_path) as f:
        hal_Config = yaml.safe_load(f)

    Dynamics_config = Config["Dynamics_config"]
    Cost_config = Config["Cost_config"]
    Sampling_config = Config["Sampling_config"]
    MPPI_config = Config["MPPI_config"]
    Map_config = Config["Map_config"]
    vehicle = Config["vehicle"]
    map_name = Config["map_name"]
    WP_file = str(Path(os.getcwd()).parent.absolute()) + "/Experiments/Waypoints/" + Config["scenarios"][0] + ".npy"
    target_WP = np.load(WP_file)
    start_pos = target_WP[0,:3]
    start_quat = target_WP[0,3:]
    map_res = Map_config["map_res"]
    map_size = Map_config["map_size"]

    bng_interface = get_beamng_default(
        car_model=vehicle["model"],
        start_pos=start_pos,
        start_quat=start_quat,
        car_make=vehicle["make"],
        map_config=Map_config,
        host_IP=args.host_IP,
        remote=args.remote,
        camera_config=hal_Config["camera"],
        lidar_config=hal_Config["lidar"],
        accel_config=hal_Config["mavros"],
        burn_time=0.04,
        run_lockstep=False,
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
                        filepath = str(Path(os.getcwd()).parent.absolute()) + "/Experiments/Waypoints/"
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name", type=str, default="Test_Config.yaml", help="name of the config file to use")
    parser.add_argument("--hal_config_name", type=str, default="offroad.yaml", help="name of the config file to use")
    parser.add_argument("--remote", type=bool, default=True, help="whether to connect to a remote beamng server")
    parser.add_argument("--host_IP", type=str, default="169.254.216.9", help="host ip address if using remote beamng")

    args = parser.parse_args()
    config_name = args.config_name
    config_path = str(Path(os.getcwd()).parent.absolute()) + "/Experiments/Configs/" + config_name

    hal_config_name = args.hal_config_name
    hal_config_path = str(Path(os.getcwd()).parent.absolute()) + "/Configs/" + hal_config_name
    main(config_path = config_path, hal_config_path = hal_config_path, args = args) ## we run for 3 iterations because science