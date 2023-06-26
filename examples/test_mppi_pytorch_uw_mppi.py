import numpy as np
from BeamNGRL.BeamNG.beamng_interface import get_beamng_default
import traceback
import torch
from BeamNGRL.control.UW_mppi.MPPI import MPPI
from BeamNGRL.control.UW_mppi.Dynamics.SimpleCarDynamicsCUDA import SimpleCarDynamics
from BeamNGRL.control.UW_mppi.Costs.SimpleCarCost import SimpleCarCost
from BeamNGRL.control.UW_mppi.Sampling.Delta_Sampling import Delta_Sampling
from BeamNGRL.utils.visualisation import costmap_vis
from BeamNGRL.utils.planning import update_goal
import yaml
import os
from pathlib import Path
import time
import cv2

def main(map_name, start_pos, start_quat, config_path, BeamNG_dir="/home/stark/", target_WP=None):
    with open(config_path + 'MPPI_config.yaml') as f:
        MPPI_config = yaml.safe_load(f)

    with open(config_path + 'Dynamics_config.yaml') as f:
        Dynamics_config = yaml.safe_load(f)

    with open(config_path + 'Cost_config.yaml') as f:
        Cost_config = yaml.safe_load(f)

    with open(config_path + 'Sampling_config.yaml') as f:
        Sampling_config = yaml.safe_load(f)

    with open(config_path + 'Map_config.yaml') as f:
        Map_config = yaml.safe_load(f)

    map_res = Map_config["map_res"]
    dtype = torch.float
    d = torch.device("cuda")
    
    with torch.no_grad():

        dynamics = SimpleCarDynamics(Dynamics_config, Map_config, MPPI_config)
        costs = SimpleCarCost(Cost_config, Map_config)
        sampling = Delta_Sampling(Sampling_config, MPPI_config)

        controller = MPPI(
            dynamics,
            costs,
            sampling,
            MPPI_config
        )

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
        bng_interface.set_lockstep(True)
        bng_interface.burn_time = Dynamics_config['dt']

        current_wp_index = 0  # initialize waypoint index with 0
        goal = None
        action = np.zeros(2)

        while True:
            try:
                bng_interface.state_poll()
                now = time.time()
                # state is np.hstack((pos, rpy, vel, A, G, st, th/br)) ## note that velocity is in the body-frame
                state = np.copy(bng_interface.state)
                # state = np.zeros(17)
                pos = np.copy(state[:2])  # example of how to get car position in world frame. All data points except for dt are 3 dimensional.
                goal, terminate, current_wp_index = update_goal(
                    goal, pos, target_WP, current_wp_index, 30, step_size=2
                )

                if terminate:
                    print("done!")
                    bng_interface.send_ctrl(np.zeros(2))
                    time.sleep(5)
                    exit()
                ## get robot_centric BEV (not rotated into robot frame)
                BEV_heght = torch.from_numpy(bng_interface.BEV_heght).to(device=d, dtype=dtype)
                BEV_normal = torch.from_numpy(bng_interface.BEV_normal).to(device=d, dtype=dtype)
                BEV_path = torch.from_numpy(bng_interface.BEV_path).to(device=d, dtype=dtype)/255
                BEV_color = bng_interface.BEV_color # this is just for visualization
                BEV_center = bng_interface.BEV_center

                controller.Dynamics.set_BEV(BEV_heght, BEV_normal)
                controller.Costs.set_BEV(BEV_heght, BEV_normal, BEV_path)
                controller.Costs.set_goal(
                    torch.from_numpy(np.copy(goal) - np.copy(pos)).to(device=d, dtype=dtype)
                )  # you can also do this asynchronously

                state[:3] -= BEV_center 

                # we use our previous control output as input for next cycle!
                state[15:17] = action ## adhoc wheelspeed.
                action = np.array(
                    controller.forward(
                        torch.from_numpy(state).to(device=d, dtype=dtype)
                    )
                    .cpu()
                    .numpy(),
                    dtype=np.float64,
                )[0]
                action[1] = np.clip(action[1], Sampling_config['min_thr'], Sampling_config['max_thr'])
                dt_ = time.time() - now
                
                costmap_vis(
                    controller.Dynamics.states.cpu().numpy(),
                    pos,
                    np.copy(goal),
                    # 1/bng_interface.BEV_normal[:,:,2]*0.1,
                    cv2.applyColorMap(((BEV_heght.cpu().numpy() + 4)*255/8).astype(np.uint8), cv2.COLORMAP_JET),
                    1 / map_res,
                )

                bng_interface.send_ctrl(action, speed_ctrl=True, speed_max = 20, Kp=1, Ki=0.05, Kd=0.0, FF_gain=0.0)

            except Exception:
                print(traceback.format_exc())

        # bng_interface.bng.close()


if __name__ == "__main__":
    # position of the vehicle for tripped_flat on grimap_v2
    start_point = np.array([-67, 336, 34.5])
    start_quat = np.array([0, 0, 0.3826834, 0.9238795])
    map_name = "small_island"
    config_path = str(Path(os.getcwd()).parent.absolute()) + "/BeamNGRL/control/UW_mppi/Configs/"
    waypoint_path = str(Path(os.getcwd()).parent.absolute()) + "/BeamNGRL/utils/waypoint_files/"
    target_WP = np.load(waypoint_path+"WP_file_offroad.npy")
    main(map_name, start_point, start_quat, config_path, target_WP=target_WP)
