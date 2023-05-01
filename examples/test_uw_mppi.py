import numpy as np

from BeamNGRL.BeamNG.beamng_interface import *
import traceback
import torch
from BeamNGRL.control.UW_mppi.MPPI import MPPI
from BeamNGRL.control.UW_mppi.Dynamics.SimpleCar import SimpleCar
from BeamNGRL.control.UW_mppi.Costs.SimpleCarCost import SimpleCarCost
from BeamNGRL.utils.visualisation import costmap_vis
from BeamNGRL.utils.planning import update_goal


def main(map_name, start_point, start_quat, target_WP=None):

    map_res = 0.1
    map_size = 64 # 16 x 16 map

    bng_interface = get_beamng_default(
        map_size=map_size,
        map_name=map_name,
        map_res=map_res,
        start_pos=start_point,
        start_quat=start_quat,
    )

    # bng_interface.set_lockstep(True)
    dtype = torch.float

    # device = torch.device('cuda')
    device = torch.device("cpu")

    ctrl_sigma = torch.zeros((2,2), device=device, dtype=dtype)
    ctrl_sigma[0,0] = 1.  # steering
    ctrl_sigma[1,1] = 1. # throttle/brake
    # ns[1,1] = 2.5 # throttle/brake
    dt = 0.04

    dyn_func = SimpleCar(
        dt=dt,
        dtype=torch.float32,
        device=device,
    )
    cost_func = SimpleCarCost(
        map_size=map_size,
        map_res=map_res,
        device=device,
    )

    controller = MPPI(
        dyn_func,
        cost_func,
        ctrl_sigma,
        ctrl_mean=None,
        n_ctrl_samples=256,
        horizon=32,
        lambda_=0.1,
        dt=dt,
        u_final=None,
        u_per_command=1,
        num_optimizations=1,
        n_state_samples=1,
        device=device,
    )

    # lookahead = 30
    lookahead = 20
    current_wp_index = 0 # initialize waypoint index with 0
    goal = None

    while True:
        try:
            # state is np.hstack((pos, rpy, vel, A, G, st, th/br))
            # Velocity is in the body-frame
            bng_interface.state_poll()
            state = bng_interface.state

            pos = np.copy(state[:2])  # example of how to get car position in world frame. All data points except for dt are 3 dimensional.

            goal, terminate, current_wp_index = update_goal(
                goal, pos, target_WP, current_wp_index, lookahead,
            )

            if terminate:
                print("done!")
                bng_interface.send_ctrl(np.zeros(2))
                time.sleep(5)
                exit()

            ## get robot_centric BEV (not rotated into robot frame)
            BEV_color = torch.from_numpy(bng_interface.BEV_color).to(device)
            BEV_heght = torch.from_numpy(bng_interface.BEV_heght).to(device)
            BEV_segmt = torch.from_numpy(bng_interface.BEV_segmt).to(device)
            BEV_path  = torch.from_numpy(bng_interface.BEV_path).to(device)  # trail/roads
            BEV_normal = torch.from_numpy(bng_interface.BEV_normal).to(device)

            cost_func.update_maps(BEV_path, BEV_heght, BEV_segmt, BEV_color, BEV_normal)

            cost_func.set_goal(torch.from_numpy(np.copy(goal) - np.copy(pos)).to(device)) # you can also do this asynchronously
            # cost_func.set_goal(torch.from_numpy(goal).to(device)) # you can also do this asynchronously

            state[:3] = np.zeros(3)  # this is for the MPPI: technically this should be state[:3] -= BEV_center
            state = torch.from_numpy(state).to(device)

            action = controller(state)

            states = controller.rollout_states.cpu().numpy()
            costmap_vis(states, pos, np.copy(goal), np.copy(BEV_path.cpu().numpy()), 1/map_res)

            action = np.array(action.cpu().numpy(), dtype=np.float64)[0]
            bng_interface.send_ctrl(action)

        except Exception:
            print(traceback.format_exc())

    bng_interface.bng.close()


if __name__ == '__main__':
    # position of the vehicle for tripped_flat on grimap_v2
    start_point = np.array([-67, 336, 0.5])
    start_quat = np.array([0, 0, 0.3826834, 0.9238795])
    map_name = "smallgrid"
    # start_point = np.array([-67, 336, 34.5])
    # start_quat = np.array([0, 0, 0.3826834, 0.9238795])
    # start_quat = np.array([0, 0, 1, 1.])
    # map_name = "small_island"
    target_WP = np.load('WP_file_offroad.npy')
    main(map_name,start_point, start_quat, target_WP=target_WP)

