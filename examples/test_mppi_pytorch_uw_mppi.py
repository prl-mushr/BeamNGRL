import numpy as np
import cv2
from BeamNGRL.BeamNG.beamng_interface import *
import traceback
import torch
from BeamNGRL.control.UW_mppi.MPPI import MPPI
from BeamNGRL.control.UW_mppi.Dynamics.SimpleCarDynamics import SimpleCarDynamics
from BeamNGRL.control.UW_mppi.Costs.SimpleCarCost import SimpleCarCost


def visualization(states, pos, goal, costmap, resolution_inv):
    goal -= pos
    map_size = costmap.shape[0] // 2
    goal_X = int((goal[0] * resolution_inv) + map_size)
    goal_Y = int((goal[1] * resolution_inv) + map_size)
    cv2.line(costmap, (map_size, map_size), (goal_X, goal_Y), (0, 1, 0), 1)
    cv2.circle(costmap, (goal_X, goal_Y), int(resolution_inv * 0.2), (1, 0, 0), -1)
    if states is not None:
        print_states = states
        x = print_states[:, :, :, 0].flatten()
        y = print_states[:, :, :, 1].flatten()
        X = np.array((x * resolution_inv) + map_size, dtype=np.int32)
        Y = np.array((y * resolution_inv) + map_size, dtype=np.int32)
        costmap[Y, X] = np.array([0, 0, 1])
    costmap = cv2.resize(costmap, (500, 500), interpolation=cv2.INTER_AREA)
    costmap = cv2.flip(costmap, 0)  # this is just for visualization
    cv2.imshow("map", costmap)
    cv2.waitKey(1)


def update_goal(goal, pos, target_WP, current_wp_index, lookahead):
    if goal is None:
        if current_wp_index == 0:
            return target_WP[current_wp_index, :2], False, current_wp_index
        else:
            print("bruh moment")
            return pos, True, current_wp_index  ## terminate
    else:
        d = np.linalg.norm(goal - pos)
        if d < lookahead and current_wp_index < len(target_WP) - 1:
            current_wp_index += 1
            return target_WP[current_wp_index, :2], False, current_wp_index  ## new goal
        if current_wp_index == len(target_WP):
            return pos, True, current_wp_index  # Terminal condition
        else:
            return goal, False, current_wp_index


def main(map_name, start_pos, start_quat, BeamNG_dir="/home/stark/", target_WP=None):
    map_res = 0.1
    map_size = 32  # 16 x 16 map
    speed_max = 17

    with torch.no_grad():
        ## BEGIN MPPI
        dtype = torch.float
        d = torch.device("cuda")

        ## potentially these things should be loaded in from some config file? Will torchscript work with that?
        dynamics = SimpleCarDynamics(
            wheelbase=0.5,
            speed_max=speed_max,
            steering_max=0.5,
            dt=0.02,
            BEVmap_size=map_size,
            BEVmap_res=map_res,
            ROLLOUTS=512,
            TIMESTEPS=32,
            BINS=1,
        )
        costs = SimpleCarCost(
            goal_w=1,
            speed_w=1.5,
            roll_w=0,
            lethal_w=1,
            speed_target=10,
            critical_z=0.5,
            critical_FA=0.5,
            BEVmap_size=map_size,
            BEVmap_res=map_res,
        )
        # dyn = torch.jit.script(dynamics)
        # dyn.save("dynamics.pt")
        # cst = torch.jit.script(costs)
        # cst.save("costs.pt")

        ns = torch.zeros((2, 2), device=d, dtype=dtype)
        ns[0, 0] = 1.0  # steering
        ns[1, 1] = 1.0  # throttle/brake

        controller = MPPI(
            dynamics,
            costs,
            CTRL_NOISE=ns,
            lambda_=0.1,
        )

        # controller = torch.jit.script(controller)
        # controller.eval()
        ## END MPPI
        bng_interface = get_beamng_default(
            car_model='Short_Course_Truck',
            start_pos=start_pos,
            start_quat=start_quat,
            map_name=map_name,
            car_make='RG_RC',
            beamng_path=BNG_HOME,
            map_res=map_res,
            map_size=map_size
        )
        bng_interface.set_lockstep(True)
        
        current_wp_index = 0  # initialize waypoint index with 0
        goal = None
        action = np.zeros(2)

        while True:
            try:
                bng_interface.state_poll()
                # state is np.hstack((pos, rpy, vel, A, G, st, th/br)) ## note that velocity is in the body-frame
                state = np.copy(bng_interface.state)
                # state = np.zeros(17)
                pos = np.copy(
                    state[:2]
                )  # example of how to get car position in world frame. All data points except for dt are 3 dimensional.
                goal, terminate, current_wp_index = update_goal(
                    goal, pos, target_WP, current_wp_index, 10
                )

                if terminate:
                    print("done!")
                    bng_interface.send_ctrl(np.zeros(2))
                    time.sleep(5)
                    exit()
                ## get robot_centric BEV (not rotated into robot frame)
                BEV_heght = torch.from_numpy(bng_interface.BEV_heght).to(device=d, dtype=dtype)
                BEV_normal = torch.from_numpy(bng_interface.BEV_normal).to(device=d, dtype=dtype)
                BEV_center = torch.from_numpy(state[:3]).to(device=d, dtype=dtype)
                
                BEV_color = bng_interface.BEV_color # this is just for visualization

                controller.Dynamics.set_BEV(BEV_heght, BEV_normal, BEV_center)
                controller.Costs.set_BEV(BEV_heght, BEV_normal, BEV_center)
                controller.Costs.set_goal(
                    torch.from_numpy(np.copy(goal) - np.copy(pos)).to(device=d, dtype=dtype)
                )  # you can also do this asynchronously

                state[:3] = np.zeros(3) # this is for the MPPI: technically this should be state[:3] -= BEV_center

                # we use our previous control output as input for next cycle!

                state[15:17] = action ## adhoc wheelspeed.
                now = time.time()
                delta_action = np.array(
                    controller.forward(
                        torch.from_numpy(state).to(device=d, dtype=dtype)
                    )
                    .cpu()
                    .numpy(),
                    dtype=np.float64,
                )[0] * 0.02
                action += delta_action
                action = np.clip(action, -1, 1)
                dt = time.time() - now
                
                visualization(
                    controller.Dynamics.states.cpu().numpy(),
                    pos,
                    np.copy(goal),
                    BEV_color,
                    1 / map_res,
                )
                
                bng_interface.send_ctrl(action, speed_ctrl=True, speed_max = speed_max, Kp=0.5, Ki=0.05, Kd=0.0, FF_gain=0.4)

            except Exception:
                print(traceback.format_exc())

        # bng_interface.bng.close()


if __name__ == "__main__":
    # position of the vehicle for tripped_flat on grimap_v2
    start_point = np.array([-67, 336, 0.5])
    start_quat = np.array([0, 0, 0.3826834, 0.9238795])
    map_name = "smallgrid"
    target_WP = np.load("WP_file_offroad.npy")
    main(map_name, start_point, start_quat, target_WP=target_WP)
