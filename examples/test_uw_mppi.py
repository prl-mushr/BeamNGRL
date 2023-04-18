from BeamNGRL.BeamNG.beamng_interface import *
import traceback
import torch
from BeamNGRL.control.UW_mppi.MPPI import MPPI
from BeamNGRL.control.UW_mppi.Dynamics.SimpleCar import SimpleCar
from BeamNGRL.control.UW_mppi.Costs.SimpleCarCost import SimpleCarCost


def visualization(states, pos, goal, costmap, resolution_inv):
    goal -= pos
    map_size = costmap.shape[0]//2
    goal_X = int((goal[0]*resolution_inv) + map_size)
    goal_Y = int((goal[1]*resolution_inv) + map_size)
    cv2.line(costmap, (map_size, map_size), (goal_X, goal_Y), (0,1,0), 1)
    cv2.circle(costmap, (goal_X, goal_Y), int(resolution_inv*0.2), (1,0,0), -1)
    if(states is not None):
        print_states = states
        x = print_states[:,:,:,0].flatten()
        y = print_states[:,:,:,1].flatten()
        X = np.array((x * resolution_inv) + map_size, dtype=np.int32)
        Y = np.array((y * resolution_inv) + map_size, dtype=np.int32)
        costmap[Y,X] = np.array([0,0,1])
    costmap = cv2.resize(costmap, (500,500), interpolation= cv2.INTER_AREA)
    costmap = cv2.flip(costmap, 0)  # this is just for visualization
    cv2.imshow("map", costmap)
    cv2.waitKey(1)


def update_goal(goal, pos, target_WP, current_wp_index, lookahead):
    if(goal is None):
        if current_wp_index == 0:
            return target_WP[current_wp_index,:2], False, current_wp_index
        else:
            print("bruh moment")
            return pos, True, current_wp_index ## terminate
    else:
        d = np.linalg.norm(goal - pos)
        if(d < lookahead and current_wp_index < len(target_WP) - 1):
            current_wp_index += 1
            return target_WP[current_wp_index,:2], False, current_wp_index ## new goal
        if current_wp_index == len(target_WP):
            return pos, True, current_wp_index # Terminal condition
        else:
            return goal, False, current_wp_index


def main(map_name, start_point, start_quat, BeamNG_dir='/home/sasha/libraries/', target_WP=None):

    map_res = 0.1
    map_size = 64 # 16 x 16 map

    bng_interface = beamng_interface(BeamNG_dir=BeamNG_dir)
    bng_interface.load_scenario(scenario_name=map_name, car_make='sunburst', car_model='RACER',
                                start_pos=start_point, start_rot=start_quat)
    bng_interface.set_map_attributes(map_size = map_size, resolution=map_res, path_to_maps='/home/sasha/projects/offroad_mbrl')

    # bng_interface.set_lockstep(True)
    dtype = torch.float

    # d = torch.device('cuda')
    d = torch.device("cpu")

    ns = torch.zeros((2,2), device=d, dtype=dtype)
    ns[0,0] = 1.  # steering
    ns[1,1] = 1. # throttle/brake
    # ns[1,1] = 2.5 # throttle/brake

    dyn_func = SimpleCar(
        wheelbase=1,
        speed_max=1,
       	steering_max=1,
       	dt=0.05,
        dtype=torch.float32,
        device=d,
    )
    cost_func = SimpleCarCost(
        BEVmap_size=map_size,
        BEVmap_res=map_res,
        device=d,
    )
    controller = MPPI(
        dyn_func,
        cost_func,
        ns,
        CTRL_NOISE_MU=None,
        N_SAMPLES=256,
        TIMESTEPS=32,
        lambda_=0.1,
        u_final=None,
        u_per_command=1,
        num_optimizations=1,
        Rollout_bins=1,
        device=d,
    )

    # lookahead = 30
    lookahead = 20
    current_wp_index = 0 # initialize waypoint index with 0
    goal = None
    action = np.zeros(2)

    while True:
        try:
            bng_interface.state_poll()
            # state is np.hstack((pos, rpy, vel, A, G, st, th/br)) ## note that velocity is in the body-frame
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
            BEV_color = torch.from_numpy(bng_interface.BEV_color).to(d)
            BEV_heght = torch.from_numpy(bng_interface.BEV_heght).to(d)
            BEV_segmt = torch.from_numpy(bng_interface.BEV_segmt).to(d)
            BEV_path  = torch.from_numpy(bng_interface.BEV_path).to(d)  # trail/roads
            BEV_normal = torch.from_numpy(bng_interface.BEV_normal).to(d)
            BEV_center = torch.from_numpy(pos).to(d)

            cost_func.update_maps(BEV_path, BEV_heght, BEV_segmt, BEV_color, BEV_normal)

            # cost_func.set_goal(torch.from_numpy(np.copy(goal) - np.copy(pos)).to(d)) # you can also do this asynchronously
            cost_func.set_goal(torch.from_numpy(goal).to(d)) # you can also do this asynchronously

            state[:3] = np.zeros(3)  # this is for the MPPI: technically this should be state[:3] -= BEV_center

            action = controller.forward(torch.from_numpy(state).to(d))

            states = controller.states.cpu().numpy()
            visualization(states, pos, np.copy(goal), np.copy(BEV_path.cpu().numpy()), 1/map_res)

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

