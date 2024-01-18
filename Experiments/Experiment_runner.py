from BeamNGRL.BeamNG.beamng_interface import *
from BeamNGRL.control.UW_mppi.MPPI import MPPI
from BeamNGRL.control.UW_mppi.Dynamics.SimpleCarNetworkDyn import SimpleCarNetworkDyn
from BeamNGRL.control.UW_mppi.Dynamics.SimpleCarDynamicsCUDA import SimpleCarDynamics
from BeamNGRL.control.UW_mppi.Dynamics.ResidualCarDynamics import ResidualCarDynamics
from BeamNGRL.control.UW_mppi.Costs.SimpleCarCost import SimpleCarCost
from BeamNGRL.control.UW_mppi.Sampling.Delta_Sampling import Delta_Sampling
from BeamNGRL.utils.visualisation import costmap_vis
from BeamNGRL.utils.planning import update_goal
from BeamNGRL import DATA_PATH, LOGS_PATH
from typing import List
import torch
import yaml
import os
import argparse
import cv2

def update_npy_datafile(buffer: List, filepath):
    buff_arr = np.array(buffer)
    if filepath.is_file():
        # Append to existing data file
        data_arr = np.load(filepath, allow_pickle=True)
        data_arr = np.concatenate((data_arr, buff_arr), axis=0)
        np.save(filepath, data_arr)
    else:
        np.save(filepath, buff_arr)
    return [] # empty buffer

## TODO: move this to some kind of utils folder because this is used both in the loop as well as open-loop. 
def get_dynamics(model, Config):
    Dynamics_config = Config["Dynamics_config"]
    MPPI_config = Config["MPPI_config"]
    Map_config = Config["Map_config"]
    if model == 'TerrainCNN':
        print("loading TerrainCNN")
        Dynamics_config["network"] = Dynamics_config["network_baseline"]
        Dynamics_config["model_weights"] = Dynamics_config["model_weights_baseline"]
        model_weights_path = str(Path(os.getcwd()).parent.absolute()) + "/logs/baseline_offroad/" + Dynamics_config["model_weights"]
        dynamics = SimpleCarNetworkDyn(Dynamics_config, Map_config, MPPI_config, model_weights_path=model_weights_path)
    elif model == "KARMA":
        print("loading KARMA")
        Dynamics_config["network"] = Dynamics_config["network_KARMA"]
        Dynamics_config["model_weights"] = Dynamics_config["model_weights_KARMA"]
        model_weights_path = str(Path(os.getcwd()).parent.absolute()) + "/logs/residual_offroad/" + Dynamics_config["model_weights"]
        dynamics = ResidualCarDynamics(Dynamics_config, Map_config, MPPI_config, model_weights_path=model_weights_path)
    elif model == 'slip3d':
        print("loading slip3d")
        Dynamics_config["type"] = "slip3d" ## just making sure 
        dynamics = SimpleCarDynamics(Dynamics_config, Map_config, MPPI_config)
    elif model == 'noslip3d':
        # temporarily change the dynamics type to noslip3d
        Dynamics_config["type"] = "noslip3d"
        dynamics = SimpleCarDynamics(Dynamics_config, Map_config, MPPI_config)
        Dynamics_config["type"] = "slip3d"
    elif model == 'slip3d_150':
        # temporarily change the dynamics type to noslip3d
        Dynamics_config["type"] = "slip3d"
        temp_D = Dynamics_config["D"]
        Dynamics_config["D"] = 1.2 ## 150 % of the original D
        dynamics = SimpleCarDynamics(Dynamics_config, Map_config, MPPI_config)
        Dynamics_config["D"] = temp_D ## change it back
    elif model == 'slip3d_LPF':
        # temporarily change the dynamics type to noslip3d
        Dynamics_config["type"] = "slip3d"
        temp_LPF = Dynamics_config["LPF_tau"]
        Dynamics_config["LPF_tau"] = 0.2 ## apply a LPF with tau = 0.2
        dynamics = SimpleCarDynamics(Dynamics_config, Map_config, MPPI_config)
        Dynamics_config["LPF_tau"] = temp_LPF ## change it back
    elif model == 'slip3d_LPF_drag':
        # temporarily change the dynamics type to noslip3d
        Dynamics_config["type"] = "slip3d"
        temp_LPF = Dynamics_config["LPF_tau"]
        Dynamics_config["LPF_tau"] = 0.2 ## apply a LPF with tau = 0.2
        temp_drag = Dynamics_config["drag_coeff"]
        temp_res = Dynamics_config["res_coeff"]
        Dynamics_config["drag_coeff"] = 0.01
        Dynamics_config["res_coeff"] = 0.01
        dynamics = SimpleCarDynamics(Dynamics_config, Map_config, MPPI_config)
        Dynamics_config["LPF_tau"] = temp_LPF ## change it back
        Dynamics_config["drag_coeff"] = temp_drag
        Dynamics_config["res_coeff"] = temp_res
    else:
        raise ValueError('Unknown model type')
    return dynamics

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
    start_pos = np.array(Config["start_pos"]) ## some default start position which will be overwritten by the scenario file
    start_quat = np.array(Config["start_quat"])
    map_res = Map_config["map_res"]
    map_size = Map_config["map_size"]

    assert len(Config["time_limit"]) == len(Config["scenarios"]), "Time limit must be specified for each scenario"
    assert len(Config["lookahead"]) == len(Config["scenarios"]), "Lookahead must be specified for each scenario"
    for scenario in Config["scenarios"]:
        WP_file = str(Path(os.getcwd()).parent.absolute()) + "/Experiments/Waypoints/" + scenario + ".npy"
        if not os.path.isfile(WP_file):
            raise ValueError("Waypoint file for scenario {} does not exist".format(scenario))
    for models in Config["models"]:
        if models not in ["TerrainCNN", "slip3d", "noslip3d", 'slip3d_LPF', 'slip3d_150', 'slip3d_LPF_drag', 'KARMA']:
            raise ValueError("Model {} not supported".format(models))
    if Config["models"].count("TerrainCNN") > 0:
        if not os.path.isfile(LOGS_PATH / "small_island" / Dynamics_config["model_weights"]):
            raise ValueError("Model weights for TerrainCNN do not exist")

    if not Config["save_data"]:
        print("WARNING: Data will not be saved!")
    if Config["run_lockstep"]:
        print("Running in lockstep mode")

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
        burn_time=0.02,
        run_lockstep=Config["run_lockstep"],
    )

    dtype = torch.float
    device = torch.device("cuda")

    output_path = DATA_PATH / 'experiment_data' / Config["output_dir"]
    output_path.mkdir(parents=True, exist_ok=True)

    timestamps = []
    state_data = []
    reset_data = []

    total_experiments = len(Config["models"]) * len(Config["scenarios"]) * Config["num_iters"]
    experiment_count = 0

    try:
        costs = SimpleCarCost(Cost_config, Map_config, device=device)
        sampling = Delta_Sampling(Sampling_config, MPPI_config, device=device)
        temp_temperature = torch.clone(sampling.temperature)
        skips = max(Dynamics_config["dt"]/bng_interface.burn_time,1)

        for model in Config["models"]:
            dynamics = get_dynamics(model, Config)

            controller = MPPI(dynamics, costs, sampling, MPPI_config, device)
            scenario_count = 0

            # if(model == "TerrainCNN"):
            #     controller.Sampling.temperature = torch.tensor(0.05, device=device)
            # else:
            #     controller.Sampling.temperature = temp_temperature

            for scenario in Config["scenarios"]:
                # load the scenario waypoints:
                WP_file = str(Path(os.getcwd()).parent.absolute()) + "/Experiments/Waypoints/" + scenario + ".npy"
                target_WP = np.load(WP_file)
                start_pos = target_WP[0,:3]
                start_quat = target_WP[0,3:]

                temp_lethal_w = torch.clone(controller.Costs.lethal_w)
                temp_roll_w = torch.clone(controller.Costs.roll_w)
                temp_scaled_dt = torch.clone(controller.Sampling.scaled_dt)

                if scenario.split('-')[0] == "race":
                    controller.Costs.lethal_w = torch.tensor(40.0, device=device)
                    # controller.Costs.roll_w = torch.tensor(5.0, device=device) ## reduce weighting on physics costs
                    # controller.Sampling.scaled_dt = torch.tensor(Dynamics_config["dt"], device=device, dtype=dtype)

                time_limit = Config["time_limit"][scenario_count]
                lookahead = Config["lookahead"][scenario_count]
                wp_radius = Config["wp_radius"][scenario_count]
                scenario_count += 1

                for trial in range(Config["num_iters"]):
                    trial_pos = np.copy(start_pos)
                    trial_pos[:2] += np.random.uniform(-Config["start_pose_noise"], Config["start_pose_noise"], size=2)
                    bng_interface.reset(start_pos=trial_pos, start_quat=start_quat)
                    current_wp_index = 0  # initialize waypoint index with 0
                    goal = None
                    action = np.zeros(2)
                    controller.reset()
                    success = False
                    result_states = []

                    last_reset_time = bng_interface.timestamp # update the last reset time
                    ts = bng_interface.timestamp - last_reset_time
                    
                    experiment_count += 1
                    print("Experiment: {}/{}".format(experiment_count, total_experiments)) #, end='\r')
                    skip_save = False
                    bng_interface.state_poll()

                    while ts < time_limit:
                        for _ in range(int(skips)):
                            bng_interface.state_poll()
                            state = np.copy(bng_interface.state)
                            ts = bng_interface.timestamp - last_reset_time
                            timestamps.append(ts)
                            state_data.append(state)
                            reset_data.append(False)
                            ## append extra data to these lists
                        now = time.time()
                        pos = np.copy(state[:2])  # example of how to get car position in world frame. All data points except for dt are 3 dimensional.
                        goal, success, current_wp_index = update_goal(goal, pos, target_WP, current_wp_index, lookahead, wp_radius=wp_radius)
                        ## get robot_centric BEV (not rotated into robot frame)
                        BEV_heght = torch.from_numpy(bng_interface.BEV_heght).to(device=device, dtype=dtype)
                        BEV_normal = torch.from_numpy(bng_interface.BEV_normal).to(device=device, dtype=dtype)
                        BEV_path = torch.from_numpy(bng_interface.BEV_path).to(device=device, dtype=dtype)/255
                        controller.Dynamics.set_BEV(BEV_heght, BEV_normal)
                        controller.Costs.set_BEV(BEV_heght, BEV_normal, BEV_path)
                        controller.Costs.set_goal(torch.from_numpy(np.copy(goal) - np.copy(pos)).to(device=device, dtype=dtype))  # you can also do this asynchronously
                        
                        state_to_ctrl = np.copy(state)
                        state_to_ctrl[:3] = np.zeros(3) # this is for the MPPI: technically this should be state[:3] -= BEV_center
                        # we use our previous control output as input for next cycle!
                        state_to_ctrl[15:17] = action ## adhoc wheelspeed.
                        action = np.array(controller.forward(torch.from_numpy(state_to_ctrl).to(device=device, dtype=dtype)).cpu().numpy(),dtype=np.float64)[0]
                        action[1] = np.clip(action[1], Sampling_config["min_thr"], Sampling_config["max_thr"])
                        costmap_vis(controller.Dynamics.states.cpu().numpy(), pos, np.copy(goal), cv2.applyColorMap(((BEV_heght.cpu().numpy() + 4)*255/8).astype(np.uint8), cv2.COLORMAP_JET), 1 / map_res)
                        bng_interface.send_ctrl(action, speed_ctrl=True, speed_max = Dynamics_config["throttle_to_wheelspeed"], Kp=2, Ki=0.05, Kd=0.0, FF_gain=0.0)
                        step_cost = controller.Costs.step_cost.cpu().numpy()

                        damage = False
                        if(type(bng_interface.broken) == dict ):
                            count = 0
                            for part in bng_interface.broken.values():
                                if part['damage'] > 0.8:
                                    count += 1
                            damage = count > 1

                        result_states.append(np.hstack ( ( np.copy(state), np.copy(goal), step_cost, np.copy(ts), success, damage) ))
                        
                        if success or bng_interface.flipped_over:
                            break ## break the for loop
                        if torch.any(torch.isnan(controller.Dynamics.states)):
                            print("naans on the run!")
                            trial -= 1
                            experiment_count -= 1
                            skip_save = True
                            break
                    if skip_save:
                        continue

                    result_states = np.array(result_states)
                    dir_name = str(Path(os.getcwd()).parent.absolute()) + "/Experiments/Results/Control/" + model
                    filename = dir_name + "/{}-trial-{}.npy".format(scenario, str(trial))
                    if(not os.path.isdir(dir_name)):
                        os.makedirs(dir_name)
                    ## add one last data point because we reset the car
                    timestamps.append(ts)
                    state_data.append(state)
                    reset_data.append(True)

                    if(Config["save_data"]):
                        np.save(filename, result_states)
                        timestamps = update_npy_datafile(timestamps, output_path / "timestamps.npy")
                        state_data = update_npy_datafile(state_data, output_path / "state.npy")
                        reset_data = update_npy_datafile(reset_data, output_path / "reset.npy")
                
                controller.Costs.lethal_w = temp_lethal_w
                controller.Costs.roll_w = temp_roll_w
                controller.Sampling.scaled_dt = temp_scaled_dt
                ## reset the weights
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(e)
        bng_interface.bng.close()
        cv2.destroyAllWindows()
        os._exit(1)
    bng_interface.bng.close()
    cv2.destroyAllWindows()
    os._exit(1)


if __name__ == "__main__":
    # do the args thingy:
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
    with torch.no_grad():
        main(config_path = config_path, hal_config_path = hal_config_path, args = args) ## we run for 3 iterations because science
