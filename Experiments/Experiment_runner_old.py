from BeamNGRL.BeamNG.beamng_interface import *
from BeamNGRL.control.UW_mppi.MPPI import MPPI
from BeamNGRL.control.UW_mppi.Dynamics.SimpleCarNetworkDyn import SimpleCarNetworkDyn
from BeamNGRL.control.UW_mppi.Dynamics.SimpleCarDynamicsCUDA import SimpleCarDynamics
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

torch.manual_seed(0)


def update_npy_datafile(buffer: List, filepath):
    buff_arr = np.array(buffer)
    if filepath.is_file():
        # Append to existing data file
        data_arr = np.load(filepath, allow_pickle=True)
        data_arr = np.concatenate((data_arr, buff_arr), axis=0)
        np.save(filepath, data_arr)
    else:
        np.save(filepath, buff_arr)
    return []  # empty buffer


def get_dynamics(model, Config):
    Dynamics_config = Config["Dynamics_config"]
    MPPI_config = Config["MPPI_config"]
    Map_config = Config["Map_config"]
    if model == "TerrainCNN":
        model_weights_path = (
            LOGS_PATH / "small_island" / Dynamics_config["model_weights"]
        )
        dynamics = SimpleCarNetworkDyn(
            Dynamics_config,
            Map_config,
            MPPI_config,
            model_weights_path=model_weights_path,
        )
    elif model == "slip3d":
        Dynamics_config["type"] = "slip3d"  ## just making sure
        dynamics = SimpleCarDynamics(Dynamics_config, Map_config, MPPI_config)
    elif model == "noslip3d":
        # temporarily change the dynamics type to noslip3d
        Dynamics_config["type"] = "noslip3d"
        dynamics = SimpleCarDynamics(Dynamics_config, Map_config, MPPI_config)
        Dynamics_config["type"] = "slip3d"
    else:
        raise ValueError("Unknown model type")
    return dynamics


def main(config_path=None, args=None):
    if config_path is None:
        print("no config file provided")
        exit()

    with open(config_path) as f:
        Config = yaml.safe_load(f)
    Dynamics_config = Config["Dynamics_config"]
    Cost_config = Config["Cost_config"]
    Sampling_config = Config["Sampling_config"]
    MPPI_config = Config["MPPI_config"]
    Map_config = Config["Map_config"]
    vehicle = Config["vehicle"]
    map_name = Config["map_name"]
    start_pos = np.array(
        Config["start_pos"]
    )  ## some default start position which will be overwritten by the scenario file
    start_quat = np.array(Config["start_quat"])
    map_res = Map_config["map_res"]
    map_size = Map_config["map_size"]

    # Check that the config file is valid because we run the experiment for a long time and don't want it to fail halfway through
    # I have had this happen to me before and it is very annoying
    assert len(Config["time_limit"]) == len(
        Config["scenarios"]
    ), "Time limit must be specified for each scenario"
    assert len(Config["lookahead"]) == len(
        Config["scenarios"]
    ), "Lookahead must be specified for each scenario"
    for scenario in Config["scenarios"]:
        WP_file = (
            str(Path(os.getcwd()).parent.absolute())
            + "/Experiments/Waypoints/"
            + scenario
            + ".npy"
        )
        if not os.path.isfile(WP_file):
            raise ValueError(
                "Waypoint file for scenario {} does not exist".format(scenario)
            )
    for models in Config["models"]:
        if models not in ["TerrainCNN", "slip3d", "noslip3d"]:
            raise ValueError("Model {} not supported".format(models))
    if Config["models"].count("TerrainCNN") > 0:
        if not os.path.isfile(
            LOGS_PATH / "small_island" / Dynamics_config["model_weights"]
        ):
            raise ValueError("Model weights for TerrainCNN do not exist")

    if not args.remote:
        bng_interface = get_beamng_default(
            car_model=vehicle["model"],
            start_pos=start_pos,
            start_quat=start_quat,
            map_name=map_name,
            car_make=vehicle["make"],
            beamng_path=BNG_HOME,
            map_res=map_res,
            map_size=map_size,
            elevation_range=Map_config["elevation_range"],
        )
        bng_interface.burn_time = Dynamics_config["dt"]

    else:
        if args.host_IP is None:
            raise ValueError("Host IP must be specified when running remote")

        bng_interface = get_beamng_remote(
            car_model=vehicle["model"],
            start_pos=start_pos,
            start_quat=start_quat,
            map_name=map_name,
            car_make=vehicle["make"],
            beamng_path=BNG_HOME,
            map_res=map_res,
            map_size=map_size,
            elevation_range=Map_config["elevation_range"],
            host_IP=args.host_IP,
        )
        bng_interface.burn_time = (
            Dynamics_config["dt"] * 0.4
        )  ## multiply by 0.4 to account for the overhead of the remote connection

    bng_interface.set_lockstep(True)

    dtype = torch.float
    device = torch.device("cuda")

    output_path = DATA_PATH / "experiment_data" / Config["output_dir"]
    output_path.mkdir(parents=True, exist_ok=True)

    timestamps = []
    state_data = []
    reset_data = []

    total_experiments = (
        len(Config["models"]) * len(Config["scenarios"]) * Config["num_iters"]
    )
    experiment_count = 0

    try:
        costs = SimpleCarCost(Cost_config, Map_config, device=device)
        sampling = Delta_Sampling(Sampling_config, MPPI_config, device=device)
        temp_temperature = torch.clone(sampling.temperature)
        for model in Config["models"]:
            dynamics = get_dynamics(model, Config)

            controller = MPPI(dynamics, costs, sampling, MPPI_config, device)
            scenario_count = 0

            if model == "TerrainCNN":
                controller.Sampling.temperature = torch.tensor(0.05, device=device)
            else:
                controller.Sampling.temperature = temp_temperature

            for scenario in Config["scenarios"]:
                # load the scenario waypoints:
                WP_file = (
                    str(Path(os.getcwd()).parent.absolute())
                    + "/Experiments/Waypoints/"
                    + scenario
                    + ".npy"
                )
                target_WP = np.load(WP_file)
                start_pos = target_WP[0, :3]
                start_quat = target_WP[0, 3:]

                temp_lethal_w = torch.clone(controller.Costs.lethal_w)
                temp_roll_w = torch.clone(controller.Costs.roll_w)

                if scenario.split("-")[0] == "race":
                    controller.Costs.lethal_w = torch.tensor(10.0, device=device)
                    controller.Costs.roll_w = torch.tensor(
                        1.0, device=device
                    )  ## reduce weighting on physics costs

                time_limit = Config["time_limit"][scenario_count]
                lookahead = Config["lookahead"][scenario_count]
                scenario_count += 1

                for trial in range(Config["num_iters"]):
                    trial_pos = np.copy(start_pos)
                    trial_pos[:2] += np.random.uniform(
                        -Config["start_pose_noise"], Config["start_pose_noise"], size=2
                    )
                    bng_interface.reset(start_pos=trial_pos, start_quat=start_quat)
                    current_wp_index = 0  # initialize waypoint index with 0
                    goal = None
                    action = np.zeros(2)
                    controller.reset()
                    success = False
                    result_states = []

                    last_reset_time = (
                        bng_interface.timestamp
                    )  # update the last reset time
                    ts = bng_interface.timestamp - last_reset_time

                    experiment_count += 1
                    print(
                        "Experiment: {}/{}".format(experiment_count, total_experiments),
                        end="\r",
                    )

                    while ts < time_limit:
                        bng_interface.state_poll()
                        state = np.copy(bng_interface.state)
                        ts = bng_interface.timestamp - last_reset_time
                        pos = np.copy(
                            state[:2]
                        )  # example of how to get car position in world frame. All data points except for dt are 3 dimensional.
                        goal, success, current_wp_index = update_goal(
                            goal, pos, target_WP, current_wp_index, lookahead
                        )
                        ## get robot_centric BEV (not rotated into robot frame)
                        BEV_heght = torch.from_numpy(bng_interface.BEV_heght).to(
                            device=device, dtype=dtype
                        )
                        BEV_normal = torch.from_numpy(bng_interface.BEV_normal).to(
                            device=device, dtype=dtype
                        )
                        BEV_path = (
                            torch.from_numpy(bng_interface.BEV_path).to(
                                device=device, dtype=dtype
                            )
                            / 255
                        )
                        controller.Dynamics.set_BEV(BEV_heght, BEV_normal)
                        controller.Costs.set_BEV(BEV_heght, BEV_normal, BEV_path)
                        controller.Costs.set_goal(
                            torch.from_numpy(np.copy(goal) - np.copy(pos)).to(
                                device=device, dtype=dtype
                            )
                        )  # you can also do this asynchronously
                        state[:3] = np.zeros(
                            3
                        )  # this is for the MPPI: technically this should be state[:3] -= BEV_center
                        # we use our previous control output as input for next cycle!
                        state[15:17] = action  ## adhoc wheelspeed.
                        action = np.array(
                            controller.forward(
                                torch.from_numpy(state).to(device=device, dtype=dtype)
                            )
                            .cpu()
                            .numpy(),
                            dtype=np.float64,
                        )[0]
                        action[1] = np.clip(
                            action[1],
                            Sampling_config["min_thr"],
                            Sampling_config["max_thr"],
                        )
                        costmap_vis(
                            controller.Dynamics.states.cpu().numpy(),
                            pos,
                            np.copy(goal),
                            cv2.applyColorMap(
                                ((BEV_path.cpu().numpy() + 4) * 255 / 8).astype(
                                    np.uint8
                                ),
                                cv2.COLORMAP_JET,
                            ),
                            1 / map_res,
                        )
                        bng_interface.send_ctrl(
                            action,
                            speed_ctrl=True,
                            speed_max=20,
                            Kp=1,
                            Ki=0.05,
                            Kd=0.0,
                            FF_gain=0.0,
                        )

                        damage = False
                        if type(bng_interface.broken) == dict:
                            count = 0
                            for part in bng_interface.broken.values():
                                if part["damage"] > 0.8:
                                    count += 1
                            damage = count > 1

                        result_states.append(
                            np.hstack(
                                (
                                    np.copy(state),
                                    np.copy(goal),
                                    np.copy(ts),
                                    success,
                                    damage,
                                )
                            )
                        )

                        if success:
                            break  ## break the for loop

                        timestamps.append(ts)
                        state_data.append(state)
                        reset_data.append(False)

                    result_states = np.array(result_states)
                    dir_name = (
                        str(Path(os.getcwd()).parent.absolute())
                        + "/Experiments/Results/Control/"
                        + model
                    )
                    filename = dir_name + "/{}-trial-{}.npy".format(
                        scenario, str(trial)
                    )
                    if not os.path.isdir(dir_name):
                        os.makedirs(dir_name)
                    np.save(filename, result_states)
                    ## add one last data point because we reset the car
                    timestamps.append(ts)
                    state_data.append(state)
                    reset_data.append(True)

                    timestamps = update_npy_datafile(
                        timestamps, output_path / "timestamps.npy"
                    )
                    state_data = update_npy_datafile(
                        state_data, output_path / "state.npy"
                    )
                    reset_data = update_npy_datafile(
                        reset_data, output_path / "reset.npy"
                    )

                controller.Costs.lethal_w = temp_lethal_w
                controller.Costs.roll_w = temp_roll_w
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
    parser.add_argument(
        "--config_name",
        type=str,
        default="Test_Config.yaml",
        help="name of the config file to use",
    )
    parser.add_argument(
        "--remote",
        type=str,
        default="False",
        help="whether to connect to a remote beamng server",
    )
    parser.add_argument(
        "--host_IP",
        type=str,
        default="10.18.172.189",
        help="host ip address if using remote beamng",
    )

    args = parser.parse_args()
    config_name = args.config_name
    config_path = (
        str(Path(os.getcwd()).parent.absolute()) + "/Experiments/Configs/" + config_name
    )
    with torch.no_grad():
        main(
            config_path=config_path, args=args
        )  ## we run for 3 iterations because science
