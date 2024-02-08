from BeamNGRL.BeamNG.beamng_interface import *
from BeamNGRL.control.UW_mppi.MPPI import MPPI
from BeamNGRL.control.UW_mppi.Dynamics.SimpleCarNetworkDyn import SimpleCarNetworkDyn
from BeamNGRL.control.UW_mppi.Dynamics.SimpleCarDynamicsCUDA import SimpleCarDynamics
from BeamNGRL.control.UW_mppi.Costs.SimpleCarCost import SimpleCarCost
from BeamNGRL.control.UW_mppi.Sampling.Delta_Sampling import Delta_Sampling
from BeamNGRL.utils.visualisation import costmap_vis
from BeamNGRL.utils.planning import update_goal, find_closest_index
from BeamNGRL import DATA_PATH, LOGS_PATH
from typing import List
import torch
import yaml
import os
import argparse
import cv2
import math as m

# torch.manual_seed(0)


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
    elif model == "slip3d" or model == "slip3d_rp":
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


def steering_limiter(
    steer=0,
    wheelspeed=0,
    roll=0,
    rotBF=np.zeros(3),
    accBF=np.zeros(3),
    wheelbase=2.6,
    t_h_ratio=0.5,
    max_steer=0.5,
    accel_gain=1.0,
    roll_rate_gain=1.0,
    steer_slack=0.1,
):
    steering_setpoint = steer * max_steer
    intervention = False
    whspd2 = max(1.0, wheelspeed)
    whspd2 *= whspd2
    Aylim = t_h_ratio * max(1.0, abs(accBF[2]))
    roll_rate = rotBF[0]
    GRAVITY = 9.8
    steering_limit_max = (
        m.atan2(wheelbase * (Aylim - GRAVITY * m.sin(roll)), whspd2)
        + steer_slack * max_steer
    )
    steering_limit_min = (
        -m.atan2(wheelbase * (Aylim + GRAVITY * m.sin(roll)), whspd2)
        - steer_slack * max_steer
    )

    temp = steering_setpoint
    steering_setpoint = min(
        steering_limit_max, max(steering_limit_min, steering_setpoint)
    )
    if abs(temp - steering_setpoint) > 0.001:
        intervention = True

    delta_steering = 0
    Ay = accBF[1]
    Ay_error = 0
    Ay_rate = -roll_rate * accBF[2] / (m.cos(roll) ** 2)

    ## this was a side experiment that we conducted to see if TTR could improve performance; TL;DR: it did not provide measureable improvement.
    ## adding TTR can also lead to more interference because it forces the system to act before something is even about to happen,
    ## and makes it more sensitive to noise from the IMU, which is more prevelant in the real world and less so in simulation.
    TTR_condition = (
        min(m.fabs(Aylim - m.fabs(Ay)) / max(m.fabs(Ay_rate), 0.01), 0.9) < 0.5
    )
    if abs(Ay) > Aylim:
        intervention = True
        if Ay >= 0:
            Ay_error = min(Aylim - Ay, 0)
            delta_steering = (
                4.0
                * (Ay_error * accel_gain - roll_rate_gain * abs(accBF[2]) * roll_rate)
                * (m.cos(steering_setpoint) ** 2)
                * wheelbase
                / whspd2
            )
            delta_steering = min(delta_steering, 0)
        else:
            Ay_error = max(-Aylim - Ay, 0)
            delta_steering = (
                4.0
                * (Ay_error * accel_gain - roll_rate_gain * abs(accBF[2]) * roll_rate)
                * (m.cos(steering_setpoint) ** 2)
                * wheelbase
                / whspd2
            )
            delta_steering = max(
                delta_steering, 0
            )  ## this prevents the car from turning in the opposite direction and causing a rollover by mistake
        steering_setpoint += delta_steering
    steering_setpoint = steering_setpoint / max_steer
    steering_setpoint = min(max(steering_setpoint, -1.0), 1.0)
    return steering_setpoint, intervention, delta_steering, TTR_condition


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
    start_pos = np.array(
        Config["start_pos"]
    )  ## some default start position which will be overwritten by the scenario file
    start_quat = np.array(Config["start_quat"])

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
        if models not in ["TerrainCNN", "slip3d", "noslip3d", "slip3d_rp"]:
            raise ValueError("Model {} not supported".format(models))
    if Config["models"].count("TerrainCNN") > 0:
        if not os.path.isfile(
            LOGS_PATH / "small_island" / Dynamics_config["model_weights"]
        ):
            raise ValueError("Model weights for TerrainCNN do not exist")

    if Config["save_data"] != True:
        print("data will not be saved!")

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

    output_path = DATA_PATH / "rollover_data" / Config["output_dir"]
    output_path.mkdir(parents=True, exist_ok=True)

    timestamps = []
    state_data = []
    reset_data = []

    total_experiments = (
        len(Config["models"]) * len(Config["scenarios"]) * Config["num_iters"]
    )
    experiment_count = 0

    vehicle = Config["vehicles"]["flux"]  ## TODO: change this to HOUND
    t_h_ratio = vehicle["track_width"] / (2 * vehicle["cg_height"])
    max_steer = vehicle["max_steer"]
    wheelbase = vehicle["wheelbase"]
    time_limit = Config["time_limit"]
    roll_rate_gain = vehicle["roll_rate_gain"]
    accel_gain = vehicle["accel_gain"]
    steer_slack = vehicle["steer_slack"]

    try:
        costs = SimpleCarCost(Cost_config, Map_config, device=device)
        sampling = Delta_Sampling(Sampling_config, MPPI_config, device=device)
        temp_temperature = torch.clone(sampling.temperature)
        skips = Dynamics_config["dt"] / bng_interface.burn_time

        for model in Config["models"]:
            dynamics = get_dynamics(model, Config)

            controller = MPPI(dynamics, costs, sampling, MPPI_config, device)
            scenario_count = 0

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
                temp_temperature = torch.clone(controller.Sampling.temperature)

                if scenario.split("-")[0] == "race":
                    controller.Costs.lethal_w = torch.tensor(10.0, device=device)
                    controller.Costs.roll_w = torch.tensor(
                        1.0, device=device
                    )  ## reduce weighting on physics costs
                    controller.Sampling.temperature = torch.tensor(0.04, device=device)

                time_limit = Config["time_limit"][scenario_count]
                lookahead = Config["lookahead"][scenario_count]
                scenario_count += 1

                use_elevation = False

                for trial in range(Config["num_iters"]):
                    trial_pos = np.copy(start_pos)
                    trial_pos[:2] += np.random.uniform(
                        -Config["start_pose_noise"], Config["start_pose_noise"], size=2
                    )
                    if Map_config["map_name"] == "smallgrid":
                        use_elevation = False
                        trial_pos[2] = 0.1
                    else:
                        use_elevation = True
                        trial_pos[2] += 0.1
                    bng_interface.reset(start_pos=trial_pos, start_quat=start_quat)
                    current_wp_index = 0  # initialize waypoint index with 0
                    goal = None
                    action = np.zeros(2)
                    controller.reset()
                    success = False
                    result_states = []

                    experiment_count += 1
                    print(
                        "Experiment: {}/{}".format(experiment_count, total_experiments),
                        end="\r",
                    )

                    # initialization
                    for i in range(int(1 / Dynamics_config["dt"])):
                        bng_interface.state_poll()
                        last_reset_time = (
                            bng_interface.timestamp
                        )  # update the last reset time
                        state = np.copy(bng_interface.state)
                        ts = bng_interface.timestamp - last_reset_time
                        timestamps.append(ts)
                        state_data.append(state)
                        reset_data.append(False)

                    while ts < time_limit:
                        for _ in range(int(skips)):
                            bng_interface.state_poll()
                            state = np.copy(bng_interface.state)
                            ts = bng_interface.timestamp - last_reset_time
                            timestamps.append(ts)
                            state_data.append(state)
                            reset_data.append(False)
                            ## append extra data to these lists
                        pos = np.copy(
                            state[:2]
                        )  # example of how to get car position in world frame. All data points except for dt are 3 dimensional.
                        goal, success, current_wp_index = update_goal(
                            goal,
                            pos,
                            target_WP,
                            current_wp_index,
                            lookahead,
                            wp_radius=Config["wp_radius"],
                        )
                        ## get robot_centric BEV (not rotated into robot frame)

                        BEV_heght = (
                            torch.from_numpy(bng_interface.BEV_heght).to(
                                device=device, dtype=dtype
                            )
                            * use_elevation
                        )
                        BEV_normal = (
                            torch.from_numpy(bng_interface.BEV_normal).to(
                                device=device, dtype=dtype
                            )
                            * use_elevation
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

                        state_to_ctrl = np.copy(state)
                        state_to_ctrl[:3] = np.zeros(
                            3
                        )  # this is for the MPPI: technically this should be state[:3] -= BEV_center
                        # we use our previous control output as input for next cycle!
                        state_to_ctrl[15:17] = action  ## adhoc wheelspeed.
                        action = np.array(
                            controller.forward(
                                torch.from_numpy(state_to_ctrl).to(
                                    device=device, dtype=dtype
                                )
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
                                ((BEV_heght.cpu().numpy() + 4) * 255 / 8).astype(
                                    np.uint8
                                ),
                                cv2.COLORMAP_JET,
                            ),
                            1 / Map_config["map_res"],
                        )

                        if model == "slip3d_rp":
                            (
                                action[0],
                                intervention,
                                delta_steering,
                                Rollover_detected,
                            ) = steering_limiter(
                                steer=action[0],
                                wheelspeed=bng_interface.avg_wheelspeed,
                                roll=state_to_ctrl[3],
                                rotBF=state_to_ctrl[12:15],
                                accBF=state_to_ctrl[9:12],
                                wheelbase=wheelbase,
                                t_h_ratio=t_h_ratio,
                                max_steer=max_steer,
                                accel_gain=accel_gain,
                                roll_rate_gain=roll_rate_gain,
                                steer_slack=steer_slack,
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

                        if bng_interface.flipped_over:
                            closest_index = find_closest_index(pos, target_WP)
                            reset_pos = target_WP[closest_index, :3]
                            if Map_config["map_name"] == "smallgrid":
                                reset_pos[2] = 0.1
                            else:
                                reset_pos[2] += 0.1
                            reset_quat = target_WP[closest_index, 3:]
                            bng_interface.reset(
                                start_pos=reset_pos, start_quat=reset_quat
                            )
                            ## give it 1 second penalty. This also helps the car stabilize after being "dropped"
                            for i in range(int(1 / Dynamics_config["dt"])):
                                bng_interface.state_poll()
                                state = np.copy(bng_interface.state)
                                ts = bng_interface.timestamp - last_reset_time
                                timestamps.append(ts)
                                state_data.append(state)
                                reset_data.append(False)
                                controller.reset()

                        if success:
                            break  ## break the for loop

                    result_states = np.array(result_states)
                    dir_name = (
                        str(Path(os.getcwd()).parent.absolute())
                        + "/Experiments/Results/Rollover/"
                        + model
                    )
                    filename = dir_name + "/{}-trial-{}.npy".format(
                        scenario, str(trial)
                    )
                    if not os.path.isdir(dir_name):
                        os.makedirs(dir_name)
                    ## add one last data point because we reset the car
                    timestamps.append(ts)
                    state_data.append(state)
                    reset_data.append(True)

                    if Config["save_data"]:
                        np.save(filename, result_states)
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
                controller.Sampling.temperature = temp_temperature
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
        default="Rollover_loop_config.yaml",
        help="name of the config file to use",
    )
    parser.add_argument(
        "--hal_config_name",
        type=str,
        default="hound.yaml",
        help="name of the config file to use",
    )
    parser.add_argument(
        "--remote",
        type=bool,
        default=True,
        help="whether to connect to a remote beamng server",
    )
    parser.add_argument(
        "--host_IP",
        type=str,
        default="169.254.216.9",
        help="host ip address if using remote beamng",
    )

    args = parser.parse_args()
    config_name = args.config_name
    config_path = (
        str(Path(os.getcwd()).parent.absolute()) + "/Experiments/Configs/" + config_name
    )

    hal_config_name = args.hal_config_name
    hal_config_path = (
        str(Path(os.getcwd()).parent.absolute()) + "/Configs/" + hal_config_name
    )
    with torch.no_grad():
        main(
            config_path=config_path, hal_config_path=hal_config_path, args=args
        )  ## we run for 3 iterations because science
