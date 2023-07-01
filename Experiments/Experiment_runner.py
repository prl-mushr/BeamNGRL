from BeamNGRL.BeamNG.beamng_interface import *
from BeamNGRL.control.UW_mppi.MPPI import MPPI
from BeamNGRL.control.UW_mppi.Dynamics.SimpleCarNetworkDyn import SimpleCarNetworkDyn
from BeamNGRL.control.UW_mppi.Dynamics.SimpleCarDynamicsCUDA import SimpleCarDynamics
from BeamNGRL.control.UW_mppi.Dynamics.SimpleCarDynamics import SimpleCarDynamics as SimpleCarDynamicsNS
from BeamNGRL.control.UW_mppi.Costs.SimpleCarCost import SimpleCarCost
from BeamNGRL.control.UW_mppi.Sampling.Delta_Sampling import Delta_Sampling
from BeamNGRL.utils.visualisation import costmap_vis
from BeamNGRL.utils.planning import update_goal
import torch
import yaml
import os
import argparse
import traceback
import cv2

def get_dynamics(model, Config):
    Dynamics_config = Config["Dynamics_config"]
    MPPI_config = Config["MPPI_config"]
    Map_config = Config["Map_config"]
    if model == 'TerrainCNN':
        model_weights_path = str(Path(os.getcwd()).parent.absolute()) + "/logs/small_island/" + Dynamics_config["model_weights"]
        dynamics = SimpleCarNetworkDyn(Dynamics_config, Map_config, MPPI_config, model_weights_path=model_weights_path)
    elif model == 'slip3d':
        Dynamics_config["type"] = "slip3d" ## just making sure 
        dynamics = SimpleCarDynamics(Dynamics_config, Map_config, MPPI_config)
    elif model == 'noslip3d':
        # temporarily change the dynamics type to noslip3d
        Dynamics_config["type"] = "noslip3d"
        dynamics = SimpleCarDynamics(Dynamics_config, Map_config, MPPI_config)
        Dynamics_config["type"] = "slip3d"
    else:
        raise ValueError('Unknown model type')
    return dynamics

def main(config_path=None, WP_file=None, scenario_name=None, num_iters=3):
    if(scenario_name is None or WP_file is None or config_path is None):
        print("you missed something")
        print("config_path: ", config_path)
        print("target_WP: ", target_WP)
        print("scenario_name: ", scenario_name)
        exit()
    
    map_name = scenario_name.split("-")[1]
    test_name = scenario_name.split("-")[2]
    test_num = scenario_name.split("-")[3]
    target_WP = np.load(WP_file)
    start_pos = target_WP[0, :3]
    start_quat = target_WP[0, 3:]
    
    with open(config_path) as f:
        Config = yaml.safe_load(f)
    Dynamics_config = Config["Dynamics_config"]
    Cost_config = Config["Cost_config"]
    Sampling_config = Config["Sampling_config"]
    MPPI_config = Config["MPPI_config"]
    Map_config = Config["Map_config"]
    vehicle = Config["vehicle"]

    bng_interface = get_beamng_default(
        car_model=vehicle["model"],
        start_pos=start_pos,
        start_quat=start_quat,
        map_name=map_name,
        car_make=vehicle["make"],
        beamng_path=BNG_HOME,
        map_res=Map_config["map_res"],
        map_size=Map_config["map_size"],
        elevation_range=Map_config["elevation_range"]
    )
    bng_interface.set_lockstep(True)
    bng_interface.burn_time = Dynamics_config["dt"]

    map_res = Map_config["map_res"]
    dtype = torch.float
    device = torch.device("cuda")
    
    with torch.no_grad():
        if Config["Model_Type"] == "noslip3d":
            dynamics = SimpleCarDynamicsNS(
                Dynamics_config, Map_config, MPPI_config,
                device=device,
            )
        elif Config["Model_Type"] == "slip3d":
            dynamics = SimpleCarDynamics(
                Dynamics_config, Map_config, MPPI_config,
                device=device,
            )
        elif Config["Model_Type"] == "TerrainCNN":
            model_weights_path = str(Path(os.getcwd()).parent.absolute()) + "/logs/small_island/" + "best_18.pt"
            print(model_weights_path)
            dynamics = SimpleCarNetworkDyn(
                Dynamics_config, Map_config, MPPI_config,
                model_weights_path=model_weights_path,
                device=device,
            )

        costs = SimpleCarCost(Cost_config, Map_config, device=device)
        sampling = Delta_Sampling(Sampling_config, MPPI_config, device=device)

        controller = MPPI(
            dynamics,
            costs,
            sampling,
            MPPI_config,
            device,
        )

        current_wp_index = 0  # initialize waypoint index with 0
        goal = None
        action = np.zeros(2)

        result_states = [] ## save the states for later analysis
        trial = 0 ## nuimber of trials

        while True:
            try:
                bng_interface.state_poll()

                state = np.copy(bng_interface.state)

                pos = np.copy(state[:2])  # example of how to get car position in world frame. All data points except for dt are 3 dimensional.
                goal, terminate, current_wp_index = update_goal(
                    goal, pos, target_WP, current_wp_index, 20
                )

                if terminate:
                    print("done!")
                    bng_interface.send_ctrl(np.zeros(2))
                    time.sleep(1)
                    # save the data:
                    result_states = np.array(result_states)
                    filename = str(Path(os.getcwd()).parent.absolute()) + "/Experiments/Results/Control/" + Config["Model_Type"] + "/{}-{}-trial-{}.npy".format(test_name, test_num, str(trial))
                    np.save(filename, result_states)
                    trial += 1
                    if(trial < num_iters):
                        bng_interface.reset()
                        current_wp_index = 0
                        controller.reset()
                        action = np.zeros(2)
                        goal = None
                        result_states = []
                        continue
                    else:
                        break
                ## get robot_centric BEV (not rotated into robot frame)
                BEV_heght = torch.from_numpy(bng_interface.BEV_heght).to(device=device, dtype=dtype)
                BEV_normal = torch.from_numpy(bng_interface.BEV_normal).to(device=device, dtype=dtype)
                BEV_path = torch.from_numpy(bng_interface.BEV_path).to(device=device, dtype=dtype)/255

                controller.Dynamics.set_BEV(BEV_heght, BEV_normal)
                controller.Costs.set_BEV(BEV_heght, BEV_normal, BEV_path)
                controller.Costs.set_goal(
                    torch.from_numpy(np.copy(goal) - np.copy(pos)).to(device=device, dtype=dtype)
                )  # you can also do this asynchronously

                state[:3] = np.zeros(3) # this is for the MPPI: technically this should be state[:3] -= BEV_center

                # we use our previous control output as input for next cycle!
                state[15:17] = action ## adhoc wheelspeed.
                action = np.array(
                    controller.forward(
                        torch.from_numpy(state).to(device=device, dtype=dtype)
                    )
                    .cpu()
                    .numpy(),
                    dtype=np.float64,
                )[0]
                action[1] = np.clip(action[1], Sampling_config["min_thr"], Sampling_config["max_thr"])
                
                costmap_vis(
                    controller.Dynamics.states.cpu().numpy(),
                    pos,
                    np.copy(goal),
                    cv2.applyColorMap(((BEV_heght.cpu().numpy() + 4)*255/8).astype(np.uint8), cv2.COLORMAP_JET),
                    1 / map_res,
                )

                bng_interface.send_ctrl(action, speed_ctrl=True, speed_max = 20, Kp=1, Ki=0.05, Kd=0.0, FF_gain=0.0)

                result_states.append(np.hstack ( ( np.copy(state), np.copy(bng_interface.timestamp) )))

            except Exception:
                print(traceback.format_exc())
            except KeyboardInterrupt:
                bng_interface.bng.close()
                cv2.destroyAllWindows()
        bng_interface.bng.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # do the args thingy:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name", type=str, default="Test_Config.yaml", help="name of the config file to use")
    parser.add_argument("--scenario_name", type=str, default="waypoints-small_island-hill-0.npy",help="name of the scenario file to use")

    args = parser.parse_args()
    config_name = args.config_name
    scenario_name = args.scenario_name

    WP_file = str(Path(os.getcwd()).parent.absolute()) + "/Experiments/Waypoints/" + scenario_name
    config_path = str(Path(os.getcwd()).parent.absolute()) + "/Experiments/Configs/" + config_name
    main(config_path=config_path, WP_file=WP_file, scenario_name=scenario_name, num_iters=3) ## we run for 3 iterations because science
