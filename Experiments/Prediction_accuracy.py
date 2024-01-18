from BeamNGRL.control.UW_mppi.Dynamics.SimpleCarDynamicsCUDA import SimpleCarDynamics
from BeamNGRL.control.UW_mppi.Dynamics.ResidualCarDynamics import ResidualCarDynamics
from BeamNGRL.control.UW_mppi.Dynamics.SimpleCarNetworkDyn import SimpleCarNetworkDyn
from BeamNGRL.dynamics.utils.exp_utils import get_dataloaders, build_nets, get_loss_func, init_exp_dir
import torch
import yaml
import os
import argparse
import traceback
import numpy as np
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import sys
import time

## the job of this script is to take ground-truth data for controls and states, run the controls through the dynamics model and compare the predicted states to the ground-truth states

def get_dynamics(model, Config):
    Dynamics_config = Config["Dynamics_config"]
    MPPI_config = Config["MPPI_config"]
    Map_config = Config["Map_config"]
    print("loading: ", model)
    if model == 'TerrainCNN':
        Dynamics_config["network"] = Dynamics_config["network_baseline"]
        Dynamics_config["model_weights"] = Dynamics_config["model_weights_baseline"]
        model_weights_path = str(Path(os.getcwd()).parent.absolute()) + "/logs/baseline_long/" + Dynamics_config["model_weights"]
        dynamics = SimpleCarNetworkDyn(Dynamics_config, Map_config, MPPI_config, model_weights_path=model_weights_path)
    elif model == "KARMA":
        Dynamics_config["type"] = "slip3d"
        Dynamics_config["network"] = Dynamics_config["network_KARMA"]
        Dynamics_config["model_weights"] = Dynamics_config["model_weights_KARMA"]## you modified this last night. Results of previous experiments indicate improvement, not so much
        model_weights_path = str(Path(os.getcwd()).parent.absolute()) + "/logs/residual_long/" + Dynamics_config["model_weights"]
        dynamics = ResidualCarDynamics(Dynamics_config, Map_config, MPPI_config, model_weights_path=model_weights_path)
    elif model == "KARMA_bad_sys":
        Dynamics_config["type"] = "slip3d"
        temp_D = Dynamics_config["D"]
        Dynamics_config["D"] = 0.4 ## 50 % of the original D
        Dynamics_config["network"] = Dynamics_config["network_KARMA"]
        Dynamics_config["model_weights"] = Dynamics_config["model_weights_KARMA_bad_sys"]## you modified this last night. Results of previous experiments indicate improvement, not so much
        model_weights_path = str(Path(os.getcwd()).parent.absolute()) + "/logs/residual_long_bad_sys/" + Dynamics_config["model_weights"]
        dynamics = ResidualCarDynamics(Dynamics_config, Map_config, MPPI_config, model_weights_path=model_weights_path)
        Dynamics_config["D"] = temp_D ## change it back

    elif model == "KARMA_noslip":
        Dynamics_config["type"] = "noslip3d"
        Dynamics_config["network"] = Dynamics_config["network_KARMA"]
        Dynamics_config["model_weights"] = Dynamics_config["model_weights_KARMA_noslip"]## you modified this last night. Results of previous experiments indicate improvement, not so much
        model_weights_path = str(Path(os.getcwd()).parent.absolute()) + "/logs/residual_long_noslip/" + Dynamics_config["model_weights"]
        dynamics = ResidualCarDynamics(Dynamics_config, Map_config, MPPI_config, model_weights_path=model_weights_path)
    elif model == "KARMA_no_change":
        Dynamics_config["type"] = "no_change"
        Dynamics_config["network"] = Dynamics_config["network_KARMA"]
        Dynamics_config["model_weights"] = Dynamics_config["model_weights_KARMA_no_change"]## you modified this last night. Results of previous experiments indicate improvement, not so much
        model_weights_path = str(Path(os.getcwd()).parent.absolute()) + "/logs/residual_long_no_change_consist/" + Dynamics_config["model_weights"]
        dynamics = ResidualCarDynamics(Dynamics_config, Map_config, MPPI_config, model_weights_path=model_weights_path)

    elif model == 'slip3d':
        Dynamics_config["type"] = "slip3d" ## just making sure 
        dynamics = SimpleCarDynamics(Dynamics_config, Map_config, MPPI_config)
    elif model == 'noslip3d' or model == "no_change" or model=="zeros":
        # temporarily change the dynamics type to noslip3d
        Dynamics_config["type"] = "noslip3d"
        dynamics = SimpleCarDynamics(Dynamics_config, Map_config, MPPI_config)
        Dynamics_config["type"] = "slip3d"
    elif model == 'slip3d_bad_sys':
        # temporarily change the dynamics type to noslip3d
        Dynamics_config["type"] = "slip3d"
        temp_D = Dynamics_config["D"]
        Dynamics_config["D"] = 0.4 ## 50 % of the original D
        dynamics = SimpleCarDynamics(Dynamics_config, Map_config, MPPI_config)
        Dynamics_config["D"] = temp_D ## change it back
    elif model == 'slip3d_LPF':
        # temporarily change the dynamics type to noslip3d
        Dynamics_config["type"] = "slip3d"
        temp_LPF = Dynamics_config["LPF_tau"]
        Dynamics_config["LPF_tau"] = 0.5 ## apply a LPF with tau = 0.2
        dynamics = SimpleCarDynamics(Dynamics_config, Map_config, MPPI_config)
        Dynamics_config["LPF_tau"] = temp_LPF ## change it back
    elif model == 'slip3d_LPF_drag':
        # temporarily change the dynamics type to noslip3d
        Dynamics_config["type"] = "slip3d"
        temp_LPF = Dynamics_config["LPF_tau"]
        Dynamics_config["LPF_tau"] = 0.5 ## apply a LPF with tau = 0.2
        temp_drag = Dynamics_config["drag_coeff"]
        temp_res = Dynamics_config["res_coeff"]
        Dynamics_config["drag_coeff"] = 0.00
        Dynamics_config["res_coeff"] = 0.00
        dynamics = SimpleCarDynamics(Dynamics_config, Map_config, MPPI_config)
        Dynamics_config["LPF_tau"] = temp_LPF ## change it back
        Dynamics_config["drag_coeff"] = temp_drag
        Dynamics_config["res_coeff"] = temp_res
    else:
        print("bruh moment")
        raise ValueError('Unknown model type')
    return dynamics

def evaluator(
        data_loader,
        config,
        tn_args,
        ):
        
        Dynamics_config = config["Dynamics_config"]
        MPPI_config = config["MPPI_config"]
        dt = Dynamics_config["dt"]
        dataset_dt = 0.02
        skip = int(dt/dataset_dt) ## please keep the dt a multiple of the dataset_dt
        TIMESTEPS = MPPI_config["TIMESTEPS"]
        np.set_printoptions(threshold=sys.maxsize)
        for model in config['models']:
            count = 0
            dynamics = get_dynamics(model, config)
            errors = np.zeros((len(data_loader), TIMESTEPS, 15))
            for i, (states_tn, controls_tn, ctx_tn_dict) in enumerate(tqdm(data_loader)):

                states_tn = states_tn.to(**tn_args)[:,::skip,:]
                controls_tn = controls_tn.to(**tn_args)[:,::skip,:]
                ctx_tn_dict = {k: tn.to(**tn_args) for k, tn in ctx_tn_dict.items()}
                gt_states = states_tn.clone().cpu().numpy()

                if config["filter"]:
                    vel_condition = np.mean(np.abs(gt_states[0,:,6])) < 5
                    acc_condition = np.max(np.abs(gt_states[0,:,9])) < 5 and np.max(np.abs(gt_states[0,:,10])) < 5
                    rat_condition = np.max(np.abs(gt_states[0,:,12])) < 0.3 and np.max(np.abs(gt_states[0,:,13])) < 0.3 and np.max(np.abs(gt_states[0,:,14])) < 0.3
                    rp_condition = np.max(np.abs(gt_states[0,:,3])) < 0.3 and np.max(np.abs(gt_states[0,:,4])) < 0.3
                    if(vel_condition or acc_condition or rat_condition or rp_condition):
                        count += 1
                        continue

                BEV_heght = ctx_tn_dict["bev_elev"].squeeze(0).squeeze(0)
                BEV_normal = ctx_tn_dict["bev_normal"].squeeze(0).squeeze(0)

                states = torch.zeros(17).to(**tn_args)
                states[:15] = states_tn[0,0,:].clone()
                states = states.repeat(dynamics.M, dynamics.K, dynamics.T, 1)
                controls = controls_tn.repeat((dynamics.K, 1, 1)).clone()
                if model == "no_change":
                    predict_states = states
                elif model == "zeros":
                    predict_states = states*0
                else:
                    dynamics.set_BEV(BEV_heght, BEV_normal)
                    predict_states = dynamics.forward(states, controls)

                pred_states = predict_states[0,0,:,:15].cpu().numpy()

                errors[i,:,:] = (pred_states - gt_states)

                if(np.any(np.isnan(pred_states))):
                    print("NaN error")
                    for time_step in range(TIMESTEPS):
                        position = pred_states[time_step, :3]
                        roll_pitch_yaw = pred_states[time_step, 3:6]  # Assuming roll, pitch, yaw are at indices 3, 4, 5
                        velocity = pred_states[time_step, 6:9]  # Assuming velocity is at indices 6, 7, 8
                        acceleration = pred_states[time_step, 9:12]  # Assuming acceleration is at indices 9, 10, 11
                        gyro = pred_states[time_step, 12:15]  # Assuming gyro is at indices 12, 13, 14

                        # Print out the components
                        print("====== TIMESTEP {}=======".format(time_step))
                        print(f"Position: {position}")
                        print(f"Roll, Pitch, Yaw: {roll_pitch_yaw}")
                        print(f"Velocity: {velocity}")
                        print(f"Acceleration: {acceleration}")
                        print(f"Gyro: {gyro}")
                    exit()

            print(count)
            dir_name = str(Path(os.getcwd()).parent.absolute()) + "/Experiments/Results/Accuracy/" + model
            if(not os.path.isdir(dir_name)):
                os.makedirs(dir_name)
            data_name = "/{}.npy".format(config["dataset"]["name"])
            filename = dir_name + data_name
            np.save(filename, errors)
            if(model == 'slip3d_LPF_drag_residual'):
                np.save(dir_name + "/{}_param.npy".format(config["dataset"]["name"]), np.array(params))
                


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="Evaluation.yaml", help='config file for training model')
    parser.add_argument('--shuffle', type=bool, required=False, default=False, help='shuffle data')
    parser.add_argument('--batchsize', type=int, required=False, default=1, help='training batch size')

    args = parser.parse_args()

    # Set torch params
    torch.manual_seed(0)
    torch.set_num_threads(1)
    
    tensor_args = {'device': torch.device('cuda'), 'dtype': torch.float32}

    # Load experiment config
    config = yaml.load(open( str(Path(os.getcwd()).parent.absolute()) + "/Experiments/Configs/" + args.config).read(), Loader=yaml.SafeLoader)
    # Dataloaders
    train_loader, valid_loader, stats, data_cfg = get_dataloaders(args, config)
    with torch.no_grad():
        evaluator(train_loader, config, tensor_args)