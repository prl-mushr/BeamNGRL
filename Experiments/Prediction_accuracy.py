from BeamNGRL.control.UW_mppi.Dynamics.SimpleCarNetworkDynCUDA import SimpleCarNetworkDyn
from BeamNGRL.control.UW_mppi.Dynamics.SimpleCarDynamicsTCUDA import SimpleCarDynamics
from BeamNGRL.dynamics.utils.exp_utils import get_dataloaders
import torch
import yaml
import os
import argparse
import traceback
import numpy as np
from tqdm import tqdm
from pathlib import Path
import sys
import time

def get_dynamics(model, Config):
    Dynamics_config = Config["Dynamics_config"]
    MPPI_config = Config["MPPI_config"]
    Map_config = Config["Map_config"]
    print("loading: ", model)
    if model != 'slip3d' and model != 'noslip3d':
        Dynamics_config["model_weights"] = Dynamics_config["model_weights_"+model]
        model_weights_path = str(Path(os.getcwd()).parent.absolute()) + "/logs/"+ model + "/" + Dynamics_config["model_weights"]
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

def evaluator(
        data_loader,
        config,
        tn_args,
        ):
        Dynamics_config = config["Dynamics_config"]
        MPPI_config = config["MPPI_config"]
        dynamics_dt = Dynamics_config["dt"]
        dataset_dt = config["dataset_dt"]
        skip = int(dynamics_dt/dataset_dt) ## please keep the dt a multiple of the dataset_dt
        TIMESTEPS = MPPI_config["TIMESTEPS"]
        np.set_printoptions(threshold=sys.maxsize)

        # technically, you should load the dataset stats from here too. Just saying.
        dataset_config = yaml.load(open( str(Path(os.getcwd()).parent.absolute()) + "/data/datasets/" + config["dataset"]["name"] + "/config.yaml" ).read(), Loader=yaml.SafeLoader)
        config["Map_config"] = dataset_config["Map_config"] # copy from dataset config

        for model in config['models']:
            count = 0
            dynamics = get_dynamics(model, config)
            errors = []
            for i, (states_tn, controls_tn, ctx_tn_dict) in enumerate(tqdm(data_loader)):
                states_tn = states_tn.to(**tn_args)[:,::skip,:]
                controls_tn = controls_tn.to(**tn_args)[:,::skip,:]
                ctx_tn_dict = {k: tn.to(**tn_args) for k, tn in ctx_tn_dict.items()}
                gt_states = states_tn.clone().cpu().numpy()

                BEV_heght = ctx_tn_dict["bev_elev"].squeeze(0).squeeze(0)
                BEV_normal = ctx_tn_dict["bev_normal"].squeeze(0).squeeze(0)

                states = torch.zeros(17).to(**tn_args)
                states[:15] = states_tn[0,0,:].clone()
                states = states.repeat(dynamics.M, dynamics.K, dynamics.T, 1)
                controls = controls_tn.repeat((dynamics.K, 1, 1)).clone()
                dynamics.set_BEV(BEV_heght, BEV_normal)

                pred_states = dynamics.forward(states, controls)[0,0,:,:15].cpu().numpy()

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
                else:
                    errors.append((pred_states - gt_states).squeeze(0))

            errors = np.array(errors)
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
    parser.add_argument('--config', type=str, default="Eval_long.yaml", help='config file for training model')
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
        evaluator(valid_loader, config, tensor_args)