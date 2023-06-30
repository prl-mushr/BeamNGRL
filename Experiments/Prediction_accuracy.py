from BeamNGRL.control.UW_mppi.Dynamics.SimpleCarNetworkDyn import SimpleCarNetworkDyn
from BeamNGRL.control.UW_mppi.Dynamics.SimpleCarDynamicsCUDA import SimpleCarDynamics
from BeamNGRL.control.UW_mppi.Dynamics.SimpleCarDynamics import SimpleCarDynamics as SimpleCarDynamicsNS
from BeamNGRL.dynamics.utils.exp_utils import get_dataloaders, build_nets, get_loss_func, init_exp_dir
import torch
import yaml
import os
import argparse
import traceback
import numpy as np
from tqdm import tqdm
from pathlib import Path
## the job of this script is to take ground-truth data for controls and states, run the controls through the dynamics model and compare the predicted states to the ground-truth states

def get_dynamics(model, Config):
    Dynamics_config = Config["Dynamics_config"]
    MPPI_config = Config["MPPI_config"]
    Map_config = Config["Map_config"]
    if model == 'TerrainCNN':
        dynamics = SimpleCarDynamics(Dynamics_config, Map_config, MPPI_config)
    elif model == 'slip3d':
        dynamics = SimpleCarDynamics(Dynamics_config, Map_config, MPPI_config)
    elif model == 'noslip3d':
        dynamics = SimpleCarDynamics(Dynamics_config, Map_config, MPPI_config)
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
        dt = Dynamics_config["dt"]
        dataset_dt = 0.02
        skip = int(dt/dataset_dt) ## please keep the dt a multiple of the dataset_dt
        TIMESTEPS = MPPI_config["TIMESTEPS"]
        if TIMESTEPS != int(50/skip):
            print("dynamics timesteps not equal to dataset timesteps after skipping frames")
            exit()
        for model in config['models']:
            dynamics = get_dynamics(model, config)
            errors = np.zeros((len(data_loader), TIMESTEPS, 15))
            for i, (states_tn, controls_tn, ctx_tn_dict) in enumerate(tqdm(data_loader)):
                # Set device
                states_tn = states_tn.to(**tn_args)[:,::skip,:]
                controls_tn = controls_tn.to(**tn_args)[:,::skip,:]
                ctx_tn_dict = {k: tn.to(**tn_args) for k, tn in ctx_tn_dict.items()}

                BEV_heght = ctx_tn_dict["bev_elev"]
                BEV_normal = ctx_tn_dict["bev_normal"]
                dynamics.set_BEV(BEV_heght, BEV_normal)

                states_tn[0,0,:3] -= states_tn[0, 0, :3].clone() ## subtract off the initial position
                states = torch.zeros(17).to(**tn_args)
                states[:15] = states_tn[0,0,:].clone()
                states = states.repeat(dynamics.M, dynamics.K, dynamics.T, 1)
                controls = controls_tn.repeat((dynamics.K, 1, 1)).clone()
                predict_states = dynamics.forward(states, controls)
                pred_states = predict_states[:,0,:,:15].cpu().numpy()
                gt_states = states_tn.clone().cpu().numpy()
                errors[i,:,:] = (pred_states - gt_states)[0,:,:]
                
                if(np.any(np.isnan(pred_states))):
                    print("NaN error")
                    print(pred_states)
                    exit()
            dir_name = str(Path(os.getcwd()).parent.absolute()) + "/Experiments/Results/Accuracy/" + model
            if(not os.path.isdir(dir_name)):
                os.makedirs(dir_name)
            data_name = "/{}.npy".format(config["dataset"]["name"])
            filename = dir_name + data_name
            np.save(filename, errors)
                
                


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
        evaluator(valid_loader, config, tensor_args)