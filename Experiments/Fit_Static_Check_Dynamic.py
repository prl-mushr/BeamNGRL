from BeamNGRL.control.UW_mppi.Dynamics.ResidualCarDynamicsSysID import ResidualCarDynamics
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
from Plot_accuracy import *
import torch.nn.functional as F
## the job of this script is to take ground-truth data for controls and states, run the controls through the dynamics model and compare the predicted states to the ground-truth states

def get_dynamics(Config):
    Dynamics_config = Config["Dynamics_config"]
    MPPI_config = Config["MPPI_config"]
    Map_config = Config["Map_config"]
    Dynamics_config["type"] = "slip3d"
    # Dynamics_config["LPF_tau"] = 0.8 ## apply a LPF with tau = 0.2
    # Dynamics_config["drag_coeff"] = 0.00
    # Dynamics_config["res_coeff"] = 0.00
    # Dynamics_config["D"] = 1.5
    # Dynamics_config["Iz"] = 2.0
    dynamics = ResidualCarDynamics(Dynamics_config, Map_config, MPPI_config)

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
        count = 0
        np.set_printoptions(threshold=sys.maxsize)
        dynamics = get_dynamics(config)
        params = []

        if 1:
            dynamics.traj_num_static = 32
            dynamics.traj_num_dyn = 32
            dynamics.static_decay_rate = 1.0

            for j in range(6):
                if j%2 == 0:
                    dynamics.setup("dynamic")
                else:
                    dynamics.setup("static")
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
                    params.append(dynamics.PARAM.cpu().numpy())
                    if i%200 == 0:
                        print(np.round(dynamics.PARAM.cpu().numpy(),3))

                    dynamics.set_BEV(BEV_heght, BEV_normal)
                    dynamics.set_GT(states_tn[0,:,:].clone(), controls[0,:,:].clone())
                    predict_states = dynamics.forward(states, controls)
                    pred_states = predict_states[0,0,:,:15].cpu().numpy()

                    if(np.any(np.isnan(pred_states))):
                        print("NaN error")
                        # print(pred_states)
                        # exit()
                        pass
                    else:
                        errors.append((pred_states - gt_states).squeeze(0))

            errors = []
            dynamics.setup("None")
            dynamics.LPF_tau *= 0.1
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
                params.append(dynamics.PARAM.cpu().numpy())
                # if i%200 == 0:
                #     print(np.round(dynamics.PARAM.cpu().numpy(),3))
                dynamics.set_BEV(BEV_heght, BEV_normal)
                dynamics.set_GT(states_tn[0,:,:].clone(), controls[0,:,:].clone())
                predict_states = dynamics.forward(states, controls)
                pred_states = predict_states[0,0,:,:15].cpu().numpy()

                if(np.any(np.isnan(pred_states))):
                    print("NaN error")
                    # print(pred_states)
                    # exit()
                    pass
                else:
                    errors.append((pred_states - gt_states).squeeze(0))

            print(dynamics.dt_avg*1e3)
            errors = np.array(errors)
            dir_name = str(Path(os.getcwd()).parent.absolute()) + "/Experiments/Results/Accuracy/" + "SysID"
            if(not os.path.isdir(dir_name)):
                os.makedirs(dir_name)
            data_name = "/{}.npy".format(config["dataset"]["name"])
            filename = dir_name + data_name
            np.save(filename, errors)
            np.save(dir_name + "/{}_param.npy".format(config["dataset"]["name"]), np.array(params))
             
        errors = []
        count = 0
        dynamics = get_dynamics(config) # reset params
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
            dynamics.set_GT(states_tn[0,:,:].clone(), controls[0,:,:].clone())
            predict_states = dynamics.forward_fixed(states, controls)
            pred_states = predict_states[0,0,:,:15].cpu().numpy()

            if(np.any(np.isnan(pred_states))):
                print("NaN error")
                # print(pred_states)
                # exit()
                pass
            else:
                errors.append((pred_states - gt_states).squeeze(0))
        print(count)
        print(dynamics.dt_avg*1e3)
        errors = np.array(errors)
        dir_name = str(Path(os.getcwd()).parent.absolute()) + "/Experiments/Results/Accuracy/" + "slip3d"
        if(not os.path.isdir(dir_name)):
            os.makedirs(dir_name)
        data_name = "/{}.npy".format(config["dataset"]["name"])
        filename = dir_name + data_name
        np.save(filename, errors)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="Evaluation_SysID_hound.yaml", help='config file for training model')
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

    plot_accuracy(config)