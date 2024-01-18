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
from Plot_accuracy import *

## the job of this script is to take ground-truth data for controls and states, run the controls through the dynamics model and compare the predicted states to the ground-truth states

def get_dynamics(Config):
    Dynamics_config = Config["Dynamics_config"]
    MPPI_config = Config["MPPI_config"]
    Map_config = Config["Map_config"]
    Dynamics_config["type"] = "slip3d"
    Dynamics_config["network"] = Dynamics_config["network_KARMA"]
    Dynamics_config["model_weights"] = Dynamics_config["model_weights_KARMA"]
    model_weights_path = str(Path(os.getcwd()).parent.absolute()) + "/logs/residual_test/" + Dynamics_config["model_weights"]
    dynamics = ResidualCarDynamics(Dynamics_config, Map_config, MPPI_config, model_weights_path=model_weights_path)

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
        errors = np.zeros((len(data_loader), TIMESTEPS, 15))
        params = []

        # time.sleep(1)
        # dynamics.setup("static")
        # for i, (states_tn, controls_tn, ctx_tn_dict) in enumerate(tqdm(data_loader)):
        #     states_tn = states_tn.to(**tn_args)[:,::skip,:]
        #     controls_tn = controls_tn.to(**tn_args)[:,::skip,:]
        #     ctx_tn_dict = {k: tn.to(**tn_args) for k, tn in ctx_tn_dict.items()}
        #     gt_states = states_tn.clone().cpu().numpy()
        #     BEV_heght = ctx_tn_dict["bev_elev"].squeeze(0).squeeze(0)
        #     BEV_normal = ctx_tn_dict["bev_normal"].squeeze(0).squeeze(0)

        #     states = torch.zeros(17).to(**tn_args)
        #     states[:15] = states_tn[0,0,:].clone()
        #     states = states.repeat(dynamics.M, dynamics.K, dynamics.T, 1)
        #     controls = controls_tn.repeat((dynamics.K, 1, 1)).clone()
        #     params.append(dynamics.PARAM.cpu().numpy())
        #     if i%50 == 0:
        #         print(np.round(dynamics.PARAM.cpu().numpy(),3))
        #     predict_states = dynamics.forward_static(states, controls, BEV_heght, BEV_normal, states_tn[0,:,:].clone().detach(), controls[0,:,:].clone().detach())
        #     pred_states = predict_states[0,0,:,:15].cpu().numpy()
            
        #     errors[i,:,:] = (pred_states - gt_states)

        #     if(np.any(np.isnan(pred_states))):
        #         print("NaN error")
        #         # print(pred_states)
        #         exit()

        # dynamics.setup("dynamic")
        for i, (states_tn, controls_tn, ctx_tn_dict) in enumerate(tqdm(data_loader)):
            states_tn = states_tn.to(**tn_args)[:,::skip,:]
            controls_tn = controls_tn.to(**tn_args)[:,::skip,:]
            ctx_tn_dict = {k: tn.to(**tn_args) for k, tn in ctx_tn_dict.items()}
            gt_states = states_tn.clone().cpu().numpy()

            if config["filter"]:
                vel_condition = np.min(gt_states[0,:,6]) < 3
                acc_condition = np.min(np.abs(gt_states[0,:,9])) < 1 and np.min(np.abs(gt_states[0,:,10])) < 1
                rat_condition = np.min(np.abs(gt_states[0,:,12])) < 0.05 and np.min(np.abs(gt_states[0,:,13])) < 0.05
                rp_condition = np.min(np.abs(gt_states[0,:,3])) < 0.05 and np.min(np.abs(gt_states[0,:,4])) < 0.05
                if(vel_condition or acc_condition):
                    count += 1
                    continue
            BEV_heght = ctx_tn_dict["bev_elev"].squeeze(0).squeeze(0)
            BEV_normal = ctx_tn_dict["bev_normal"].squeeze(0).squeeze(0)

            states = torch.zeros(17).to(**tn_args)
            states[:15] = states_tn[0,0,:].clone()
            states = states.repeat(dynamics.M, dynamics.K, dynamics.T, 1)
            controls = controls_tn.repeat((dynamics.K, 1, 1)).clone()
            dynamics.set_BEV(BEV_heght, BEV_normal)
            predict_states = dynamics.forward(states, controls)

            pred_states = predict_states[0,0,:,:15].cpu().numpy()

            errors[i,:,:] = (pred_states - gt_states)

            if(np.any(np.isnan(pred_states))):
                print("NaN error")
                # print(pred_states)
                exit()
        ## this will only work with batch size of 1 for now.
        # network = dynamics.dyn_model    
        # for i, (states_tn, controls_tn, ctx_tn_dict) in enumerate(tqdm(data_loader)):
        #     BEV_heght = ctx_tn_dict["bev_elev"].squeeze(1)
        #     BEV_normal = ctx_tn_dict["bev_normal"].squeeze(1)

        #     states = torch.zeros(1,states_tn.shape[0], states_tn.shape[1], 17).to(**tn_args)
        #     states[0,:,0,:15] = states_tn[:,0,:].clone().detach()
        #     controls = controls_tn.clone().detach().unsqueeze(0)
        #     predict_states = dynamics.forward_train(states, controls, BEV_heght, BEV_normal)
        #     while torch.any(torch.isnan(predict_states)):
        #         predict_states = dynamics.forward_train(states, controls, BEV_heght, BEV_normal, print_something="fixing nans")
        #     ctx_data={'rotate_crop': dynamics.bev_input_train}
        #     states_input = predict_states[...,:15].squeeze(0)
        #     states_input = rotate_traj(states_input)
        #     pred = dynamics.dyn_model(
        #         states_input,
        #         controls_tn,
        #         ctx_data,
        #     )
        #     pred_states = pred[0,0,:,:15].cpu().numpy()
            
        #     errors[i,:,:] = (pred_states - gt_states)

        #     if(np.any(np.isnan(pred_states))):
        #         print("NaN error")
        #         print("pred_states:", pred_states)
        #         exit()

        print(dynamics.dt_avg*1e3)
       
        dir_name = str(Path(os.getcwd()).parent.absolute()) + "/Experiments/Results/Accuracy/" + "KARMA"
        if(not os.path.isdir(dir_name)):
            os.makedirs(dir_name)
        data_name = "/{}.npy".format(config["dataset"]["name"])
        filename = dir_name + data_name
        np.save(filename, errors)
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
        evaluator(valid_loader, config, tensor_args)

    plot_accuracy(config)