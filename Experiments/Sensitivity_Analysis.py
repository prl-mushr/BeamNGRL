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

def evaluator(
        data_loader,
        config,
        tn_args,
        ):
        global gt_rollout
        gt_rollout = torch.jit.script(gt_rollout)
        Dynamics_config = config["Dynamics_config"]
        MPPI_config = config["MPPI_config"]
        dt = Dynamics_config["dt"]
        dataset_dt = 0.02
        skip = int(dt/dataset_dt) ## please keep the dt a multiple of the dataset_dt
        TIMESTEPS = MPPI_config["TIMESTEPS"]
        np.set_printoptions(threshold=sys.maxsize)
        position_errors = []
        rpy_errors = []
        velocity_errors = []
        acceleration_errors = []
        rotation_rate_errors = []
        std = np.array(Dynamics_config["network_KARMA"]["net_kwargs"]["std_state"])
        
        for k in range(2,11):
            errors = []
            noise = torch.zeros(15).to(**tn_args)
            if k < 2:
                noise[k+3] = 0.1*torch.rand(1)*std[k+3]
            if k >= 5 and k < 8:
                continue
            else:
                noise[k+4] = 0.2*torch.rand(1)*std[k+4]
            for i, (states_tn, controls_tn, ctx_tn_dict) in enumerate(tqdm(data_loader)):
                states_tn = states_tn.to(**tn_args)[:,::skip,:]
                gt_states = states_tn.clone().cpu().numpy()

                predict_states = gt_rollout(states_tn, noise)

                pred_states = predict_states[0,0,:,:15].cpu().numpy()
                errors.append((pred_states - gt_states).squeeze(0))

            errors = np.array(errors)
            position_errors.append(abs(errors[..., :3].mean()))
            rpy_errors.append(abs(errors[..., 3:6].mean()))
            velocity_errors.append(abs(errors[..., 6:9].mean()))
            acceleration_errors.append(abs(errors[..., 9:12].mean()))
            rotation_rate_errors.append(abs(errors[..., 12:15].mean()))

        print("Position errors: ", position_errors)
        print("Velocity errors: ", velocity_errors)
        # normalize errors by their standard deviations
        normalized_position_errors = position_errors / std[:3].mean()
        normalized_acceleration_errors = acceleration_errors / std[6:9].mean()
        normalized_velocity_errors = velocity_errors / std[6:9].mean()
        normalized_weights = normalized_position_errors + normalized_acceleration_errors
        loss_weights = 1/normalized_weights
        loss_weights /= loss_weights.sum()
        loss_weights[:3] *= std[9:12]
        loss_weights[3:] *= std[12:15]
        print("loss weights: ", np.round(loss_weights, 3))



def gt_rollout(states, noise):
    states = states.clone().detach()
    t = states.shape[-2]
    cr = torch.cos(states[..., 0, 3])
    sr = torch.sin(states[..., 0, 3])
    cp = torch.cos(states[..., 0, 4])
    sp = torch.sin(states[..., 0, 4])
    cy = torch.cos(states[..., 0, 5])
    sy = torch.sin(states[..., 0, 5])

    dt = 0.04
    dV = torch.zeros_like(states[..., 6:9])
    dV[..., :-1, :] = torch.diff(states[..., 6:9], dim=-2)/dt
    dV[..., -1,:] = dV[..., -2,:]
    for i in range(1, t):
        states[..., i, :] += noise
        wx = states[..., i, 12]
        wy = states[..., i, 13]
        wz = states[..., i, 14]
        # states[..., i, 3] = states[..., i-1, 3] + dt*( wx*1 + wy*(sr*sp/cp) + wz*(sp*cr/cp) )
        # states[..., i, 4] = states[..., i-1, 4] + dt*( wx*0 + wy*cr         + wz*(-sr)      )
        states[..., i, 5] = states[..., i-1, 5] + dt*( wx*0 + wy*(sr/cp)    + wz*(cr/cp)    )
        cr = torch.cos(states[..., i, 3])
        sr = torch.sin(states[..., i, 3])
        cp = torch.cos(states[..., i, 4])
        sp = torch.sin(states[..., i, 4])
        cy = torch.cos(states[..., i, 5])
        sy = torch.sin(states[..., i, 5])

        states[..., 9]  = dV[..., 0] - (states[..., i, 7]*states[..., i, 14] - states[..., i, 8]*states[..., i, 13] + sp*9.81)
        states[..., 10] = dV[..., 1] - (-states[..., i, 6]*states[..., i, 14] + states[..., i, 8]*states[..., i, 12] - sr*cp*9.81)
        states[..., 11] = dV[..., 2] - (states[..., i, 6]*states[..., i, 13] - states[..., i, 7]*states[..., i, 12] - cp*cr*9.81)

        # vx = states[..., i-1, 6]
        # vy = states[..., i-1, 7]
        # vz = states[..., i-1, 8]

        # ## using the same rotation matrix, just do G*R where G is [0,0,9.8] and then subtract.
        # ## Details in "Vehicle Models and Optimal Control on a Nonplanar Surface"
        # ax = states[..., i, 9]  + vy*wz - vz*wy + sp*9.8
        # ay = states[..., i, 10] - vx*wz + vz*wx - sr*cp*9.8
        # az = states[..., i, 11] + vx*wy - vy*wx - cp*cr*9.8

        # states[..., i, 6] = states[..., i-1, 6] + ax * dt;
        # states[..., i, 7] = states[..., i-1, 7] + ay * dt;
        # states[..., i, 8] = states[..., i-1, 8] + az * dt;

        states[..., i, 0] = states[..., i-1, 0] + dt*( states[..., i, 6]*cp*cy + states[..., i, 7]*(sr*sp*cy - cr*sy) + states[..., i, 8]*(cr*sp*cy + sr*sy) )
        states[..., i, 1] = states[..., i-1, 1] + dt*( states[..., i, 6]*cp*sy + states[..., i, 7]*(sr*sp*sy + cr*cy) + states[..., i, 8]*(cr*sp*sy - sr*cy) )
        states[..., i, 2] = states[..., i-1, 2] + dt*( states[..., i, 6]*(-sp) + states[..., i, 7]*(sr*cp)            + states[..., i, 8]*(cr*cp)            )

    return states.unsqueeze(0)


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