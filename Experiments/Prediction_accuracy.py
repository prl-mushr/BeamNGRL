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
import cv2

## the job of this script is to take ground-truth data for controls and states, run the controls through the dynamics model and compare the predicted states to the ground-truth states

def get_dynamics(model, Config):
    Dynamics_config = Config["Dynamics_config"]
    MPPI_config = Config["MPPI_config"]
    Map_config = Config["Map_config"]
    print("loading: ", model)
    if model == 'TerrainCNN':
        Dynamics_config["network"] = Dynamics_config["network_baseline"]
        Dynamics_config["model_weights"] = Dynamics_config["model_weights_baseline"]
        model_weights_path = str(Path(os.getcwd()).parent.absolute()) + "/logs/baseline/" + Dynamics_config["model_weights"]
        dynamics = SimpleCarNetworkDyn(Dynamics_config, Map_config, MPPI_config, model_weights_path=model_weights_path)
    elif model == 'TerrainCNN_easy':
        Dynamics_config["network"] = Dynamics_config["network_baseline"]
        Dynamics_config["model_weights"] = Dynamics_config["model_weights_baseline_easy"]
        model_weights_path = str(Path(os.getcwd()).parent.absolute()) + "/logs/baseline_long_easy/" + Dynamics_config["model_weights"]
        dynamics = SimpleCarNetworkDyn(Dynamics_config, Map_config, MPPI_config, model_weights_path=model_weights_path)
    elif model == 'TerrainCNN_hard':
        Dynamics_config["network"] = Dynamics_config["network_baseline"]
        Dynamics_config["model_weights"] = Dynamics_config["model_weights_baseline_hard"]
        model_weights_path = str(Path(os.getcwd()).parent.absolute()) + "/logs/baseline_long_tough/" + Dynamics_config["model_weights"]
        dynamics = SimpleCarNetworkDyn(Dynamics_config, Map_config, MPPI_config, model_weights_path=model_weights_path)
    elif model == "KARMA":
        Dynamics_config["type"] = "slip3d"
        Dynamics_config["network"] = Dynamics_config["network_KARMA"]
        Dynamics_config["model_weights"] = Dynamics_config["model_weights_KARMA"]## you modified this last night. Results of previous experiments indicate improvement, not so much
        model_weights_path = str(Path(os.getcwd()).parent.absolute()) + "/logs/residual/" + Dynamics_config["model_weights"]
        dynamics = ResidualCarDynamics(Dynamics_config, Map_config, MPPI_config, model_weights_path=model_weights_path)
    elif model == "KARMA_hound":
        Dynamics_config["type"] = "slip3d"
        Dynamics_config["network"] = Dynamics_config["network_KARMA"]
        Dynamics_config["model_weights"] = Dynamics_config["model_weights_KARMA_hound"]## you modified this last night. Results of previous experiments indicate improvement, not so much
        model_weights_path = str(Path(os.getcwd()).parent.absolute()) + "/logs/residual_hound/" + Dynamics_config["model_weights"]
        dynamics = ResidualCarDynamics(Dynamics_config, Map_config, MPPI_config, model_weights_path=model_weights_path)
    elif model == "KARMA_hound_noslip":
        Dynamics_config["type"] = "noslip3d"
        Dynamics_config["network"] = Dynamics_config["network_KARMA"]
        Dynamics_config["model_weights"] = Dynamics_config["model_weights_KARMA_hound_noslip"]## you modified this last night. Results of previous experiments indicate improvement, not so much
        model_weights_path = str(Path(os.getcwd()).parent.absolute()) + "/logs/residual_hound_noslip/" + Dynamics_config["model_weights"]
        dynamics = ResidualCarDynamics(Dynamics_config, Map_config, MPPI_config, model_weights_path=model_weights_path)
    elif model == "KARMA_bad_sys":
        Dynamics_config["type"] = "slip3d"
        temp_D = Dynamics_config["D"]
        Dynamics_config["D"] = 0.4 ## 50 % of the original D
        Dynamics_config["network"] = Dynamics_config["network_KARMA"]
        Dynamics_config["model_weights"] = Dynamics_config["model_weights_KARMA_bad_sys"]## you modified this last night. Results of previous experiments indicate improvement, not so much
        model_weights_path = str(Path(os.getcwd()).parent.absolute()) + "/logs/residual_bad_sys/" + Dynamics_config["model_weights"]
        dynamics = ResidualCarDynamics(Dynamics_config, Map_config, MPPI_config, model_weights_path=model_weights_path)
        Dynamics_config["D"] = temp_D ## change it back

    elif model == "KARMA_noslip":
        Dynamics_config["type"] = "noslip3d"
        Dynamics_config["network"] = Dynamics_config["network_KARMA"]
        Dynamics_config["model_weights"] = Dynamics_config["model_weights_KARMA_noslip"]## you modified this last night. Results of previous experiments indicate improvement, not so much
        model_weights_path = str(Path(os.getcwd()).parent.absolute()) + "/logs/residual_noslip/" + Dynamics_config["model_weights"]
        dynamics = ResidualCarDynamics(Dynamics_config, Map_config, MPPI_config, model_weights_path=model_weights_path)
    elif model == "KARMA_no_change":
        Dynamics_config["type"] = "no_change"
        Dynamics_config["network"] = Dynamics_config["network_KARMA"]
        Dynamics_config["model_weights"] = Dynamics_config["model_weights_KARMA_no_change"]## you modified this last night. Results of previous experiments indicate improvement, not so much
        model_weights_path = str(Path(os.getcwd()).parent.absolute()) + "/logs/residual_long_no_change/" + Dynamics_config["model_weights"]
        dynamics = ResidualCarDynamics(Dynamics_config, Map_config, MPPI_config, model_weights_path=model_weights_path)

    elif model == 'slip3d' or model == "gt_rollout":
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

def find_and_plot_centers(small_images):
    # Load the big image
    big_image = cv2.imread('/root/catkin_ws/src/BeamNGRL/data/map_data/color_map.png')
    shape = big_image.shape[0]
    big_image = cv2.resize(big_image, (int(shape*0.4), int(shape*0.4)))
    big_gray = cv2.cvtColor(big_image, cv2.COLOR_BGR2GRAY)

    # Iterate over each small image
    best_matches = []
    for small_image in small_images:
        # small_gray = cv2.cvtColor(small_image, cv2.COLOR_BGR2GRAY)
        # Perform template matching
        result = cv2.matchTemplate(big_image, small_image, cv2.TM_CCOEFF_NORMED)
        
        # Find the location of the best match
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        # Store the center of the best match for the current small image
        match = (max_loc[0] + small_image.shape[1] // 2, max_loc[1] + small_image.shape[0] // 2)
        best_matches.append(match)
    # Plot the centers of the best matches on the big image
    for center in best_matches:
        cv2.circle(big_image, center, radius=5, color=(0, 255, 0), thickness=-1)
    
    cv2.imwrite('/root/catkin_ws/src/BeamNGRL/Experiments/locations.png', big_image)

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

        if config["filter"]:
            print("DATASET_NAME:", config["dataset"]["name"])
            # calculate the std of the states from the dataset
            mean_state = np.zeros(15)
            std_state = np.zeros(15)
            for i, (states_tn, controls_tn, ctx_tn_dict) in enumerate(tqdm(data_loader)):
                mean_state += np.mean(states_tn[0,...].cpu().numpy(), axis=0)
                std_state += np.std(states_tn[0,...].cpu().numpy(), axis=0)
            std_state /= i
            mean_state /= i
            print("Mean state: ", mean_state)
            print("Std state: ", std_state)

            scores_list = []
            for i, (states_tn, controls_tn, ctx_tn_dict) in enumerate(tqdm(data_loader)):
                gt_states = states_tn.clone().cpu().numpy()
                controls = controls_tn.clone().cpu().numpy()

                roll_rate = np.mean(np.abs(gt_states[..., 12]))/std_state[12]
                pitch_rate = np.mean(np.abs(gt_states[..., 13]))/std_state[13]
                yaw_rate = np.mean(np.abs(gt_states[..., 14]))/std_state[14]
                roll = np.mean(np.abs(gt_states[..., 3]))/std_state[3]
                pitch = np.mean(np.abs(gt_states[..., 4]))/std_state[4]
                ax = np.mean(np.abs(gt_states[..., 9]))/std_state[9]
                ay = np.mean(np.abs(gt_states[..., 10]))/std_state[10]
                az = np.mean(np.abs(gt_states[..., 11] - 9.8))/std_state[11]
                vx = np.mean(np.abs(gt_states[..., 6] - controls[..., 1]*20.0))
                vy = np.mean(np.abs(gt_states[..., 7]))/std_state[7]
                # NRMM based score
                score = roll_rate + pitch_rate + roll + pitch #+ yaw_rate
                scores_list.append(score)

            # Find the indices corresponding to the top 100 scores
            length = len(scores_list)
            indices = np.argsort(scores_list)
            indices = np.array(indices)
            np.save('indices.npy', indices)
            indices = np.load('indices.npy')
            indices = indices[-int(length*0.2):]

            # # plotting the locations where this happens:
            # color_list = []
            # for i, (states_tn, controls_tn, ctx_tn_dict) in enumerate(tqdm(data_loader)):
            #     if np.isin(i, indices):
            #         ctx_tn_dict = {k: tn for k, tn in ctx_tn_dict.items()}
            #         color = ctx_tn_dict["bev_color"].squeeze(0).squeeze(0).permute(1,2,0).numpy()
            #         color_list.append(color)

            # find_and_plot_centers(color_list)
            freq_list = []
            for i, (states_tn, controls_tn, ctx_tn_dict) in enumerate(tqdm(data_loader)):
                if i in indices:
                    states_tn = states_tn[0,...].cpu().numpy()
                    # also calculate the fft for each state
                    freq = np.zeros(12)
                    for j in range(3, 15):
                        # subtract mean from value because otherwise you will get strong component at 0 hz
                        fft_result = np.fft.fft(states_tn[..., j] - mean_state[j])
                        # Frequencies corresponding to the FFT result
                        frequencies = np.fft.fftfreq(len(states_tn[..., j]), dataset_dt)
                        # highest amplitude frequency:
                        max_freq = np.abs(frequencies[np.argmax(np.abs(fft_result))])
                        freq[j-3] = max_freq
                    freq_list.append(freq)

            freq_list = np.array(freq_list)
            print("Frequency Analysis:")
            print(freq_list.mean(axis=0))
            print(len(indices))
        else:
            indices = None
        for model in config['models']:
            count = 0
            dynamics = get_dynamics(model, config)
            # errors = np.zeros((len(data_loader), TIMESTEPS, 15))
            errors = []
            for i, (states_tn, controls_tn, ctx_tn_dict) in enumerate(tqdm(data_loader)):
                if indices is not None:
                    if not np.isin(i, indices):
                        continue
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
                if model == "no_change":
                    predict_states = states
                elif model == "zeros":
                    predict_states = states*0
                elif model == "gt_rollout":
                    predict_states = gt_rollout(states_tn, torch.tensor(config["Dynamics_config"]["dt"]).to(**tn_args))
                else:
                    dynamics.set_BEV(BEV_heght, BEV_normal)
                    predict_states = dynamics.forward(states, controls)

                pred_states = predict_states[0,0,:,:15].cpu().numpy()

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
                

def gt_rollout(states, dt):
    states = states.clone().detach()
    t = states.shape[-2]
    cr = torch.cos(states[..., 0, 3])
    sr = torch.sin(states[..., 0, 3])
    cp = torch.cos(states[..., 0, 4])
    sp = torch.sin(states[..., 0, 4])
    cy = torch.cos(states[..., 0, 5])
    sy = torch.sin(states[..., 0, 5])

    for i in range(1, t):
        wx = states[..., i, 12]
        wy = states[..., i, 13]
        wz = states[..., i, 14]
        states[..., i, 3] = states[..., i-1, 3] + dt*( wx*1 + wy*(sr*sp/cp) + wz*(sp*cr/cp) )
        states[..., i, 4] = states[..., i-1, 4] + dt*( wx*0 + wy*cr         + wz*(-sr)      )
        states[..., i, 5] = states[..., i-1, 5] + dt*( wx*0 + wy*(sr/cp)    + wz*(cr/cp)    )
        cr = torch.cos(states[..., i, 3])
        sr = torch.sin(states[..., i, 3])
        cp = torch.cos(states[..., i, 4])
        sp = torch.sin(states[..., i, 4])
        cy = torch.cos(states[..., i, 5])
        sy = torch.sin(states[..., i, 5])
        vx = states[..., i-1, 6]
        vy = states[..., i-1, 7]
        vz = states[..., i-1, 8]

        # ## using the same rotation matrix, just do G*R where G is [0,0,9.8] and then subtract.
        # ## Details in "Vehicle Models and Optimal Control on a Nonplanar Surface"
        ax = states[..., i, 9]  + vy*wz - vz*wy + sp*9.8
        ay = states[..., i, 10] - vx*wz + vz*wx - sr*cp*9.8
        az = states[..., i, 11] + vx*wy - vy*wx - cp*cr*9.8

        states[..., i, 6] = states[..., i-1, 6] + ax * dt;
        states[..., i, 7] = states[..., i-1, 7] + ay * dt;
        states[..., i, 8] = states[..., i-1, 8] + az * dt;

        states[..., i, 0] = states[..., i-1, 0] + dt*( states[..., i, 6]*cp*cy + states[..., i, 7]*(sr*sp*cy - cr*sy) + states[..., i, 8]*(cr*sp*cy + sr*sy) )
        states[..., i, 1] = states[..., i-1, 1] + dt*( states[..., i, 6]*cp*sy + states[..., i, 7]*(sr*sp*sy + cr*cy) + states[..., i, 8]*(cr*sp*sy - sr*cy) )
        states[..., i, 2] = states[..., i-1, 2] + dt*( states[..., i, 6]*(-sp) + states[..., i, 7]*(sr*cp)            + states[..., i, 8]*(cr*cp)            )
        


    return states.unsqueeze(0)


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