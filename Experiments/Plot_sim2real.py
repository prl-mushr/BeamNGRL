import numpy as np
import matplotlib.pyplot as plt
import yaml
from pathlib import Path
import os
import argparse
import seaborn as sns
from scipy import signal
import torch
from BeamNGRL.control.UW_mppi.Dynamics.SimpleCarNetworkDyn import SimpleCarNetworkDyn
from BeamNGRL.control.UW_mppi.Dynamics.SimpleCarDynamicsCUDA import SimpleCarDynamics
import sys

def get_dynamics(model, Config):
    Dynamics_config = Config["Dynamics_config"]
    MPPI_config = Config["MPPI_config"]
    Map_config = Config["Map_config"]
    if model == 'slip3d':
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

def evaluator(config, real, tn_args):
    Dynamics_config = config["Dynamics_config"]
    MPPI_config = config["MPPI_config"]
    dt = Dynamics_config["dt"]
    dataset_dt = 0.02
    np.set_printoptions(threshold=sys.maxsize)
    real_data = torch.from_numpy(real)
    for model in config['models']:
        dynamics = get_dynamics(model, config)
        errors = np.zeros((len(real_data), 15))
        predict_states = np.zeros_like(errors)
        predict_states[0,:] = real[0,:15]
        for i in range(0,len(real_data),MPPI_config["TIMESTEPS"]):
            states_tn = real_data[i:i+MPPI_config["TIMESTEPS"],:15].to(**tn_args)
            controls_tn = real_data[i:i+MPPI_config["TIMESTEPS"],17:19].to(**tn_args)
            gt_states = states_tn.clone().cpu().numpy()
            BEV_heght = torch.zeros((64, 64))
            BEV_normal = torch.zeros((64,64,3))
            dynamics.set_BEV(BEV_heght, BEV_normal)

            states = torch.zeros(17).to(**tn_args)
            states[:15] = torch.from_numpy(predict_states[i,:]).to(**tn_args)
            states = states.repeat(dynamics.M, dynamics.K, dynamics.T, 1)
            controls = controls_tn.repeat((dynamics.K, 1, 1)).clone()
            pred_states = dynamics.forward(states, controls)[0,0,:,:15].cpu().numpy()
            
            errors[i:i+MPPI_config["TIMESTEPS"],:] = (pred_states - gt_states)
            predict_states[i:i+MPPI_config["TIMESTEPS"],:] = pred_states
            if(np.any(np.isnan(pred_states))):
                print("NaN error")
                print(pred_states)
                exit()
        dir_name = str(Path(os.getcwd()).parent.absolute()) + "/Experiments/Results/Sim2Real/"
        if(not os.path.isdir(dir_name)):
            os.makedirs(dir_name)
        data_name = "/{}.npy".format(model)
        filename = dir_name + data_name
        np.save(filename, predict_states)
        data_name = "/err_{}.npy".format(model)
        filename = dir_name + data_name
        np.save(filename, errors)

def conf(data):
	return np.fabs(np.percentile(data,97.5) - np.percentile(data,2.5))/2.0


def Plot_metrics(Config):

	order = 2  # Filter order
	cutoff_freq = 0.04  # Cutoff frequency as a fraction of Nyquist frequency
	b, a = signal.butter(order, cutoff_freq, btype='low', analog=False)
	bag_length = 25

	tensor_args = {'device': torch.device('cuda'), 'dtype': torch.float32}
	dir_name = str(Path(os.getcwd()).parent.absolute()) + "/Experiments/Results/Sim2Real/"
	real_0 = np.load(dir_name + 'bag_20.npy')[:50*50,:]
	real_1 = np.load(dir_name + 'bag_21.npy')[:25*50,:]
	real_2 = np.load(dir_name + 'bag_23.npy')[:24*50,:]
	BeamNG_0 = np.load(dir_name + 'BeamNG_20.npy')[:50*50,:]
	BeamNG_1 = np.load(dir_name + 'BeamNG_21.npy')[:25*50,:]
	BeamNG_2 = np.load(dir_name + 'BeamNG_23.npy')[:24*50,:]
	real = np.concatenate((real_0, real_1, real_2), axis=0)
	BeamNG = np.concatenate((BeamNG_0, BeamNG_1, BeamNG_2), axis=0)
	np.save(dir_name + 'BeamNG.npy', BeamNG)
	# evaluator(Config, real, tensor_args)

	acc = real[:,9:12]
	gyro= real[:,12:15]
	vel = real[:,6:9]
	max_acc = np.max(acc)
	max_gyro= np.max(gyro)
	max_vel = np.max(vel)

	i = 0
	fig = plt.figure()
	fig.set_size_inches(5, 5)
	X = np.arange(3)
	methods = ["BeamNG", "slip3d", "noslip3d"]
	metrics = ["Acceleration", "Rotation rate", "Velocity"]
	bar_width = 0.3
	for data in methods:
		error = np.load(dir_name + '{}.npy'.format(data))
		acc_err = np.linalg.norm(signal.lfilter(b, a, error[:,9:12], axis=0), axis=1)/max_acc
		rot_err = np.linalg.norm(signal.lfilter(b, a, error[:,12:15], axis=0), axis=1)/max_gyro
		vel_err = np.linalg.norm(signal.lfilter(b, a, error[:,6:9], axis=0), axis=1)/max_vel
		color = plt.cm.tab10(i)  # Choose the same color from the 'tab10' colormap
		plt.bar(0 + bar_width*(1 - i), acc_err.mean(), yerr=conf(acc_err), width=bar_width, alpha=0.5, ecolor='black', capsize=10, color=color)
		plt.bar(1 + bar_width*(1 - i), rot_err.mean(), yerr=conf(rot_err), width=bar_width, alpha=0.5, ecolor='black', capsize=10, color=color)
		plt.bar(2 + bar_width*(1 - i), vel_err.mean(), yerr=conf(acc_err), width=bar_width, alpha=0.5, ecolor='black', capsize=10, label=data, color=color)
		i += 1

	plt.ylabel("normalized error")
	plt.xticks(X, metrics)
	plt.grid(True, linestyle='--', alpha=0.7)
	plt.legend()
	plt.show()
	plt.savefig(str(Path(os.getcwd()).parent.absolute()) + "/Experiments/Results/Sim2Real/sim2real_comp.png")


if __name__ == "__main__":
    ## add a parser:
    parser = argparse.ArgumentParser(description="Plot the accuracy of the models")
    parser.add_argument("--config_name", "-c", default="sim2real_Config.yaml", type=str, help="Path to the config file. Keep the same as the one used for evaluation")

    args = parser.parse_args()
    config_name = args.config_name
    config_path = str(Path(os.getcwd()).parent.absolute()) + "/Experiments/Configs/" + config_name
    with open(config_path, "r") as f: 
        Config = yaml.safe_load(f)
    ## call the plotting function, we'll extract data in there.
    Plot_metrics(Config)