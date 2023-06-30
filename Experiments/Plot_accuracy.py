import numpy as np
import matplotlib.pyplot as plt
import yaml
from pathlib import Path
import os
import argparse

def create_error_plot(errors, timesteps, model, ax):
    mean = np.mean(np.mean(np.abs(errors), axis=0), axis=1)
    std = np.mean(np.std(np.abs(errors), axis=0), axis=1)
    ax.plot(np.arange(0, timesteps), mean, label=model)
    ax.fill_between(np.arange(0, timesteps), mean - std, mean + std, alpha=0.2)

def plot_accuracy(config):
    pos = slice(0,3)
    rp = slice(3,5)
    yaw = slice(5,6)
    vel = slice(6,9)
    drp = slice(12,14)
    dyaw = slice(14,15)

    fig = plt.figure()
    fig.suptitle("Error vs Timestep on {} dataset with dt {} seconds".format(config["dataset"]["name"], config["Dynamics_config"]["dt"]))
    # create 4 subplots for each of the error types
    ax1 = fig.add_subplot(2,3,1)
    ax1.set_title("Position (m)")
    ax1.xaxis.set_label_text("Timesteps")
    ax2 = fig.add_subplot(2,3,2)
    ax2.set_title("Roll-Pitch (rad)")
    ax2.xaxis.set_label_text("Timesteps")
    ax3 = fig.add_subplot(2,3,3)
    ax3.set_title("Velocity (m/s)")
    ax3.xaxis.set_label_text("Timesteps")
    ax4 = fig.add_subplot(2,3,4)
    ax4.set_title("Roll-Pitch rate (rad/s)")
    ax4.xaxis.set_label_text("Timesteps")
    ax5 = fig.add_subplot(2,3,5)
    ax5.set_title("Yaw (rad)")
    ax5.xaxis.set_label_text("Timesteps")
    ax6 = fig.add_subplot(2,3,6)
    ax6.set_title("Yaw rate (rad/s)")
    ax6.xaxis.set_label_text("Timesteps")
    
    timesteps = config["MPPI_config"]["TIMESTEPS"]

    for model in config["models"]:
        data = str(Path(os.getcwd()).parent.absolute()) + "/Experiments/Results/Accuracy/" + model
        errors = np.load(data + "/{}.npy".format(config["dataset"]["name"]))
        create_error_plot(errors[:,:,pos], timesteps, model, ax1)
        create_error_plot(errors[:,:,rp], timesteps, model, ax2)
        create_error_plot(errors[:,:,vel], timesteps, model, ax3)
        create_error_plot(errors[:,:,drp], timesteps, model, ax4)
        ## need to warp the yaw errors between -pi and pi. yaw error is on position 5
        errors[:,:,yaw] = np.arctan2(np.sin(errors[:,:,yaw]), np.cos(errors[:,:,yaw]))
        create_error_plot(errors[:,:,yaw], timesteps, model, ax5)
        create_error_plot(errors[:,:,dyaw], timesteps, model, ax6)

    ax1.legend()
    ax2.legend()
    ax3.legend()
    ax4.legend()
    ax5.legend()
    ax6.legend()

    plt.show()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="Evaluation.yaml", help='config file for training model')
    parser.add_argument('--shuffle', type=bool, required=False, default=False, help='shuffle data')
    parser.add_argument('--batchsize', type=int, required=False, default=1, help='training batch size')

    args = parser.parse_args()
    config = yaml.load(open( str(Path(os.getcwd()).parent.absolute()) + "/Experiments/Configs/" + args.config).read(), Loader=yaml.SafeLoader)
    plot_accuracy(config)
