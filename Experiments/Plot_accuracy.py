import numpy as np
import matplotlib.pyplot as plt
import yaml
from pathlib import Path
import os
import argparse

def remove_outliers(data, threshold=3):
    # Calculate the z-scores for each data point
    z_scores = (data - np.mean(data, axis=1, keepdims=True)) / np.std(data, axis=1, keepdims=True)

    # Find the indices of outliers
    outlier_indices = np.where(np.abs(z_scores) > threshold)

    # Replace outlier values with NaN
    data[outlier_indices] = np.nan

    # Remove rows with NaN values
    data = data[~np.isnan(data).any(axis=1)]

    return data

def create_error_plot(errors, timesteps, model, ax, plot='mean'):
    plot_error = np.linalg.norm(errors, axis=2)
    # plot_error = remove_outliers(plot_error)
    mean = np.mean(plot_error, axis=0)
    std = np.std(plot_error, axis=0)
    # calculate the RMSE:
    rmse = np.sqrt(np.mean(plot_error**2, axis=0))
    if plot == 'mean':
        ax.plot(np.arange(0, timesteps), mean, label=model)
    elif plot == 'std':
        ax.plot(np.arange(0, timesteps), std, label=model)
    elif plot == 'both':
        ax.plot(np.arange(0, timesteps), mean, label=model)
        ax.fill_between(np.arange(0, timesteps), mean - std, mean + std, alpha=0.2)
    elif plot == 'rmse':
        ax.plot(np.arange(0, timesteps), rmse, label=model)
    else:
        raise ValueError("Unknown plot type")

def plot_accuracy(config):
    pos = slice(0,2)
    rp = slice(3,5)
    yaw = slice(5,6)
    vel = slice(6,8)
    drp = slice(12,14)
    dyaw = slice(14,15)

    fig = plt.figure()
    fig.suptitle("Error vs Timestep on {} dataset with dt {} seconds".format(config["dataset"]["name"], config["Dynamics_config"]["dt"]))
    # create 4 subplots for each of the error types
    ax1 = fig.add_subplot(2,5,1)
    ax1.set_title("Position")
    ax1.xaxis.set_label_text("Timesteps")
    ax2 = fig.add_subplot(2,5,2)
    ax2.set_title("Roll-Pitch")
    ax2.xaxis.set_label_text("Timesteps")
    ax3 = fig.add_subplot(2,5,3)
    ax3.set_title("Velocity")
    ax3.xaxis.set_label_text("Timesteps")
    ax4 = fig.add_subplot(2,5,4)
    ax4.set_title("Roll-Pitch rate")
    ax4.xaxis.set_label_text("Timesteps")
    ax5 = fig.add_subplot(2,5,5)
    ax5.set_title("Yaw")
    ax5.xaxis.set_label_text("Timesteps")
    ax6 = fig.add_subplot(2,5,6)
    ax6.set_title("Yaw rate")
    ax6.xaxis.set_label_text("Timesteps")
    
    ax7 = fig.add_subplot(2,5,7)
    ax7.set_title("accel X")
    ax7.xaxis.set_label_text("Timesteps")
    ax8 = fig.add_subplot(2,5,8)
    ax8.set_title("accel Y")
    ax8.xaxis.set_label_text("Timesteps")
    ax9 = fig.add_subplot(2,5,9)
    ax9.set_title("accel Z")
    ax9.xaxis.set_label_text("Timesteps")


    timesteps = config["MPPI_config"]["TIMESTEPS"]

    for model in config["models"]:
        data = str(Path(os.getcwd()).parent.absolute()) + "/Experiments/Results/Accuracy/" + model
        errors = np.load(data + "/{}.npy".format(config["dataset"]["name"]))
        create_error_plot(errors[:,:,pos], timesteps, model, ax1, plot='both')
        create_error_plot(errors[:,:,rp], timesteps, model, ax2, plot='both')
        create_error_plot(errors[:,:,vel], timesteps, model, ax3, plot='both')
        create_error_plot(errors[:,:,drp], timesteps, model, ax4, plot='both')
        ## need to warp the yaw errors between -pi and pi. yaw error is on position 5
        errors[:,:,yaw] = np.arctan2(np.sin(errors[:,:,yaw]), np.cos(errors[:,:,yaw]))
        create_error_plot(errors[:,:,yaw], timesteps, model, ax5, plot='both')
        create_error_plot(errors[:,:,dyaw], timesteps, model, ax6, plot='both')
        create_error_plot(errors[:,:,[9]], timesteps, model, ax7, plot='both')
        create_error_plot(errors[:,:,[10]], timesteps, model, ax8, plot='both')
        create_error_plot(errors[:,:,[11]], timesteps, model, ax9, plot='both')

    ax1.legend()
    ax2.legend()
    ax3.legend()
    ax4.legend()
    ax5.legend()
    ax6.legend()
    ax7.legend()
    ax8.legend()
    ax9.legend()

    plt.show()

    # fig = plt.figure()
    # fig.suptitle("Error vs Timestep on {} dataset with dt {} seconds".format(config["dataset"]["name"], config["Dynamics_config"]["dt"]))
    # # create 4 subplots for each of the error types
    # ax1 = fig.add_subplot(2,5,1)
    # ax1.set_title("Position")
    # ax1.xaxis.set_label_text("Timesteps")
    # ax2 = fig.add_subplot(2,5,2)
    # ax2.set_title("Roll-Pitch")
    # ax2.xaxis.set_label_text("Timesteps")
    # ax3 = fig.add_subplot(2,5,3)
    # ax3.set_title("Velocity")
    # ax3.xaxis.set_label_text("Timesteps")
    # ax4 = fig.add_subplot(2,5,4)
    # ax4.set_title("Roll-Pitch rate")
    # ax4.xaxis.set_label_text("Timesteps")
    # ax5 = fig.add_subplot(2,5,5)
    # ax5.set_title("Yaw")
    # ax5.xaxis.set_label_text("Timesteps")
    # ax6 = fig.add_subplot(2,5,6)
    # ax6.set_title("Yaw rate")
    # ax6.xaxis.set_label_text("Timesteps")
    
    # ax7 = fig.add_subplot(2,5,7)
    # ax7.set_title("accel X")
    # ax7.xaxis.set_label_text("Timesteps")
    # ax8 = fig.add_subplot(2,5,8)
    # ax8.set_title("accel Y")
    # ax8.xaxis.set_label_text("Timesteps")
    # ax9 = fig.add_subplot(2,5,9)
    # ax9.set_title("accel Z")
    # ax9.xaxis.set_label_text("Timesteps")
    # for model in config["models"]:
    #     data = str(Path(os.getcwd()).parent.absolute()) + "/Experiments/Results/Accuracy/" + model
    #     errors = np.load(data + "/{}.npy".format(config["dataset"]["name"]))
    #     create_error_plot(errors[:,:,pos], timesteps, model, ax1, plot='std')
    #     create_error_plot(errors[:,:,rp], timesteps, model, ax2, plot='std')
    #     create_error_plot(errors[:,:,vel], timesteps, model, ax3, plot='std')
    #     create_error_plot(errors[:,:,drp], timesteps, model, ax4, plot='std')
    #     ## need to warp the yaw errors between -pi and pi. yaw error is on position 5
    #     errors[:,:,yaw] = np.arctan2(np.sin(errors[:,:,yaw]), np.cos(errors[:,:,yaw]))
    #     create_error_plot(errors[:,:,yaw], timesteps, model, ax5, plot='std')
    #     create_error_plot(errors[:,:,dyaw], timesteps, model, ax6, plot='std')
    #     create_error_plot(errors[:,:,[9]], timesteps, model, ax7, plot='std')
    #     create_error_plot(errors[:,:,[10]], timesteps, model, ax8, plot='std')
    #     create_error_plot(errors[:,:,[11]], timesteps, model, ax9, plot='std')

    # ax1.legend()
    # ax2.legend()
    # ax3.legend()
    # ax4.legend()
    # ax5.legend()
    # ax6.legend()
    # ax7.legend()
    # ax8.legend()
    # ax9.legend()

    # plt.show()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="Evaluation.yaml", help='config file for training model')
    parser.add_argument('--shuffle', type=bool, required=False, default=False, help='shuffle data')
    parser.add_argument('--batchsize', type=int, required=False, default=1, help='training batch size')

    args = parser.parse_args()
    config = yaml.load(open( str(Path(os.getcwd()).parent.absolute()) + "/Experiments/Configs/" + args.config).read(), Loader=yaml.SafeLoader)
    plot_accuracy(config)
