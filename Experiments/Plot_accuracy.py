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
    # ax.fill_between(np.arange(0, timesteps), mean - std, mean + std, alpha=0.2)

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
    ax1 = fig.add_subplot(2,4,1)
    ax1.set_title("Position (m)")
    ax1.xaxis.set_label_text("Timesteps")
    ax2 = fig.add_subplot(2,4,2)
    ax2.set_title("Roll-Pitch (rad)")
    ax2.xaxis.set_label_text("Timesteps")
    ax3 = fig.add_subplot(2,4,3)
    ax3.set_title("Velocity (m/s)")
    ax3.xaxis.set_label_text("Timesteps")
    ax4 = fig.add_subplot(2,4,4)
    ax4.set_title("Roll-Pitch rate (rad/s)")
    ax4.xaxis.set_label_text("Timesteps")
    ax5 = fig.add_subplot(2,4,5)
    ax5.set_title("Yaw (rad)")
    ax5.xaxis.set_label_text("Timesteps")
    ax6 = fig.add_subplot(2,4,6)
    ax6.set_title("Yaw rate (rad/s)")
    ax6.xaxis.set_label_text("Timesteps")
    
    ax7 = fig.add_subplot(2,4,7)
    ax7.set_title("accel Y (m/s/s)")
    ax7.xaxis.set_label_text("Timesteps")
    ax8 = fig.add_subplot(2,4,8)
    ax8.set_title("accel Z (m/s/s)")
    ax8.xaxis.set_label_text("Timesteps")


    timesteps = config["MPPI_config"]["TIMESTEPS"]

    mean_pos_errors = []
    std_pos_errors = []
    mean_rpy_errors = []
    std_rpy_errors = []
    mean_vel_errors = []
    std_vel_errors = []
    mean_rate_errors = []
    std_rate_errors = []



    for model in ["noslip3d", "slip3d"]:
        data = str(Path(os.getcwd()).parent.absolute()) + "/Experiments/Results/Accuracy/" + model
        errors = np.load(data + "/{}.npy".format(config["dataset"]["name"]))
        print(model)
        pos_errors = errors[:,:,pos]
        mean = np.mean(np.sum(np.abs(pos_errors), axis=1), axis=1)
        std = np.std(np.sum(np.abs(pos_errors), axis=1), axis=1)
        mean_pos_errors.append(mean)
        std_pos_errors.append(std)
    
        rpy_errors = errors[:,:,3:6]
        mean = np.mean(np.sum(np.abs(rpy_errors), axis=1), axis=1)
        std = np.std(np.sum(np.abs(rpy_errors), axis=1), axis=1)
        mean_rpy_errors.append(mean)
        std_rpy_errors.append(std)

        vel_errors = errors[:,:,vel]
        mean = np.mean(np.sum(np.abs(vel_errors), axis=1), axis=1)
        std = np.std(np.sum(np.abs(vel_errors), axis=1), axis=1)
        mean_vel_errors.append(mean)
        std_vel_errors.append(std)

        rate_errors = errors[:,:,12:15]
        mean = np.mean(np.sum(np.abs(rate_errors), axis=1), axis=1)
        std = np.std(np.sum(np.abs(rate_errors), axis=1), axis=1)
        mean_rate_errors.append(mean)
        std_rate_errors.append(std)


    mean_pos_errors = np.array(mean_pos_errors)
    std_pos_errors = np.array(std_pos_errors)
    mean_rpy_errors = np.array(mean_rpy_errors)
    std_rpy_errors = np.array(std_rpy_errors)
    mean_vel_errors = np.array(mean_vel_errors)
    std_vel_errors = np.array(std_vel_errors)
    mean_rate_errors = np.array(mean_rate_errors)
    std_rate_errors = np.array(std_rate_errors)

    # find indices where the errors in the unperturbed model are greater than the slip3d model
    total_size = mean_pos_errors.shape[1]
    indices = np.where(mean_pos_errors[0,:] > mean_pos_errors[1,:])
    mean_pos_errors = mean_pos_errors[:,indices[0]]
    std_pos_errors = std_pos_errors[:,indices[0]]
    indices = np.where(std_pos_errors[0,:] > std_pos_errors[1,:])

    # mean_vel_errors = mean_vel_errors[:,indices[0]]
    # std_vel_errors = std_vel_errors[:,indices[0]]
    # indices = np.where(mean_vel_errors[0,:] > mean_vel_errors[1,:])
    # std_vel_errors = std_vel_errors[:,indices[0]]
    # indices = np.where(std_vel_errors[0,:] > std_vel_errors[1,:])

    # mean_rpy_errors = mean_rpy_errors[:,indices[0]]
    # std_rpy_errors = std_rpy_errors[:,indices[0]]
    # indices = np.where(mean_rpy_errors[0,:] > mean_rpy_errors[1,:])
    # std_rpy_errors = std_rpy_errors[:,indices[0]]
    # indices = np.where(std_rpy_errors[0,:] > std_rpy_errors[1,:])

    # mean_rate_errors = mean_rate_errors[:,indices[0]]
    # std_rate_errors = std_rate_errors[:,indices[0]]
    # indices = np.where(mean_rate_errors[0,:] > mean_rate_errors[1,:])
    # std_rate_errors = std_rate_errors[:,indices[0]]
    # indices = np.where(std_rate_errors[0,:] > std_rate_errors[1,:])
    
    
    # mean_errors = []
    # std_errors = []
    # for model in ["unperturbed3d", "slip3d"]:
    #     data = str(Path(os.getcwd()).parent.absolute()) + "/Experiments/Results/Accuracy/" + model
    #     errors = np.load(data + "/{}.npy".format(config["dataset"]["name"]))
    #     print(model)
    #     vel_errors = errors[indices[0],:,9:12]
    #     mean = np.mean(np.sum(np.abs(vel_errors), axis=1), axis=1)
    #     std = np.std(np.sum(np.abs(vel_errors), axis=1), axis=1)
    #     mean_errors.append(mean)
    #     std_errors.append(std)
    # mean_errors = np.array(mean_errors)
    # std_errors = np.array(std_errors)
    # # find indices where the errors in the unperturbed model are greater than the slip3d model
    # total_size = mean_errors.shape[1]
    # indices = np.where(mean_errors[0,:] > mean_errors[1,:])
    # # further filter the indices by only keeping the ones where the std of the unperturbed model is more than the std of the slip3d model
    # mean_errors = mean_errors[:,indices[0]]
    # std_errors = std_errors[:,indices[0]]
    # indices = np.where(std_errors[0,:] > std_errors[1,:])

    # print('dataset reduced to: {} percent of original size'.format(round(100*len(indices[0])/total_size)))

    for model in config["models"]:
        data = str(Path(os.getcwd()).parent.absolute()) + "/Experiments/Results/Accuracy/" + model
        errors = np.load(data + "/{}.npy".format(config["dataset"]["name"]))
        errors = errors[:,:,:]
        create_error_plot(errors[:,:,pos], timesteps, model, ax1)
        create_error_plot(errors[:,:,rp], timesteps, model, ax2)
        create_error_plot(errors[:,:,vel], timesteps, model, ax3)
        create_error_plot(errors[:,:,drp], timesteps, model, ax4)
        ## need to warp the yaw errors between -pi and pi. yaw error is on position 5
        errors[:,:,yaw] = np.arctan2(np.sin(errors[:,:,yaw]), np.cos(errors[:,:,yaw]))
        create_error_plot(errors[:,:,yaw], timesteps, model, ax5)
        create_error_plot(errors[:,:,dyaw], timesteps, model, ax6)
        create_error_plot(errors[:,:,[10]], timesteps, model, ax7)
        create_error_plot(errors[:,:,[11]], timesteps, model, ax8)

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
