import numpy as np
import matplotlib.pyplot as plt
import yaml
from pathlib import Path
import os
import argparse

# import seaborn as sns
# # Set Seaborn color palette to "colorblind"
# sns.set_palette("colorblind")
from scipy.stats import mannwhitneyu, t as student_t
from matplotlib import rc

# rc('font', family='Times New Roman', size=16)


def conf(data):
    # Sample size
    n = len(data)
    s = np.std(data, ddof=1)  # Use ddof=1 to get the sample standard deviation
    # Confidence level
    C = 0.95  # 95%
    # Significance level, α
    alpha = 1 - C
    # Number of tails
    tails = 2
    # Quantile (the cumulative probability)
    q = 1 - (alpha / tails)
    # Degrees of freedom
    dof = n - 1
    # Critical t-statistic, calculated using the percent-point function (aka the
    # quantile function) of the t-distribution
    t_star = student_t.ppf(q, dof)
    # Confidence interval
    return t_star * s / np.sqrt(n)


def remove_outliers(data, threshold=2):
    # Calculate the z-scores for each data point
    z_scores = (data - np.mean(data, axis=1, keepdims=True)) / np.std(
        data, axis=1, keepdims=True
    )

    # Find the indices of outliers
    outlier_indices = np.where(np.abs(z_scores) > threshold)

    # Replace outlier values with NaN
    data[outlier_indices] = np.nan

    # Remove rows with NaN values
    data = data[~np.isnan(data).any(axis=1)]

    return data


def create_error_plot(errors, timesteps, model, ax, plot="conf", max_err = None):
    plot_error = np.linalg.norm(errors, axis=2)
    mean = np.mean(plot_error, axis=0)
    std = np.zeros_like(mean)
    for i in range(timesteps):
        std[i] = conf(errors[:, i, :])
    if max_err is not None:
        max_mean = np.mean(np.linalg.norm(max_err, axis=2),axis=0)
        mean /= max_mean[-1]
        std /= max_mean[-1]
    ax.plot(np.arange(0, timesteps), mean, label=model)
    ax.fill_between(np.arange(0, timesteps), mean - std, mean + std, alpha=0.2)


def plot_accuracy(config):
    pos = slice(0, 3)
    rpy = slice(3, 6)
    yaw = slice(5, 6)
    vel = slice(6, 9)
    drpy = slice(12, 15)
    dyaw = slice(14, 15)
    acc = slice(9, 12)

    skip = int(config["Dynamics_config"]["dt"]/0.02)

    fig = plt.figure()
    fig.suptitle(
        "Error vs Timestep on {} dataset with dt {} seconds".format(
            config["dataset"]["name"], config["Dynamics_config"]["dt"]
        )
    )
    # create 4 subplots for each of the error types
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.set_title("Position (x,y,z)")
    ax1.xaxis.set_label_text("Timesteps")
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.set_title("Roll-Pitch-Yaw")
    ax2.xaxis.set_label_text("Timesteps")
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.set_title("Velocity (x,y,z)")
    ax3.xaxis.set_label_text("Timesteps")
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.set_title("Roll-Pitch-Yaw rate")
    ax4.xaxis.set_label_text("Timesteps")
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.set_title("Acceleration (x,y,z)")
    ax5.xaxis.set_label_text("Timesteps")

    # ax7 = fig.add_subplot(2, 5, 7)
    # ax7.set_title("accel X-Y")
    # ax7.xaxis.set_label_text("Timesteps")
    # ax8 = fig.add_subplot(2, 5, 8)
    # ax8.set_title("accel Y")
    # ax8.xaxis.set_label_text("Timesteps")
    # ax9 = fig.add_subplot(2, 5, 9)
    # ax9.set_title("accel Z")
    # ax9.xaxis.set_label_text("Timesteps")
    # ax10 = fig.add_subplot(2, 5, 10)
    # ax10.set_title("vel Z")
    # ax10.xaxis.set_label_text("Timesteps")
    timesteps = config["MPPI_config"]["TIMESTEPS"]


    data = (
        str(Path(os.getcwd()).parent.absolute())
        + "/Experiments/Results/Accuracy/"
        + "no_change"
    )
    errors = np.load(data + "/{}.npy".format(config["dataset"]["name"]))
    if errors.shape[1] > config["MPPI_config"]["TIMESTEPS"]:
        skip = errors.shape[1]//config["MPPI_config"]["TIMESTEPS"]
        print("skips: ",skip)
        errors = errors[:,::skip,:]
    max_error = np.abs(errors)

    for model in config["models"]:
        data = (
            str(Path(os.getcwd()).parent.absolute())
            + "/Experiments/Results/Accuracy/"
            + model
        )
        errors = np.load(data + "/{}.npy".format(config["dataset"]["name"]))
        if errors.shape[1] > config["MPPI_config"]["TIMESTEPS"]:
            skip = errors.shape[1]//config["MPPI_config"]["TIMESTEPS"]
            print("skips: ",skip)
            errors = errors[:,::skip,:]

        create_error_plot(errors[:, :timesteps, pos], timesteps, model, ax1, plot="conf") #, max_err =  max_error[..., pos])
        create_error_plot(errors[:, :timesteps, vel],timesteps, model, ax3, plot="conf") #, max_err =  max_error[..., vel],)
        ## need to warp the yaw errors between -pi and pi. yaw error is on position 5
        errors[:, :timesteps, yaw] = np.arctan2(
            np.sin(errors[:, :timesteps, yaw]), np.cos(errors[:, :timesteps, yaw])
        )
        create_error_plot(errors[:, :timesteps, rpy],  timesteps, model, ax2, plot="conf") #, max_err = max_error[..., rpy])
        create_error_plot(errors[:, :timesteps, drpy], timesteps, model, ax4, plot="conf") #, max_err =  max_error[..., drpy])
        create_error_plot(errors[:, :timesteps, acc], timesteps, model, ax5, plot="conf") #, max_err =  max_error[..., acc])
        # create_error_plot(errors[:, :timesteps, dyaw], timesteps, model, ax6, plot="conf")
        # create_error_plot(errors[:, :timesteps, 9:12], timesteps, model, ax7, plot="conf")
        # create_error_plot(errors[:, :timesteps, [10]], timesteps, model, ax8, plot="conf")
        # create_error_plot(errors[:, :timesteps, [11]], timesteps, model, ax9, plot="conf")
        # create_error_plot(errors[:, :timesteps, [8]], timesteps, model, ax10, plot="conf")


    # pos_error_list = np.array(pos_error_list)
    # p1 = pos_error_list[0]
    # p2 = pos_error_list[1]
    # statistic, p_value = mannwhitneyu(p1[:,0:2], p1[:,0:2])
    # print("p value: ", p_value)
    # ax1.plot(np.arange(timesteps), 1.5*np.ones(timesteps), label="crop size", linestyle="--")
    # ax1.plot(np.ones(timesteps)*22, 0.1*np.arange(timesteps), label="crossover", linestyle="--")
    # ax2.plot(np.ones(timesteps)*22, 0.003*np.arange(timesteps), label="crossover", linestyle="--")
    # ax3.plot(np.ones(timesteps)*22, 0.05*np.arange(timesteps), label="crossover", linestyle="--")
    # ax4.plot(np.ones(timesteps)*22, 0.01*np.arange(timesteps), label="crossover", linestyle="--")
    # ax5.plot(np.ones(timesteps)*22, 0.01*np.arange(timesteps), label="crossover", linestyle="--")
    # ax6.plot(np.ones(timesteps)*22, 0.01*np.arange(timesteps), label="crossover", linestyle="--")
    # ax7.plot(np.ones(timesteps)*22, 0.1*np.arange(timesteps), label="crossover", linestyle="--")
    # ax8.plot(np.ones(timesteps)*22, 0.1*np.arange(timesteps), label="crossover", linestyle="--")
    # ax9.plot(np.ones(timesteps)*22, 0.1*np.arange(timesteps), label="crossover", linestyle="--")
    # ax10.plot(np.ones(timesteps)*22, 0.01*np.arange(timesteps), label="crossover", linestyle="--")

    ax1.legend()
    ax2.legend()
    ax3.legend()
    ax4.legend()
    ax5.legend()
    # ax6.legend()
    # ax7.legend()
    # ax8.legend()
    # ax9.legend()
    # ax10.legend()
    plt.show()
    ## save the figure in the results/accuracy folder:
    fig.savefig(str(Path(os.getcwd()).parent.absolute()) + "/Experiments/Results/Accuracy/" + config["dataset"]["name"] + "_" + str(config["MPPI_config"]["TIMESTEPS"]) + ".png")

    # for model in config["models"]:
    #     data = (
    #         str(Path(os.getcwd()).parent.absolute())
    #         + "/Experiments/Results/Accuracy/"
    #         + model
    #     )
    #     try:
    #         params = np.load(data + "/{}_param.npy".format(config["dataset"]["name"]))
    #     except:
    #         continue
    #     # params are of shape (iterations, num_params)
    #     # use shape to the number of params, then programmatically create subplots
    #     fig = plt.figure()
    #     fig.suptitle(
    #         "estimated parameter value over iterations (each iteration is {} seconds long)".format(
    #             config["Dynamics_config"]["dt"] * config["MPPI_config"]["TIMESTEPS"]
    #         )
    #     )
    #     ax = []
    #     friction = 0.85
    #     param_list = ["friction", "resistance coeff", "drag coeff", "roll-pitch low-pass coeff", "steering low-pass coeff", "throttle low-pass coeff", "normalized yaw MoI"]
    #     for i in range(params.shape[1]):
    #         ax.append(fig.add_subplot(2, 5, i + 1))
    #         ax[i].set_title("Param {}".format(param_list[i]))
    #         ax[i].xaxis.set_label_text("Iterations")
    #         ax[i].plot(np.arange(0, params.shape[0]), params[:, i], label="estimate")
    #         if i == 0:
    #             ax[i].plot(np.arange(0, params.shape[0]), np.ones(params.shape[0])*friction, label="GT")
    #         ax[i].legend()
    #     plt.show()

    # fig = plt.figure()
    # fig.suptitle("estimated parameter value over iterations (each iteration is {} seconds long)".format(config["dataset"]["dt"]*config["MPPI_config"]["TIMESTEPS"]))
    # param_data = np.load
    # ax1 = fig.add_subplot(2,5,1)
    # ax1.set_title("Friction")
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

def plot_accuracy_new(config):
    pos = slice(0, 3)
    rp = slice(3, 5)
    yaw = slice(5, 6)
    vel = slice(6, 8)
    drp = slice(12, 14)
    dyaw = slice(14, 15)
    dvel = slice(9, 12)

    skip = int(config["Dynamics_config"]["dt"]/0.02)

    fig = plt.figure()
    fig.suptitle(
        "Error vs Timestep on {} dataset with dt {} seconds".format(
            config["dataset"]["name"], config["Dynamics_config"]["dt"]
        )
    )
    # create 4 subplots for each of the error types
    ax1 = fig.add_subplot(1, 5, 1)
    ax1.set_title("Position")
    ax1.xaxis.set_label_text("Timesteps")
    ax2 = fig.add_subplot(1, 5, 2)
    ax2.set_title("Velocity")
    ax2.xaxis.set_label_text("Timesteps")
    ax3 = fig.add_subplot(1, 5, 3)
    ax3.set_title("Roll-Pitch ")
    ax3.xaxis.set_label_text("Timesteps")
    ax4 = fig.add_subplot(1, 5, 4)
    ax4.set_title("Yaw")
    ax4.xaxis.set_label_text("Timesteps")
    ax5 = fig.add_subplot(1, 5, 5)
    ax5.set_title("accelerations m/s/s")
    ax5.xaxis.set_label_text("Timesteps")
    timesteps = config["MPPI_config"]["TIMESTEPS"]

    data = (
        str(Path(os.getcwd()).parent.absolute())
        + "/Experiments/Results/Accuracy/slip3d"
    )

    errors = np.load(data + "/{}.npy".format(config["dataset"]["name"]))
    mean = np.mean(np.abs(errors), axis=0)
    normalization = 1 #np.max(mean, axis=0)
    print(normalization)

    for model in config["models"]:
        data = (
            str(Path(os.getcwd()).parent.absolute())
            + "/Experiments/Results/Accuracy/"
            + model
        )
        errors = np.load(data + "/{}.npy".format(config["dataset"]["name"]))/normalization
        if errors.shape[1] > config["MPPI_config"]["TIMESTEPS"]:
            errors = errors[:,::skip,:]
        create_error_plot(errors[:, :, pos], timesteps, model, ax1, plot="conf")
        create_error_plot(errors[:, :, vel], timesteps, model, ax2, plot="conf")
        create_error_plot(errors[:, :, rp], timesteps, model, ax3, plot="conf")
        ## need to warp the yaw errors between -pi and pi. yaw error is on position 5
        errors[:, :, yaw] = np.arctan2(
            np.sin(errors[:, :, yaw]), np.cos(errors[:, :, yaw])
        )
        create_error_plot(errors[:, :, yaw], timesteps,  model, ax4, plot="conf")
        create_error_plot(errors[:, :, dvel], timesteps, model, ax5, plot="conf")


    ax1.legend()
    ax2.legend()
    ax3.legend()
    ax4.legend()
    ax5.legend()

    plt.show()
    ## save the figure in the results/accuracy folder:
    fig.savefig(str(Path(os.getcwd()).parent.absolute()) + "/Experiments/Results/Accuracy/" + config["dataset"]["name"] + "_" + str(config["MPPI_config"]["TIMESTEPS"]) + ".png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="Evaluation.yaml",
        help="config file for training model",
    )
    parser.add_argument(
        "--shuffle", type=bool, required=False, default=False, help="shuffle data"
    )
    parser.add_argument(
        "--batchsize", type=int, required=False, default=1, help="training batch size"
    )

    args = parser.parse_args()
    config = yaml.load(
        open(
            str(Path(os.getcwd()).parent.absolute())
            + "/Experiments/Configs/"
            + args.config
        ).read(),
        Loader=yaml.SafeLoader,
    )
    plot_accuracy(config)
