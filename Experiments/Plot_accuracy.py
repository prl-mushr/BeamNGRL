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
    pos = slice(0, 2)
    rpy = slice(3, 6)
    yaw = slice(5, 6)
    vel = slice(6, 8)
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


    ax1.legend()
    ax2.legend()
    ax3.legend()
    ax4.legend()
    ax5.legend()
    plt.show()
    ## save the figure in the results/accuracy folder:
    fig.savefig(str(Path(os.getcwd()).parent.absolute()) + "/Experiments/Results/Accuracy/" + config["dataset"]["name"] + "_" + str(config["MPPI_config"]["TIMESTEPS"]) + ".png")

    data = (
        str(Path(os.getcwd()).parent.absolute())
        + "/Experiments/Results/Accuracy/SysID"
    )
    if os.path.isfile(data + "/{}_param.npy".format(config["dataset"]["name"])):
        params = np.load(data + "/{}_param.npy".format(config["dataset"]["name"]))
        steps = len(params)
        axes = []
        fig2 = plt.figure()
        ax = fig2.add_subplot(2, 4, 1)
        ax.set_title("D")
        ax.xaxis.set_label_text("Timesteps")
        axes.append(ax)
        ax = fig2.add_subplot(2, 4, 2)
        ax.set_title("res_coeff")
        ax.xaxis.set_label_text("Timesteps")
        axes.append(ax)
        ax = fig2.add_subplot(2, 4, 3)
        ax.set_title("drag_coeff")
        ax.xaxis.set_label_text("Timesteps")
        axes.append(ax)
        ax = fig2.add_subplot(2, 4, 4)
        ax.set_title("LPF_tau")
        ax.xaxis.set_label_text("Timesteps")
        axes.append(ax)
        ax = fig2.add_subplot(2, 4, 5)
        ax.set_title("LPF_st")
        ax.xaxis.set_label_text("Timesteps")
        axes.append(ax)
        ax = fig2.add_subplot(2, 4, 6)
        ax.set_title("LPF_th")
        ax.xaxis.set_label_text("Timesteps")
        axes.append(ax)
        ax = fig2.add_subplot(2, 4, 7)
        ax.set_title("Iz")
        ax.xaxis.set_label_text("Timesteps")
        axes.append(ax)
        for i in range(7):
            axes[i].plot(np.arange(steps), params[:, i])
        plt.show()
        fig2.savefig(
            str(Path(os.getcwd()).parent.absolute())
            + "/Experiments/Results/Accuracy/"
            + config["dataset"]["name"]
            + "_params.png"
        )

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
