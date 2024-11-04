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
from scipy.stats import mannwhitneyu


def conf(data):
    '''
    Calculate the confidence interval for the data
    :param data: the data to calculate the confidence interval for
    :return: the confidence interval
    Code/math adopted from: https://rowannicholls.github.io/python/statistics/confidence_intervals.html
    '''
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


def create_error_plot(errors, timesteps, model, ax, plot="conf", max_err = None, color=None, linestyle=None):
    plot_error = np.linalg.norm(errors, axis=2)
    mean = np.mean(plot_error, axis=0)
    std = np.zeros_like(mean)
    for i in range(timesteps):
        std[i] = conf(errors[:, i, :])
    if max_err is not None:
        max_mean = np.mean(np.linalg.norm(max_err, axis=2),axis=0)
        mean /= max_mean[-1]
        std /= max_mean[-1]
    if color is not None and linestyle is not None:
        ax.plot(np.arange(0, timesteps)*0.08, mean, label=model, color=color, linestyle=linestyle)
    else:
        ax.plot(np.arange(0, timesteps), mean, label=model)
    ax.fill_between(np.arange(0, timesteps)*0.08, mean - std, mean + std, alpha=0.2, color=color)


def plot_accuracy(config):
    pos  = slice(0, 3)
    rpy  = slice(3, 5)
    yaw  = slice(5, 6)
    vel  = slice(6, 9)
    drpy = slice(12, 15)
    dyaw = slice(14, 15)
    acc  = slice(9, 12)

    skip = int(config["Dynamics_config"]["dt"]/0.02)

    fig = plt.figure()

    rc('font', family='Times New Roman', size=10)
    fig.set_size_inches(10, 3)
    fig.subplots_adjust(left=0.05, right=0.98, top=0.8, bottom=0.15)  # Adjust the values as needed
    ax1 = fig.add_subplot(1, 5, 1)
    ax1.set_title("Position Error")
    ax1.yaxis.set_label_text("Error in m")
    ax1.xaxis.set_label_text("Time (s)")
    # plt.xticks(fontsize=12)
    # plt.yticks(fontsize=12)
    ax2 = fig.add_subplot(1, 5, 2)
    ax2.set_title("Rotation Rate Error")
    ax2.yaxis.set_label_text("Error in rad/s")
    ax2.xaxis.set_label_text("Time (s)")
    ax3 = fig.add_subplot(1, 5, 3)
    ax3.set_title("Velocity")
    ax3.yaxis.set_label_text("errors in m/s")
    ax3.xaxis.set_label_text("Timesteps")
    ax4 = fig.add_subplot(1, 5, 4)
    ax4.set_title("Tilt Error")
    ax4.yaxis.set_label_text("Error in rad")
    ax4.xaxis.set_label_text("Time (s)")
    ax5 = fig.add_subplot(1, 5, 5)
    ax5.set_title("Acceleration Error")
    ax5.yaxis.set_label_text("Errors in m/s/s")
    ax5.xaxis.set_label_text("Time (s)")
    # plt.xticks(fontsize=12)
    # plt.yticks(fontsize=12)
    timesteps = config["MPPI_config"]["TIMESTEPS"]

    ablation = config["ablation"]
    color_palette = 'inferno'
    original_colors = plt.get_cmap(color_palette)(range(256))

    skips = 256 // len(config["models"])
    colors = original_colors[::skips]

    count = 0

    print("dataset: ", config["dataset"]["name"])
    for model in config["models"]:
        data = (
            str(Path(os.getcwd()).parent.absolute())
            + "/Experiments/Results/Accuracy/"
            + model
        )
        errors = np.load(data + "/{}.npy".format(config["dataset"]["name"]))
        if errors.shape[1] > config["MPPI_config"]["TIMESTEPS"]:
            skip = errors.shape[1]//config["MPPI_config"]["TIMESTEPS"]
            errors = errors[:,::skip,:]
        
        color = colors[count]
        linestyle = '-'

        create_error_plot(errors[:, :timesteps, pos], timesteps, model, ax1, plot="conf", color=color, linestyle=linestyle)
        errors[:, :timesteps, yaw] = np.arctan2(
            np.sin(errors[:, :timesteps, yaw]), np.cos(errors[:, :timesteps, yaw])
        )
        create_error_plot(errors[:, :timesteps, drpy], timesteps, model, ax2, plot="conf", color=color, linestyle=linestyle)
        create_error_plot(errors[:, :timesteps, vel], timesteps, model, ax3, plot="conf", color=color, linestyle=linestyle)
        create_error_plot(errors[:, :timesteps, rpy], timesteps, model, ax4, plot="conf", color=color, linestyle=linestyle) 
        create_error_plot(errors[:, :timesteps, acc], timesteps, model, ax5, plot="conf", color=color, linestyle=linestyle)
        count += 1

    plt.legend(loc='upper center', bbox_to_anchor=(-0.8,1.3), ncol=len(config["models"]), fontsize=10)
    plt.show()
    ## save the figure in the results/accuracy folder:
    fig.savefig(str(Path(os.getcwd()).parent.absolute()) + "/Experiments/Results/Accuracy/" + config["dataset"]["name"] + "_" + str(config["MPPI_config"]["TIMESTEPS"]) + ".png")


def plot_accuracy_final(config):
    pos  = slice(0, 3)
    rpy  = slice(3, 5)
    yaw  = slice(5, 6)
    vel  = slice(6, 9)
    drpy = slice(12, 15)
    dyaw = slice(14, 15)
    acc  = slice(9, 12)

    skip = int(config["Dynamics_config"]["dt"]/0.02)

    normalize = True
    num_models = len(config["models"])
    plt.figure().set_size_inches(2*num_models, 3)
    plt.subplots_adjust(left=0.07, right=0.99, top=0.9, bottom=0.1)  # Adjust the values as needed
    if normalize:
        plt.title("Normalized Errors for Each Model over a 2 second horizon")
        plt.ylabel("Average Normalised Error")
    else:
        plt.title("Absolute Errors for Each Model")
        plt.ylabel("Absolute Error")
    timesteps = config["MPPI_config"]["TIMESTEPS"]
    rc('font', family='Times New Roman', size=10)

    ablation = config["ablation"]
    color_palette = 'plasma'
    original_colors = plt.get_cmap(color_palette)(range(256))

    skips = 256 // len(config["models"])
    colors = original_colors[::skips]
    
    count = 0
    bar_width = 0.1

    # normalize using the max error. Max error happens in noslip model, so use the mean error of noslip model to normalize
    data = (
        str(Path(os.getcwd()).parent.absolute())
        + "/Experiments/Results/Accuracy/"
        + "noslip3d"
    )
    max_error = np.load(data + "/{}.npy".format(config["dataset"]["name"]))
    max_error[:, :timesteps, yaw] = np.arctan2(
            np.sin(max_error[:, :timesteps, yaw]), np.cos(max_error[:, :timesteps, yaw])
        )
    max_pos = np.mean(np.linalg.norm(max_error[..., pos], axis=-1),axis=-1).mean()
    max_rpy = np.mean(np.linalg.norm(max_error[..., rpy], axis=-1),axis=-1).mean()
    max_vel = np.mean(np.linalg.norm(max_error[..., vel], axis=-1),axis=-1).mean()
    max_acc = np.mean(np.linalg.norm(max_error[..., acc], axis=-1),axis=-1).mean()
    print(max_pos)
    if normalize:
        max_error = np.array([max_pos, max_rpy, max_vel, max_acc])
    else:
        max_error = np.array([1, 1, 1, 1])
    for model in config["models"]:
        data = (
            str(Path(os.getcwd()).parent.absolute())
            + "/Experiments/Results/Accuracy/"
            + model
        )
        errors = np.load(data + "/{}.npy".format(config["dataset"]["name"]))
        if errors.shape[1] > config["MPPI_config"]["TIMESTEPS"]:
            skip = errors.shape[1]//config["MPPI_config"]["TIMESTEPS"]
            errors = errors[:,::skip,:]
        if model == "baseline":
            model = "Learned_baseline"
        if model[:5] == "KARMA":
            model = "Method" + model[5:]
 
        pos_error = np.mean(np.linalg.norm(errors[:, :timesteps, pos], axis=-1), axis=-1)/max_error[0]
        rpy_error = np.mean(np.linalg.norm(errors[:, :timesteps, rpy], axis=-1), axis=-1)/max_error[1]
        vel_error = np.mean(np.linalg.norm(errors[:, :timesteps, vel], axis=-1), axis=-1)/max_error[2]
        acc_error = np.mean(np.linalg.norm(errors[:, :timesteps, acc], axis=-1), axis=-1)/max_error[3]
        bar_positions = np.arange(4) + count*bar_width
        plt.bar(bar_positions, np.array([np.mean(pos_error), np.mean(rpy_error), np.mean(vel_error), np.mean(acc_error)]), bar_width, label=model, color=colors[count])
        plt.errorbar(bar_positions, np.array([np.mean(pos_error), np.mean(rpy_error), np.mean(vel_error), np.mean(acc_error)]), yerr=np.array([conf(pos_error), conf(rpy_error), conf(vel_error), conf(acc_error)]), fmt='none', capsize=5)
        count += 1
    plt.ylim(0, 1.2)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5,1.0), ncol=len(config["models"]))
    if normalize:
        plt.xticks(np.arange(4) + bar_width*(count-1)/2, ["Position Error", "Tilt Error", "Velocity Error", "Acceleration Error"])
    else:
        plt.xticks(np.arange(4) + bar_width*(count-1)/2, ["Position Error (m)", "Tilt Error (rad)", "Velocity Error (m/s)", "Acceleration Error (m/s^2)"])
    plt.savefig(str(Path(os.getcwd()).parent.absolute()) + "/Experiments/Results/Accuracy/" + config["dataset"]["name"] + "_final" + ".png")
    plt.show()


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
