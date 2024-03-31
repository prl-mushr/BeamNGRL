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
        ax.plot(np.arange(0, timesteps), mean, label=model, color=color, linestyle=linestyle)
    else:
        ax.plot(np.arange(0, timesteps), mean, label=model)
    ax.fill_between(np.arange(0, timesteps), mean - std, mean + std, alpha=0.2, color=color)


def plot_accuracy(config):
    pos  = slice(0, 3)
    rpy  = slice(3, 6)
    yaw  = slice(5, 6)
    vel  = slice(6, 9)
    drpy = slice(12, 15)
    dyaw = slice(14, 15)
    acc  = slice(9, 12)

    skip = int(config["Dynamics_config"]["dt"]/0.02)

    fig = plt.figure()
    fig.suptitle(
        "Error vs Timestep on {} dataset with dt {} seconds".format(
            config["dataset"]["name"], config["Dynamics_config"]["dt"]
        )
    )
    # create 4 subplots for each of the error types
    # ax1 = fig.add_subplot(2, 3, 1)
    # ax1.set_title("Position (x,y,z) m")
    # ax1.xaxis.set_label_text("Timesteps")
    # ax2 = fig.add_subplot(2, 3, 2)
    # ax2.set_title("Roll-Pitch-Yaw radians")
    # ax2.xaxis.set_label_text("Timesteps")
    # ax3 = fig.add_subplot(2, 3, 3)
    # ax3.set_title("Velocity (x,y,z) m/s")
    # ax3.xaxis.set_label_text("Timesteps")
    # ax4 = fig.add_subplot(2, 3, 4)
    # ax4.set_title("Roll-Pitch-Yaw rate rad/s")
    # ax4.xaxis.set_label_text("Timesteps")
    # ax5 = fig.add_subplot(2, 3, 5)
    # ax5.set_title("Acceleration (x,y,z) m/s/s")
    # ax5.xaxis.set_label_text("Timesteps")
    fig.suptitle(
        "Error vs Timestep with dt={} seconds".format(
            config["Dynamics_config"]["dt"]
        )
    )
    fig.set_size_inches(20, 5)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.9, bottom=0.1)  # Adjust the values as needed
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.set_title("Position")
    ax1.yaxis.set_label_text("error in m")
    ax1.xaxis.set_label_text("Timesteps")
    # ax3 = fig.add_subplot(1, 4, 2)
    # ax3.set_title("Velocity")
    # ax3.yaxis.set_label_text("errors in m/s")
    # ax3.xaxis.set_label_text("Timesteps")
    ax4 = fig.add_subplot(1, 3, 2)
    ax4.set_title("Orientation rate")
    ax4.yaxis.set_label_text("errors in rad/s")
    ax4.xaxis.set_label_text("Timesteps")
    ax5 = fig.add_subplot(1, 3, 3)
    ax5.set_title("Acceleration")
    ax5.yaxis.set_label_text("errors in m/s/s")
    ax5.xaxis.set_label_text("Timesteps")
    timesteps = config["MPPI_config"]["TIMESTEPS"]
    rc('font', family='Times New Roman', size=8)

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
        if model == "TerrainCNN":
            model = "Learned_baseline"
        
        color = colors[count]
        linestyle = '-'

        create_error_plot(errors[:, :timesteps, pos], timesteps, model, ax1, plot="conf", color=color, linestyle=linestyle) #, max_err =  max_error[..., pos])
        # create_error_plot(errors[:, :timesteps, vel],timesteps, model, ax3, plot="conf", color=color, linestyle=linestyle) #, max_err =  max_error[..., vel],)
        ## need to warp the yaw errors between -pi and pi. yaw error is on position 5
        errors[:, :timesteps, yaw] = np.arctan2(
            np.sin(errors[:, :timesteps, yaw]), np.cos(errors[:, :timesteps, yaw])
        )
        # create_error_plot(errors[:, :timesteps, rpy],  timesteps, model, ax2, plot="conf", color=color, linestyle=linestyle) #, max_err = max_error[..., rpy])
        create_error_plot(errors[:, :timesteps, drpy], timesteps, model, ax4, plot="conf", color=color, linestyle=linestyle) #, max_err =  max_error[..., drpy])
        create_error_plot(errors[:, :timesteps, acc], timesteps, model, ax5, plot="conf", color=color, linestyle=linestyle) #, max_err =  max_error[..., acc])
        count += 1
        # print("model: ", model)
        # std_state = config["std_state"]
        # MSE_mean = (np.linalg.norm(errors[...,:]/std_state[:], axis = -1)).max()
        # print("MSE: ", MSE_mean)

    ax1.legend()
    # ax2.legend()
    # ax3.legend()
    ax4.legend()
    ax5.legend()
    plt.show()
    ## save the figure in the results/accuracy folder:
    fig.savefig(str(Path(os.getcwd()).parent.absolute()) + "/Experiments/Results/Accuracy/" + config["dataset"]["name"] + "_" + str(config["MPPI_config"]["TIMESTEPS"]) + ".png")


def plot_accuracy_final(config):
    pos  = slice(0, 3)
    rpy  = slice(3, 6)
    yaw  = slice(5, 6)
    vel  = slice(6, 9)
    drpy = slice(12, 15)
    dyaw = slice(14, 15)
    acc  = slice(9, 12)

    skip = int(config["Dynamics_config"]["dt"]/0.02)

    # in this function, I want to plot a bar-graph of the errors for each model for each attribute (position, velocity, orientation, orientation rate, acceleration)
    # I want to have the option to switch between final error and mean error. Also want the confidence interval
    fig = plt.figure()
    fig.suptitle(
        "Normalised Errors for different attributes".format()
    )
    fig.set_size_inches(12, 4)
    #spacing :
    fig.subplots_adjust(left=0.06, right=0.99, top=0.9, bottom=0.1)  # Adjust the values as needed
    # all errors in one graph. I basically want to plot a bar graph for each attribute, for each model (same graph). 
    # I will have 1 plot, with 5 bars for each model. Each bar will represent the error for a specific attribute
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.yaxis.set_label_text("Normalised Error")
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
    max_pos = np.linalg.norm(max_error[..., pos], axis=2).mean()
    max_rpy = np.linalg.norm(max_error[..., rpy], axis=2).mean()
    max_vel = np.linalg.norm(max_error[..., vel], axis=2).mean()
    max_drpy = np.linalg.norm(max_error[..., drpy], axis=2).mean()
    max_acc = np.linalg.norm(max_error[..., acc], axis=2).mean()
    max_error = np.array([max_pos, max_rpy, max_vel, max_drpy, max_acc])

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
        if model == "TerrainCNN":
            model = "Learned_baseline"
        
        # color = colors[count]
        # linestyle = '-'

        errors[:, :timesteps, yaw] = np.arctan2(
            np.sin(errors[:, :timesteps, yaw]), np.cos(errors[:, :timesteps, yaw])
        )
        pos_error = np.linalg.norm(errors[:, :timesteps, pos], axis=2)/max_error[0]
        rpy_error = np.linalg.norm(errors[:, :timesteps, rpy], axis=2)/max_error[1]
        vel_error = np.linalg.norm(errors[:, :timesteps, vel], axis=2)/max_error[2]
        drpy_error = np.linalg.norm(errors[:, :timesteps, drpy], axis=2)/max_error[3]
        acc_error = np.linalg.norm(errors[:, :timesteps, acc], axis=2)/max_error[4]
        # now plot the errors for each attribute for this model
        bar_positions = np.arange(5) + count*bar_width
        # confidence interval:
        ax1.errorbar(bar_positions, [np.mean(pos_error), np.mean(rpy_error), np.mean(vel_error), np.mean(drpy_error), np.mean(acc_error)],
                            yerr=[conf(pos_error), conf(rpy_error), conf(vel_error), conf(drpy_error), conf(acc_error)], fmt='o', color='black')
        # also mean:
        ax1.bar(bar_positions, [np.mean(pos_error), np.mean(rpy_error), np.mean(vel_error), np.mean(drpy_error), np.mean(acc_error)], bar_width, label=model, color=colors[count])
        count += 1
    ax1.legend()
    ax1.set_xticks(np.arange(5) + bar_width*(count-1)/2)
    ax1.set_xticklabels(["Position", "Orientation", "Velocity", "Orientation rate", "Acceleration"])
    plt.show()
    ## save the figure in the results/accuracy folder:
    fig.savefig(str(Path(os.getcwd()).parent.absolute()) + "/Experiments/Results/Accuracy/" + config["dataset"]["name"] + "_final" + ".png")

    pos_error_list = []
    rpy_error_list = []
    vel_error_list = []
    drpy_error_list = []
    acc_error_list = []

    count = 0
    for model in ["KARMA", "slip3d", "KARMA_bad_sys", "slip3d_bad_sys", "KARMA_noslip", "noslip3d"]:
        data = (
            str(Path(os.getcwd()).parent.absolute())
            + "/Experiments/Results/Accuracy/"
            + model
        )
        errors = np.load(data + "/{}.npy".format(config["dataset"]["name"]))
        if errors.shape[1] > config["MPPI_config"]["TIMESTEPS"]:
            skip = errors.shape[1]//config["MPPI_config"]["TIMESTEPS"]
            errors = errors[:,::skip,:]
        if model == "TerrainCNN":
            model = "Learned_baseline"
        
        errors[:, :timesteps, yaw] = np.arctan2(
            np.sin(errors[:, :timesteps, yaw]), np.cos(errors[:, :timesteps, yaw])
        )
        pos_error_list.append(np.mean(np.linalg.norm(errors[:, :timesteps, pos], axis=2)/max_error[0], axis=1))
        rpy_error_list.append(np.mean(np.linalg.norm(errors[:, :timesteps, rpy], axis=2)/max_error[1], axis=1))
        vel_error_list.append(np.mean(np.linalg.norm(errors[:, :timesteps, vel], axis=2)/max_error[2], axis=1))
        drpy_error_list.append(np.mean(np.linalg.norm(errors[:, :timesteps, drpy], axis=2)/max_error[3], axis=1))
        acc_error_list.append(np.mean(np.linalg.norm(errors[:, :timesteps, acc], axis=2)/max_error[4], axis=1))

        count += 1
    
    # Perform p-test for position error
    print("Position Error:")
    # KARMA vs slip3d

    stat, p = mannwhitneyu(pos_error_list[0], pos_error_list[1])
    print(f"KARMA vs slip3d: p-value = {p:.3f}")

    # KARMA vs KARMA_bad_sys
    stat, p = mannwhitneyu(pos_error_list[0], pos_error_list[2])
    print(f"KARMA vs KARMA_bad_sys: p-value = {p:.3f}")

    print(f"KARMA vs slip3d: p-value = {p:.3f}")

    # KARMA vs KARMA_bad_sys
    stat, p = mannwhitneyu(rpy_error_list[0], rpy_error_list[2])
    print(f"KARMA vs KARMA_bad_sys: p-value = {p:.3f}")

    # KARMA vs KARMA_noslip
    stat, p = mannwhitneyu(rpy_error_list[0], rpy_error_list[3])
    print(f"KARMA vs KARMA_noslip: p-value = {p:.3f}")

    # KARMA_bad_sys vs slip3d_bad_sys
    stat, p = mannwhitneyu(rpy_error_list[2], rpy_error_list[4])
    print(f"KARMA_bad_sys vs slip3d_bad_sys: p-value = {p:.3f}")

    # KARMA_noslip vs noslip3d
    stat, p = mannwhitneyu(rpy_error_list[3], rpy_error_list[5])
    print(f"KARMA_noslip vs noslip3d: p-value = {p:.3f}")

    print("=====")

    # Perform p-test for velocity error
    print("Velocity Error:")
    # KARMA vs slip3d
    stat, p = mannwhitneyu(vel_error_list[0], vel_error_list[1])
    print(f"KARMA vs slip3d: p-value = {p:.3f}")

    # KARMA vs KARMA_bad_sys
    stat, p = mannwhitneyu(vel_error_list[0], vel_error_list[2])
    print(f"KARMA vs KARMA_bad_sys: p-value = {p:.3f}")

    # KARMA vs KARMA_noslip
    stat, p = mannwhitneyu(vel_error_list[0], vel_error_list[3])
    print(f"KARMA vs KARMA_noslip: p-value = {p:.3f}")

    # KARMA_bad_sys vs slip3d_bad_sys
    stat, p = mannwhitneyu(vel_error_list[2], vel_error_list[4])
    print(f"KARMA_bad_sys vs slip3d_bad_sys: p-value = {p:.3f}")

    # KARMA_noslip vs noslip3d
    stat, p = mannwhitneyu(vel_error_list[3], vel_error_list[5])
    print(f"KARMA_noslip vs noslip3d: p-value = {p:.3f}")

    print("=====")

    # Perform p-test for drpy error
    print("DRPY Error:")
    # KARMA vs slip3d
    stat, p = mannwhitneyu(drpy_error_list[0], drpy_error_list[1])
    print(f"KARMA vs slip3d: p-value = {p:.3f}")

    # KARMA vs KARMA_bad_sys
    stat, p = mannwhitneyu(drpy_error_list[0], drpy_error_list[2])
    print(f"KARMA vs KARMA_bad_sys: p-value = {p:.3f}")

    # KARMA vs KARMA_noslip
    stat, p = mannwhitneyu(drpy_error_list[0], drpy_error_list[3])
    print(f"KARMA vs KARMA_noslip: p-value = {p:.3f}")

    # KARMA_bad_sys vs slip3d_bad_sys
    stat, p = mannwhitneyu(drpy_error_list[2], drpy_error_list[4])
    print(f"KARMA_bad_sys vs slip3d_bad_sys: p-value = {p:.3f}")

    # KARMA_noslip vs noslip3d
    stat, p = mannwhitneyu(drpy_error_list[3], drpy_error_list[5])
    print(f"KARMA_noslip vs noslip3d: p-value = {p:.3f}")

    print("=====")

    # Perform p-test for acceleration error
    print("Acceleration Error:")
    # KARMA vs slip3d
    stat, p = mannwhitneyu(acc_error_list[0], acc_error_list[1])
    print(f"KARMA vs slip3d: p-value = {p:.3f}")

    # KARMA vs KARMA_bad_sys
    stat, p = mannwhitneyu(acc_error_list[0], acc_error_list[2])
    print(f"KARMA vs KARMA_bad_sys: p-value = {p:.3f}")

    # KARMA vs KARMA_noslip
    stat, p = mannwhitneyu(acc_error_list[0], acc_error_list[3])
    print(f"KARMA vs KARMA_noslip: p-value = {p:.3f}")

    # KARMA_bad_sys vs slip3d_bad_sys
    stat, p = mannwhitneyu(acc_error_list[2], acc_error_list[4])
    print(f"KARMA_bad_sys vs slip3d_bad_sys: p-value = {p:.3f}")

    # KARMA_noslip vs noslip3d
    stat, p = mannwhitneyu(acc_error_list[3], acc_error_list[5])
    print(f"KARMA_noslip vs noslip3d: p-value = {p:.3f}")

    print("=====")
 # KARMA vs KARMA_noslip
    stat, p = mannwhitneyu(pos_error_list[0], pos_error_list[3])
    print(f"KARMA vs KARMA_noslip: p-value = {p:.3f}")

    # KARMA_bad_sys vs slip3d_bad_sys
    stat, p = mannwhitneyu(pos_error_list[2], pos_error_list[4])
    print(f"KARMA_bad_sys vs slip3d_bad_sys: p-value = {p:.3f}")

    # KARMA_noslip vs noslip3d
    stat, p = mannwhitneyu(pos_error_list[3], pos_error_list[5])
    print(f"KARMA_noslip vs noslip3d: p-value = {p:.3f}")

    print("=====")

    # Perform p-test for rpy error
    print("RPY Error:")
    # KARMA vs slip3d
    stat, p = mannwhitneyu(rpy_error_list[0], rpy_error_list[1])
    print(f"KARMA vs slip3d: p-value = {p:.3f}")

    # KARMA vs KARMA_bad_sys
    stat, p = mannwhitneyu(rpy_error_list[0], rpy_error_list[2])
    print(f"KARMA vs KARMA_bad_sys: p-value = {p:.3f}")

    # KARMA vs KARMA_noslip
    stat, p = mannwhitneyu(rpy_error_list[0], rpy_error_list[3])
    print(f"KARMA vs KARMA_noslip: p-value = {p:.3f}")

    # KARMA_bad_sys vs slip3d_bad_sys
    stat, p = mannwhitneyu(rpy_error_list[2], rpy_error_list[4])
    print(f"KARMA_bad_sys vs slip3d_bad_sys: p-value = {p:.3f}")

    # KARMA_noslip vs noslip3d
    stat, p = mannwhitneyu(rpy_error_list[3], rpy_error_list[5])
    print(f"KARMA_noslip vs noslip3d: p-value = {p:.3f}")

    print("=====")

    # Perform p-test for velocity error
    print("Velocity Error:")
    # KARMA vs slip3d
    stat, p = mannwhitneyu(vel_error_list[0], vel_error_list[1])
    print(f"KARMA vs slip3d: p-value = {p:.3f}")

    # KARMA vs KARMA_bad_sys
    stat, p = mannwhitneyu(vel_error_list[0], vel_error_list[2])
    print(f"KARMA vs KARMA_bad_sys: p-value = {p:.3f}")

    # KARMA vs KARMA_noslip
    stat, p = mannwhitneyu(vel_error_list[0], vel_error_list[3])
    print(f"KARMA vs KARMA_noslip: p-value = {p:.3f}")

    # KARMA_bad_sys vs slip3d_bad_sys
    stat, p = mannwhitneyu(vel_error_list[2], vel_error_list[4])
    print(f"KARMA_bad_sys vs slip3d_bad_sys: p-value = {p:.3f}")

    # KARMA_noslip vs noslip3d
    stat, p = mannwhitneyu(vel_error_list[3], vel_error_list[5])
    print(f"KARMA_noslip vs noslip3d: p-value = {p:.3f}")

    print("=====")

    # Perform p-test for drpy error
    print("DRPY Error:")
    # KARMA vs slip3d
    stat, p = mannwhitneyu(drpy_error_list[0], drpy_error_list[1])
    print(f"KARMA vs slip3d: p-value = {p:.3f}")

    # KARMA vs KARMA_bad_sys
    stat, p = mannwhitneyu(drpy_error_list[0], drpy_error_list[2])
    print(f"KARMA vs KARMA_bad_sys: p-value = {p:.3f}")

    # KARMA vs KARMA_noslip
    stat, p = mannwhitneyu(drpy_error_list[0], drpy_error_list[3])
    print(f"KARMA vs KARMA_noslip: p-value = {p:.3f}")

    # KARMA_bad_sys vs slip3d_bad_sys
    stat, p = mannwhitneyu(drpy_error_list[2], drpy_error_list[4])
    print(f"KARMA_bad_sys vs slip3d_bad_sys: p-value = {p:.3f}")

    # KARMA_noslip vs noslip3d
    stat, p = mannwhitneyu(drpy_error_list[3], drpy_error_list[5])
    print(f"KARMA_noslip vs noslip3d: p-value = {p:.3f}")

    print("=====")

    # Perform p-test for acceleration error
    print("Acceleration Error:")
    # KARMA vs slip3d
    stat, p = mannwhitneyu(acc_error_list[0], acc_error_list[1])
    print(f"KARMA vs slip3d: p-value = {p:.3f}")

    # KARMA vs KARMA_bad_sys
    stat, p = mannwhitneyu(acc_error_list[0], acc_error_list[2])
    print(f"KARMA vs KARMA_bad_sys: p-value = {p:.3f}")

    # KARMA vs KARMA_noslip
    stat, p = mannwhitneyu(acc_error_list[0], acc_error_list[3])
    print(f"KARMA vs KARMA_noslip: p-value = {p:.3f}")

    # KARMA_bad_sys vs slip3d_bad_sys
    stat, p = mannwhitneyu(acc_error_list[2], acc_error_list[4])
    print(f"KARMA_bad_sys vs slip3d_bad_sys: p-value = {p:.3f}")

    # KARMA_noslip vs noslip3d
    stat, p = mannwhitneyu(acc_error_list[3], acc_error_list[5])
    print(f"KARMA_noslip vs noslip3d: p-value = {p:.3f}")

    print("=====")



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
    plot_accuracy_final(config)
