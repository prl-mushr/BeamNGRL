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
from scipy.stats import mannwhitneyu, t as student_t
from matplotlib import rc

rc("font", family="Times New Roman", size=14)


def get_dynamics(model, Config):
    Dynamics_config = Config["Dynamics_config"]
    MPPI_config = Config["MPPI_config"]
    Map_config = Config["Map_config"]
    if model == "slip3d":
        Dynamics_config["type"] = "slip3d"  ## just making sure
        dynamics = SimpleCarDynamics(Dynamics_config, Map_config, MPPI_config)
    elif model == "noslip3d":
        # temporarily change the dynamics type to noslip3d
        Dynamics_config["type"] = "noslip3d"
        dynamics = SimpleCarDynamics(Dynamics_config, Map_config, MPPI_config)
        Dynamics_config["type"] = "slip3d"
    else:
        raise ValueError("Unknown model type")
    return dynamics


def evaluator(config, real, tn_args):
    Dynamics_config = config["Dynamics_config"]
    MPPI_config = config["MPPI_config"]
    dt = Dynamics_config["dt"]
    dataset_dt = 0.02
    np.set_printoptions(threshold=sys.maxsize)
    real_data = torch.from_numpy(real)
    for model in config["models"]:
        dynamics = get_dynamics(model, config)
        errors = np.zeros((len(real_data), 15))
        predict_states = np.zeros_like(errors)
        predict_states[0, :] = real[0, :15]
        for i in range(0, len(real_data), MPPI_config["TIMESTEPS"]):
            states_tn = real_data[i : i + MPPI_config["TIMESTEPS"], :15].to(**tn_args)
            controls_tn = real_data[i : i + MPPI_config["TIMESTEPS"], 17:19].to(
                **tn_args
            )
            gt_states = states_tn.clone().cpu().numpy()
            BEV_heght = torch.zeros((64, 64))
            BEV_normal = torch.zeros((64, 64, 3))
            dynamics.set_BEV(BEV_heght, BEV_normal)

            states = torch.zeros(17).to(**tn_args)
            states[:15] = torch.from_numpy(predict_states[i, :]).to(**tn_args)
            states = states.repeat(dynamics.M, dynamics.K, dynamics.T, 1)
            controls = controls_tn.repeat((dynamics.K, 1, 1)).clone()
            pred_states = dynamics.forward(states, controls)[0, 0, :, :15].cpu().numpy()
            # print(pred_states[:,11])

            errors[i : i + MPPI_config["TIMESTEPS"], :] = pred_states - gt_states
            predict_states[i : i + MPPI_config["TIMESTEPS"], :] = pred_states
            if np.any(np.isnan(pred_states)):
                print("NaN error")
                print(pred_states)
                exit()
        dir_name = (
            str(Path(os.getcwd()).parent.absolute()) + "/Experiments/Results/Sim2Real/"
        )
        if not os.path.isdir(dir_name):
            os.makedirs(dir_name)
        data_name = "/{}.npy".format(model)
        filename = dir_name + data_name
        np.save(filename, predict_states)
        data_name = "/err_{}.npy".format(model)
        filename = dir_name + data_name
        np.save(filename, errors)


def conf(data):
    # return np.fabs(np.percentile(data,97.5) - np.percentile(data,2.5))/2.0
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


def Plot_metrics(Config):
    order = 2  # Filter order
    cutoff_freq = 0.04  # Cutoff frequency as a fraction of Nyquist frequency
    b, a = signal.butter(order, cutoff_freq, btype="low", analog=False)
    bag_length = 25

    tensor_args = {"device": torch.device("cuda"), "dtype": torch.float32}
    dir_name = (
        str(Path(os.getcwd()).parent.absolute()) + "/Experiments/Results/Sim2Real/"
    )
    real_0 = np.load(dir_name + "bag_20.npy")[: 50 * 50, :]
    real_1 = np.load(dir_name + "bag_21.npy")[: 25 * 50, :]
    real_2 = np.load(dir_name + "bag_23.npy")[: 24 * 50, :]
    BeamNG_0 = np.load(dir_name + "BeamNG_20.npy")[: 50 * 50, :]
    BeamNG_1 = np.load(dir_name + "BeamNG_21.npy")[: 25 * 50, :]
    BeamNG_2 = np.load(dir_name + "BeamNG_23.npy")[: 24 * 50, :]
    real = np.concatenate((real_0, real_1, real_2), axis=0)
    BeamNG = np.concatenate((BeamNG_0, BeamNG_1, BeamNG_2), axis=0)
    np.save(dir_name + "err_BeamNG.npy", BeamNG)
    evaluator(Config, real, tensor_args)

    acc = signal.lfilter(b, a, real[:, 9:12], axis=0)
    gyro = real[:, 12:15]
    vel = real[:, 6:9]
    max_acc = np.max(acc)
    max_gyro = np.max(gyro)
    max_vel = np.max(vel)
    print(max_acc)
    print(max_vel)
    print(max_gyro)

    i = 0
    fig = plt.figure()
    fig.set_size_inches(8, 3)
    plt.subplots_adjust(left=0.08, right=0.99, top=0.99, bottom=0.1)
    X = np.arange(3)
    methods = ["BeamNG", "slip3d", "noslip3d"]
    metrics = ["Acceleration", "Rotation rate", "Velocity"]
    bar_width = 0.25
    acc_err = np.zeros((len(BeamNG), 3))
    rot_err = np.zeros((len(BeamNG), 3))
    vel_err = np.zeros((len(BeamNG), 3))
    for data in methods:
        error = np.load(dir_name + "err_{}.npy".format(data))
        acc_err[..., i] = np.linalg.norm(error[:, 9:12], axis=1)
        rot_err[..., i] = np.linalg.norm(error[:, 12:15], axis=1)
        vel_err[..., i] = np.linalg.norm(error[:, 6:8], axis=1)
        print(data)
        print("vel_err:", vel_err[..., i].mean())
        print("acc_err:", acc_err[..., i].mean())
        print("rot_err:", rot_err[..., i].mean())
        i += 1
    acc_max = max(acc_err.mean(axis=0))
    vel_max = max(vel_err.mean(axis=0))
    rot_max = max(rot_err.mean(axis=0))

    i = 0
    print(real.shape)
    for data in methods:
        error = np.load(dir_name + "err_{}.npy".format(data))
        color = plt.cm.tab10(i)  # Choose the same color from the 'tab10' colormap
        plt.bar(
            0 + bar_width * (1 - i),
            acc_err[..., i].mean() / acc_max,
            yerr=conf(acc_err[..., i]) / acc_max,
            width=bar_width,
            alpha=1.0,
            ecolor="black",
            capsize=10,
            color=color,
        )
        plt.bar(
            1 + bar_width * (1 - i),
            rot_err[..., i].mean() / rot_max,
            yerr=conf(rot_err[..., i]) / rot_max,
            width=bar_width,
            alpha=1.0,
            ecolor="black",
            capsize=10,
            color=color,
        )
        plt.bar(
            2 + bar_width * (1 - i),
            vel_err[..., i].mean() / vel_max,
            yerr=conf(vel_err[..., i]) / vel_max,
            width=bar_width,
            alpha=1.0,
            ecolor="black",
            capsize=10,
            label=data,
            color=color,
        )
        i += 1

    print(mannwhitneyu(acc_err[..., 0], acc_err[..., 1]))
    print(mannwhitneyu(acc_err[..., 2], acc_err[..., 1]))
    print(mannwhitneyu(acc_err[..., 0], acc_err[..., 2]))

    print(mannwhitneyu(rot_err[..., 0], rot_err[..., 1]))
    print(mannwhitneyu(rot_err[..., 2], rot_err[..., 1]))
    print(mannwhitneyu(rot_err[..., 0], rot_err[..., 2]))

    print(mannwhitneyu(vel_err[..., 0], vel_err[..., 1]))
    print(mannwhitneyu(vel_err[..., 2], vel_err[..., 1]))
    print(mannwhitneyu(vel_err[..., 0], vel_err[..., 2]))

    plt.ylabel("Relative error")
    plt.xticks(X, metrics)
    plt.grid(True, linestyle="--", alpha=0.7)
    legend_position = (0.55, 0.6)  # Specify the position as (x, y)
    plt.legend(loc=legend_position)
    plt.savefig(
        str(Path(os.getcwd()).parent.absolute())
        + "/Experiments/Results/Sim2Real/sim2real_comp.png"
    )
    # plt.show()
    plt.close()


if __name__ == "__main__":
    ## add a parser:
    parser = argparse.ArgumentParser(description="Plot the accuracy of the models")
    parser.add_argument(
        "--config_name",
        "-c",
        default="sim2real_Config.yaml",
        type=str,
        help="Path to the config file. Keep the same as the one used for evaluation",
    )

    args = parser.parse_args()
    config_name = args.config_name
    config_path = (
        str(Path(os.getcwd()).parent.absolute()) + "/Experiments/Configs/" + config_name
    )
    with open(config_path, "r") as f:
        Config = yaml.safe_load(f)
    ## call the plotting function, we'll extract data in there.
    Plot_metrics(Config)
