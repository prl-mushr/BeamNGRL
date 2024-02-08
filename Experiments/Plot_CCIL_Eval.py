import numpy as np
import matplotlib.pyplot as plt
import yaml
from pathlib import Path
import os
import argparse
from scipy.stats import mannwhitneyu, t as student_t
from matplotlib import rc
import seaborn as sns

rc("font", family="Times New Roman", size=16)


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


def Plot_metircs(Config):
    # create a new graph for each scenario:
    critical_SA = Config["Cost_config"]["critical_SA"]
    critical_RI = Config["Cost_config"]["critical_RI"]
    GRAVITY = 9.8

    time_limit = Config["time_limit"]
    scenario_count = 0

    critical_SA = Config["Cost_config"]["critical_SA"]
    critical_RI = Config["Cost_config"]["critical_RI"]
    critical_vert_acc = Config["Cost_config"]["critical_vert_acc"]
    critical_vert_spd = Config["Cost_config"]["critical_vert_spd"]

    for scenario in Config["scenarios"]:
        fig, axs = plt.subplots(1, 3)
        fig.set_size_inches(25.5, 5.5)
        fig.suptitle(scenario)
        axs[0].set_title("cross track error")
        axs[1].set_title("Time Taken")
        axs[2].set_title("Cost accrued normalized by maximum time allotted")

        scenario_time_limit = time_limit[scenario_count]
        scenario_count += 1

        WP_file = (
            str(Path(os.getcwd()).parent.absolute())
            + "/Experiments/Waypoints/"
            + scenario
            + ".npy"
        )
        target_WP = np.load(WP_file)
        target_pos = target_WP[:, :2]

        for policy in Config["policies"]:
            cross_track_error = []
            time_taken = []
            cost_per_unit_time = []

            for trial in range(Config["num_iters"]):
                dir_name = (
                    str(Path(os.getcwd()).parent.absolute())
                    + "/Experiments/Results/CCIL/"
                    + policy
                )
                filename = dir_name + "/{}-trial-{}.npy".format(scenario, str(trial))
                ## data has structure: state(17), goal(2), timestamp(1), success(1), damage(1)
                data = np.load(filename)
                ## find cross-track error between target_pos and pos from state, where pos is state[:, :2]
                pos = data[:, :2]
                ## pos data may be shorter, or divergent from target, so run a for-loop, where you find the closest point in target_pos to pos:
                cte = []
                for i in range(len(pos)):
                    dist = np.linalg.norm(target_pos - pos[i], axis=1)
                    idx = np.argmin(dist)
                    cte.append(dist[idx])
                cte = np.array(cte)
                cross_track_error.append(cte.mean())  ## take mean of cte for each trial
                ## extract the time taken using the last timestamp:
                time_taken.append(data[:, -3].max().mean() / scenario_time_limit)
                roll = data[:, 7]
                pitch = data[:, 8]

                ay = data[:, 10]
                az = data[:, 11]
                ct = np.sqrt(
                    np.clip(
                        1 - (np.square(np.sin(roll)) + np.square(np.sin(pitch))),
                        0.01,
                        1,
                    )
                )
                cost = (
                    np.clip((1 / ct) - critical_SA, 0, 10)
                    + np.clip(np.fabs(ay / az) - critical_RI, 0, 10)
                    + np.clip(np.fabs(az - GRAVITY) - critical_vert_acc, 0, 10.0)
                    + 5 * np.clip(np.fabs(data[:, 6]) - critical_vert_spd, 0, 10.0)
                )
                cost_per_unit_time.append(cost.sum())
            ## take mean and std for all:
            cross_track_error = np.array(cross_track_error)
            time_taken = np.array(time_taken)
            cost_per_unit_time = np.array(cost_per_unit_time)
            ## now take mean and std:
            cross_track_error_mean = cross_track_error.mean()
            time_taken_mean = time_taken.mean()
            cost_per_unit_time_mean = np.array(cost_per_unit_time).mean()

            cross_track_error_conf = conf(cross_track_error)
            time_taken_conf = conf(time_taken)
            cost_per_unit_time_conf = conf(cost_per_unit_time)

            ## now plot the data:
            ## success and damage don't need standard deviation:
            axs[0].bar(
                policy,
                cross_track_error_mean,
                yerr=cross_track_error_conf,
                align="center",
                alpha=0.5,
                ecolor="black",
                capsize=10,
            )
            axs[1].bar(
                policy,
                time_taken_mean,
                yerr=time_taken_conf,
                align="center",
                alpha=0.5,
                ecolor="black",
                capsize=10,
            )
            axs[2].bar(
                policy,
                cost_per_unit_time_mean,
                yerr=cost_per_unit_time_conf,
                align="center",
                alpha=0.5,
                ecolor="black",
                capsize=10,
            )

        fig.legend()

        # fig.savefig(str(Path(os.getcwd()).parent.absolute()) + "/Experiments/Results/CCIL/" + scenario + ".png")
        ## close the figure:
        plt.show()
        plt.close(fig)


if __name__ == "__main__":
    ## add a parser:
    parser = argparse.ArgumentParser(description="Plot the accuracy of the policies")
    parser.add_argument(
        "--config_name",
        "-c",
        default="CCIL_Eval_Config.yaml",
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
    Plot_metircs(Config)
