import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import yaml
from pathlib import Path
import os
import argparse
import matplotlib.style as style
style.use('seaborn-colorblind')
from scipy.stats import mannwhitneyu
from matplotlib import rc

def Plot_metircs(Config):
    # create a new graph for each scenario:
    rc('font', family='Times New Roman', size=12)
    plt.figure().set_size_inches(3, 3)
    plt.subplots_adjust(left=0.2, right=0.99, top=0.99, bottom=0.11)  # Adjust the values as needed
    combinations = []
    for scenario in Config["scenarios"]:
        if scenario == "race-4":
            scn = "Shallow turns"
        if scenario == "roll-0":
            scn = "Tight turns"
        combinations.append(scn)

    models = len(Config["models"])
    X = np.arange(len(Config["scenarios"]))
    bar_width = 0.2
    model_count = 0
    time_limit = Config["time_limit"]

    time_taken_list = []
    speed_max_shallow = 0
    speed_max_tight = 0
    plt.ylabel("Time taken (s)")
    for model in Config["models"]:
        time_taken_mean = []
        time_taken_std = []
        scenario_count = 0
        if(model == "slip3d_rp"):
            model_name = "RPS ON"
        if(model == "slip3d"):
            model_name = "RPS OFF"
        for scenario in Config["scenarios"]:
            scenario_time_limit = time_limit[scenario_count]
            scenario_count += 1
            time_taken = []
            for trial in range(Config["num_iters"]):
                dir_name = str(Path(os.getcwd()).parent.absolute()) + "/Experiments/Results/Rollover/" + model
                filename = dir_name + "/{}-trial-{}.npy".format(scenario, str(trial))
                ## data has structure: state(17), goal(2), timestamp(1), success(1), damage(1)
                data = np.load(filename)
                ## extract the success and damage:
                time_taken.append((data[-1, -3] - data[0,-3]))
                ay = data[:, 10]
                az = data[:, 11]
                az_denom = np.clip(np.fabs(az),1,25)
                lat_ratio = np.fabs(ay/az)
                speed = data[:,6].max()
                if(scenario=="race-4"):
                    speed_max_shallow = max(speed_max_shallow, speed)
                else:
                    speed_max_tight = max(speed_max_tight, speed)
            ## take mean and std for all:
            time_taken = np.array(time_taken)
            time_taken_mean.append(time_taken.mean())
            time_taken_std.append(abs(np.percentile(time_taken, 2.5) - np.percentile(time_taken,97.5))/2)
            time_taken_list.append(time_taken)
        plt.bar(X + bar_width*(models - model_count - 1), time_taken_mean, yerr=time_taken_std, width=bar_width, alpha=0.5, ecolor='black', capsize=10, label=model_name)
        plt.grid(True, linestyle='--', alpha=0.7)
        model_count += 1
    plt.xticks(X, combinations)
    plt.legend()
    plt.savefig(str(Path(os.getcwd()).parent.absolute()) + "/Experiments/Results/Rollover/In_the_loop.png")
    # plt.show()
    plt.close()
    print("speed_max_tight: ", speed_max_tight)
    print("speed_max_shallow: ", speed_max_shallow)

    statistic, p_value = mannwhitneyu(time_taken_list[0], time_taken_list[2])
    print("p value tight turns: ", p_value)
    statistic, p_value = mannwhitneyu(time_taken_list[1], time_taken_list[3])
    print("p value shallow turns: ", p_value)

    plt.figure().set_size_inches(3, 3)
    plt.subplots_adjust(left=0.2, right=0.99, top=0.99, bottom=0.11)  # Adjust the values as needed
    combinations = []
    for scenario in Config["scenarios"]:
        if scenario == "race-4":
            scn = "Shallow turns"
        if scenario == "roll-0":
            scn = "Tight turns"
        combinations.append(scn)

    models = len(Config["models"])
    X = np.arange(len(Config["scenarios"]))
    bar_width = 0.2
    model_count = 0
    yaw_alpha_list = []
    plt.ylabel("Yaw acceleration (rad/s/s)")
    for model in Config["models"]:
        yaw_alpha_mean = []
        yaw_alpha_std = []
        scenario_count = 0
        if(model == "slip3d_rp"):
            model_name = "RPS ON"
        if(model == "slip3d"):
            model_name = "RPS OFF"
        for scenario in Config["scenarios"]:
            scenario_time_limit = time_limit[scenario_count]
            scenario_count += 1
            yaw_alpha = []
            for trial in range(Config["num_iters"]):
                dir_name = str(Path(os.getcwd()).parent.absolute()) + "/Experiments/Results/Rollover/" + model
                filename = dir_name + "/{}-trial-{}.npy".format(scenario, str(trial))
                ## data has structure: state(17), goal(2), timestamp(1), success(1), damage(1)
                data = np.load(filename)
                ## extract the success and damage:
                dwz = np.diff(data[:,14])/0.02
                yaw_alpha.append(np.max(np.fabs(dwz)))
            ## take mean and std for all:
            yaw_alpha = np.array(yaw_alpha)
            yaw_alpha_mean.append(yaw_alpha.mean())
            yaw_alpha_std.append(abs(np.percentile(yaw_alpha, 2.5) - np.percentile(yaw_alpha,97.5))/2)
            yaw_alpha_list.append(yaw_alpha)
        bar = plt.bar(X + bar_width*(models - model_count - 1), yaw_alpha_mean, yerr=yaw_alpha_std, width=bar_width, alpha=0.5, ecolor='black', capsize=10, label=model_name)
        # print(mcolors.rgb2hex(bar[0].get_facecolor()))
        plt.grid(True, linestyle='--', alpha=0.7)
        model_count += 1
    plt.xticks(X, combinations)
    plt.legend()
    plt.savefig(str(Path(os.getcwd()).parent.absolute()) + "/Experiments/Results/Rollover/yaw_alpha.png")
    # plt.show()
    plt.close()
    statistic, p_value = mannwhitneyu(yaw_alpha_list[0], yaw_alpha_list[2])
    print("yaw alpha p value tight turns: ", p_value)
    statistic, p_value = mannwhitneyu(yaw_alpha_list[1], yaw_alpha_list[3])
    print("yaw alpha p value shallow turns: ", p_value)


    rc('font', family='Times New Roman', size=24)
    fig, axs = plt.subplots(1,2)
    fig.set_size_inches(15, 6)
    plt.subplots_adjust(left=0.07, right=1.0, bottom=0.14, top=0.92)  # Adjust the values as needed
    for scenario in Config["scenarios"]:
        if scenario == "race-4":
            scn = "Shallow turns"
        if scenario == "roll-0":
            scn = "Tight turns"
        axs[Config["scenarios"].index(scenario)].set_title(scn)
        axs[Config["scenarios"].index(scenario)].axis('equal')

    damage_on_tight = 0
    damage_on_shallow = 0
    damage_off_tight = 0
    damage_off_shallow = 0
    for scenario in Config["scenarios"]:
        WP_file = str(Path(os.getcwd()).parent.absolute()) + "/Experiments/Waypoints/" + scenario + ".npy"
        target_WP = np.load(WP_file)[:,:2]
        wp_x = target_WP[:, 0] - target_WP[0, 0]
        wp_y = target_WP[:, 1] - target_WP[0, 1]

        start = target_WP[0,:] - target_WP[0,:]
        damage_x_ON =[]
        damage_y_ON =[]
        damage_x_OFF =[]
        damage_y_OFF =[]
        axs[Config["scenarios"].index(scenario)].plot(wp_x[::10], wp_y[::10], linewidth=2, label="Target trajectory", color='red')

        for model in Config["models"]:
            x = []
            y = []
            if(model == "slip3d_rp"):
                model_name = "RPS ON"
            if(model == "slip3d"):
                model_name = "RPS OFF"
            for trial in range(Config["num_iters"]):
                dir_name = str(Path(os.getcwd()).parent.absolute()) + "/Experiments/Results/Rollover/" + model
                filename = dir_name + "/{}-trial-{}.npy".format(scenario, str(trial))
                ## data has structure: state(17), goal(2), timestamp(1), success(1), damage(1)
                data = np.load(filename)
                X = data[:, 0] - target_WP[0, 0]
                Y = data[:, 1] - target_WP[0, 1]
                damage_index = np.where(np.fabs(data[:, 3]) > np.pi/2 - 0.1)[0]
                if model_name == "RPS ON":
                    damage_x_ON.extend(X[damage_index].tolist())
                    damage_y_ON.extend(Y[damage_index].tolist())
                if model_name == "RPS OFF":
                    damage_x_OFF.extend(X[damage_index].tolist())
                    damage_y_OFF.extend(Y[damage_index].tolist())

                x.extend(X[::5].tolist())
                y.extend(Y[::5].tolist())
            axs[Config["scenarios"].index(scenario)].scatter(x,y, s=40, label=model_name)

        # axs[Config["scenarios"].index(scenario)].scatter(start[0], start[1], s=100, label="Start")
        axs[Config["scenarios"].index(scenario)].scatter(damage_x_ON, damage_y_ON, s=100, label="Reset for RPS ON")
        axs[Config["scenarios"].index(scenario)].scatter(damage_x_OFF, damage_y_OFF, s=100, label="Reset for RPS OFF")

        axs[Config["scenarios"].index(scenario)].set_xlabel("X (East) (m)", size='large')
        axs[Config["scenarios"].index(scenario)].set_ylabel("Y (North) (m)", rotation=90, size='large')

        if(scenario == "race-4"):
            damage_on_shallow = len(damage_x_ON)
            damage_off_shallow = len(damage_x_OFF)
            axs[Config["scenarios"].index(scenario)].legend(bbox_to_anchor=(0, 1), loc='upper left', borderaxespad=0., fontsize="24")
        else:
            damage_on_tight = len(damage_x_ON)
            damage_off_tight = len(damage_x_OFF)

    plt.savefig(str(Path(os.getcwd()).parent.absolute()) + "/Experiments/Results/Rollover/trajectories.png")
    # plt.show()
    plt.close(fig)

    rc('font', family='Times New Roman', size=12)

    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(3, 3)
    plt.subplots_adjust(left=0.2, right=0.99, top=0.99, bottom=0.1)  # Adjust the values as needed
    ax.set_ylabel("Rollovers")
    legend_dict = {}
    rps_off = "RPS_OFF"
    rps_on = "RPS_ON"
    legend_dict[rps_on] = ax.bar(0 + bar_width * 1, damage_on_tight, width=bar_width, alpha=0.5, ecolor='black', capsize=10, label=rps_on, color='#0072b2')
    legend_dict[rps_off] = ax.bar(0 + bar_width * 0, damage_off_tight, width=bar_width, alpha=0.5, ecolor='black', capsize=10, label=rps_off, color='#009e73')
    legend_dict[rps_on] = ax.bar(1 + bar_width * 1, damage_on_shallow, width=bar_width, alpha=0.5, ecolor='black', capsize=10, label=rps_on, color='#0072b2')
    legend_dict[rps_off] = ax.bar(1 + bar_width * 0, damage_off_shallow, width=bar_width, alpha=0.5, ecolor='black', capsize=10, label=rps_off, color='#009e73')
    plt.xticks([0, 1], ["Tight turns", "Shallow turns"])
    unique_labels, unique_handles = zip(*[(label, handle) for label, handle in legend_dict.items()])
    ax.legend(unique_handles, unique_labels)
    plt.savefig(str(Path(os.getcwd()).parent.absolute()) + "/Experiments/Results/Rollover/rollover_rate_closed.png")
    # plt.show()
    plt.close()


if __name__ == "__main__":
    ## add a parser:
    parser = argparse.ArgumentParser(description="Plot the accuracy of the models")
    parser.add_argument("--config_name", "-c", default="Rollover_loop_config.yaml", type=str, help="Path to the config file. Keep the same as the one used for evaluation")

    args = parser.parse_args()
    config_name = args.config_name
    config_path = str(Path(os.getcwd()).parent.absolute()) + "/Experiments/Configs/" + config_name
    with open(config_path, "r") as f: 
        Config = yaml.safe_load(f)
    ## call the plotting function, we'll extract data in there.
    Plot_metircs(Config)