import numpy as np
import matplotlib.pyplot as plt
import yaml
from pathlib import Path
import os
import argparse
import seaborn as sns
# Set Seaborn color palette to "colorblind"
sns.set_palette("colorblind")
from scipy.stats import mannwhitneyu

def conf(data):
    return np.fabs(np.percentile(data,97.5) - np.percentile(data,2.5))/2.0

def Plot_metrics(Config, Config_crash):
    plt.figure().set_size_inches(6, 4)
    plt.subplots_adjust(left=0.09, right=0.99)  # Adjust the values as needed
    RPS = ["Static limiter", "No prevention", "Full RPS"]
    bar_width = 0.2
    combinations = []
    for scenario in Config["scenarios"]:
        if scenario == "smallgrid":
            scn = "Flat"
        if scenario == "small_island":
            scn = "Offroad"
        for vehicle_name in Config["vehicle_list"]:
            if vehicle_name == "flux":
                vn = "Small car"
            if vehicle_name == "rollover_test":
                vn = "Big car"
            combinations.append("{}-{}".format(scn, vn))
    rps_count = 0

    lat_acc_list = []

    for RP in RPS:
        if RP == "No prevention":
            rollover_prevention = 0
        if RP == "Static limiter":
            rollover_prevention = 2
        if RP == "Full RPS":
            rollover_prevention = 1
        scenario_count = 0
        total_scenarios = len(Config["scenarios"])*len(Config["vehicle_list"])
        X = np.arange(total_scenarios)
        mean_bar = []
        std_bar = []
        for scenario in Config["scenarios"]:
            for vehicle_name in Config["vehicle_list"]:
                average_lat_ratio = []
                rollover = []
                min_az = []
                track_width = Config["vehicles"][vehicle_name]["track_width"]/2
                for trial in range(Config["num_iters"]):
                    dir_name = str(Path(os.getcwd()).parent.absolute()) + "/Experiments/Results/Rollover/{}/".format(str(rollover_prevention))
                    filename = dir_name + "/{}-{}-{}.npy".format(scenario, vehicle_name, str(trial))
                    ## data has structure: state(17) flipped_over(1) rollover_prevention(1) intervention(1) Rollover_detected(1) delta_steering(1) turn_time(1) rotate_speed(1) ts(1) ))
                    data = np.load(filename)
                    turn_index = np.where(data[:,-1] >= data[-1, 22])[0]
                    ay = data[turn_index, 10]
                    az = data[turn_index, 11]
                    roll = data[turn_index, 3]
                    index = np.where(np.abs(roll) < 1.5)[0]
                    ay = ay[index] + track_width*data[index, 12]**2
                    az = az[index]
                    average_lat_ratio.append(np.mean(np.abs(ay))/np.mean(np.abs(az)) )
                    min_az.append(np.min(np.abs(az)))
                    rollover.append(data[-1, 17])\
                ## this isn't 95 percentile!
                mean_LTR = np.mean(np.array(average_lat_ratio))
                std_LTR = conf(np.array(average_lat_ratio))
                rollover_rate = np.mean(np.array(rollover))
                min_az = np.mean(np.array(min_az))
                mean_bar.append(mean_LTR)
                std_bar.append(std_LTR)
                lat_acc_list.append(np.array(average_lat_ratio))
        plt.bar(X + bar_width*(len(RPS) - rps_count - 2), mean_bar, yerr=std_bar, width=bar_width, alpha=0.5, ecolor='black', capsize=10, label=RP)
        plt.grid(True, linestyle='--', alpha=0.7)
        rps_count += 1
    plt.xticks(X, combinations)
    plt.ylabel("Ratio of peak lateral to vertical acceleration")
    plt.legend()
    # plt.title("Ratio of Lateral Acceleration to Vertical Acceleration on different surfaces in different vehicles")
    plt.savefig(str(Path(os.getcwd()).parent.absolute()) + "/Experiments/Results/Rollover/rollover_isolated.png")
    plt.show()

    num_scenarios = len(Config["scenarios"])*len(Config["vehicle_list"])
    for i in range(num_scenarios):
        statistic, p_value = mannwhitneyu(lat_acc_list[i+num_scenarios], lat_acc_list[i+2*num_scenarios])
        print("p value: ", p_value)
    

    plt.figure().set_size_inches(6, 4)
    plt.subplots_adjust(left=0.09, right=0.99)  # Adjust the values as needed

    RPS = ["Static limiter", "No prevention", "Full RPS"]
    bar_width = 0.2
    combinations = []
    for scenario in Config["scenarios"]:
        if scenario == "smallgrid":
            scn = "Flat"
        if scenario == "small_island":
            scn = "Offroad"
        for vehicle_name in Config["vehicle_list"]:
            if vehicle_name == "flux":
                vn = "Small car"
            if vehicle_name == "rollover_test":
                vn = "Big car"
            combinations.append("{}-{}".format(scn, vn))
    rps_count = 0
    for RP in RPS:
        if RP == "No prevention":
            rollover_prevention = 0
        if RP == "Static limiter":
            rollover_prevention = 2
        if RP == "Full RPS":
            rollover_prevention = 1
        scenario_count = 0
        total_scenarios = len(Config["scenarios"])*len(Config["vehicle_list"])
        X = np.arange(total_scenarios)
        mean_bar = []
        std_bar = []
        for scenario in Config["scenarios"]:
            for vehicle_name in Config["vehicle_list"]:
                average_lat_ratio = []
                rollover = []
                min_az = []
                track_width = Config["vehicles"][vehicle_name]["track_width"]/2
                for trial in range(Config["num_iters"]):
                    dir_name = str(Path(os.getcwd()).parent.absolute()) + "/Experiments/Results/Rollover/{}/".format(str(rollover_prevention))
                    filename = dir_name + "/{}-{}-{}.npy".format(scenario, vehicle_name, str(trial))
                    ## data has structure: state(17) flipped_over(1) rollover_prevention(1) intervention(1) Rollover_detected(1) delta_steering(1) turn_time(1) rotate_speed(1) ts(1) ))
                    data = np.load(filename)
                    rollover.append(data[-1, 17])
                rollover_rate = np.mean(np.array(rollover))
                mean_bar.append(rollover_rate)
        plt.bar(X + bar_width*(len(RPS) - rps_count - 2), mean_bar, width=bar_width, alpha=0.5, ecolor='black', capsize=10, label=RP)
        plt.grid(True, linestyle='--', alpha=0.7)
        rps_count += 1
    plt.xticks(X, combinations)
    plt.ylabel("Rollover Rate")
    legend_position = (0.5, 0.75)  # Specify the position as (x, y)
    plt.legend(loc=legend_position)
    # plt.title("Ratio of Lateral Acceleration to Vertical Acceleration on different surfaces in different vehicles")
    plt.savefig(str(Path(os.getcwd()).parent.absolute()) + "/Experiments/Results/Rollover/rollover_rate_isolated.png")
    plt.show()


    SI_vert_acc_list = []
    full_vert_acc_list = []
    # fig.suptitle("Minimum Vertical Acceleration vs Rollover Rate")
    for vn in ["small car", "big car"]:
        if vn == "small car":
            vehicle_name = "flux"
        if vn == "big car":
            vehicle_name = "rollover_test"
        for scenario in Config["scenarios"]:
            for RP in ["static", "static+dynamic"]:
                if RP == "none":
                    rollover_prevention = 0
                if RP == "static":
                    rollover_prevention = 2
                if RP == "static+dynamic":
                    rollover_prevention = 1
                min_az = []
                rollover = []
                for trial in range(Config["num_iters"]):
                    dir_name = str(Path(os.getcwd()).parent.absolute()) + "/Experiments/Results/Rollover/{}/".format(str(rollover_prevention))
                    filename = dir_name + "/{}-{}-{}.npy".format(scenario, vehicle_name, str(trial))
                    ## data has structure: state(17) flipped_over(1) rollover_prevention(1) intervention(1) Rollover_detected(1) delta_steering(1) turn_time(1) rotate_speed(1) ts(1) ))
                    data = np.load(filename)
                    turn_index = np.where(data[:,-1] >= data[-1, 22])[0]
                    az = data[turn_index, 11]
                    roll = data[turn_index, 3]
                    index = np.where(np.abs(roll) < 10/57.3)[0]
                    az = az[index]
                    min_az.append(np.min(np.abs(az)))
                    rollover.append(data[-1, 17])
                    if(scenario == "small_island"):
                        SI_vert_acc_list.append(np.min(np.abs(az)))
                    else:
                        full_vert_acc_list.append(np.min(np.abs(az)))

                min_az = np.mean(np.array(min_az))
                rollover_rate = np.mean(np.array(rollover))

    print("SI_min_vert_acc", np.mean(np.array(SI_vert_acc_list)))
    print("full_min_vert_acc", np.mean(np.array(full_vert_acc_list)))

    fig, ax1 = plt.subplots(1,1)
    fig.set_size_inches(5, 5)
    plt.subplots_adjust(left=0.12, right=0.99)  # Adjust the values as needed

    ax2 = ax1.twiny()
    ax1.set_ylabel("Rollover Rate")
    ax1.set_xlabel("Operational speed for small car (m/s)")
    ax2.set_xlabel("Operational speed for big car (m/s)")
    line_list = []
    color_count = 0
    for vn in ["small car", "big car"]:
        if vn == "small car":
            vehicle_name = "flux"
        if vn == "big car":
            vehicle_name = "rollover_test"
        for scenario in Config_crash["scenarios"]:
            if scenario == "smallgrid":
                scn = "Flat"
            if scenario == "small_island":
                scn = "Offroad"
            for RP in ["static+dynamic"]:
                if RP == "none":
                    rollover_prevention = 0
                if RP == "static":
                    rollover_prevention = 2
                if RP == "static+dynamic":
                    rollover_prevention = 1

                rollover_rate = []
                rotate_speed = []
                for trial in range(Config_crash["num_iters"]):
                    rollover = []
                    for attempts in range(Config_crash["speed_iters"]):
                        dir_name = str(Path(os.getcwd()).parent.absolute()) + "/Experiments/Results/Rollover/speed_crash_{}/".format(str(rollover_prevention))
                        filename = dir_name + "/{}-{}-{}-{}.npy".format(scenario, vehicle_name, str(trial), str(attempts))
                        ## data has structure: state(17) flipped_over(1) rollover_prevention(1) intervention(1) Rollover_detected(1) delta_steering(1) turn_time(1) rotate_speed(1) ts(1) ))
                        data = np.load(filename)
                        rollover.append(data[-1, 17])
                        speed = data[-1,-2] # - Config_crash["vehicles"][vehicle_name]["rotate_speed"])/Config_crash["vehicles"][vehicle_name]["rollover_speed_max"]
                    rollover_rate.append(np.mean(np.array(rollover)))
                    rotate_speed.append(speed)
            # rotate_speed = np.arange(Config_crash["speed_iters"])/Config_crash["speed_iters"]
            color = plt.cm.tab10(color_count)  # Choose the same color from the 'tab10' colormap
            color_count += 1
            if vn == "small car":
                line, = ax1.plot(rotate_speed, rollover_rate, label="{}-{}".format(scn, vn), color=color)
            else:
                line, = ax2.plot(rotate_speed, rollover_rate, label="{}-{}".format(scn, vn), color=color)
            line_list.append(line)
    labels = [line.get_label() for line in line_list]
    ax1.legend(line_list, labels, loc='upper left')
    plt.savefig(str(Path(os.getcwd()).parent.absolute()) + "/Experiments/Results/Rollover/rollover_speed_crash.png")
    plt.show()


if __name__ == "__main__":
    ## add a parser:
    parser = argparse.ArgumentParser(description="Plot the accuracy of the models")
    parser.add_argument("--config_name", "-c", default="Rollover_Config.yaml", type=str, help="Path to the config file. Keep the same as the one used for evaluation")
    parser.add_argument("--config_crash_name", "-cc", default="Rollover_speed_crash_Config.yaml", type=str, help="Path to the config file. Keep the same as the one used for evaluation")

    args = parser.parse_args()
    config_name = args.config_name
    config_path = str(Path(os.getcwd()).parent.absolute()) + "/Experiments/Configs/" + config_name
    with open(config_path, "r") as f: 
        Config = yaml.safe_load(f)
    config_crash_name = args.config_crash_name
    config_crash_path = str(Path(os.getcwd()).parent.absolute()) + "/Experiments/Configs/" + config_crash_name
    with open(config_crash_path, "r") as f: 
        Config_crash = yaml.safe_load(f)
    ## call the plotting function, we'll extract data in there.
    Plot_metrics(Config, Config_crash)