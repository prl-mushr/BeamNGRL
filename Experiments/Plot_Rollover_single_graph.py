import numpy as np
import matplotlib.pyplot as plt
import yaml
from pathlib import Path
import os
import argparse
import seaborn as sns

# Set Seaborn color palette to "colorblind"
sns.set_palette("colorblind")

def Plot_metrics(Config):
    plt.figure().set_size_inches(10.5, 5.5)

    RPS = ["Static limiter", "No prevention", "Static limiter with feedback"]
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
        if RP == "Static limiter with feedback":
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
                    rollover.append(data[-1, 17])
                mean_LTR = np.mean(np.array(average_lat_ratio))
                std_LTR = np.std(np.array(average_lat_ratio))
                rollover_rate = np.mean(np.array(rollover))
                min_az = np.mean(np.array(min_az))
                mean_bar.append(mean_LTR)
                std_bar.append(std_LTR)
        plt.bar(X + bar_width*(len(RPS) - rps_count - 2), mean_bar, yerr=std_bar, width=bar_width, alpha=0.5, ecolor='black', capsize=10, label=RP)
        plt.grid(True, linestyle='--', alpha=0.7)
        rps_count += 1
    plt.xticks(X, combinations)
    plt.ylabel("Ratio of peak lateral to vertical acceleration")
    plt.legend()
    # plt.title("Ratio of Lateral Acceleration to Vertical Acceleration on different surfaces in different vehicles")
    plt.savefig(str(Path(os.getcwd()).parent.absolute()) + "/Experiments/Results/Rollover/rollover_isolated.png")
    plt.show()


    plt.figure().set_size_inches(10.5, 5.5)

    RPS = ["Static limiter", "No prevention", "Static limiter with feedback"]
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
        if RP == "Static limiter with feedback":
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
    plt.legend()
    # plt.title("Ratio of Lateral Acceleration to Vertical Acceleration on different surfaces in different vehicles")
    plt.savefig(str(Path(os.getcwd()).parent.absolute()) + "/Experiments/Results/Rollover/rollover_rate_isolated.png")
    plt.show()


    # fig, axs = plt.subplots(1,1)
    # fig.set_size_inches(18.5, 10.5)
    # # fig.suptitle("Minimum Vertical Acceleration vs Rollover Rate")
    # for vn in ["small car", "big car"]:
    #     if vn == "small car":
    #         vehicle_name = "flux"
    #     if vn == "big car":
    #         vehicle_name = "rollover_test"
    #     axs.set_title("{}".format(vn))
    #     axs.set_ylabel("Rollover Rate")
    #     axs.set_xlabel("Minimum Vertical Acceleration")
    #     for scenario in Config["scenarios"]:
    #         for RP in ["static", "static+dynamic", "none"]:
    #             if RP == "none":
    #                 rollover_prevention = 0
    #             if RP == "static":
    #                 rollover_prevention = 2
    #             if RP == "static+dynamic":
    #                 rollover_prevention = 1
    #             min_az = []
    #             rollover = []
    #             for trial in range(Config["num_iters"]):
    #                 dir_name = str(Path(os.getcwd()).parent.absolute()) + "/Experiments/Results/Rollover/{}/".format(str(rollover_prevention))
    #                 filename = dir_name + "/{}-{}-{}.npy".format(scenario, vehicle_name, str(trial))
    #                 ## data has structure: state(17) flipped_over(1) rollover_prevention(1) intervention(1) Rollover_detected(1) delta_steering(1) turn_time(1) rotate_speed(1) ts(1) ))
    #                 data = np.load(filename)
    #                 turn_index = np.where(data[:,-1] >= data[-1, 22])[0]
    #                 az = data[turn_index, 11]
    #                 roll = data[turn_index, 3]
    #                 index = np.where(np.abs(roll) < 0.5)[0]
    #                 az = az[index]
    #                 min_az.append(np.min(np.abs(az)))
    #                 rollover.append(data[-1, 17])
    #             min_az = np.mean(np.array(min_az))
    #             rollover_rate = np.mean(np.array(rollover))
    #             if(RP != "none"):
    #                 axs.scatter(min_az, rollover_rate, label=RP)
    #             print("Scenario: {}, Vehicle: {}, Minimum Vertical Acceleration: {}, Rollover Rate: {}".format(scenario, vehicle_name, min_az, rollover_rate))
    # # axs.legend()
    # plt.show()

if __name__ == "__main__":
    ## add a parser:
    parser = argparse.ArgumentParser(description="Plot the accuracy of the models")
    parser.add_argument("--config_name", "-c", default="Rollover_Config.yaml", type=str, help="Path to the config file. Keep the same as the one used for evaluation")

    args = parser.parse_args()
    config_name = args.config_name
    config_path = str(Path(os.getcwd()).parent.absolute()) + "/Experiments/Configs/" + config_name
    with open(config_path, "r") as f: 
        Config = yaml.safe_load(f)
    ## call the plotting function, we'll extract data in there.
    Plot_metrics(Config)