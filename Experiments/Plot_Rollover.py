import numpy as np
import matplotlib.pyplot as plt
import yaml
from pathlib import Path
import os
import argparse


def Plot_metrics(Config):

    rows = len(Config["scenarios"])
    cols = len(Config["vehicle_list"])
    fig, axs = plt.subplots(rows,cols)
    fig.set_size_inches(18.5, 10.5)
    fig.suptitle("Ratio of Lateral Acceleration to Vertical Acceleration on different surfaces in different vehicles")
    for sc in ["offroad", "flat_ground"]:
        if sc == "offroad":
            scenario = "small_island"
        if sc == "flat_ground":
            scenario = "smallgrid"
        for vn in ["small car", "big car"]:
            if vn == "small car":
                vehicle_name = "flux"
            if vn == "big car":
                vehicle_name = "rollover_test"
            axs[Config["scenarios"].index(scenario), Config["vehicle_list"].index(vehicle_name)].set_title("{}-{}".format(sc, vn))
            axs[Config["scenarios"].index(scenario), Config["vehicle_list"].index(vehicle_name)].set_ylabel("Average Lateral Ratio")

    for RP in ["static", "static+dynamic", "none"]:
        if RP == "none":
            rollover_prevention = False
        if RP == "static":
            rollover_prevention = 2
        if RP == "static+dynamic":
            rollover_prevention = True
        for scenario in Config["scenarios"]:
            for vehicle_name in Config["vehicle_list"]:
                average_lat_ratio = []
                rollover = []
                for trial in range(Config["num_iters"]):
                    dir_name = str(Path(os.getcwd()).parent.absolute()) + "/Experiments/Results/Rollover/{}/".format(str(rollover_prevention))
                    filename = dir_name + "/{}-{}-{}.npy".format(scenario, vehicle_name, str(trial))
                    ## data has structure: state(17) flipped_over(1) rollover_prevention(1) intervention(1) Rollover_detected(1) delta_steering(1) turn_time(1) rotate_speed(1) ts(1) ))
                    data = np.load(filename)
                    turn_index = np.where(data[:,-1] >= data[-1, 22])[0]
                    ay = data[turn_index, 10]
                    az = data[turn_index, 11]
                    roll = data[turn_index, 3]
                    index = np.where(np.abs(roll) < 30/57.3)[0]
                    ay = ay[index]
                    az = az[index]
                    average_lat_ratio.append(np.mean(np.abs(ay))/np.mean(np.abs(az)) )
                    rollover.append(data[-1, 17])
                mean_LTR = np.mean(np.array(average_lat_ratio))
                std_LTR = np.std(np.array(average_lat_ratio))
                rollover_rate = np.mean(np.array(rollover))
                axs[Config["scenarios"].index(scenario), Config["vehicle_list"].index(vehicle_name)].bar(RP, mean_LTR, yerr=std_LTR, align='center', alpha=0.5, ecolor='black', capsize=10)
                print("Scenario: {}, Vehicle: {}, Rollover Prevention: {}, Mean Lateral Ratio: {}, Std Lateral Ratio: {}, Rollover Rate: {}".format(scenario, vehicle_name, RP, mean_LTR, std_LTR, rollover_rate))
    plt.savefig(str(Path(os.getcwd()).parent.absolute()) + "/Experiments/Results/Rollover/Results.png")
    plt.show()

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