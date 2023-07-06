import numpy as np
import matplotlib.pyplot as plt
import yaml
from pathlib import Path
import os
import argparse

def Plot_metircs(Config):
    # create a new graph for each scenario:
    critical_SA = Config["Cost_config"]["critical_SA"]
    critical_RI = Config["Cost_config"]["critical_RI"]
    GRAVITY = 9.8

    for scenario in Config["scenarios"]:
        fig, axs = plt.subplots(1,4)
        fig.set_size_inches(18.5, 5.5)
        fig.suptitle(scenario)
        axs[0].set_title("Success Rate")
        axs[1].set_title("Damage Rate")
        axs[2].set_title("Time Taken")
        axs[3].set_title("Cost accrued")

        for model in Config["models"]:
            success_rate = []
            damage_rate = []
            time_taken = []
            cost_per_unit_time = []

            for trial in range(Config["num_iters"]):
                dir_name = str(Path(os.getcwd()).parent.absolute()) + "/Experiments/Results/Control/" + model
                filename = dir_name + "/{}-trial-{}.npy".format(scenario, str(trial))
                ## data has structure: state(17), goal(2), timestamp(1), success(1), damage(1)
                data = np.load(filename)
                ## extract the success and damage:
                success_rate.append(data[:, -2].max().mean())
                damage_rate.append(data[:, -1].max().mean())

                ## extract the time taken using the last timestamp:
                time_taken.append(data[:, -3].max().mean())
                roll = data[:, 3]
                pitch = data[:, 4]
                ay = data[:, 10]
                az = data[:, 11]
                az_denom = np.clip(np.fabs(az),1,25)
                ct = np.sqrt(np.clip(1 - (np.square(np.sin(roll)) + np.square(np.sin(pitch))), 0.01,1) )
                cost = np.clip((1/ct) - critical_SA, 0, 10) + np.clip(np.abs(ay/az_denom) - critical_RI, 0, 10) + np.clip(np.abs(az - GRAVITY) - 5.0, 0, 10.0)
                # cost /= data[:, -3].max().mean()
                cost_per_unit_time.append(cost.sum())
            ## take mean and std for all:
            success_rate = np.array(success_rate)
            damage_rate = np.array(damage_rate)
            time_taken = np.array(time_taken)
            ## now take mean and std:
            success_rate_mean = success_rate.mean()
            damage_rate_mean = damage_rate.mean()
            time_taken_mean = time_taken.mean()
            time_taken_std = time_taken.std()
            cost_per_unit_time_mean = np.array(cost_per_unit_time).mean()
            cost_per_unit_time_std = np.array(cost_per_unit_time).std()
            ## now plot the data:
            ## success and damage don't need standard deviation:
            axs[0].bar(model, success_rate_mean, align='center', alpha=0.5, ecolor='black', capsize=10) 
            axs[1].bar(model, damage_rate_mean, align='center', alpha=0.5, ecolor='black', capsize=10) 
            ## other metrics need standard deviation:
            axs[2].bar(model, time_taken_mean, yerr=time_taken_std, align='center', alpha=0.5, ecolor='black', capsize=10)
            axs[3].bar(model, cost_per_unit_time_mean, yerr=cost_per_unit_time_std, align='center', alpha=0.5, ecolor='black', capsize=10)

        fig.legend()

        fig.savefig(str(Path(os.getcwd()).parent.absolute()) + "/Experiments/Results/Control/" + scenario + ".png")
        ## close the figure:
        # plt.show()
        plt.close(fig)


if __name__ == "__main__":
    ## add a parser:
    parser = argparse.ArgumentParser(description="Plot the accuracy of the models")
    parser.add_argument("--config_name", "-c", default="Control_Eval_Config.yaml", type=str, help="Path to the config file. Keep the same as the one used for evaluation")

    args = parser.parse_args()
    config_name = args.config_name
    config_path = str(Path(os.getcwd()).parent.absolute()) + "/Experiments/Configs/" + config_name
    with open(config_path, "r") as f: 
        Config = yaml.safe_load(f)
    ## call the plotting function, we'll extract data in there.
    Plot_metircs(Config)