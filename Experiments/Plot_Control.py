import numpy as np
import matplotlib.pyplot as plt
import yaml
from pathlib import Path
import os
import argparse

def Plot_metircs(Config):
    # create a new graph for each scenario:
    for scenario in Config["scenarios"]:
        fig, axs = plt.subplots(2,4)
        fig.set_size_inches(18.5, 10.5)
        fig.suptitle(scenario)
        axs[0,0].set_title("Success Rate")
        axs[0,1].set_title("Damage Rate")
        axs[0,2].set_title("Time Taken")
        axs[0,3].set_title("Roll Rate")
        axs[1,0].set_title("Pitch Rate")
        axs[1,1].set_title("Acc X")
        axs[1,2].set_title("Acc Y")
        axs[1,3].set_title("Acc Z")
        ## for each model, we need to extract the data:

        for model in Config["models"]:
            success_rate = []
            damage_rate = []
            time_taken = []
            roll_rate = []
            pitch_rate = []
            acc_x = []
            acc_y = []
            acc_z = []

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
                ## extract the rp rate, acceleration in xyz:
                roll_rate.append(np.abs(data[:, 12]).mean())
                pitch_rate.append(np.abs(data[:, 13]).mean())
                acc_x.append(np.abs(data[:, 9]).mean())
                acc_y.append(np.abs(data[:, 10]).mean())
                acc_z.append(np.abs(data[:, 11]).mean())
            ## take mean and std for all:
            success_rate = np.array(success_rate)
            damage_rate = np.array(damage_rate)
            time_taken = np.array(time_taken)
            roll_rate = np.array(roll_rate)
            pitch_rate = np.array(pitch_rate)
            acc_x = np.array(acc_x)
            acc_y = np.array(acc_y)
            acc_z = np.array(acc_z)
            ## now take mean and std:
            success_rate_mean = success_rate.mean()
            damage_rate_mean = damage_rate.mean()
            time_taken_mean = time_taken.mean()
            time_taken_std = time_taken.std()
            roll_rate_mean = roll_rate.mean()
            roll_rate_std = roll_rate.std()
            pitch_rate_mean = pitch_rate.mean()
            pitch_rate_std = pitch_rate.std()
            acc_x_mean = acc_x.mean()
            acc_x_std = acc_x.std()
            acc_y_mean = acc_y.mean()
            acc_y_std = acc_y.std()
            acc_z_mean = acc_z.mean()
            acc_z_std = acc_z.std()
            ## now plot the data:
            ## success and damage don't need standard deviation:
            axs[0,0].bar(model, success_rate_mean, align='center', alpha=0.5, ecolor='black', capsize=10)  #.set_title("Success Rate")
            axs[0,1].bar(model, damage_rate_mean, align='center', alpha=0.5, ecolor='black', capsize=10)  #.set_title("Damage Rate")
            ## other metrics need standard deviation:
            axs[0,2].bar(model, time_taken_mean, yerr=time_taken_std, align='center', alpha=0.5, ecolor='black', capsize=10)  #.set_title("Time Taken")
            axs[0,3].bar(model, roll_rate_mean, yerr=roll_rate_std, align='center', alpha=0.5, ecolor='black', capsize=10)  #.set_title("Roll Rate")
            axs[1,0].bar(model, pitch_rate_mean, yerr=pitch_rate_std, align='center', alpha=0.5, ecolor='black', capsize=10)  #.set_title("Pitch Rate")
            axs[1,1].bar(model, acc_x_mean, yerr=acc_x_std, align='center', alpha=0.5, ecolor='black', capsize=10)  #.set_title("Acc X")
            axs[1,2].bar(model, acc_y_mean, yerr=acc_y_std, align='center', alpha=0.5, ecolor='black', capsize=10)  #.set_title("Acc Y")
            axs[1,3].bar(model, acc_z_mean, yerr=acc_z_std, align='center', alpha=0.5, ecolor='black', capsize=10)  #.set_title("Acc Z")
        fig.legend()

        fig.savefig(str(Path(os.getcwd()).parent.absolute()) + "/Experiments/Results/Control/" + scenario + ".png")
        ## close the figure:
        # plt.show()
        plt.close(fig)


if __name__ == "__main__":
    ## add a parser:
    parser = argparse.ArgumentParser(description="Plot the accuracy of the models")
    parser.add_argument("--config_name", "-c", default="Test_Config.yaml", type=str, help="Path to the config file. Keep the same as the one used for evaluation")

    args = parser.parse_args()
    config_name = args.config_name
    config_path = str(Path(os.getcwd()).parent.absolute()) + "/Experiments/Configs/" + config_name
    with open(config_path, "r") as f: 
        Config = yaml.safe_load(f)
    ## call the plotting function, we'll extract data in there.
    Plot_metircs(Config)