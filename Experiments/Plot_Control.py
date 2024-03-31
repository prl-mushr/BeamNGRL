import numpy as np
import matplotlib.pyplot as plt
import yaml
from pathlib import Path
import os
import argparse
from scipy.stats import mannwhitneyu, t as student_t
from scipy.stats import norm

# TODO: I should have a utils directory that takes care of these statistical functions no?

def conf(data):
    # Sample size
    n = len(data)*10
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

def binomial_proportion_ci(success_arr, confidence_level=0.95):
    successes = len(np.where(success_arr)[0])
    total_trials = len(success_arr)
    # Calculate the binomial proportion
    p_hat = successes / total_trials

    # Calculate the standard error
    standard_error = np.sqrt(p_hat * (1 - p_hat) / total_trials)

    # Calculate the Z-score for the confidence level
    z_score = norm.ppf(1 - (1 - confidence_level) / 2)

    # Calculate the margin of error
    margin_of_error = z_score * standard_error

    # Calculate the confidence interval

    return margin_of_error

def Plot_metircs(Config):
    # for each scenario, plot a graph of success rate vs time taken for each model, where we set the time limit to be the maximum time taken by any model if there is damage:
    # the plot will be a scatter plot with the x-axis being the time taken and the y-axis being the success rate
    time_limit = Config["time_limit"]
    scenario_count = 0
    color_palette = 'plasma'
    original_colors = plt.get_cmap(color_palette)(range(256))

    skips = 256 // len(Config["models"])
    colors = original_colors[::skips]

    for scenario in Config["scenarios"]:
        fig, axs = plt.subplots(1,1)
        fig.set_size_inches(4, 3)
        scn_type = scenario.split('-')[0]
        
        fig.subplots_adjust(left=0.15, right=0.99, top=0.9, bottom=0.15)

        axs.set_title(scn_type.title())
        scenario_time_limit = time_limit[scenario_count]
        wp_radius = Config["wp_radius"][scenario_count]
        scenario_count += 1

        # load the waypoints corresponding to this scenario:
        waypoints = np.load(str(Path(os.getcwd()).parent.absolute()) + "/Experiments/Waypoints/" + scenario + ".npy")
        # make the waypoints evenly spaced. The waypoints are in the form [x, y, z, q_x, q_y, q_z, q_w]. We only care about making them evenly spaced in the x-y plane:
        waypoints = waypoints[:, :2]
        # I want the waypoint spacing to be 1 meter. So, create a new list of waypoints where we only add the new waypoint if it is at least 1 meter away from the last one:
        new_waypoints = [waypoints[0]]
        for i in range(1, waypoints.shape[0]):
            if np.linalg.norm(waypoints[i] - new_waypoints[-1]) >= 1:
                new_waypoints.append(waypoints[i])
        waypoints = np.array(new_waypoints)
        # find the waypoint that is wp_radius away from the goal:
        goal = waypoints[-1]
        for i in range(waypoints.shape[0]):
            if np.linalg.norm(waypoints[i] - goal) <= wp_radius:
                break
        print(i)
        waypoints = waypoints[:i+1]
        length_waypoints = waypoints.shape[0]

        count = 0
        for model in Config["models"]:
            max_progress_list = []
            time_taken = np.arange(0, scenario_time_limit, 0.04)
            for trial in range(Config["num_iters"]):
                dir_name = str(Path(os.getcwd()).parent.absolute()) + "/Experiments/Results/Control/" + model
                filename = dir_name + "/{}-trial-{}.npy".format(scenario, str(trial))
                ## data has structure: state(17), goal(2), timestamp(1), success(1), damage(1)
                data = np.load(filename)
                ## for every point in the agent's trajectory, find the closest waypoint in the list of waypoints. Do this in a vectorized manner:
                agent_trajectory = data[:, :2]
                # I want the maximum progress made vs timestep
                max_prog = np.zeros(int(scenario_time_limit*25))
                for i in range(0, int(scenario_time_limit*25)):
                    # progress is the index of the closest waypoint to the agent's position at this timestep:
                    if i < agent_trajectory.shape[0]:
                        progress = (np.argmin(np.linalg.norm(waypoints - agent_trajectory[i], axis=1)))/length_waypoints
                        # what I want to track is the maximum progress made by the agent at any timestep, so append a new progress value only if it is greater than the last one, otherwise, keep the last one:
                        if i == 0:
                            max_progress = progress
                        else:
                            max_progress = max(max_progress, progress)
                        # the time taken is the timestamp of the last timestep:
                    max_prog[i] = max_progress
                max_progress_list.append(max_prog)
            # convert progress and time taken to numpy arrays:
            max_progress_list = np.array(max_progress_list)
            # calculate the mean and std of max_progress_list:
            mean_progress = np.mean(max_progress_list, axis=0)
            # use the conf function for confidence intervals:
            conf_progress = conf(max_progress_list)
            # plot the mean progress vs time taken, with alpha = 0.5:
            color = colors[count]
            axs.plot(time_taken, mean_progress, label=model, color=color)
            axs.fill_between(time_taken, mean_progress - conf_progress, mean_progress + conf_progress, alpha=0.2, color= color)
            count += 1
        axs.set_xlabel("Time Taken")
        axs.set_ylabel("Progress Made")
        axs.legend()

        fig.savefig(str(Path(os.getcwd()).parent.absolute()) + "/Experiments/Results/Control/" + scenario + "_test.png")
        ## close the figure:
        # plt.show()
        # plt.close(fig)


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