from random import uniform
import gym
import BeamNGRL.gym
from pathlib import Path
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--hal_config_name", type=str, default="offroad.yaml", help="name of the HAL config file to use")
parser.add_argument("--config_name", type=str, default="gym_experiment.yaml", help="name of the config used for experiment specifics")
parser.add_argument("--remote", type=bool, default=True, help="whether to connect to a remote beamng server")
parser.add_argument("--host_IP", type=str, default="169.254.216.9", help="host ip address if using remote beamng")

args = parser.parse_args()
hal_config_name = args.hal_config_name
hal_config_path = str(Path(os.getcwd()).parent.absolute()) + "/Configs/" + hal_config_name
config_name = args.config_name
config_path = str(Path(os.getcwd()).parent.absolute()) + "/Experiments/Configs/" +  config_name ## anything experiment specific should be put here

env = gym.make('offroad-small-island-v0', hal_config_path=hal_config_path, config_path=config_path, args=args)
## env.config would be the experiment config dict
env.reset()
total_reward, done = 0, False
# Drive around randomly until finishing
while not done:
    obs, reward, done, aux = env.step((uniform(-1, 1), uniform(-1, 1)))
    total_reward += reward
print('Achieved reward:', total_reward)