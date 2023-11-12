from random import uniform
import gym
import BeamNGRL.gym
from pathlib import Path
import os
import argparse
import torch
import numpy as np
import cv2

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

dtype = torch.float
device = torch.device("cuda")

with torch.no_grad():
    for model_name in env.Config["model"]:
        env.reset()
        total_reward, done = 0, False
        # Drive around randomly until finishing
        model_path = "/root/catkin_ws/src/BeamNGRL/data/CCIL/" +  model_name + ".pt"
        model = torch.jit.load(model_path)
        model.eval()
        actions = np.zeros(2)
        counter = 0
        while not done and counter < 1000:
            obs, reward, done, aux = env.step(actions)
            cv2.imshow('path', obs['path'])
            cv2.waitKey(1)
            actions = model.forward(torch.from_numpy(obs['path']).to(device=device, dtype=dtype), torch.from_numpy(obs['state']).to(device=device, dtype=dtype)).cpu().numpy()
            actions = np.array(actions, dtype=np.float64)
            print(obs['state'][5])
            total_reward += reward
            counter += 1
        print('Achieved reward:', total_reward)