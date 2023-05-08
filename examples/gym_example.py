from random import uniform
import gym
import BeamNGRL.gym
from pathlib import Path
import os

config_path = str(Path(os.getcwd()).parent.absolute()) + "/BeamNGRL/control/UW_mppi/Configs/"
waypoint_path = str(Path(os.getcwd()).parent.absolute()) + "/BeamNGRL/utils/waypoint_files/"
env = gym.make('offroad-small-island-v0', config_path=config_path, waypoint_path=waypoint_path)
env.reset()
total_reward, done = 0, False
# Drive around randomly until finishing
while not done:
    obs, reward, done, aux = env.step((uniform(-1, 1), uniform(-1, 1)))
    total_reward += reward
print('Achieved reward:', total_reward)