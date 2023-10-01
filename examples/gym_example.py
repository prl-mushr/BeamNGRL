from random import uniform
import gym
import BeamNGRL.gym
from pathlib import Path
import os

parser = argparse.ArgumentParser()
parser.add_argument("--hal_config_name", type=str, default="offroad.yaml", help="name of the HAL config file to use")
parser.add_argument("--remote", type=bool, default=False, help="whether to connect to a remote beamng server")
parser.add_argument("--host_IP", type=str, default="169.254.216.9", help="host ip address if using remote beamng")

args = parser.parse_args()
hal_config_name = args.hal_config_name
hal_config_path = str(Path(os.getcwd()).parent.absolute()) + "/Configs/" + hal_config_name
with torch.no_grad():
    main(config_path = config_path, hal_config_path = hal_config_path, args = args) ## we run for 3 iterations because science

waypoint_path = str(Path(os.getcwd()).parent.absolute()) + "/BeamNGRL/utils/waypoint_files/" ## this should come from test-specific file!
camera_config = dict()
lidar_config = dict()
IMU_config = dict()
Map_config = dict()

env = gym.make('offroad-small-island-v0', config_path=config_path, waypoint_path=waypoint_path, Map_config=Map_config, camera_config=camera_config, lidar_config=lidar_config, IMU_config=IMU_config, args=args)
## TODO: fix gym backend.
## TODO: load configs from Config folder.
## TODO: add the parser shit

## TODO: move minimal example to this directory
env.reset()
total_reward, done = 0, False
# Drive around randomly until finishing
while not done:
    obs, reward, done, aux = env.step((uniform(-1, 1), uniform(-1, 1)))
    total_reward += reward
print('Achieved reward:', total_reward)