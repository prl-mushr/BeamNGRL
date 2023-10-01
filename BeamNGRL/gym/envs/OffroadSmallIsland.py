import gym
import yaml
import numpy as np
from BeamNGRL.BeamNG.beamng_interface import *
from BeamNGRL.utils.planning import update_goal
from gym import spaces

class OffroadSmallIsland(gym.Env):
    def __init__(self, config_path, waypoint_path,  Map_config, camera_config, lidar_config, IMU_config, args):        
        self.target_WP = np.load(waypoint_path + "WP_file_offroad.npy")[:100]
        with open(config_path + 'Map_config.yaml') as f:
            Map_config = yaml.safe_load(f)

        self.map_res = Map_config["map_res"]
        self.map_size = Map_config["map_size"]
        self.map_size_px = int(self.map_size/self.map_res)

        self.bng_interface = get_beamng_default(
            car_model="offroad",
            start_pos=start_pos, ## start position in ENU (east north up). Center of the map is usually 0,0, height is terrain dependent. TODO: use the map model to estimate terrain height.
            start_quat=start_quat, ## start quaternion -- TODO: there should be a ROS to BeamNG to ROS conversion system for reference frames.
            car_make="sunburst", ## car make (company/manufacturer)
            map_config=Map_config, ## Map config; this is "necessary"
            remote=args.remote, ## are you running the simulator remotely (on a separate computer or on the same computer but outside the docker)? 
            host_IP=args.host_IP, ## if using a remote connection (usually the case when running sim on a separate computer)
            camera_config=camera_config, ## currently, camera only works on windows, so you can only use this if you have the sim running remotely or you're using windows as the host
            lidar_config=lidar_config, ## currently, lidar only works on windows, so you can only use this if the sim is running remotely or you're using a windows host
            accel_config=IMU_config, ## IMU config. if left blank, a default config is used.
            burn_time=args.step_time, ## step or dt time
            run_lockstep=True ## whether the simulator waits for control input to move forward in time. Set to true to have a gym "step" like functionality
        )
        
        self.current_wp_index = 0  # initialize waypoint index with 0
        self.goal = None
        action = np.zeros(2)
        self.lookahead = 15
        for i in range(10):
            self.bng_interface.send_ctrl(action)
            self.bng_interface.state_poll()

        action_low = np.array([-1.0, -1.0])  # Minimum speed and steering values
        action_high = np.array([1.0, 1.0])  # Maximum speed and steering values
        action_space = spaces.Box(low=action_low, high=action_high, shape=(2,), dtype=np.float64)
        self.action_space = action_space

        self.observation_space = spaces.Dict({
            'height': spaces.Box(low=-Map_config["elevation_range"], high=Map_config["elevation_range"], shape=(self.map_size_px, self.map_size_px, 1), dtype=np.float32),
            'normal': spaces.Box(low=-1.0, high=1.0, shape=(self.map_size_px, self.map_size_px, 3), dtype=np.float32),
            'color':  spaces.Box(low=0, high=255, shape=(self.map_size_px, self.map_size_px, 3), dtype=np.uint8),
            'pos': spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            'rpy': spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            'lin_vel': spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            'ang_vel': spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            'accel': spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            'last_action': spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
        })

    def reset(self, seed):
        self.bng_interface.reset()
        obs, _, _, info = self.step(np.zeros(2,dtype=np.float64))
        return obs, info

    def step(self, action):
        self.bng_interface.send_ctrl(action)
        self.bng_interface.state_poll()
        state = self.bng_interface.state
        
        observations = {
            'height': self.bng_interface.BEV_heght,
            'normal': self.bng_interface.BEV_normal,
            'color':  self.bng_interface.BEV_color,
            'pos': state[:3],
            'rpy': state[3:6],
            'lin_vel': state[6:9],
            'ang_vel': state[9:12],
            'accel': state[12:15],
            'last_action': state[15:17],
        }

        pos = np.copy(state[:2])  # example of how to get car position in world frame. All data points except for dt are 3 dimensional.
        goal, done, self.current_wp_index = update_goal(
            self.goal, pos, self.target_WP, self.current_wp_index, self.lookahead, step_size=10
        )

        damage = len(self.bng_interface.broken)

        reward = np.linalg.norm(goal - pos) + 100.0 * damage

        info = dict({'timestamp': self.bng_interface.timestamp})

        return observations, reward, done, info