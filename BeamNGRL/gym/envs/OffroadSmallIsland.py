import gym
import yaml
import numpy as np
from BeamNGRL.BeamNG.beamng_interface import *
from BeamNGRL.utils.planning import update_goal
from gym import spaces

class OffroadSmallIsland(gym.Env):
    def __init__(self, config_path, waypoint_path):        
        self.target_WP = np.load(waypoint_path + "WP_file_offroad.npy")[:100]
        with open(config_path + 'Map_config.yaml') as f:
            Map_config = yaml.safe_load(f)

        self.map_res = Map_config["map_res"]
        self.map_size = Map_config["map_size"]
        self.map_size_px = int(self.map_size/self.map_res)


        self.bng_interface = get_beamng_default(
            car_model='RACER',
            start_pos=np.array([-67, 336, 34.5]),
            start_quat=np.array([0, 0, 0.3826834, 0.9238795]),
            map_name="small_island",
            car_make='sunburst',
            beamng_path=BNG_HOME,
            map_res=Map_config["map_res"],
            map_size=Map_config["map_size"]
        )
        self.bng_interface.set_lockstep(True)
        
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
            'height': spaces.Box(low=-2.0, high=2.0, shape=(self.map_size_px, self.map_size_px, 1), dtype=np.float32),
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