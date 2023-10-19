import gym
import yaml
import numpy as np
from BeamNGRL.BeamNG.beamng_interface import *
from BeamNGRL.utils.planning import update_goal
from gym import spaces

class OffroadSmallIsland(gym.Env):
    def __init__(self, hal_config_path = None, config_path=None, args=None):
        if config_path is None:
            print("no config file provided!")
            exit()
        if hal_config_path is None:
            print("no hal config file provided!")
            exit()

        with open(config_path) as f:
            Config = yaml.safe_load(f)
        with open(hal_config_path) as f:
            hal_Config = yaml.safe_load(f)

        vehicle = Config["vehicle"]
        start_pos = np.array(Config["start_pos"]) ## some default start position which will be overwritten by the scenario file
        start_quat = np.array(Config["start_quat"])
        Map_config = Config["Map_config"]
        scenario = Config["scenario"] ## TODO: WP_file_offroad does not contain heading data

        self.speed_max = Config["speed_max"]
        self.speed_kp = Config["speed_kp"]
        self.speed_ki = Config["speed_ki"]
        self.speed_kd = Config["speed_kd"]
        self.speed_FF = Config["speed_FF"]

        WP_file = str(Path(os.getcwd()).parent.absolute()) + "/Experiments/Waypoints/" + scenario + ".npy"
        self.target_WP = np.load(WP_file)[:100] ## this was up to the first 100 points to prevent trying to perform the full loop.

        self.map_res = Map_config["map_res"]
        self.map_size = Map_config["map_size"]
        self.map_size_px = int(self.map_size/self.map_res)
        
        self.current_wp_index = 0  # initialize waypoint index with 0
        self.goal = None
        action = np.zeros(2)
        self.lookahead = 15

        action_low = np.array([-1.0, -1.0])  # Minimum speed and steering values
        action_high = np.array([1.0, 1.0])  # Maximum speed and steering values
        action_space = spaces.Box(low=action_low, high=action_high, shape=(2,), dtype=np.float64)
        self.action_space = action_space
        ## TODO: add camera image to observation space?
        self.observation_space = spaces.Dict({
            'height': spaces.Box(low=-Map_config["elevation_range"], high=Map_config["elevation_range"], shape=(self.map_size_px, self.map_size_px, 1), dtype=np.float32),
            'normal': spaces.Box(low=-1.0, high=1.0, shape=(self.map_size_px, self.map_size_px, 3), dtype=np.float64),
            'color':  spaces.Box(low=0, high=255, shape=(self.map_size_px, self.map_size_px, 3), dtype=np.uint8),
            'pos': spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64),
            'rpy': spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64),
            'lin_vel': spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64),
            'ang_vel': spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64),
            'accel': spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64),
            'last_action': spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float64),
        })
        self.bng_interface = get_beamng_default(
            car_model=vehicle["model"],
            start_pos=start_pos,
            start_quat=start_quat,
            car_make=vehicle["make"],
            map_config=Map_config,
            host_IP=args.host_IP,
            remote=args.remote,
            camera_config=hal_Config["camera"],
            lidar_config=hal_Config["lidar"],
            accel_config=hal_Config["mavros"],
            burn_time=Config["burn_time"],
            run_lockstep=Config["run_lockstep"],
        )
        for i in range(10):
            self.bng_interface.send_ctrl(action)
            self.bng_interface.state_poll()

    def reset(self):
        self.bng_interface.reset()
        obs, _, _, info = self.step(np.zeros(2,dtype=np.float64))
        return obs, info

    def step(self, action):
        try:
            self.bng_interface.send_ctrl(action, speed_ctrl=True, speed_max = self.speed_max, Kp=self.speed_kp, Ki=self.speed_ki, Kd=self.speed_kd, FF_gain=self.speed_FF)
            self.bng_interface.state_poll() ## this steps the environment forward (it is the equivalent of doing a "step")
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

            pos = np.copy(state[:2])
            goal, done, self.current_wp_index = update_goal(
                self.goal, pos, self.target_WP, self.current_wp_index, self.lookahead, step_size=10
            )

            damage = len(self.bng_interface.broken)

            reward = -(np.linalg.norm(goal - pos) + 100.0 * damage)

            info = dict({'timestamp': self.bng_interface.timestamp})

            return observations, reward, done, info

        except KeyboardInterrupt:
            self.bng_interface.bng.close()
            os._exit(1)
        except Exception as e:
            print(e)