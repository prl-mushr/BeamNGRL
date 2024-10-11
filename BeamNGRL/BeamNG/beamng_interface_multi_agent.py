import cv2
import torch
import numpy as np
# from pyquaternion import Quaternion

import traceback
import time
from pathlib import Path
import os
from sys import platform

import BeamNGRL
from BeamNGRL.utils.visualisation import Vis
from beamngpy import BeamNGpy, Scenario, Vehicle
from beamngpy.sensors import Lidar, Camera, Electrics, Accelerometer, Timer, Damage
from BeamNGRL.BeamNG.agent import *
import threading

ROOT_PATH = Path(BeamNGRL.__file__).parent
DATA_PATH = ROOT_PATH.parent / ('BeamNGRL/data' if platform == "win32" else 'data')
BNG_HOME = os.environ.get('BNG_HOME')


def get_beamng_default(
        vids=None,
        ego_vid=None,
        traffic_vids=None,
        car_model='offroad',
        start_pos=None,
        start_quat=None,
        car_make='sunburst',
        beamng_path=BNG_HOME,
        map_config=None,
        path_to_maps=DATA_PATH.__str__(),
        remote=False,
        host_IP=None,
        camera_config=None,
        lidar_config=None,
        accel_config=None,
        vesc_config=None,
        burn_time=0.02,
        run_lockstep=False,
):

    if(start_pos is None):
        print("please provide a start pos! I can not spawn a car in the ether!")
        exit()
    if(start_quat is None):
        print("please provide a start quat! I can not spawn a car's rotation in the ether!")
        exit()
    if(map_config is None):
        print("please provide a map_config! I can not spawn a car in the ether!")
        exit()
    map_rotate = False
    if "rotate" in map_config:
        map_rotate = map_config["rotate"]

    bng = beamng_interface_multi_agent(BeamNG_path=beamng_path, remote=remote, host_IP=host_IP)
    bng.set_map_attributes(
        map_size=map_config["map_size"], resolution=map_config["map_res"], elevation_range=map_config["elevation_range"], path_to_maps=path_to_maps, rotate=map_rotate, map_name=map_config["map_name"]
    )
    bng.load_scenario(
        scenario_name=map_config["map_name"], vids=vids, ego_vid=ego_vid, traffic_vids=traffic_vids, car_make=car_make, car_model=car_model,
        start_pos=start_pos, start_rot=start_quat,
        camera_config=camera_config, lidar_config=lidar_config, accel_config=accel_config, vesc_config=vesc_config
    )
    bng.burn_time = burn_time
    bng.set_lockstep(run_lockstep)
    return bng

## this is the equivalent of None pizza with left beef joke. Yes I'd like one beamng simulator without the beamng simulator.
## https://en.wikipedia.org/wiki/None_pizza_with_left_beef
def get_beamng_nobeam(
        Dynamics,
        vids=None,
        ego_vid=None,
        traffic_vids=None,
        car_model='offroad',
        start_pos=None,
        start_quat=None,
        car_make='sunburst',
        beamng_path=BNG_HOME,
        map_config=None,
        path_to_maps=DATA_PATH.__str__(),
        remote=False, ## these options have no effect but are here for "compatibility"
        host_IP=None,
        camera_config=None,
        lidar_config=None,
        accel_config=None,
        vesc_config=None,
        burn_time=0.02,
        run_lockstep=False,
):

    if(start_pos is None):
        print("please provide a start pos! I can not spawn a car in the ether!")
        exit()
    if(start_quat is None):
        print("please provide a start quat! I can not spawn a car's rotation in the ether!")
        exit()
    if(map_config is None):
        print("please provide a map_config! I can not spawn a car in the ether!")
        exit()
    map_rotate = False
    if "rotate" in map_config:
        map_rotate = map_config["rotate"]

    bng = beamng_interface_multi_agent(BeamNG_path=beamng_path, use_beamng=False, dyn=Dynamics)
    bng.set_map_attributes(
        map_size=map_config["map_size"], resolution=map_config["map_res"], elevation_range=map_config["elevation_range"], path_to_maps=path_to_maps, rotate=map_rotate
    )
    bng.load_scenario(
        scenario_name=map_config["map_name"], vids=vids, ego_vid=ego_vid, traffic_vids=traffic_vids, car_make=car_make, car_model=car_model,
        start_pos=start_pos, start_rot=start_quat,
    )
    bng.burn_time = burn_time
    bng.set_lockstep(run_lockstep)
    return bng


class beamng_interface_multi_agent():
    def __init__(self, BeamNG_path=BNG_HOME, host='localhost', port=64256, use_beamng=True, dyn=None, remote=False, host_IP=None, shell_mode=False, HITL_mode=False, async_mode=False):
        self.lockstep   = False
        self.BEV_center = np.zeros(3)
        self.elev_map_hgt = 2.0
        self.paused = False
        self.remote = remote

        self.agents = {}

        self.use_beamng = use_beamng
        if self.use_beamng:
            if remote==True and host_IP is not None:
                self.bng = BeamNGpy(host_IP, 64256, remote=True)
                self.bng.open(launch=False, deploy=False)
            elif remote==True and host_IP is None:
                print("~Ara Ara! Trying to run BeamNG remotely without providing any host IP?")
                exit()
            else:
                self.bng = BeamNGpy(host, port, home=BeamNG_path, user=BeamNG_path + '/userfolder')
                self.bng.open()
        elif shell_mode:
            self.dyn = dyn
            self.state = torch.zeros(17, dtype=dyn.dtype, device=dyn.d)
            self.vis = Vis()

    def load_scenario(self, scenario_name='small_island', vids=["default"], ego_vid="default", traffic_vids=[], car_make='sunburst', car_model='offroad',
                      start_pos=[np.array([-67, 336, 34.5])], start_rot=[np.array([0, 0, 0.3826834, 0.9238795])],
                      camera_config=None, lidar_config=None, accel_config=None, vesc_config=None,
                      time_of_day=1200, hide_hud=False):
        self.scenario = Scenario(scenario_name, name="test integration")
        self.traffic_vids = traffic_vids

        # creates vehicles
        num_agents = len(start_pos)
        if num_agents != len(start_rot) or num_agents != len(vids):
            raise IndexError("The lists defining agents (start_pos, start_rot, vids) don't have the same length. Make sure the lists have the same length and the coresponding indexes refer to the same agents")
        if not set(traffic_vids).issubset(vids):
            raise IndexError("The traffic_vids is not a subset of all agent vids")
        if ego_vid not in vids:
            raise IndexError("The given ego_vid is not in the list of all vids")


        self.ego_vid = ego_vid
        for i in range(len(vids)):
            start_pos[i][2] = self.get_height(start_pos[i])
            self.agents[vids[i]] = get_agent(vid=vids[i], use_beamng=self.use_beamng, remote=self.remote)
            vehicle = self.agents[vids[i]].create_vehicle(car_make='sunburst', car_model='offroad', start_pos=start_pos[i], 
                start_rot=start_rot[i], camera_config=camera_config, lidar_config=lidar_config, accel_config=accel_config, 
                vesc_config=vesc_config, bng=self.bng)
            self.scenario.add_vehicle(vehicle, pos=start_pos[i], rot_quat=start_rot[i])

        self.bng.set_tod(time_of_day/2400)

        self.bng.set_deterministic()

        self.scenario.make(self.bng)
        if(hide_hud):
            self.bng.hide_hud()

        self.bng.load_scenario(self.scenario)
        self.bng.start_scenario()

        for vid, agent in self.agents.items():
            agent.load_vehicle(car_make='sunburst', car_model='offroad', start_pos=start_pos, 
            start_rot=start_rot, camera_config=camera_config, lidar_config=lidar_config, accel_config=accel_config, 
            vesc_config=vesc_config, bng=self.bng)

        self.bng.start_traffic(self.agents[vid].vehicle for vid in traffic_vids)
        self.bng.switch_vehicle(self.agents[self.ego_vid].vehicle)

        
    def set_map_attributes(self, map_size = 16, resolution = 0.25, path_to_maps=DATA_PATH.__str__(), rotate=False, elevation_range=2.0, map_name="small_island"):
        self.elevation_map_full = np.load(path_to_maps + f'/map_data/{map_name}/elevation_map.npy', allow_pickle=True)
        self.color_map_full = cv2.imread(path_to_maps + f'/map_data/{map_name}/color_map.png')
        self.segmt_map_full = cv2.imread(path_to_maps + f'/map_data/{map_name}/segmt_map.png')
        self.path_map_full  = cv2.imread(path_to_maps + f'/map_data/{map_name}/paths.png')
        self.image_shape    = self.color_map_full.shape
        self.image_resolution = 0.1  # this is the original meters per pixel resolution of the image
        self.resolution     = resolution  # meters per pixel of the target map
        self.resolution_inv = 1/self.resolution  # pixels per meter
        self.map_size       = map_size/2  # 16 x 16 m grid around the car by default
        self.rotate = rotate
        self.elev_map_hgt = elevation_range

        if(self.image_resolution != self.resolution):
            scale_factor = self.image_resolution/self.resolution
            new_shape = np.array(np.array(self.image_shape) * scale_factor, dtype=np.int32)
            self.elevation_map_full = cv2.resize(self.elevation_map_full, (new_shape[0], new_shape[1]), cv2.INTER_AREA)
            self.color_map_full = cv2.resize(self.color_map_full, (new_shape[0], new_shape[1]), cv2.INTER_AREA)
            self.segmt_map_full = cv2.resize(self.segmt_map_full, (new_shape[0], new_shape[1]), cv2.INTER_AREA)
            self.path_map_full  = cv2.resize(self.path_map_full, (new_shape[0], new_shape[1]), cv2.INTER_AREA)
            self.image_shape    = (new_shape[0], new_shape[1])

        self.map_size_px = int(self.map_size*self.resolution_inv)
        self.map_size_px = (self.map_size_px, self.map_size_px)
        self.mask_size   = (2 * self.map_size_px[0], 2 * self.map_size_px[1])
        mask             = np.zeros(self.mask_size, np.uint8)
        self.mask        = cv2.circle(mask, self.map_size_px, self.map_size_px[0], 255, thickness=-1)
        self.mask_center = (self.map_size_px[0], self.map_size_px[1])

        self.inpaint_mask = np.zeros_like(self.elevation_map_full, dtype=np.uint8)
        index = np.where(self.elevation_map_full == 0)
        self.inpaint_mask[index] = 255

        # creates marker image 
        self.marker_width = int(self.map_size*self.resolution_inv/8)
        self.overlay_image = np.zeros([self.marker_width, self.marker_width, 3])
        cv2.rectangle(self.overlay_image, (int(self.marker_width / 3), 0), (int(self.marker_width * 2 / 3), self.marker_width), (255, 255, 255), -1) 
        cv2.circle(self.overlay_image, (int(self.marker_width / 2), int(self.marker_width / 4)), int(self.marker_width / 4), (255, 255, 255), -1)

    def get_map_bf_no_rp(self, map_img, gen_mask=False, inpaint_mask = None):
        ch = len(map_img.shape)
        if(ch==3):
            BEV = map_img[self.Y_min:self.Y_max, self.X_min:self.X_max, :]
        else:
            BEV = map_img[self.Y_min:self.Y_max, self.X_min:self.X_max]

        if inpaint_mask is not None:
            BEV = cv2.inpaint(BEV, inpaint_mask, ch, cv2.INPAINT_TELEA)
        
        if(self.rotate):
            # get rotation matrix using yaw:
            rotate_matrix = cv2.getRotationMatrix2D(center=self.mask_center, angle= self.agents[self.ego_vid].rpy[2]*57.3, scale=1)
            # rotate the image using cv2.warpAffine
            BEV = cv2.warpAffine(src=BEV, M=rotate_matrix, dsize=self.mask_size)
            # mask:
            if not gen_mask:
                BEV = cv2.bitwise_and(BEV, BEV, mask=self.mask)

        return BEV

    def transform_world_to_bodyframe(x, y, xw, yw, th):
        x -= xw
        y -= yw
        R = np.zeros((2,2))
        ct, st = np.cos(-th), np.sin(-th)
        R[0,0], R[0,1], R[1,0], R[1,1] = ct, -st, st, ct
        X = np.array(x)
        Y = np.array(y)
        V = np.array([X,Y])
        O = np.matmul(R, V)
        x, y = O[0,:], O[1,:]
        return x, y
 
    def get_height(self, pos):
        elevation_img_X = np.clip(int( pos[0]*self.resolution_inv + self.image_shape[0]//2), self.map_size*self.resolution_inv, self.image_shape[0] - 1 - self.map_size*self.resolution_inv)
        elevation_img_Y = np.clip(int( pos[1]*self.resolution_inv + self.image_shape[1]//2), self.map_size*self.resolution_inv, self.image_shape[0] - 1 - self.map_size*self.resolution_inv)

        return self.elevation_map_full[int( np.round(elevation_img_Y) ), int( np.round(elevation_img_X) )]

    def gen_BEVmap(self):
        self.img_X = np.clip(int( self.agents[self.ego_vid].pos[0]*self.resolution_inv + self.image_shape[0]//2), self.map_size*self.resolution_inv, self.image_shape[0] - 1 - self.map_size*self.resolution_inv)
        self.img_Y = np.clip(int( self.agents[self.ego_vid].pos[1]*self.resolution_inv + self.image_shape[1]//2), self.map_size*self.resolution_inv, self.image_shape[0] - 1 - self.map_size*self.resolution_inv)

        self.Y_min = int(self.img_Y - self.map_size*self.resolution_inv)
        self.Y_max = int(self.img_Y + self.map_size*self.resolution_inv)

        self.X_min = int(self.img_X - self.map_size*self.resolution_inv)
        self.X_max = int(self.img_X + self.map_size*self.resolution_inv)

        ## inputs:
        local_inpaint = self.get_map_bf_no_rp(self.inpaint_mask, gen_mask=True)
        self.BEV_color = self.get_map_bf_no_rp(self.color_map_full, inpaint_mask=local_inpaint)  # crops circle, rotates into body frame
        self.BEV_heght = self.get_map_bf_no_rp(self.elevation_map_full, inpaint_mask=local_inpaint)
        self.BEV_segmt = self.get_map_bf_no_rp(self.segmt_map_full, inpaint_mask=local_inpaint)
        self.BEV_path  = self.get_map_bf_no_rp(self.path_map_full)


        # car overlay on map
        marker_size = int(self.map_size*self.resolution_inv/16)
        car_shapes = np.zeros_like(self.BEV_color, np.uint8)
        car_shapes = np.pad(car_shapes, ((marker_size, marker_size), (marker_size, marker_size), (0, 0)), 'edge')

        self.img_X_agents = {} 
        self.img_Y_agents = {}

        self.Y_min_agents = {}
        self.Y_max_agents = {}

        self.X_min_agents = {}
        self.X_max_agents = {}
        for vid, agent in self.agents.items():
            self.img_X_agents[vid] = np.clip(int( agent.pos[0]*self.resolution_inv + self.image_shape[0]//2), self.map_size*self.resolution_inv, self.image_shape[0] - 1 - self.map_size*self.resolution_inv)
            self.img_Y_agents[vid] = np.clip(int( agent.pos[1]*self.resolution_inv + self.image_shape[1]//2), self.map_size*self.resolution_inv, self.image_shape[0] - 1 - self.map_size*self.resolution_inv)

            if ((self.img_X_agents[vid] - self.img_X) ** 2 < (self.map_size*self.resolution_inv) ** 2 and 
                (self.img_Y_agents[vid] - self.img_Y) ** 2 < (self.map_size*self.resolution_inv) ** 2):
                self.Y_min_agents[vid] = int(self.img_Y_agents[vid] - self.Y_min - marker_size) + marker_size
                self.Y_max_agents[vid] = int(self.img_Y_agents[vid] - self.Y_min + marker_size) + marker_size

                self.X_min_agents[vid] = int(self.img_X_agents[vid] - self.X_min - marker_size) + marker_size
                self.X_max_agents[vid] = int(self.img_X_agents[vid] - self.X_min + marker_size) + marker_size

                agent_marker_rotation = - agent.rpy[2] * 180 / np.pi + 90

                image_center = tuple(np.array(self.overlay_image.shape[1::-1]) / 2)
                rot_mat = cv2.getRotationMatrix2D(image_center, int(agent_marker_rotation), 1.0)
                result = cv2.warpAffine(self.overlay_image, rot_mat, self.overlay_image.shape[1::-1], flags=cv2.INTER_LINEAR)

                car_shapes[self.Y_min_agents[vid]:self.Y_max_agents[vid], self.X_min_agents[vid]:self.X_max_agents[vid]] = result

        # applies mask
        car_shapes = car_shapes[marker_size:-marker_size, marker_size:-marker_size]
        mask = car_shapes.astype(bool)

        self.BEV_color[mask] = cv2.addWeighted(car_shapes, 0.7, self.BEV_color, 0.3, 0)[mask]

        self.BEV_center[:2] = self.agents[self.ego_vid].pos[:2]
        self.BEV_center[2] = self.BEV_heght[self.map_size_px[0], self.map_size_px[1]]
        self.BEV_heght -= self.BEV_center[2]
        self.BEV_heght = np.clip(self.BEV_heght, -self.elev_map_hgt, self.elev_map_hgt)
        self.BEV_heght = np.nan_to_num(self.BEV_heght, copy=False, nan=0.0, posinf=self.elev_map_hgt, neginf=-self.elev_map_hgt)
        self.BEV_normal = self.compute_surface_normals()


    def increase_brightness(self, img, value=30):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        lim = 255 - value
        v[v > lim] = 255
        v[v <= lim] += value

        final_hsv = cv2.merge((h, s, v))
        img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        return img

    def compute_surface_normals(self):
        # Compute the gradient of the elevation map using the Sobel operator
        BEV_normal = np.copy(self.BEV_heght)
        BEV_normal = cv2.resize(BEV_normal, (int(self.map_size_px[0]*4), int(self.map_size_px[0]*4)), cv2.INTER_AREA)
        BEV_normal = cv2.GaussianBlur(BEV_normal, (3,3), 0)
        normal_x = -cv2.Sobel(BEV_normal, cv2.CV_64F, 1, 0, ksize=3)
        normal_y = -cv2.Sobel(BEV_normal, cv2.CV_64F, 0, 1, ksize=3)
        # Compute the normal vector as the cross product of the x and y gradients
        normal_z = np.ones_like(BEV_normal)
        normals = np.stack([normal_x, normal_y, normal_z], axis=-1)
        # Normalize the normal vectors
        norms = np.linalg.norm(normals, axis=-1, keepdims=True)
        normals = normals / norms
        normals = cv2.resize(normals, (int(self.map_size_px[0]*2), int(self.map_size_px[0]*2)), cv2.INTER_AREA)
        return normals

    def send_ctrl(self, actions, speed_ctrl=False, speed_max = 1, Kp = 1, Ki =  1, Kd = 0, FF_gain = 1):
        for vid, action in actions.items():
            if (vid not in self.traffic_vids):
                self.agents[vid].send_ctrl(action, speed_ctrl=speed_ctrl, speed_max=speed_max, Kp=Kp, Ki=Ki, Kd=Kd, FF_gain=FF_gain)

    def reset(self, vids):
        for vid in vids:
            self.agents[vid].reset()

    def get_state(self, vid):
        return self.agents[vid].state;

    def set_lockstep(self, lockstep):
        self.lockstep = lockstep
        if self.lockstep:
            self.paused = False ## assume initially not paused
        else:
            self.paused = True ## assume initially paused

    def handle_timing(self):
        if(self.lockstep):
            if not self.paused:
                self.bng.pause()
                self.paused = True
            self.bng.step(1)
        else:
            if self.paused:
                self.bng.resume()
                self.paused = False

        for vid, agent in self.agents.items():
            agent.state_poll()

        self.gen_BEVmap()

    def step(self, action):
        for vid, agent in self.agents.items():
            agent.step(action)

        self.gen_BEVmap()

        BEV_heght = torch.from_numpy(self.BEV_heght).to(device=self.dyn.d, dtype=self.dyn.dtype)
        BEV_normal = torch.from_numpy(self.BEV_normal).to(device=self.dyn.d, dtype=self.dyn.dtype)
        self.dyn.set_BEV(BEV_heght, BEV_normal)


    def render(self, goal):
        vis_state = self.get_state(ego_vid).cpu().numpy()
        self.vis.setcar(pos=np.zeros(3), rpy=vis_state[3:6])
        self.vis.setgoal(goal - vis_state[:2])
        self.vis.set_terrain(self.BEV_heght, self.resolution, self.map_size)
