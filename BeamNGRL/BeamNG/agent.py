import cv2
import torch
import numpy as np
from pyquaternion import Quaternion

import traceback
import time
from pathlib import Path
import os

import BeamNGRL
from BeamNGRL.utils.visualisation import Vis
from beamngpy import BeamNGpy, Scenario, Vehicle
from beamngpy.sensors import Lidar, Camera, Electrics, Accelerometer, Timer, Damage
import threading

def get_agent(vid="default", use_beamng=True, remote=False):
    return agent(vid=vid, use_beamng=use_beamng, remote=remote)

class agent():
    def __init__(self, vid="default", use_beamng=True, remote=False):
        self.vid = vid

        self.lidar_list = []
        self.lidar_fps = 10
        self.new_lidar = False
        self.last_lidar_time = 0
        self.camera_list = []
        self.camera_fps = 30
        self.new_cam = False
        self.last_cam_time = 0
        self.cam_segmt = False
        self.state_init = False
        self.A = np.array([0,0,9.81])
        self.last_A     = np.copy(self.A)
        self.vel_wf     = np.zeros(3)
        self.last_vel_wf= np.zeros(3)
        self.quat       = np.array([1,0,0,0])
        self.Tnb, self.Tbn = self.calc_Transform(self.quat)
        self.depth      = None
        self.pos        = None
        self.color      = None
        self.segmt      = None
        self.lidar_pts  = None
        self.Gravity    = np.array([0,0,9.81])
        self.state      = None
        self.avg_wheelspeed = 0
        self.dt = 0.02
        self.last_whspd_error = 0
        self.whspd_error_sigma = 0
        self.whspd_error_diff = 0
        self.burn_time = 0.02
        self.use_vel_diff = True
        self.remote = remote
        self.lidar_config = None
        self.camera_config = None
        self.camera = False
        self.lidar = False
        self.use_sgmt = False
        self.steering_max = 260.0

        self.use_beamng = use_beamng

    def create_vehicle(self, car_make='sunburst', car_model='offroad', start_pos=np.array([-67, 336, 34.5]), 
            start_rot=np.array([0, 0, 0.3826834, 0.9238795]), camera_config=None, lidar_config=None, 
            accel_config=None, vesc_config=None, bng=None):
        print(f"init agent {self.vid}")

        self.bng = bng
        self.start_pos = start_pos
        self.start_quat = start_rot
        if not self.use_beamng:
            self.state[:3] = torch.from_numpy(start_pos)
            self.state[3:6] = torch.from_numpy(self.rpy_from_quat(self.convert_beamng_to_REP103(start_rot)))
            return

        self.vehicle = Vehicle(self.vid, model=car_make, partConfig='vehicles/'+ car_make + '/' + car_model + '.pc')

        return self.vehicle

    def load_vehicle(self, car_make='sunburst', car_model='offroad', start_pos=np.array([-67, 336, 34.5]), 
            start_rot=np.array([0, 0, 0.3826834, 0.9238795]), camera_config=None, lidar_config=None, 
            accel_config=None, vesc_config=None, bng=None):
        self.electrics = Electrics()
        self.timer = Timer()
        self.damage = Damage()
        self.vehicle.attach_sensor('electrics', self.electrics)
        self.vehicle.attach_sensor('timer', self.timer)
        self.vehicle.attach_sensor('damage', self.damage)
        
        if accel_config == None:
            base_pos = (0,0,0.8)
        else:
            base_pos = self.ROS2BNG_bf_pos(accel_config["pos"],(0,0,0))

        self.camera_config = camera_config
        self.lidar_config = lidar_config
        self.vesc_config = vesc_config
        if self.vesc_config is not None:
            self.steering_max = self.vesc_config["steering_degrees"]
            
        if self.camera_config is not None and self.camera_config["enable"]:
            self.camera = True
            self.camera_fps = self.camera_config["fps"]
            cam_pos = self.ROS2BNG_bf_pos(self.camera_config["pos"], base_pos)
            self.use_sgmt = self.camera_config["annotation"]
            self.attach_camera(name='camera', pos=cam_pos, update_frequency=self.camera_fps, dir=self.camera_config["dir"], up=self.camera_config["up"], 
                               field_of_view_y=self.camera_config["fov"], resolution=(self.camera_config["width"],self.camera_config["height"]),
                               annotation=self.use_sgmt)

        if self.lidar_config is not None and self.lidar_config["enable"]:
            self.lidar = True
            self.lidar_fps = self.lidar_config["fps"]
            lidar_pos = self.ROS2BNG_bf_pos(self.lidar_config["pos"], base_pos)
            self.attach_lidar("lidar", pos=lidar_pos, dir=self.lidar_config["dir"], up=self.lidar_config["up"], vertical_resolution=self.lidar_config["channels"],
                             vertical_angle = self.lidar_config["vertical_angle"], rays_per_second_per_scan=self.lidar_config["rays_per_second_per_scan"],
                             update_frequency=self.lidar_fps, max_distance=self.lidar_config["max_distance"])

        self.state_poll()
        self.flipped_over = False

        return self.vehicle


    def ROS2BNG_bf_pos(self, pos, base_pos):
        return  (pos[1] + base_pos[1], -pos[0] + base_pos[1], pos[2] + base_pos[2])

    def rpy_from_quat(self, quat):
        y = np.zeros(3)
        y[0] = np.arctan2((2.0*(quat[2]*quat[3]+quat[0]*quat[1])) , (quat[0]**2 - quat[1]**2 - quat[2]**2 + quat[3]**2))
        y[1] = -np.arcsin(2.0*(quat[1]*quat[3]-quat[0]*quat[2]));
        y[2] = np.arctan2((2.0*(quat[1]*quat[2]+quat[0]*quat[3])) , (quat[0]**2 + quat[1]**2 - quat[2]**2 - quat[3]**2))
        return y

    def quat_from_rpy(self, rpy):
        u1 = np.cos(0.5*rpy[0]);
        u2 = np.cos(0.5*rpy[1]);
        u3 = np.cos(0.5*rpy[2]);
        u4 = np.sin(0.5*rpy[0]);
        u5 = np.sin(0.5*rpy[1]);
        u6 = np.sin(0.5*rpy[2]);
        quat = np.zeros(4)
        quat[0] = u1*u2*u3+u4*u5*u6;
        quat[1] = u4*u2*u3-u1*u5*u6;
        quat[2] = u1*u5*u3+u4*u2*u6;
        quat[3] = u1*u2*u6-u4*u5*u3;
        return quat

    def convert_beamng_to_REP103(self, rot):
        rot = Quaternion(rot[2], -rot[0], -rot[1], -rot[3])
        new = Quaternion([0,np.sqrt(2)/2,np.sqrt(2)/2,0])*rot
        rot = Quaternion(-new[1], -new[3], -new[0], -new[2])
        return rot

    def calc_Transform(self, quat):
        q00 = quat[0]**2;
        q11 = quat[1]**2;
        q22 = quat[2]**2;
        q33 = quat[3]**2;
        q01 =  quat[0]*quat[1];
        q02 =  quat[0]*quat[2];
        q03 =  quat[0]*quat[3];
        q12 =  quat[1]*quat[2];
        q13 =  quat[1]*quat[3];
        q23 =  quat[2]*quat[3];

        Tbn = np.zeros((3,3)) # transform body->ned
        Tbn[0][0] = q00 + q11 - q22 - q33;
        Tbn[1][1] = q00 - q11 + q22 - q33;
        Tbn[2][2] = q00 - q11 - q22 + q33;
        Tbn[0][1] = 2*(q12 - q03);
        Tbn[0][2] = 2*(q13 + q02);
        Tbn[1][0] = 2*(q12 + q03);
        Tbn[1][2] = 2*(q23 - q01);
        Tbn[2][0] = 2*(q13 - q02);
        Tbn[2][1] = 2*(q23 + q01);

        Tnb = Tbn.transpose(); # transform ned->body
        return Tnb, Tbn

    def attach_lidar(self, name, pos=(0,0,1.5), dir=(0,-1,0), up=(0,0,1), vertical_resolution=3, vertical_angle=26.9,
                     rays_per_second_per_scan=5000, update_frequency=10, max_distance=10.0):
        lidar = Lidar(name, self.bng, self.vehicle, pos = pos, dir=dir, up=up,requested_update_time=0.001, is_visualised=False,
                        vertical_resolution=3, vertical_angle=5, rays_per_second=vertical_resolution*rays_per_second_per_scan, max_distance=max_distance,
                        frequency=update_frequency, update_priority = 0,is_using_shared_memory=(not self.remote))
        self.lidar_list.append(lidar)
        print("lidar attached")

    def attach_camera(self, name, pos=(0,-2,1.4), dir=(0,-1,0), up=(0,0,1), field_of_view_y=87, resolution=(640,480),
                      depth=True, color=True, annotation=False, instance=False, near_far_planes=(0.15,60.0), update_frequency = 30, static=False):
        camera = Camera(name, self.bng, self.vehicle, pos=pos, dir=dir, up=up, field_of_view_y=field_of_view_y, resolution=resolution, update_priority=0,
                         is_render_colours=color, is_render_depth=depth, is_render_annotations=annotation,is_visualised=True,
                         requested_update_time=0.01, near_far_planes=near_far_planes, is_using_shared_memory=(not self.remote),
                         is_render_instance=instance,  is_static=static)
        self.camera_list.append(camera)
        print("camera attached")

    def attach_accelerometer(self, pos=(0, 0.0,0.8)):
        self.accel = Accelerometer('accel', self.bng, self.vehicle, pos =pos, requested_update_time=0.1, is_using_gravity=False)
        print("accel attached")

    def camera_poll(self, index):
        ## TODO: this function should "return" the images corresponding to that sensor, not just store them in "self.color/depth"
        try:
            camera_readings = self.camera_list[index].poll()
            color = camera_readings['colour']
            self.color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
            self.depth = camera_readings['depth']
            if self.use_sgmt:
                self.segmt = camera_readings['annotation']
        except Exception as e:
            print(traceback.format_exc())

    def lidar_poll(self, index):
        ## TODO: this function should "return" the images corresponding to that sensor
        try:
            points = self.lidar_list[index].poll()
            self.lidar_pts = np.copy(points['pointCloud'])
        except Exception as e:
            print(traceback.format_exc())

    def Accelerometer_poll(self):
        ## TODO: this function should return the readings, not store them in a class variable to accomodate multi-agent simulation in the future.
        if not self.use_vel_diff:
            try:
                acc = self.accel.poll()
                if 'axis1' in acc: ## simulator tends to provide empty message on failure
                    temp_acc = np.array([acc['axis1'], acc['axis3'], -acc['axis2']])
                    if np.all(temp_acc) != 0: ## in case you're using my modified version of the simulator which sends all 0s on fault.
                        g_bf = np.matmul(self.Tnb, self.Gravity)
                        self.last_A = self.A
                        self.A = temp_acc + g_bf
                ## only update acceleration if we have a valid new reading.
                ## the accelerometer in bng 0.26 is unreliable, so I recommend using vel_diff method
            except Exception:
                print(traceback.format_exc())
        else:
            acc = (self.vel_wf - self.last_vel_wf)/self.dt
            self.last_vel_wf = np.copy(self.vel_wf)
            self.A = 0.2*np.matmul(self.Tnb, acc + self.Gravity) + 0.8*self.last_A
            self.last_A = np.copy(self.A)



    def state_poll(self):
        try:
            if(self.state_init == False):
                assert self.burn_time != 0, "step time can't be 0"
                self.bng.set_steps_per_second(int(1/self.burn_time)) ## maximum steps per second; we can only guarantee this if running on a high perf. system.
                self.Accelerometer_poll()
                self.vehicle.poll_sensors() # Polls the data of all sensors attached to the vehicle
                self.state_init = True
                self.last_quat = self.convert_beamng_to_REP103(self.vehicle.state['rotation'])
                self.timestamp = self.vehicle.sensors['timer']['time']
                print("beautiful day, __init__?") ## being cheeky are we?
            else:
                self.Accelerometer_poll()
                self.vehicle.poll_sensors() # Polls the data of all sensors attached to the vehicle
                self.timestamp = self.vehicle.sensors['timer']['time'] ## time in seconds since the start of the simulation -- does not care about resets
                self.dt = max(self.vehicle.sensors['timer']['time'] - self.timestamp, 0.02)
                self.timestamp = self.vehicle.sensors['timer']['time'] ## time in seconds since the start of the simulation -- does not care about resets
                self.broken = self.vehicle.sensors['damage']['part_damage'] ## this is useful for reward functions
                self.pos = np.copy(self.vehicle.state['pos'])
                self.vel = np.copy(self.vehicle.state['vel'])
                self.quat = self.convert_beamng_to_REP103(np.copy(self.vehicle.state['rotation']))
                self.rpy = self.rpy_from_quat(self.quat)
                self.Tnb, self.Tbn = self.calc_Transform(self.quat)
                self.vel_wf = np.copy(self.vel)
                self.vel = np.matmul(self.Tnb, self.vel)
                diff = self.quat/self.last_quat
                self.last_quat = self.quat
                self.G = np.array([diff[1]*2/self.dt, diff[2]*2/self.dt, diff[3]*2/self.dt])  # gx gy gz
                self.G = np.matmul(self.Tnb, self.G)
                sign = np.sign(self.vehicle.sensors['electrics']['gear_index'])
                if sign == 0:
                    sign = 1 ## special case just to make sure we don't consider 0 speed in neutral gear
                self.avg_wheelspeed = self.vehicle.sensors['electrics']['wheelspeed'] * sign
                self.steering = float(self.vehicle.sensors['electrics']['steering']) / self.steering_max
                throttle = float(self.vehicle.sensors['electrics']['throttle'])
                brake = float(self.vehicle.sensors['electrics']['brake'])
                self.thbr = throttle - brake
                self.state = np.hstack((self.pos, self.rpy, self.vel, self.A, self.G, self.steering, self.thbr))
                if(abs(self.rpy[0]) > np.pi/2 or abs(self.rpy[1]) > np.pi/2):
                    self.flipped_over = True

                if self.camera:
                    if self.timestamp - self.last_cam_time > 1/self.camera_fps:
                        self.camera_poll(0)
                        self.last_cam_time = self.timestamp
                if self.lidar:
                    if self.timestamp - self.last_lidar_time > 1/self.lidar_fps:
                        self.lidar_poll(0)
                        self.last_lidar_time = self.timestamp
                        self.lidar_pts -= self.pos
                        self.lidar_pts = np.matmul(self.lidar_pts, self.Tnb.T)
        except Exception:
            print(traceback.format_exc())


    def scaled_PID_FF(self, Kp, Ki, Kd, FF_gain, FF, error, error_sigma, error_diff, last_error):
        error_sigma += error * self.dt
        error_sigma = np.clip(error_sigma, -1, 1) ## clip error_sigma to 10%
        ## innovation in error_derivative:
        diff_innov = np.clip((error - last_error)/self.dt, -1, 1) - error_diff
        ## smoothing error derivative:
        error_diff += diff_innov * 0.5
        PI = Kp * error + Ki * error_sigma + Kd * error_diff + FF * FF_gain
        return PI, error_sigma, error_diff

    def send_ctrl(self, action, speed_ctrl=False, speed_max = 1, Kp = 1, Ki =  1, Kd = 0, FF_gain = 1):
        st, th = -action[0], action[1]
        if(speed_ctrl):
            speed_err = th - (self.avg_wheelspeed/speed_max)
            th, self.whspd_error_sigma, self.whspd_error_diff = self.scaled_PID_FF(Kp, Ki, Kd, FF_gain, th, speed_err, self.whspd_error_sigma, self.whspd_error_diff, self.last_whspd_error)
            self.last_whspd_error = speed_err
            th = np.clip(th, -1,1)
        br = 0
        th_out = th
        if(th < 0):
            br = -th
            th_out = 0

        self.vehicle.control(throttle = th_out, brake = br, steering = st)

    def reset(self, start_pos = None, start_quat=None):
        if(start_pos is None):
            start_pos = np.copy(self.start_pos)
        if(start_quat is None):
            start_quat = np.copy(self.start_quat)
            self.vehicle.teleport(pos=(start_pos[0], start_pos[1], start_pos[2]) )
        else:
            self.vehicle.teleport(pos=(start_pos[0], start_pos[1], start_pos[2]), rot_quat= (start_quat[0], start_quat[1], start_quat[2], start_quat[3]) )
        self.vehicle.control(throttle = 0, brake = 0, steering = 0)
        self.flipped_over = False

        self.avg_wheelspeed = 0
        self.last_whspd_error = 0
        self.whspd_error_sigma = 0
        self.whspd_error_diff = 0


    def step(self, action):
        self.pos = self.state[:3].cpu().numpy()

        offset = torch.clone(self.state[:3])
        self.state[:3] = 0
        padded_state = self.state[None, None, None, :]
        padded_action = action[None, None, None, :]

        self.state = self.dyn.forward(padded_state, padded_action)
        self.state = self.state.squeeze()
        self.state[:3] += offset
