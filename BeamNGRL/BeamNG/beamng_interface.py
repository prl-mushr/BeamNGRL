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


ROOT_PATH = Path(BeamNGRL.__file__).parent
DATA_PATH = ROOT_PATH.parent / 'data'
BNG_HOME = os.environ.get('BNG_HOME')


def get_beamng_default(
        car_model='RACER',
        start_pos=None,
        start_quat=None,
        map_name=None,
        car_make='sunburst',
        beamng_path=BNG_HOME,
        map_res=0.05,
        map_size=16, # 16 x 16 map
        path_to_maps=DATA_PATH.__str__(),
):

    if(start_pos is None):
        print("please provide a start pos! I can not spawn a car in the ether!")
        exit()
    if(start_quat is None):
        print("please provide a start quat! I can not spawn a car's rotation in the ether!")
        exit()
    if(map_name is None):
        print("please provide a map_name! I can not spawn a car in the ether!")
        exit()
        
    bng = beamng_interface(BeamNG_path=beamng_path)

    bng.load_scenario(
        scenario_name=map_name, car_make=car_make, car_model=car_model,
        start_pos=start_pos, start_rot=start_quat,
    )
    bng.set_map_attributes(
        map_size=map_size, resolution=map_res, path_to_maps=path_to_maps,
    )

    return bng

def get_beamng_remote(
        car_model='RACER',
        start_pos=None,
        start_quat=None,
        map_name=None,
        car_make='sunburst',
        beamng_path=BNG_HOME,
        map_res=0.05,
        map_size=16, # 16 x 16 map
        path_to_maps=DATA_PATH.__str__(),
        remote=True,
        host_IP=None,
):

    if(host_IP is None):
        print("please provide a host IP!")
        exit()
    if(start_pos is None):
        print("please provide a start pos! I can not spawn a car in the ether!")
        exit()
    if(start_quat is None):
        print("please provide a start quat! I can not spawn a car's rotation in the ether!")
        exit()
    if(map_name is None):
        print("please provide a map_name! I can not spawn a car in the ether!")
        exit()
        
    bng = beamng_interface(BeamNG_path=beamng_path, remote=remote, host_IP=host_IP)

    bng.load_scenario(
        scenario_name=map_name, car_make=car_make, car_model=car_model,
        start_pos=start_pos, start_rot=start_quat,
    )
    bng.set_map_attributes(
        map_size=map_size, resolution=map_res, path_to_maps=path_to_maps,
    )

    return bng

## this is the equivalent of None pizza with left beef joke. Yes I'd like one beamng simulator without the beamng simulator.
def get_beamng_nobeam(
        Dynamics,
        car_model='RACER',
        start_pos=None,
        start_quat=None,
        map_name=None,
        car_make='sunburst',
        beamng_path=BNG_HOME,
        map_res=0.05,
        map_size=16, # 16 x 16 map
        path_to_maps=DATA_PATH.__str__(),
):

    if(start_pos is None):
        print("please provide a start pos! I can not spawn a car in the ether!")
        exit()
    if(start_quat is None):
        print("please provide a start quat! I can not spawn a car's rotation in the ether!")
        exit()
    if(map_name is None):
        print("please provide a map_name! I can not spawn a car in the ether!")
        exit()
        
    bng = beamng_interface(BeamNG_path=beamng_path, use_beamng=False, dyn=Dynamics)

    bng.load_scenario(
        scenario_name=map_name, car_make=car_make, car_model=car_model,
        start_pos=start_pos, start_rot=start_quat,
    )
    bng.set_map_attributes(
        map_size=map_size, resolution=map_res, path_to_maps=path_to_maps,
    )

    return bng


class beamng_interface():
    def __init__(self, BeamNG_path=BNG_HOME, host='localhost', port=64256, use_beamng=True, dyn=None, remote=False, host_IP=None):
        self.lockstep   = False
        self.lidar_list = []
        self.camera_list = []
        self.state_init = False
        self.last_A     = np.zeros(3)
        self.quat       = np.array([1,0,0,0])
        self.Tnb, self.Tbn = self.calc_Transform(self.quat)
        self.depth      = None
        self.pos        = None
        self.color      = None
        self.segmt      = None
        self.lidar_pts  = None
        self.Gravity    = np.array([0,0,9.81])
        self.state      = None
        self.BEV_center = np.zeros(3)
        self.avg_wheelspeed = 0
        self.dt = 0.016
        self.last_whspd_error = 0
        self.whspd_error_sigma = 0
        self.whspd_error_diff = 0

        self.use_beamng = use_beamng
        if self.use_beamng:
            if(remote==True and host_IP is not None):
                self.bng = BeamNGpy(host_IP, 64256, remote=True)
                self.bng.open(launch=False, deploy=False)
            else:
                self.bng = BeamNGpy(host, port, home=BeamNG_path, user=BeamNG_path + '/userfolder')
                self.bng.open()
        else:
            self.dyn = dyn
            self.state = torch.zeros(17, dtype=dyn.dtype, device=dyn.d)
            self.vis = Vis()

    def load_scenario(self, scenario_name='small_island', car_make='sunburst', car_model='RACER',
                      start_pos=np.array([-67, 336, 34.5]), start_rot=np.array([0, 0, 0.3826834, 0.9238795]),
                      time_of_day=1200, hide_hud=False, fps=60):
        self.start_pos = start_pos
        if not self.use_beamng:
            self.state[:3] = torch.from_numpy(start_pos)
            self.state[3:6] = torch.from_numpy(self.rpy_from_quat(self.convert_beamng_to_REP103(start_rot)))
            return

        self.scenario = Scenario(scenario_name, name="test integration")

        self.vehicle = Vehicle('ego_vehicle', model=car_make, partConfig='vehicles/'+ car_make + '/' + car_model + '.pc')

        self.scenario.add_vehicle(self.vehicle, pos=(start_pos[0], start_pos[1], start_pos[2]),
                             rot_quat=(start_rot[0], start_rot[1], start_rot[2], start_rot[3]))
        self.bng.set_tod(time_of_day/2400)
        self.scenario.make(self.bng)
        if(hide_hud):
            self.bng.hide_hud()
        # self.bng.set_steps_per_second(fps)  # Set simulator to 60hz temporal resolution
        # Create an Electrics sensor and attach it to the vehicle
        self.electrics = Electrics()
        self.timer = Timer()
        self.damage = Damage()
        self.vehicle.attach_sensor('electrics', self.electrics)
        self.vehicle.attach_sensor('timer', self.timer)
        self.vehicle.attach_sensor('damage', self.damage)
        self.bng.load_scenario(self.scenario)
        self.bng.start_scenario()
        # time.sleep(2)
        self.attach_accelerometer()
        # time.sleep(2)
        # self.attach_camera(name='camera')
        # time.sleep(2)
        # self.attach_lidar(name='lidar')
        # time.sleep(2)
        self.state_poll()
        self.flipped_over = False


    def set_map_attributes(self, map_size = 16, resolution = 0.25, path_to_maps='', rotate=False):
        self.elevation_map_full = np.load(path_to_maps + '/map_data/elevation_map.npy', allow_pickle=True)
        self.color_map_full = cv2.imread(path_to_maps + '/map_data/color_map.png')
        self.segmt_map_full = cv2.imread(path_to_maps + '/map_data/segmt_map.png')
        self.path_map_full  = cv2.imread(path_to_maps + '/map_data/paths.png')
        self.image_shape    = self.color_map_full.shape
        self.image_resolution = 0.1  # this is the original meters per pixel resolution of the image
        self.resolution     = resolution  # meters per pixel of the target map
        self.resolution_inv = 1/self.resolution  # pixels per meter
        self.map_size       = map_size/2  # 16 x 16 m grid around the car by default
        self.rotate = rotate

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

    ## this "nested" function uses variables from the intrinsic data, be careful if you move this function out
    def get_map_bf_no_rp(self, map_img):
        if(len(map_img.shape)==3):
            BEV = map_img[self.Y_min:self.Y_max, self.X_min:self.X_max, :]
        else:
            BEV = map_img[self.Y_min:self.Y_max, self.X_min:self.X_max]
        if(self.rotate):
            BEV = cv2.bitwise_and(BEV, BEV, mask=self.mask)
            # get rotation matrix using yaw:
            rotate_matrix = cv2.getRotationMatrix2D(center=self.mask_center, angle= -self.rpy[2]*57.3, scale=1)
            # rotate the image using cv2.warpAffine
            BEV = cv2.warpAffine(src=BEV, M=rotate_matrix, dsize=self.mask_size)
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
 
    def gen_BEVmap(self):
        self.img_X = int( self.pos[0]*self.resolution_inv + self.image_shape[0]//2)
        self.img_Y = int( self.pos[1]*self.resolution_inv + self.image_shape[1]//2)

        self.Y_min = int(self.img_Y - self.map_size*self.resolution_inv)
        self.Y_max = int(self.img_Y + self.map_size*self.resolution_inv)

        self.X_min = int(self.img_X - self.map_size*self.resolution_inv)
        self.X_max = int(self.img_X + self.map_size*self.resolution_inv)

        ## inputs:
        self.BEV_color = self.get_map_bf_no_rp(self.color_map_full)  # crops circle, rotates into body frame
        self.BEV_heght = self.get_map_bf_no_rp(self.elevation_map_full)
        self.BEV_segmt = self.get_map_bf_no_rp(self.segmt_map_full)
        self.BEV_path  = self.get_map_bf_no_rp(self.path_map_full)

        mask = np.zeros_like(self.BEV_heght, dtype=np.uint8)
        index = np.where(self.BEV_heght == 0)
        mask[index] = 255

        self.BEV_color = cv2.inpaint(self.BEV_color, mask, 3,cv2.INPAINT_TELEA)
        self.BEV_segmt = cv2.inpaint(self.BEV_segmt, mask, 3,cv2.INPAINT_TELEA)
        self.BEV_heght = cv2.inpaint(self.BEV_heght, mask, 1,cv2.INPAINT_TELEA)
        self.BEV_center[:2] = self.pos[:2]
        self.BEV_center[2] = self.BEV_heght[self.map_size_px[0], self.map_size_px[1]]
        self.BEV_heght -= self.BEV_center[2]
        self.BEV_heght = np.clip(self.BEV_heght, -2.0, 2.0)

        self.BEV_normal = self.compute_surface_normals()


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

    def rpy_from_quat(self, quat):
        y = np.zeros(3)
        y[0] = np.arctan2((2.0*(quat[2]*quat[3]+quat[0]*quat[1])) , (quat[0]**2 - quat[1]**2 - quat[2]**2 + quat[3]**2));
        y[1] = -np.arcsin(2.0*(quat[1]*quat[3]-quat[0]*quat[2]));
        y[2] = np.arctan2((2.0*(quat[1]*quat[2]+quat[0]*quat[3])) , (quat[0]**2 + quat[1]**2 - quat[2]**2 - quat[3]**2));
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

    def increase_brightness(self, img, value=30):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        lim = 255 - value
        v[v > lim] = 255
        v[v <= lim] += value

        final_hsv = cv2.merge((h, s, v))
        img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        return img

    def attach_lidar(self, name, pos=(0,0,1.5), dir=(0,-1,0), up=(0,0,1), vertical_resolution=3, vertical_angle=26.9,
                     rays_per_second_per_scan=5000, update_frequency=10):
        lidar = Lidar(name, self.bng, self.vehicle, pos = pos, dir=dir, up=up,requested_update_time=0.001, is_visualised=False,
                        vertical_resolution=3, vertical_angle=5, rays_per_second=vertical_resolution*rays_per_second_per_scan,
                        frequency=update_frequency, update_priority = 0,is_using_shared_memory=True)
        self.lidar_list.append(lidar)
        print("lidar attached")

    def attach_camera(self, name, pos=(0,-2,1.4), dir=(0,-1,0), field_of_view_y=87, resolution=(640,480),
                      depth=True, color=True, annotation=False, instance=False, near_far_planes=(1,60.0), update_frequency = 30, static=False):
        camera = Camera(name, self.bng, self.vehicle, pos=pos, field_of_view_y=field_of_view_y, resolution=resolution, update_priority=0,
                         is_render_colours=color, is_render_depth=depth, is_render_annotations=annotation,is_visualised=False,
                         requested_update_time=0.01, near_far_planes=near_far_planes, is_using_shared_memory=True,
                         is_render_instance=instance,  is_static=static)
        self.camera_list.append(camera)
        print("camera attached")

    def attach_accelerometer(self):
        self.accel = Accelerometer('accel', self.bng, self.vehicle, pos = (0, 0.0,0.8), requested_update_time=0.01, is_using_gravity=False)
        print("accel attached")

    def camera_poll(self, index):
        try:
            camera_readings = self.camera_list[index].poll()
            color = camera_readings['colour']
            self.color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
            self.depth = camera_readings['depth']
            self.segmt = camera_readings['annotation']
        except Exception as e:
            print(traceback.format_exc())

    def lidar_poll(self, index):
        try:
            self.lidar_pts = np.copy(self.lidar_list[index].poll())
        except:
            pass

    def Accelerometer_poll(self):
        try:
            acc = self.accel.poll()
            self.A = np.array([acc['axis1'], acc['axis3'], acc['axis2']])
            g_bf = np.matmul(self.Tnb, self.Gravity)
            if( np.all(self.A) == 0):
                self.A = self.last_A - g_bf
            else:
                self.last_A = self.A
            self.A += g_bf
        except Exception:
            print(traceback.format_exc())

    def set_lockstep(self, lockstep):
        self.lockstep = lockstep

    def state_poll(self):
        if(self.state_init == False):
            # self.camera_poll(0)
            # self.lidar_poll(0)
            self.Accelerometer_poll()
            self.vehicle.poll_sensors() # Polls the data of all sensors attached to the vehicle
            self.state_init = True
            self.last_quat = self.convert_beamng_to_REP103(self.vehicle.state['rotation'])
            self.timestamp = self.vehicle.sensors['timer']['time']
            print("beautiful day, __init__?")
            time.sleep(0.02)
        else:
            if(self.lockstep):
                self.bng.resume()
                time.sleep(0.001)
            # self.camera_poll(0)
            # self.lidar_poll(0)                
            self.vehicle.poll_sensors() # Polls the data of all sensors attached to the vehicle
            self.Accelerometer_poll()
            if(self.lockstep):
                self.bng.pause()
            
            self.dt = self.vehicle.sensors['timer']['time'] - self.timestamp
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

            ## wheel ordering is FR BR FL BL
            wheeldownforce = self.vehicle.sensors['electrics']['wheeldownforce']
            wheelhorizontalforce = self.vehicle.sensors['electrics']['wheelhorizontalforce']
            wheelslip = self.vehicle.sensors['electrics']['wheelslip']
            wheelsideslip = self.vehicle.sensors['electrics']['wheelsideslip']
            wheelspeed = self.vehicle.sensors['electrics']['wheelspeed_individual']
            self.wheeldownforce = np.array([wheeldownforce[0.0], wheeldownforce[1.0], wheeldownforce[2.0], wheeldownforce[3.0]])
            self.wheelhorizontalforce = np.array([wheelhorizontalforce[0.0], wheelhorizontalforce[1.0], wheelhorizontalforce[2.0], wheelhorizontalforce[3.0]])
            self.wheelslip = np.array([wheelslip[0.0], wheelslip[1.0], wheelslip[2.0], wheelslip[3.0]])
            self.wheelsideslip = np.array([wheelsideslip[0.0], wheelsideslip[1.0], wheelsideslip[2.0], wheelsideslip[3.0]])
            self.wheelspeed = np.array([wheelspeed[0.0], wheelspeed[1.0], wheelspeed[2.0], wheelspeed[3.0]])
            self.avg_wheelspeed = self.vehicle.sensors['electrics']['wheelspeed']

            self.steering = float(self.vehicle.sensors['electrics']['steering']) / 260.0
            throttle = float(self.vehicle.sensors['electrics']['throttle'])
            brake = float(self.vehicle.sensors['electrics']['brake'])
            self.thbr = throttle - brake
            self.state = np.hstack((self.pos, self.rpy, self.vel, self.A, self.G, self.steering, self.thbr))
            self.gen_BEVmap()
            if(abs(self.rpy[0]) > np.pi/2 or abs(self.rpy[1]) > np.pi/2):
                self.flipped_over = True


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

    def reset(self, start_pos = None):
        if(start_pos is None):
            start_pos = np.copy(self.start_pos)
        self.vehicle.teleport(pos=(start_pos[0], start_pos[1], start_pos[2]))# rot_quat= (start_quat[0], start_quat[1], start_quat[2], start_quat[3]) )
        self.vehicle.control(throttle = 0, brake = 0, steering = 0)
        self.flipped_over = False

        self.avg_wheelspeed = 0
        self.dt = 0.016
        self.last_whspd_error = 0
        self.whspd_error_sigma = 0
        self.whspd_error_diff = 0


    def step(self, action):
        self.pos = self.state[:3].cpu().numpy()
        self.gen_BEVmap()

        BEV_heght = torch.from_numpy(self.BEV_heght).to(device=self.dyn.d, dtype=self.dyn.dtype)
        BEV_normal = torch.from_numpy(self.BEV_normal).to(device=self.dyn.d, dtype=self.dyn.dtype)
        self.dyn.set_BEV(BEV_heght, BEV_normal)

        offset = torch.clone(self.state[:3])
        self.state[:3] = 0
        padded_state = self.state[None, None, None, :]
        padded_action = action[None, None, None, :]

        self.state = self.dyn.forward(padded_state, padded_action)
        self.state = self.state.squeeze()
        self.state[:3] += offset

    def render(self, goal):
        vis_state = self.state.cpu().numpy()
        self.vis.setcar(pos=np.zeros(3), rpy=vis_state[3:6])
        self.vis.setgoal(goal - vis_state[:2])
        self.vis.set_terrain(self.BEV_heght, self.resolution, self.map_size)
