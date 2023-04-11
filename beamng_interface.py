import time
from pyquaternion import Quaternion
from beamngpy import BeamNGpy, Scenario, Vehicle
from beamngpy.sensors import Lidar, Camera, Electrics, Accelerometer
import numpy as np
import traceback
import cv2

## BeamNG Interface Class

class beamng_interface():
    def __init__(self, homedir='G:/BeamNG/BeamNG',userfolder='G:/BeamNG/BeamNG/userfolder', host='localhost', port=64256):
        self.bng = BeamNGpy(host, port, home = homedir, user=userfolder)
        # Launch BeamNG.tech
        self.bng.open()
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

    def load_scenario(self, scenario_name='small_island', car_make='sunburst', car_model='RACER',
                      start_pos=np.array([-67, 336, 34.5]), start_rot=np.array([0, 0, 0.3826834, 0.9238795]),
                      time_of_day=1200, hide_hud=False, fps=60):
        self.scenario = Scenario('small_island', name="test integration")

        self.vehicle = Vehicle('ego_vehicle', model=car_make, partConfig='vehicles/'+ car_make + '/' + car_model + '.pc')

        self.start_pos = start_pos
        self.scenario.add_vehicle(self.vehicle, pos=(start_pos[0], start_pos[1], start_pos[2]),
                             rot_quat=(start_rot[0], start_rot[1], start_rot[2], start_rot[3]))
        self.bng.set_tod(time_of_day/2400)
        self.scenario.make(self.bng)
        if(hide_hud):
            self.bng.hide_hud()
        # self.bng.set_steps_per_second(fps)  # Set simulator to 60hz temporal resolution
        # Create an Electrics sensor and attach it to the vehicle
        self.electrics = Electrics()
        self.vehicle.attach_sensor('electrics', self.electrics)
        self.bng.load_scenario(self.scenario)
        self.bng.start_scenario()
        time.sleep(2)
        self.attach_accelerometer()
        time.sleep(2)
        # self.attach_camera(name='camera')
        # time.sleep(2)
        # self.attach_lidar(name='lidar')
        # time.sleep(2)
        self.state_poll()
        self.flipped_over = False

    def set_map_attributes(self, map_size = 16, resolution = 0.25):
        self.elevation_map_full = np.load('map_data/elevation_map.npy', allow_pickle=True)
        self.color_map_full = cv2.imread('map_data/color_map.png')
        self.segmt_map_full = cv2.imread('map_data/segmt_map.png')
        self.path_map_full  = cv2.imread('map_data/paths.png')
        self.image_shape    = self.color_map_full.shape
        self.image_resolution = 0.1  # this is the original meters per pixel resolution of the image
        self.resolution     = resolution  # meters per pixel of the target map
        self.resolution_inv = 1/self.resolution  # pixels per meter
        self.map_size       = map_size/2  # 16 x 16 m grid around the car by default

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
    def get_map_bf_no_rp(self, map_img, rotate = False, rpy=None):
        if(len(map_img.shape)==3):
            BEV = map_img[self.Y_min:self.Y_max, self.X_min:self.X_max, :]
        else:
            BEV = map_img[self.Y_min:self.Y_max, self.X_min:self.X_max]
        if(rotate):
            BEV = cv2.bitwise_and(BEV, BEV, mask=mask)
            # get rotation matrix using yaw:
            rotate_matrix = cv2.getRotationMatrix2D(center=mask_center, angle= -rpy[i,2]*57.3, scale=1)
            # rotate the image using cv2.warpAffine
            BEV = cv2.warpAffine(src=BEV, M=rotate_matrix, dsize=mask_size)
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
        self.BEV_heght -= self.BEV_heght[self.map_size_px[0], self.map_size_px[1]]
        self.BEV_heght = np.clip(self.BEV_heght, -2.0, 2.0)

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
                self.A = self.last_A
            else:
                self.last_A = self.A
            self.A += g_bf
        except Exception:
            print(traceback.format_exc())

    def set_lockstep(self, lockstep):
        self.lockstep = lockstep

    def state_poll(self):
        try:
            if(self.state_init == False):
                # self.camera_poll(0)
                # self.lidar_poll(0)
                self.Accelerometer_poll()
                self.vehicle.poll_sensors() # Polls the data of all sensors attached to the vehicle
                self.state_init = True
                self.last_quat = self.convert_beamng_to_REP103(self.vehicle.state['rotation'])
                self.now = time.time()
                print("beautiful day, __init__?")
                time.sleep(0.02)
            else:
                dt = time.time() - self.now
                self.now = time.time()
                # self.camera_poll(0)
                # self.lidar_poll(0)
                self.Accelerometer_poll()
                self.vehicle.poll_sensors() # Polls the data of all sensors attached to the vehicle
                self.pos = np.copy(self.vehicle.state['pos'])
                self.vel = np.copy(self.vehicle.state['vel'])
                self.quat = self.convert_beamng_to_REP103(np.copy(self.vehicle.state['rotation']))
                self.rpy = self.rpy_from_quat(self.quat)
                self.Tnb, self.Tbn = self.calc_Transform(self.quat)
                self.vel = np.matmul(self.Tnb, self.vel)
                diff = self.quat/self.last_quat
                self.last_quat = self.quat
                self.G = np.array([diff[1]*2/dt, diff[2]*2/dt, diff[3]*2/dt])  # gx gy gz
                # wheeldownforce = vehicle.sensors['electrics']['wheeldownforce']
                # wheelhorizontalforce = vehicle.sensors['electrics']['wheelhorizontalforce']
                # wheelslip = vehicle.sensors['electrics']['wheelslip']
                # wheelsideslip = vehicle.sensors['electrics']['wheelsideslip']
                # wheelspeed = vehicle.sensors['electrics']['wheelspeed_individual']
                # self.wheeldownforce = np.array([wheeldownforce[0.0], wheeldownforce[1.0], wheeldownforce[2.0], wheeldownforce[3.0]])
                # self.wheelhorizontalforce = np.array([wheelhorizontalforce[0.0], wheelhorizontalforce[1.0], wheelhorizontalforce[2.0], wheelhorizontalforce[3.0]])
                # self.wheelslip = np.array([wheelslip[0.0], wheelslip[1.0], wheelslip[2.0], wheelslip[3.0]])
                # self.wheelsideslip = np.array([wheelsideslip[0.0], wheelsideslip[1.0], wheelsideslip[2.0], wheelsideslip[3.0]])
                # self.wheelspeed = np.array([wheelspeed[0.0], wheelspeed[1.0], wheelspeed[2.0], wheelspeed[3.0]])
                self.steering = float(self.vehicle.sensors['electrics']['steering']) / 260.0
                throttle = float(self.vehicle.sensors['electrics']['throttle'])
                brake = float(self.vehicle.sensors['electrics']['brake'])
                self.thbr = throttle - brake
                self.state = np.hstack((self.pos, self.rpy, self.vel, self.A, self.G, self.steering, self.thbr))
                self.gen_BEVmap()
                if(abs(self.rpy[0]) > np.pi/2 or abs(self.rpy[1]) > np.pi/2):
                    self.flipped_over = True
                if(self.lockstep):
                    self.bng.pause()
        except Exception:
            print(traceback.format_exc())

    def send_ctrl(self, action):
        st, th = -action[0], action[1]
        br = 0
        th_out = th
        if(th < 0):
            br = -th
            th_out = 0
        if(self.lockstep):
            self.bng.resume()

        self.vehicle.control(throttle = th_out, brake = br, steering = st)

    def reset(self, start_pos = None):
        if(start_pos is None):
            start_pos = np.copy(self.start_pos)
        self.vehicle.teleport(pos=(start_pos[0], start_pos[1], start_pos[2]))# rot_quat= (start_quat[0], start_quat[1], start_quat[2], start_quat[3]) )
        self.vehicle.control(throttle = 0, brake = 0, steering = 0)
        self.flipped_over = False