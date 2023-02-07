import time
from time import sleep
from beamngpy import BeamNGpy, Scenario, Vehicle, set_up_simple_logging
from beamngpy.sensors import Lidar, Camera, Electrics, Accelerometer
import math as m
import cv2
import numpy as np
import traceback
import threading
from pyquaternion import Quaternion

## BeamNG Interface Class

class beamng_interface():
    def __init__(self, homedir='G:/BeamNG/BeamNG',userfolder='G:/BeamNG/BeamNG/userfolder', host='localhost', port=64256, state_update_rate = 50):
        self.bng = BeamNGpy(host, port, home = homedir, user=userfolder)
        # Launch BeamNG.tech
        self.bng.open()
        self.lidar_list = []
        self.camera_list = []
        self.state_init = False
        self.last_A = np.zeros(3)
        self.quat = np.array([1,0,0,0])
        self.Tnb, self.Tbn = self.calc_Transform(self.quat)
        self.state_update_rate = state_update_rate
        self.depth = None
        self.pos = None
        self.color = None
        self.segmt = None

    def load_scenario(self, scenario_name='small_island', car_make='sunburst', car_model='RACER', time_of_day=1200, hide_hud=False, fps=60):
        self.scenario = Scenario('small_island', name="test integration")

        self.vehicle = Vehicle('ego_vehicle', model='sunburst', partConfig='vehicles/sunburst/RACER.pc')

        self.scenario.add_vehicle(self.vehicle, pos=(-67, 336, 34.5),
                             rot_quat=(0, 0, 0.3826834, 0.9238795))
        self.bng.set_tod(time_of_day/2400)
        self.scenario.make(self.bng)
        if(not hide_hud):
            self.bng.hide_hud()
        self.bng.set_steps_per_second(fps)  # Set simulator to 60hz temporal resolution
        # Create an Electrics sensor and attach it to the vehicle
        self.electrics = Electrics()
        self.vehicle.attach_sensor('electrics', self.electrics)
        self.bng.load_scenario(self.scenario)
        self.bng.start_scenario()

        # proc_state = threading.Thread(target = self.state_loop, args=(50,)) # daemon -> kills thread when main program is stopped
        # proc_state.daemon = True
        # proc_state.start()
        time.sleep(2)
        self.attach_accelerometer()
        time.sleep(2)
        ## top down camera:
        self.attach_camera(name='camera', pos=(0, -30, 30), dir = (0,0,-1),field_of_view_y=90, resolution=(300,300), annotation=True)
        ## standard camera:
        # self.attach_camera(name='camera')
        time.sleep(2)
        # self.attach_lidar(name='lidar')
        # time.sleep(2)

        # proc_camera = threading.Thread(target = self.camera_loop, args=(0, 30, )) # daemon -> kills thread when main program is stopped
        # proc_camera.daemon = True
        # proc_camera.start()
        # print("camera started")
        # time.sleep(2)

        # proc_accel = threading.Thread(target = self.Accelerometer_loop)
        # proc_accel.daemon = True
        # proc_accel.start()
        # print("accelerometer started")
        # time.sleep(2) # wait for accel to start

        # proc_lidar = threading.Thread(target = self.lidar_loop, args=(0, 10,)) # daemon -> kills thread when main program is stopped
        # proc_lidar.daemon = True
        # proc_lidar.start()
        # print("lidar started")
        # time.sleep(2)
        self.state_loop(self.state_update_rate)

    def rpy_from_quat(self, quat):
        y = np.zeros(3)
        y[0] = m.atan2((2.0*(quat[2]*quat[3]+quat[0]*quat[1])) , (quat[0]**2 - quat[1]**2 - quat[2]**2 + quat[3]**2));
        y[1] = -m.asin(2.0*(quat[1]*quat[3]-quat[0]*quat[2]));
        y[2] = m.atan2((2.0*(quat[1]*quat[2]+quat[0]*quat[3])) , (quat[0]**2 + quat[1]**2 - quat[2]**2 - quat[3]**2));
        return y

    def quat_from_rpy(self, rpy):
        u1 = m.cos(0.5*rpy[0]);
        u2 = m.cos(0.5*rpy[1]);
        u3 = m.cos(0.5*rpy[2]);
        u4 = m.sin(0.5*rpy[0]);
        u5 = m.sin(0.5*rpy[1]);
        u6 = m.sin(0.5*rpy[2]);
        quat = np.zeros(4)
        quat[0] = u1*u2*u3+u4*u5*u6;
        quat[1] = u4*u2*u3-u1*u5*u6;
        quat[2] = u1*u5*u3+u4*u2*u6;
        quat[3] = u1*u2*u6-u4*u5*u3;
        return quat

    def convert_beamng_to_REP103(self, rot):
        rot = Quaternion(rot[2], -rot[0], -rot[1], -rot[3])
        new = Quaternion([0,m.sqrt(2)/2,m.sqrt(2)/2,0])*rot
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

    def camera_loop(self, index, update_frequency):
        update_time = 1.0/update_frequency
        # print("camera_loop started")
        # while True:
        try:
            # now = time.time()
            camera_readings = self.camera_list[index].poll()
            # dt = time.time() - now
            color = camera_readings['colour']
            self.color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
            # color = self.increase_brightness(color, 60)  # only needed if lidar running
            self.depth = camera_readings['depth']
            self.segmt = camera_readings['annotation']
            # print("got camera: ", dt*1000)
            # while(time.time() - now < update_time):
            #     time.sleep(0.001) 
        except Exception as e:
            print(traceback.format_exc())
            time.sleep(update_time)

    def lidar_loop(self, index, update_frequency):
        update_time = 1.0/update_frequency
        print("lidar_loop started")
        while True:
            try:
                now = time.time()
                lidar_readings = np.copy(self.lidar_list[index].poll)
                dt = time.time() - now
                print("got lidar: ", dt*1000)
                while(time.time() - now < update_time):
                    time.sleep(0.001)
            except:
                time.sleep(update_time)

    def Accelerometer_loop(self):
        update_time = 0.01
        # while True:
        try:
            now = time.time()
            acc = self.accel.poll()
            self.A = np.array([acc['axis1'], acc['axis3'], acc['axis2']])
            g_bf = np.matmul(self.Tnb, np.array([0,0,-9.8]))
            if( np.all(self.A) == 0):
                self.A = self.last_A
            else:
                self.last_A = self.A
            self.A += g_bf
            # while time.time() - now < update_time:
            #     time.sleep(0.001)
        except Exception:
            print(traceback.format_exc())
            time.sleep(update_time)

    def state_loop(self, update_frequency):
        delta = 1.0/update_frequency
        while True:
            try:
                if(self.state_init == False):
                    print("state_init")
                    self.camera_loop(0, update_frequency)
                    self.Accelerometer_loop()
                    self.vehicle.poll_sensors() # Polls the data of all sensors attached to the vehicle
                    self.state_init = True
                    self.last_quat = self.convert_beamng_to_REP103(self.vehicle.state['rotation'])
                    print("state_init done")
                    time.sleep(0.02)
                else:
                    now = time.time()
                    self.camera_loop(0, update_frequency)
                    self.Accelerometer_loop()
                    self.vehicle.poll_sensors() # Polls the data of all sensors attached to the vehicle
                    self.pos = np.copy(self.vehicle.state['pos'])
                    self.vel = np.copy(self.vehicle.state['vel'])
                    self.quat = self.convert_beamng_to_REP103(np.copy(self.vehicle.state['rotation']))
                    self.Tnb, self.Tbn = self.calc_Transform(self.quat)
                    self.vel = np.matmul(self.Tnb, self.vel)
                    diff = self.quat/self.last_quat
                    self.last_quat = self.quat
                    self.G = np.array([diff[1]*2*50, diff[2]*2*50, diff[3]*2*50])  # gx gy gz
                    dt = time.time() - now
                    # print("got state: ", dt*1000)
                    # while(time.time() - now < delta):
                    #     time.sleep(0.001)
            except Exception:
                print(traceback.format_exc())
                time.sleep(delta)
            except KeyboardInterrupt:
                exit()

# main function:
if __name__ == "__main__":
    interface = beamng_interface()
    # load scenario:
    interface.load_scenario()
    time.sleep(60)
