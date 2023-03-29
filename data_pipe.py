import time
from time import sleep

from beamngpy import BeamNGpy, Scenario, Vehicle, set_up_simple_logging
from beamngpy.sensors import Electrics, Accelerometer, Camera
from pyquaternion import Quaternion
import math as m
import numpy as np
import traceback
import cv2

def convert_beamng_to_REP103(rot):
    rot = Quaternion(rot[2], -rot[0], -rot[1], -rot[3])
    new = Quaternion([0,m.sqrt(2)/2,m.sqrt(2)/2,0])*rot
    rot = Quaternion(-new[1], -new[3], -new[0], -new[2])
    return rot

def calc_Transform(quat):
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

def rpy_from_quat(quat):
    y = np.zeros(3)
    y[0] = m.atan2((2.0*(quat[2]*quat[3]+quat[0]*quat[1])) , (quat[0]**2 - quat[1]**2 - quat[2]**2 + quat[3]**2));
    y[1] = -m.asin(2.0*(quat[1]*quat[3]-quat[0]*quat[2]));
    y[2] = m.atan2((2.0*(quat[1]*quat[2]+quat[0]*quat[3])) , (quat[0]**2 + quat[1]**2 - quat[2]**2 - quat[3]**2));
    return y



def main(start_point, start_quat, turn_point, folder_name, map_name, speed_target, episode_time, num_episodes=2):
    bng = BeamNGpy('localhost', 64256, home='G:/BeamNG/BeamNG', user='G:/BeamNG/BeamNG/userfolder')
    bng.open()
    scenario = Scenario(map_name, name="test integration")
    vehicle = Vehicle('ego_vehicle', model='sunburst', partConfig='vehicles/sunburst/RACER.pc')
    scenario.add_vehicle(vehicle, pos=(start_point[0], start_point[1], start_point[2] + 0.5),
                     rot_quat=(start_quat[0], start_quat[1], start_quat[2], start_quat[3]))
    bng.set_tod(0.5)
    scenario.make(bng)
    electrics = Electrics()
    vehicle.attach_sensor('electrics', electrics)
    bng.load_scenario(scenario)
    # bng.hide_hud()
    bng.start_scenario()
    vehicle.poll_sensors()
    sensors = vehicle.sensors
    accel = Accelerometer('accel', bng, vehicle, pos = (0, 0.0,0), requested_update_time=0.01, is_using_gravity=False)
    camera = Camera('camera', bng, vehicle, pos=(0, -0.2, 0.2), dir=(0,-1,0), field_of_view_y=87.5, resolution=(640, 480), update_priority=0,
                    is_render_colours=True, is_render_depth=True, is_render_annotations=True,is_visualised=False,
                    requested_update_time=0.001, near_far_planes=(0.1, 200.0), is_using_shared_memory=True, is_static=False) # get camera data

    print("accel attached")
    sleep(2)
    print("starting")
    start = time.time()
    last_A = 0
    intrinsic_data = []
    color_data = []
    depth_data = []
    segmt_data = []

    acc = accel.poll()
    last_A = np.array([acc['axis1'], acc['axis3'], acc['axis2']])
    vehicle.poll_sensors() # Polls the data of all sensors attached to the vehicle
    last_quat = convert_beamng_to_REP103(vehicle.state['rotation'])
    attempts = 0
    now = time.time()
    time.sleep(0.02)
    while True:
        try:
            now = time.time()
            vehicle.poll_sensors() # Polls the data of all sensors attached to the vehicle
            acc = accel.poll()
            camera_readings = camera_BEV.poll()
            dt = time.time() - now 
            A = np.array([acc['axis1'], acc['axis3'], acc['axis2']])
            pos = np.copy(vehicle.state['pos'])
            vel = np.copy(vehicle.state['vel'])
            vel_wf = np.copy(vehicle.state['vel'])
            quat = convert_beamng_to_REP103(np.copy(vehicle.state['rotation']))
            rpy = rpy_from_quat(quat)
            Tnb, Tbn = calc_Transform(quat)
            vel_bf = np.matmul(Tnb, vel)
            diff = quat/last_quat
            last_quat = quat
            g_bf = np.matmul(Tnb, np.array([0,0,9.8]))
            if( np.all(A) == 0):
                A = last_A
            else:
                last_A = A
            A += g_bf
            G = np.array([diff[1]*2/dt, diff[2]*2/dt, diff[3]*2/dt])  # gx gy gz
            wheeldownforce = vehicle.sensors['electrics']['wheeldownforce']
            wheelhorizontalforce = vehicle.sensors['electrics']['wheelhorizontalforce']
            wheeldownforce = np.array([wheeldownforce[0.0], wheeldownforce[1.0], wheeldownforce[2.0], wheeldownforce[3.0]])
            wheelhorizontalforce = np.array([wheelhorizontalforce[0.0], wheelhorizontalforce[1.0], wheelhorizontalforce[2.0], wheelhorizontalforce[3.0]])
            wheelslip = vehicle.sensors['electrics']['wheelslip']
            wheelslip = np.array([wheelslip[0.0], wheelslip[1.0], wheelslip[2.0], wheelslip[3.0]])
            wheelsideslip = vehicle.sensors['electrics']['wheelsideslip']
            wheelsideslip = np.array([wheelsideslip[0.0], wheelsideslip[1.0], wheelsideslip[2.0], wheelsideslip[3.0]])
            wheelspeed = vehicle.sensors['electrics']['wheelspeed_individual']
            wheelspeed = np.array([wheelspeed[0.0], wheelspeed[1.0], wheelspeed[2.0], wheelspeed[3.0]])
            steering = float(vehicle.sensors['electrics']['steering'])
            throttle = float(vehicle.sensors['electrics']['throttle'])
            brake = float(vehicle.sensors['electrics']['brake'])
            
            color = camera_readings['colour']
            color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
            # color = increase_brightness(color, 60)  # only needed if lidar running
            depth = camera_readings['depth']
            segmt = camera_readings['annotation']
            
            t = time.time() - start
            print(dt)
            data = np.hstack((pos, rpy, vel_bf, vel_wf, A, G, steering, throttle, brake, wheeldownforce, wheelhorizontalforce, wheelslip, wheelsideslip, wheelspeed, t, dt))


        except Exception:
            print(traceback.format_exc())
    bng.close()



if __name__ == '__main__':
    start_point = np.array([-86.52589376, 322.26751955,  35.33346797]) 
    # start_quat = np.array([ 0.02423989, -0.05909005,  0.19792375,  0.97813445])
    start_quat = np.array([0,0,0,1])
    turn_point = np.array([-101.02775679,  291.77741613,   37.28218909])
    folder_name = "IRL_data"
    map_name = "small_island"
    speed_target = 12.0
    episode_time = 90.0
    main(start_point, start_quat, turn_point, folder_name, map_name, speed_target, episode_time)