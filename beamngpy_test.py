
import random
import time
from time import sleep

from beamngpy import BeamNGpy, Scenario, Vehicle, set_up_simple_logging
from beamngpy.sensors import Lidar, Camera, Electrics, Accelerometer, IMU
import math as m
import cv2
import numpy as np
import traceback
from pyquaternion import Quaternion


def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

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

def main():
    # random.seed(1703)
    # set_up_simple_logging()

    bng = BeamNGpy('localhost', 64256, home='/home/stark/BeamNG/BeamNG', user='/home/stark/BeamNG/BeamNG/userfolder')
    # Launch BeamNG.tech
    bng.open()

    scenario = Scenario('gridmap_v2', name="test integration")

    vehicle = Vehicle('ego_vehicle', model='sunburst', partConfig='vehicles/sunburst/RACER.pc')
    # vehicle = Vehicle('ego_vehicle', model='cardboard_box', partConfig='vehicles/cardboard_box/large.pc')
    # scenario.add_vehicle(vehicle, pos=(-67, 336, 100),
    #                      rot_quat=(0, 0, 0.3826834, 0.9238795))
    scenario.add_vehicle(vehicle, pos=(-340, -260,  100.5),
                         rot_quat=(0, 0, 0, 1))

    # scenario.add_vehicle(vehicle, pos=(-771, -700,  100.5 ),
    #                  rot_quat=(0, 0, 0, 1))
    bng.set_tod(0.5)
    scenario.make(bng)
    # bng.hide_hud()
    # bng.set_deterministic()
    # bng.set_steps_per_second(100)  # Set simulator to 60hz temporal resolution
    # Create an Electrics sensor and attach it to the vehicle
    electrics = Electrics()
    vehicle.attach_sensor('electrics', electrics)
    imu = IMU(name='imu', pos=(0,0,0.8))
    vehicle.attach_sensor('imu',imu)
    bng.load_scenario(scenario)
    # bng.hide_hud()
    bng.start_scenario()

    vehicle.poll_sensors()
    accel = Accelerometer('accel', bng, vehicle, pos = (0, 0.0,0), requested_update_time=0.01, is_using_gravity=False)
    sensors = vehicle.sensors
    # NOTE: Create sensor after scenario has started.
    # lidar = Lidar('lidar1', bng, vehicle, requested_update_time=0.01, is_using_shared_memory=True)     # Send data via shared memory.
    # camera1 = Camera('camera1', bng, vehicle, pos=(-67, 336, 100), dir=(0,0,-1), field_of_view_y=90, resolution=(400, 400), update_priority=1,
    #                  is_render_colours=True, is_render_depth=True, is_render_annotations=False,is_visualised=False,
    #                  requested_update_time=0.01, near_far_planes=(0.1, 200.0), is_using_shared_memory=True, is_static=True) # get camera data
    # vehicle.ai_set_mode('span')
    # sleep(5)
    print("starting")
    start = time.time()
    attempt = 0
    episode_time = 10.0
    # time.sleep(10)
    print("start!")
    wp_list = []

    while True:
        try:
            # now = time.time()
            # # camera_readings = camera1.poll() #
            # dt = time.time() - now
            vehicle.poll_sensors() # Polls the data of all sensors attached to the vehicle
            acc = accel.poll()
            A = np.array([acc['axis1'], acc['axis3'], acc['axis2']])
            raw_A = np.copy(A)
            pos = np.copy(vehicle.state['pos'])
            vel = np.copy(vehicle.state['vel'])
            vel_wf = np.copy(vehicle.state['vel'])
            quat = convert_beamng_to_REP103(np.copy(vehicle.state['rotation']))
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
            steering = vehicle.sensors['electrics']['steering']
            speed = np.linalg.norm(vel)
            mean_wheelspeed = max(np.mean(np.abs(wheelspeed)),0.1)
            ratio = speed/mean_wheelspeed
            data = np.hstack((speed,ratio))
            # vehicle.control(throttle = 0.2, brake=br, steering = st)
            print(speed)
            # print("wheelspeed", wheelspeed)
            # print("wheelslip", wheelslip)
            # print("wheelsideslip", wheelsideslip)
            # print("wheeldownforce", wheeldownforce)
            # print("wheelhorizontalforce", wheelhorizontalforce)
            # print("got here")
            # print(vehicle.sensors['electrics']['engine_load'])
            # time.sleep(10)
            # wp_list.append(pos)
            # steering = vehicle.sensors['electrics']['steering']
            # wheelspeed = vehicle.sensors['electrics']['wheelspeed']
            # wheeltorque = vehicle.sensors['electrics']['wheeltorque']

            # print("rotation: ", rotation)
            # color = camera_readings['colour']
            # color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
            # depth = camera_readings['depth']
            # cv2.imshow('colour', color)
            # # cv2.imshow('color', increase_brightness(color, 50))
            # cv2.imshow('depth', (depth - 50)/50)
            # cv2.waitKey(1)

        except Exception:
            print(traceback.format_exc())
        except KeyboardInterrupt:
            break
    # np.save("dynamics_dirt.npy")
    # np.save("WP_file_arena_plan.npy", np.array(wp_list))
    # bng.close()
# [-65.74681675 339.62223993  34.69057929]
# [-67.32520802 333.68821688  33.9343761 ]


if __name__ == '__main__':
    main()
'''
ENU
NE [273.16721393  414.49434783   25.7465919 ]
SW: [-359.92681776 -329.09107363   32.47618165]
ZMAx: [ -49.16995027 -214.5803518    96.46927808]
'''

'''
C: [-302 -324  100]

MNE: [-304 -276  100]
MNW: [-336 -276  100]
MSW: [-336 -372  100]
MSE: [-304 -372  100]

MNE: [-268 -276  100]
MNW: [-300 -276  100]
SSW: [-300 -372  100]
SSE: [-268 -372  100]

'''