"""
.. module:: west_coast_lidar
    :platform: Windows
    :synopsis: Example code making a scenario in west_coast_usa and having the
               vehicle span the map while emitting Lidar.
.. moduleauthor:: Marc Müller <mmueller@beamng.gmbh>
"""
import random
import time
from time import sleep
from beamngpy import BeamNGpy, Scenario, Vehicle
from beamngpy.sensors import Accelerometer, Electrics
import numpy as np
import traceback
import math as m
from pyquaternion import Quaternion

def rpy_from_quat(quat):
    y = np.zeros(3)
    y[0] = m.atan2((2.0*(quat[2]*quat[3]+quat[0]*quat[1])) , (quat[0]**2 - quat[1]**2 - quat[2]**2 + quat[3]**2));
    y[1] = -m.asin(2.0*(quat[1]*quat[3]-quat[0]*quat[2]));
    y[2] = m.atan2((2.0*(quat[1]*quat[2]+quat[0]*quat[3])) , (quat[0]**2 + quat[1]**2 - quat[2]**2 - quat[3]**2));
    return y

def quat_from_rpy(rpy):
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

def main():
    bng = BeamNGpy('localhost', 64256, home='G:/BeamNG/BeamNG', user='G:/BeamNG/BeamNG/userfolder')
    # Launch BeamNG.tech
    bng.open()

    # scenario = Scenario('small_island', name="test integration")
    scenario = Scenario('smallgrid', name="test integration")

    vehicle = Vehicle('ego_vehicle', model='sunburst', partConfig='vehicles/sunburst/RACER.pc')

    scenario.add_vehicle(vehicle, pos=(-67, 336, 34.5),
                         rot_quat=(0, 0, 0.3826834, 0.9238795))
    electrics = Electrics()
    vehicle.attach_sensor('electrics', electrics)
    bng.set_tod(0.5)
    scenario.make(bng)
    bng.load_scenario(scenario)
    # bng.hide_hud()
    bng.start_scenario()
    # time.sleep(1)
    acc = Accelerometer('accel', bng, vehicle, pos = (0, 0.0,0.8),requested_update_time=0.0001, is_using_gravity=True)
    # time.sleep(1)
    print("starting")
    vehicle.poll_sensors() # Polls the data of all sensors attached to the vehicle
    sensors = vehicle.sensors
    last_rotation = convert_beamng_to_REP103(vehicle.state['rotation'])

    while True:
        try:
            now = time.time()
            vehicle.poll_sensors() # Polls the data of all sensors attached to the vehicle
            sensors = vehicle.sensors
            rotation = convert_beamng_to_REP103(vehicle.state['rotation'])
            vel = vehicle.state['vel']
            A = acc.poll()
            A = np.array([A['axis1'], A['axis3'], A['axis2']])
            Tnb, Tbn = calc_Transform(rotation)
            vel = np.matmul(Tnb,vel)
            g_bf = np.matmul(Tnb, np.array([0,0,-9.8]))
            A += g_bf
            print(np.round(A,3))
            # print(np.round(vehicle.state['dir'],3))
            last_rotation = rotation

            while(time.time() - now < 0.02):
                time.sleep(0.001)

        except Exception:
            print(traceback.format_exc())
    acc.remove()
    bng.close()


if __name__ == '__main__':
    main()