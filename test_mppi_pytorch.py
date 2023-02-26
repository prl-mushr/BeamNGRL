import time
from time import sleep

from beamngpy import BeamNGpy, Scenario, Vehicle, set_up_simple_logging
from beamngpy.sensors import Electrics, Accelerometer
from pyquaternion import Quaternion
import math as m
import numpy as np
from Bezier import *
import traceback
from mppi_controller import *

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

def steering_limiter(steering_setpoint, wheelspeed, accBF, wheelbase, critical_angle, steering_max, roll_rate, roll):
    steering_setpoint = steering_setpoint*steering_max
    intervention = False
    whspd2 = max(1.0, wheelspeed)
    whspd2 *= whspd2
    t_h = m.tan(critical_angle)
    Aylim_static = t_h * max(1.0, abs(accBF[2]))
    Aylim_static -= min(Aylim_static, abs(accBF[0]))
    Aylim_dynamic = Aylim_static*max(m.cos(roll*2),0)
    steering_limit = abs(m.atan2(wheelbase * Aylim_static, whspd2))

    if(abs(steering_setpoint) > steering_limit):
        intervention = True
        steering_setpoint = min(steering_limit, max(-steering_limit, steering_setpoint))
    # print(steering_limit*57.3, t_h, wheelspeed,steering_setpoint*57.3, st_init*57.3)
    Ay = accBF[1]
    Ay_error = 0
    if(abs(accBF[1]) > Aylim_dynamic):
        intervention = True
        if(Ay >= 0):
            Ay_error = Ay - Aylim_dynamic
        else:
            Ay_error = Ay + Aylim_dynamic
        Ay_rate_error = -roll_rate/(m.fabs(accBF[2])/(accBF[1]**2 + accBF[2]**2))
        steering_projection_inverse = m.fabs((1/max(m.cos(roll*2),0.01)**2))
        delta_steering = steering_projection_inverse*(Ay_error + Ay_rate_error*10) * (m.cos(steering_setpoint)**2) * wheelbase / whspd2
        steering_setpoint += delta_steering
    steering_setpoint = steering_setpoint/steering_max
    steering_setpoint = min(max(steering_setpoint, -1.0),1.0)
    return steering_setpoint, intervention


def get_WP(filename):
    target_WP = np.load(filename,allow_pickle=True)
    target_WP = np.array(target_WP,dtype=float)
    target_Vhat = np.zeros_like(target_WP) # 0 out everything
    cruise_speed = 50
    for i in range(1,len(target_WP)-1): # all points except first and last
        V_prev = target_WP[i] - target_WP[i-1]
        V_next = target_WP[i+1] - target_WP[i]
        target_Vhat[i] = (V_next + V_prev)/np.linalg.norm(V_next + V_prev)
    N = 200
    wp_list = np.zeros((N*len(target_WP),6)) # x,y,z, x^, y^, z^
    for i in range(len(target_WP)-1):
        P0 = target_WP[i]
        P3 = target_WP[i+1]
        P1,P2 = get_Intermediate_Points_generic(P0,P3,target_Vhat[i],target_Vhat[i+1],cruise_speed,compliment=False)
        bx,by,bz = get_bezier(P0,P1,P2,P3,float(N))
        for j in range(N):
            Curvature,Direction,Normal = get_CTN(P0,P1,P2,P3,float(j)/float(N))
            wp_list[N*i + j] = np.array([bx[j],by[j],bz[j],Direction[0],Direction[1],Direction[2]])
    return wp_list

def main(start_point, start_quat, turn_point, folder_name, map_name, speed_target, episode_time, num_episodes=4):
    bng = BeamNGpy('localhost', 64256, home='/home/stark/BeamNG/BeamNG', user='/home/stark/BeamNG/BeamNG/userfolder')
    # Launch BeamNG.tech
    bng.open()
    scenario = Scenario(map_name, name="test integration")
    vehicle = Vehicle('ego_vehicle', model='sunburst', partConfig='vehicles/sunburst/RACER.pc')
    # vehicle = Vehicle('ego_vehicle', model='RG_RC', partConfig='vehicles/RG_RC/Short_Course_Truck.pc')

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
    print("accel attached")
    sleep(2)
    print("starting")
    start = time.time()
    attempt = 0
    last_A = 0
    episode = []

    acc = accel.poll()
    last_A = np.array([acc['axis1'], acc['axis3'], acc['axis2']])
    vehicle.poll_sensors() # Polls the data of all sensors attached to the vehicle
    last_quat = convert_beamng_to_REP103(vehicle.state['rotation'])
    start_turning = False
    fraction = 1/float(num_episodes)
    now = time.time()
    time.sleep(0.02)

    WP_file = "WP_file_offroad.npy"
    wp_list = get_WP(WP_file)
    controller = control_system(trajectory = wp_list)
    st = 0
    th = 1
    recording = False
    while True:
        try:
            dt = time.time() - now 
            now = time.time()
            print("dt: ",dt*1000)
            vehicle.poll_sensors() # Polls the data of all sensors attached to the vehicle
            acc = accel.poll()
            A = np.array([acc['axis1'], acc['axis3'], acc['axis2']])
            raw_A = np.copy(A)
            pos = np.copy(vehicle.state['pos'])
            vel = np.copy(vehicle.state['vel'])
            vel_wf = np.copy(vehicle.state['vel'])
            quat = convert_beamng_to_REP103(np.copy(vehicle.state['rotation']))
            rpy = rpy_from_quat(quat)
            Tnb, Tbn = calc_Transform(quat)
            vel = np.matmul(Tnb, vel)
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
            steering = vehicle.sensors['electrics']['steering']
            t = time.time() - start

            data = np.hstack((pos, rpy, vel, A, G)) #, t, dt, raw_A, start_turning, wheeldownforce, wheelhorizontalforce, wheelslip, wheelsideslip, wheelspeed, steering))


            action = controller.update(data.copy())
            action = action.numpy()
            action = np.array(action, dtype=np.float64)
            st, th = -action[0], action[1]
            br = 0
            th_out = th
            if(th < 0):
                br = -th
                th_out = 0
            vehicle.control(throttle = th_out, brake=br, steering = st)
        except Exception:
            print(traceback.format_exc())
    # bng.close()



if __name__ == '__main__':
    # position of the vehicle for tripped_flat on grimap_v2
    # start_point = np.array([-814.33583743, -669.04329255,  100.56901635])
    # start_quat = np.array([-0.00396234, -0.002702,   -0.16445144,  0.9863735])
    # turn_point = np.array([-798.569961,  -716.46442828,  100.87442045])
    # folder_name = "tripped_flat_with_correction"
    # map_name = "gridmap_v2"
    # speed_target = 18.0
    # episode_time = 10.0
    # main(start_point, start_quat, turn_point, folder_name, map_name, speed_target, episode_time)
    # time.sleep(2)
    # # position of the vehicle for untripped_flat on smallgrid:

    # start_point = np.array([-86.52589376, 321.26751955, 0.0])
    # # start_point = np.array([0,0, 0.0])
    # # start_quat = np.array([0.0, 0.0, 1.0, 0.0])
    # start_quat = np.array([ 0.02423989, -0.05909005,  0.19792375,  0.97813445])
    # turn_point = np.array([0, 50.0 ,0])
    # folder_name = "controller_test"
    # map_name = "smallgrid"
    # speed_target = 18.0
    # episode_time = 10.0
    # main(start_point, start_quat, turn_point, folder_name, map_name, speed_target, episode_time)
    # time.sleep(2)
    # position of the vehicle for mixed_offroad on small_island:
    start_point = np.array([-86.52589376, 322.26751955,  35.33346797]) 
    start_quat = np.array([ 0.02423989, -0.05909005,  0.19792375,  0.97813445])
    turn_point = np.array([-101.02775679,  291.77741613,   37.28218909])
    folder_name = "mixed_offroad_with_correction"
    map_name = "small_island"
    speed_target = 12.0
    episode_time = 15.0
    main(start_point, start_quat, turn_point, folder_name, map_name, speed_target, episode_time)