from BeamNGRL.BeamNG.beamng_interface import *
from typing import List
import numpy as np
import yaml
import os
import argparse
import math as m
import rosbag
import os
from tf.transformations import euler_from_quaternion

## TODO: make this script run over the collected data in one shot.
def main(config_path=None, hal_config_path=None, args=None):
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

    dir_name = str(Path(os.getcwd()).parent.absolute()) + "/Experiments/Results/Sim2Real/"

    real_name = dir_name + 'bag_23.npy'

    try:
        states = np.load(real_name)
    except:
        bng_interface = None
        bagdir = Config["bagdir"]
        bag_name = Config["bag_name"]
        source = bagdir + '/' + bag_name
        bag = rosbag.Bag(source, 'r')

        states = []
        state = np.zeros(20)
        got_speed = False
        got_odom = False
        got_imu = False
        got_rc = False
        got_st = False

        for topic, msg, t in bag.read_messages(topics=['/sensors/core', '/mavros/imu/data_raw', '/mavros/local_position/odom', '/mavros/manual_control/send', '/mavros/rc/in']):
            if topic == '/sensors/core':
                state[16] = msg.state.speed/3166
                got_speed = True
            elif topic == '/mavros/imu/data_raw':
                state[9:12] = [msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z]
                got_imu = True
            elif topic == '/mavros/local_position/odom':
                state[0:3] = [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z]
                rpy = euler_from_quaternion([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])
                state[3:6] = [rpy[0], rpy[1], rpy[2]]
                state[6:9] = [msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z]
                state[12:15] = [msg.twist.twist.angular.x, msg.twist.twist.angular.y, msg.twist.twist.angular.z]
                state[19] = msg.header.stamp.to_sec()
                got_odom = True
            elif topic == '/mavros/manual_control/send':
                state[15] = -msg.y/1000.0
                got_st = True
            elif topic == '/mavros/rc/in':
                state[17] = (msg.channels[0] - 1500.0)/500.0
                print(state[17])
                state[18] = (msg.channels[2] - 1000.0)/1000.0
                got_rc = True

            if got_odom and got_speed and got_imu and got_rc:
                states.append(state)
                state = np.zeros(20)
                got_speed = False
                got_odom = False
                got_imu = False
                got_rc = False
                got_st = False

        states = np.array(states)
        states[:,-1] -= states[0,-1] ## 0 out the timestamps
        np.save(real_name, states)


    scenario = Config["scenarios"]
    vehicle = Config["vehicles"]["flux"]
    start_pos = np.zeros(3)
    start_quat = np.array([0,0,0,1])

    bng_interface = get_beamng_default(
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
        burn_time=0.02,
        run_lockstep=Config["run_lockstep"],
    )

    for i in range(50):
        bng_interface.state_poll() ## burn through the first few frames to stabilize the car

    dynamics_error = []
    bng_states = []
    for i in range(len(states)):
        action = states[i, 17:19]
        bng_interface.send_ctrl(action, speed_ctrl=False)
        bng_interface.state_poll()
        state = np.copy(bng_interface.state)
        ts = bng_interface.timestamp
        bng_state = np.hstack((state[:15], action[0], bng_interface.avg_wheelspeed))
        bng_states.append(bng_state)
        error = np.hstack((bng_state - states[i,:17], action))
        dynamics_error.append(error)

    dynamics_error = np.array(dynamics_error)
    bng_states = np.array(bng_states)
    dir_name = str(Path(os.getcwd()).parent.absolute()) + "/Experiments/Results/Sim2Real/"
    filename = dir_name + "BeamNG_23.npy"
    if(not os.path.isdir(dir_name)):
        os.makedirs(dir_name)
    np.save(filename, dynamics_error)

    filename = dir_name + "bng_23.npy"
    if(not os.path.isdir(dir_name)):
        os.makedirs(dir_name)
    np.save(filename, bng_states)

    bng_interface.send_ctrl(np.zeros(2), speed_ctrl=True, speed_max = 20, Kp=2, Ki=0.05, Kd=0.0, FF_gain=0.0)
    bng_interface.set_lockstep(False)
    bng_interface.bng.resume()
    bng_interface.bng.close()
    os._exit(0)
                


if __name__ == "__main__":
    # TODO: Test this script
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name", type=str, default="sim2real_Config.yaml", help="name of the config file to use")
    parser.add_argument("--hal_config_name", type=str, default="hound.yaml", help="name of the config file to use")
    parser.add_argument("--remote", type=bool, default=True, help="whether to connect to a remote beamng server")
    parser.add_argument("--host_IP", type=str, default="169.254.216.9", help="host ip address if using remote beamng")


    args = parser.parse_args()
    config_name = args.config_name
    config_path = str(Path(os.getcwd()).parent.absolute()) + "/Experiments/Configs/" + config_name
    hal_config_name = args.hal_config_name
    hal_config_path = str(Path(os.getcwd()).parent.absolute()) + "/Configs/" + hal_config_name
    main(config_path=config_path, hal_config_path=hal_config_path, args=args) ## we run for 3 iterations because science
