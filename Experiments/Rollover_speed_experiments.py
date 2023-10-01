from BeamNGRL.BeamNG.beamng_interface import *
from typing import List
import numpy as np
import yaml
import os
import argparse
import math as m


'''
in this experiment i just need the car to drive straight until it reaches a specific speed and then start turning
We basically increase the turn angle with each "iteration" of the experiment and test the rollover prevention
I wont be using the MPPI controller here since it is not needed.
'''
def create_bng_scenario(Map_config, make, model, start_pos, start_quat, args, hal_Config):
    bng_interface = get_beamng_default(
        car_model=model,
        start_pos=start_pos,
        start_quat=start_quat,
        car_make=make,
        map_config=Map_config,
        host_IP=args.host_IP,
        remote=args.remote,
        camera_config=hal_Config[model]["camera"],
        lidar_config=hal_Config[model]["lidar"],
        accel_config=hal_Config[model]["mavros"],
        burn_time=0.02,
        run_lockstep=True,
    )
    return bng_interface

def steering_limiter(steer=0, wheelspeed=0, roll=0, roll_rate=0,  accBF=np.zeros(3), dA_dt=np.zeros(3), wheelbase=2.6, t_h_ratio=0.5, max_steer=0.5, accel_gain=1.0, roll_rate_gain=1.0, rollover_prevention=1, steer_slack=0.2):
    steering_setpoint = steer*max_steer
    intervention = False
    whspd2 = max(1.0, wheelspeed)
    whspd2 *= whspd2
    Aylim = t_h_ratio * max(1.0, abs(accBF[2]))
    delta_steering = 0
    GRAVITY = 9.81

    steering_limit_max = m.atan2(wheelbase * (Aylim - GRAVITY*m.sin(roll)), whspd2)
    steering_limit_min = -m.atan2(wheelbase * (Aylim + GRAVITY*m.sin(roll)), whspd2)
    if(rollover_prevention == 2): ## only static limiter
        temp = steering_setpoint
        steering_setpoint = min(steering_limit_max, max(steering_limit_min, steering_setpoint))
        if abs(temp - steering_setpoint) > 0.001:
            intervention = True
        delta_steering = steering_setpoint - steer*max_steer
        steering_setpoint = steering_setpoint/max_steer
        steering_setpoint = min(max(steering_setpoint, -1.0),1.0)
        return steering_setpoint, intervention, delta_steering, 0

    steering_limit_max += steer_slack*max_steer
    steering_limit_min -= steer_slack*max_steer

    temp = steering_setpoint
    steering_setpoint = min(steering_limit_max, max(steering_limit_min, steering_setpoint))
    if abs(temp - steering_setpoint) > 0.001:
        intervention = True

    delta_steering = 0
    Ay = accBF[1]
    Ay_error = 0
    Ay_rate = -roll_rate*accBF[2]/(m.cos(roll)**2)
    TTR_condition = min(m.fabs(Aylim - m.fabs(Ay))/max(m.fabs(Ay_rate),0.01), 0.9) < 0.5
    if(abs(Ay) > Aylim):
        intervention = True
        if(Ay >= 0):
            Ay_error = min(Aylim - Ay,0)
            delta_steering = 4.0*(Ay_error*accel_gain - roll_rate_gain*abs(accBF[2])*roll_rate) * (m.cos(steering_setpoint)**2) * wheelbase / whspd2
            delta_steering = min(delta_steering, 0)
        else:
            Ay_error = max(-Aylim - Ay,0) 
            delta_steering = 4.0*(Ay_error*accel_gain - roll_rate_gain*abs(accBF[2])*roll_rate) * (m.cos(steering_setpoint)**2) * wheelbase / whspd2
            delta_steering = max(delta_steering, 0) ## this prevents the car from turning in the opposite direction and causing a rollover by mistake
        steering_setpoint += delta_steering
    steering_setpoint = steering_setpoint/max_steer
    steering_setpoint = min(max(steering_setpoint, -1.0),1.0)
    return steering_setpoint, intervention, delta_steering, TTR_condition


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

    Map_config = Config["Map_config"]
    total_experiments = len(Config["vehicle_list"]) * len(Config["scenarios"]) * Config["num_iters"] * 2
    experiment_count = 0
    bng_interface = None

    if Config["save_data"] == False:
        print("data will not be saved!")

    try:
        for scenario in Config["scenarios"]:
            ## check the map for the scenario:
            Map_config["map_name"] = scenario

            for vehicle_name in Config["vehicle_list"]:
                
                WP_file = str(Path(os.getcwd()).parent.absolute()) + "/Experiments/Waypoints/Rollover-" + scenario + ".npy"
                target_WP = np.load(WP_file)
                start_pos = target_WP[0,:3]
                start_quat = target_WP[0,3:]

                vehicle = Config["vehicles"][vehicle_name]
                make = vehicle["make"]
                model = vehicle["model"]
                dt = float(Config["dt"])
                
                track_width = vehicle["track_width"]
                t_h_ratio = vehicle["track_width"]/(2*vehicle["cg_height"])
                max_steer = vehicle["max_steer"]
                wheelbase = vehicle["wheelbase"]
                time_limit = Config["time_limit"]
                rotate_speed_default = vehicle["rotate_speed"]
                max_speed = vehicle["max_speed"]
                num_iters = Config["num_iters"]
                roll_rate_gain = vehicle["roll_rate_gain"]
                accel_gain = vehicle["accel_gain"]
                speed_Kp = vehicle["speed_Kp"]
                steer_slack = vehicle["steer_slack"]
                rollover_speed_max = vehicle["rollover_speed_max"]
                speed_iters = Config["speed_iters"]

                bng_interface = create_bng_scenario(Map_config, make, model, start_pos, start_quat, args, hal_Config)

                for attempt in range(speed_iters):
                    for trial in range(num_iters):
                        # run once with rollover prevention and once without
                        for rollover_prevention in [1]:
                            trial_pos = np.copy(start_pos)
                            bng_interface.reset(start_pos=trial_pos, start_quat=start_quat)
                            action = np.zeros(2)
                            bng_interface.set_lockstep(True)
                            for i in range(50):
                                bng_interface.state_poll() ## burn through the first few frames to stabilize the car

                            last_reset_time = bng_interface.timestamp # update the last reset time
                            ts = bng_interface.timestamp - last_reset_time

                            experiment_count += 1
                            print("Experiment: {}/{}".format(experiment_count, total_experiments)) #, end='\r')

                            f = trial/num_iters
                            rotate_speed = (1-f)*rotate_speed_default + f*rollover_speed_max 

                            start_turning = False
                            last_A = np.array([0,0,9.81])
                            last_dA_dt = np.zeros(3)
                            result_states = []
                            avg_ay = []
                            turn_time = 0 ## useful for post processing
                            while ts < time_limit:
                                bng_interface.state_poll()
                                state = np.copy(bng_interface.state)
                                ts = bng_interface.timestamp - last_reset_time
                                action[1] = (rotate_speed + 2)/max_speed ## this is just so that we get to the desired speed quickly
                                if state[6] > rotate_speed  and not start_turning: ## start turning anyway at 5 seconds
                                    start_turning = True
                                    turn_time = ts
                                if start_turning:    
                                    action[0] = -1.0 * min((ts - turn_time)*2, 1.0) ## swing over 0.5 seconds to full lock
                                else:
                                    action[0] = 0.0
                                
                                intervention = False

                                ## calc the rate of change of the acceleration using a low pass filtered version of the acceleration
                                Acc = np.copy(state[9:12])
                                dA_dt = 0.1*((Acc - last_A)/dt) + 0.9*last_dA_dt
                                last_dA_dt = np.copy(dA_dt)
                                last_A = np.copy(Acc)
                                delta_steering = 0
                                if rollover_prevention:
                                    action[0], intervention, delta_steering, Rollover_detected = steering_limiter(
                                                                                            steer=action[0], 
                                                                                            wheelspeed=bng_interface.avg_wheelspeed, 
                                                                                            roll = state[3], 
                                                                                            roll_rate= state[12],
                                                                                            accBF = Acc,
                                                                                            dA_dt = dA_dt,
                                                                                            wheelbase = wheelbase,
                                                                                            t_h_ratio = t_h_ratio, 
                                                                                            max_steer = max_steer,
                                                                                            accel_gain=accel_gain,
                                                                                            roll_rate_gain=roll_rate_gain,
                                                                                            rollover_prevention=rollover_prevention,
                                                                                            steer_slack = steer_slack)

                                else:
                                    Rollover_detected = False
                                # prevent car from doing a backflip
                                if state[4] < -0.5:
                                    action[1] = 0 ## bring the nose down please
                                bng_interface.send_ctrl(action, speed_ctrl=True, speed_max = max_speed, Kp=speed_Kp, Ki=0.05, Kd=0.0, FF_gain=0.0)
                                
                                result_states.append(np.hstack ( ( np.copy(state), bng_interface.flipped_over, rollover_prevention, intervention, Rollover_detected, delta_steering, turn_time, rotate_speed, ts) ))
                                if(start_turning): ## max value obtained before rollover.
                                    avg_ay.append(abs(state[10] + track_width*state[12]**2))
                                if bng_interface.flipped_over:
                                    print("rollover!")
                                    break ## break the while loop and reset the car

                            print("Average ay: ", np.mean(np.array(avg_ay)) )
                            result_states = np.array(result_states)
                            dir_name = str(Path(os.getcwd()).parent.absolute()) + "/Experiments/Results/Rollover/speed_crash_{}/".format(str(rollover_prevention))
                            filename = dir_name + "/{}-{}-{}-{}.npy".format(scenario, vehicle_name, str(trial), str(attempt))
                            bng_interface.send_ctrl(np.zeros(2), speed_ctrl=True, speed_max = max_speed, Kp=speed_Kp, Ki=0.05, Kd=0.0, FF_gain=0.0)
                            if Config["save_data"]:
                                if(not os.path.isdir(dir_name)):
                                    os.makedirs(dir_name)
                                np.save(filename, result_states)
                bng_interface.set_lockstep(False)
                bng_interface.bng.resume()
                
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(e)
        cv2.destroyAllWindows()
        if(bng_interface is not None):
            bng_interface.bng.close()
        os._exit(1)
    if(bng_interface is not None):
        bng_interface.bng.close()
    cv2.destroyAllWindows()
    os._exit(1)


if __name__ == "__main__":
    # do the args thingy:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name", type=str, default="Rollover_speed_crash_Config.yaml", help="name of the config file to use")
    parser.add_argument("--hal_config_name", type=str, default="combined_rollover.yaml", help="name of the config file to use")
    parser.add_argument("--remote", type=bool, default=True, help="whether to connect to a remote beamng server")
    parser.add_argument("--host_IP", type=str, default="169.254.216.9", help="host ip address if using remote beamng")

    args = parser.parse_args()
    config_name = args.config_name
    config_path = str(Path(os.getcwd()).parent.absolute()) + "/Experiments/Configs/" + config_name
    hal_config_name = args.hal_config_name
    hal_config_path = str(Path(os.getcwd()).parent.absolute()) + "/Configs/" + hal_config_name
    main(config_path=config_path, hal_config_path=hal_config_path, args=args) ## we run for 3 iterations because science
