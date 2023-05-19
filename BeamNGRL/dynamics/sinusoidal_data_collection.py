import numpy as np
from BeamNGRL.BeamNG.beamng_interface import get_beamng_default
import traceback
import torch
from BeamNGRL.control.UW_mppi.MPPI import MPPI
from BeamNGRL.control.UW_mppi.Dynamics.SimpleCarDynamics import SimpleCarDynamics
from BeamNGRL.control.UW_mppi.Costs.SimpleCarCost import SimpleCarCost
from BeamNGRL.control.UW_mppi.Sampling.Delta_Sampling import Delta_Sampling
from BeamNGRL.utils.visualisation import costmap_vis
from BeamNGRL.utils.planning import update_goal
import yaml
import argparse
from datetime import datetime
from BeamNGRL import MPPI_CONFIG_PTH, DATA_PATH, ROOT_PATH
import time
from typing import List
import gc


def update_npy_datafile(buffer: List, filepath):
    buff_arr = np.array(buffer)
    if filepath.is_file():
        # Append to existing data file
        data_arr = np.load(filepath, allow_pickle=True)
        data_arr = np.concatenate((data_arr, buff_arr), axis=0)
        np.save(filepath, data_arr)
    else:
        np.save(filepath, buff_arr)
    return [] # empty buffer


def collect_mppi_data(args):

    with open(MPPI_CONFIG_PTH / 'MPPI_config.yaml') as f:
        MPPI_config = yaml.safe_load(f)

    with open(MPPI_CONFIG_PTH / 'Dynamics_config.yaml') as f:
        Dynamics_config = yaml.safe_load(f)

    with open(MPPI_CONFIG_PTH / 'Cost_config.yaml') as f:
        Cost_config = yaml.safe_load(f)

    with open(MPPI_CONFIG_PTH / 'Sampling_config.yaml') as f:
        Sampling_config = yaml.safe_load(f)

    with open(MPPI_CONFIG_PTH / 'Map_config.yaml') as f:
        Map_config = yaml.safe_load(f)

    speed_max = Dynamics_config["throttle_to_wheelspeed"]
    map_res = Map_config["map_res"]
    dtype = torch.float
    device = torch.device("cuda")

    output_dir = args.output_dir
    if output_dir is None:
        date_time = datetime.now().strftime("%m_%d_%Y")
        output_dir = f'{args.map_name}_{date_time}'

    output_path = DATA_PATH / 'sinu_data' / output_dir
    output_path.mkdir(parents=True, exist_ok=True)

    bng = get_beamng_default(
        car_model='RACER',
        start_pos=np.array(args.start_pos),
        start_quat=np.array(args.start_quat),
        map_name=args.map_name,
        car_make='sunburst',
        # map_res=Map_config["map_size"]//2, ## make the map useless
        map_res=Map_config["map_res"],
        map_size=Map_config["map_size"]
    )

    action = np.zeros(2)

    bng.set_lockstep(True)

    timestamps = []
    state_data = []
    color_data = []
    elev_data = []
    segmt_data = []
    path_data = []
    normal_data = []

    start = None
    running = True
    save_prompt_time = float(args.save_every_n_sec)

    f_min = 0.1
    f_max = 1.0
    scaler = 1/args.duration

    while running:
        try:
            bng.state_poll()
            state = bng.state
            ts = bng.timestamp

            if not start:
                start = ts
            T = ts - start

            f = f_min + (f_max - f_min)*T/args.duration

            # get robot_centric BEV (not rotated into robot frame)
            BEV_color = bng.BEV_color
            BEV_height = bng.BEV_heght
            BEV_segmt = bng.BEV_segmt
            BEV_path  = bng.BEV_path  # trail/roads
            BEV_normal  = bng.BEV_normal  # trail/roads

            state_to_ctrl = state.copy()
            state_to_ctrl[:3] = np.zeros(3) # this is for the MPPI: technically this should be state[:3] -= BEV_center

            # we use our previous control output as input for next cycle!
            state_to_ctrl[15:17] = action ## adhoc wheelspeed.
            action[0] = np.sin(f*T*np.pi*3) ## change steering 50% faster than throttle so that you don't get PLL
            action[1] = np.sin(f*T*np.pi*2)*0.25 + 0.25

            # Aggregate Data
            timestamps.append(ts)
            state_data.append(state)
            color_data.append(BEV_color)
            elev_data.append(BEV_height)
            segmt_data.append(BEV_segmt)
            path_data.append(BEV_path)
            normal_data.append(BEV_normal)

            if ts >= save_prompt_time or \
                T > args.duration:

                print("\nSaving data...")
                print(f"time: {ts}")
                timestamps = update_npy_datafile(timestamps, output_path / "timestamps.npy")
                state_data = update_npy_datafile(state_data, output_path / "state.npy")
                path_data = update_npy_datafile(path_data, output_path / "bev_path.npy")
                color_data = update_npy_datafile(color_data, output_path / "bev_color.npy")
                segmt_data = update_npy_datafile(segmt_data, output_path / "bev_segmt.npy")
                elev_data = update_npy_datafile(elev_data, output_path / "bev_elev.npy")
                normal_data = update_npy_datafile(normal_data, output_path / "bev_normal.npy")

                gc.collect()
                save_prompt_time += float(args.save_every_n_sec)

            if T > args.duration:
                break

            bng.send_ctrl(
                action,
                speed_ctrl=True,
                speed_max=speed_max,
                Kp=5,
                Ki=0.05,
                Kd=0.0,
                FF_gain=0.0,
            )

        except Exception:
            print(traceback.format_exc())

    bng.bng.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default=None, help='location to store test results')
    parser.add_argument('--start_pos', type=float, default=[-67, 336, 0.5], nargs=3, help='Starting position of the vehicle for tripped_flat on grimap_v2')
    parser.add_argument('--start_quat', type=float, default=[0, 0, 0.3826834, 0.9238795], nargs=4, help='Starting rotation (quat) of the vehicle.')
    parser.add_argument('--map_name', type=str, default='smallgrid', help='Map name.')
    parser.add_argument('--waypoint_file', type=str, default='WP_file_offroad.npy', help='Map name.')
    parser.add_argument('--duration', type=int, default=30)
    parser.add_argument('--save_every_n_sec', type=int, default=15)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    collect_mppi_data(args)
