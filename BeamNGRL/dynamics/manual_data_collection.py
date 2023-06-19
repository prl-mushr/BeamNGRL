import numpy as np
from BeamNGRL.BeamNG.beamng_interface import get_beamng_default
import traceback
import torch
from BeamNGRL.control.UW_mppi.MPPI import MPPI
from BeamNGRL.control.UW_mppi.Dynamics.SimpleCarNetworkDyn import SimpleCarNetworkDyn
from BeamNGRL.control.UW_mppi.Costs.SimpleCarCost import SimpleCarCost
from BeamNGRL.control.UW_mppi.Sampling.Delta_Sampling import Delta_Sampling
from BeamNGRL.utils.visualisation import costmap_vis
from BeamNGRL.utils.planning import update_goal
import yaml
import argparse
from datetime import datetime
from BeamNGRL import MPPI_CONFIG_PTH, DATA_PATH, ROOT_PATH, LOGS_PATH
import time
from typing import List
import gc ## import group chat?


## I used the MPPI to train a model which I then use to collect more data. makes perfect sense.

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

    dtype = torch.float
    device = torch.device("cuda")

    output_dir = args.output_dir
    if output_dir is None:
        date_time = datetime.now().strftime("%m_%d_%Y")
        output_dir = f'{args.map_name}_{date_time}'

    output_path = DATA_PATH / 'manual_data' / output_dir
    output_path.mkdir(parents=True, exist_ok=True)

    bng = get_beamng_default(
        car_model='offroad',
        start_pos=np.array(args.start_pos),
        start_quat=np.array(args.start_quat),
        map_name=args.map_name,
        car_make='sunburst',
        map_res=0.25,
        map_size=32
    )## map params here make no difference!
    bng.set_lockstep(True)
    
    current_wp_index = 0  # initialize waypoint index with 0
    goal = None
    action = np.zeros(2)

    timestamps = []
    state_data = []
    reset_data = []
    reset_counter = 0
    reset_limit = 100

    start = None
    running = True
    save_prompt_time = float(args.save_every_n_sec)

    while running:
        try:
            bng.state_poll()
            state = bng.state
            ts = bng.timestamp

            if not start:
                start = ts

            T = ts - start

            damage = False
            if(type(bng.broken) == dict ):
                count = 0
                for part in bng.broken.values():
                    if part['damage'] > 0.8:
                        count += 1
                damage = count > 1
            reset = False
            if(damage or bng.flipped_over):
                reset_counter += 1
                if reset_counter >= reset_limit:
                    reset = True
                    reset_counter = 0
                    bng.reset()

            
            state[16] = bng.avg_wheelspeed/20 ## divide by max wheelspeed because that is how we do it with the MPPI's output.
            print(state[:3], state[15:])
            # Aggregate Data
            timestamps.append(ts)
            state_data.append(state)
            reset_data.append(reset)

            if ts >= save_prompt_time or \
                ts - start > args.duration:

                print("\nSaving data...")
                print(f"time: {ts}")
                timestamps = update_npy_datafile(timestamps, output_path / "timestamps.npy")
                state_data = update_npy_datafile(state_data, output_path / "state.npy")
                reset_data = update_npy_datafile(reset_data, output_path / "reset.npy")
                gc.collect()
                save_prompt_time += float(args.save_every_n_sec)

            if ts - start > args.duration:
                break

        except Exception:
            print(traceback.format_exc())

    bng.bng.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default=None, help='location to store test results')
    parser.add_argument('--start_pos', type=float, default=[-67, 336, 34.5], nargs=3, help='Starting position of the vehicle for tripped_flat on grimap_v2')
    parser.add_argument('--start_quat', type=float, default=[0, 0, 0.3826834, 0.9238795], nargs=4, help='Starting rotation (quat) of the vehicle.')
    parser.add_argument('--map_name', type=str, default='small_island', help='Map name.')
    parser.add_argument('--waypoint_file', type=str, default='WP_file_offroad.npy', help='Map name.')
    parser.add_argument('--duration', type=int, default=600)
    parser.add_argument('--save_every_n_sec', type=int, default=15)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    collect_mppi_data(args)
