from BeamNGRL.BeamNG.beamng_interface import *
import traceback
import argparse
from datetime import datetime
from BeamNGRL import *


def collect_data(args):

    output_dir = args.output_dir
    if output_dir is None:
        date_time = datetime.now().strftime("%m_%d_%Y")
        output_dir = f'{args.map_name}_{date_time}'

    output_path = DATA_PATH / 'manual_data' / output_dir
    output_path.mkdir(parents=True, exist_ok=True)

    bng = get_beamng_default(
        map_name=args.map_name,
        start_pos=np.array(args.start_pos),
        start_quat=np.array(args.start_quat),
        car_make='sunburst',
        car_model='drift'
    )

    bng.set_lockstep(True)
    start = None

    timestamps = []
    state_data = []
    color_data = []
    elev_data = []
    segmt_data = []
    path_data = []
    normal_data = []

    while True:
        try:
            bng.state_poll()

            # state is np.hstack((pos, rpy, vel, A, G, st, th/br)) ## note that velocity is in the body-frame
            # state information follows ROS REP103 standards (so basically ROS standards): world refernce frame for (x,y,z) is east-north-up(ENU). Body frame ref is front-left-up(FLU)
            state = bng.state
            ts = bng.timestamp
            state[16] = bng.avg_wheelspeed/20.0
            print(state[-2:])
            if not start:
                start = ts

            # get robot_centric BEV (not rotated into robot frame)
            BEV_color = bng.BEV_color
            BEV_height = (bng.BEV_heght + 2.0)/4.0  # note that BEV_heght (elevation) has a range of +/- 2 meters around the center of the elevation.
            BEV_segmt = bng.BEV_segmt
            BEV_path  = bng.BEV_path  # trail/roads
            BEV_normal  = bng.BEV_normal  # trail/roads

            timestamps.append(ts)
            state_data.append(state)
            color_data.append(BEV_color)
            elev_data.append(BEV_height)
            segmt_data.append(BEV_segmt)
            path_data.append(BEV_path)
            normal_data.append(BEV_normal)

            if ts - start > args.duration:
                print("Saving data...")
                np.save(output_path / "timestamps.npy", np.array(timestamps))
                np.save(output_path / "state.npy", np.array(state_data))
                np.save(output_path / "bev_path.npy", np.array(path_data))
                np.save(output_path / "bev_color.npy", np.array(color_data))
                np.save(output_path / "bev_segmt.npy", np.array(segmt_data))
                np.save(output_path / "bev_elev.npy", np.array(elev_data))
                np.save(output_path / "bev_normal.npy", np.array(normal_data))
                break

        except Exception:
            print(traceback.format_exc())

    bng.bng.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default=None, help='location to store test results')
    parser.add_argument('--start_pos', type=float, default=[-67, 336, 0.5], nargs=3, help='Starting position of the vehicle for tripped_flat on grimap_v2')
    parser.add_argument('--start_quat', type=float, default=[0, 0, 0.3826834, 0.9238795], nargs=4, help='Starting rotation (quat) of the vehicle.')
    parser.add_argument('--map_name', type=str, default='smallgrid', help='Map name.')
    parser.add_argument('--duration', type=int, default=300)
    args = parser.parse_args()

    collect_data(args)