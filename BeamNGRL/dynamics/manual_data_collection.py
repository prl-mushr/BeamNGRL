from BeamNGRL.BeamNG.beamng_interface import *
import traceback
import argparse
import BeamNGRL
from pathlib import Path


ROOT_PATH = Path(BeamNGRL.__file__).parent
DATA_PATH = ROOT_PATH.parent / 'data'


def collect_data(args):

    output_path = DATA_PATH / args.output_dir
    output_path.mkdir(parents=True, exist_ok=True)

    bng = get_beamng_default(
        map_name=args.map_name,
        start_pos=np.array(args.start_pos),
        start_quat=np.array(args.start_quat),
    )

    start = time.time()

    intrinsic_data = []
    color_data = []
    elevt_data = []
    segmt_data = []
    path_data = []

    while True:
        try:
            bng.state_poll()

            # state is np.hstack((pos, rpy, vel, A, G, st, th/br)) ## note that velocity is in the body-frame
            # state information follows ROS REP103 standards (so basically ROS standards): world refernce frame for (x,y,z) is east-north-up(ENU). Body frame ref is front-left-up(FLU)
            state = bng.state

            ## get robot_centric BEV (not rotated into robot frame)
            BEV_color = bng.BEV_color
            BEV_heght = (bng.BEV_heght + 2.0)/4.0  # note that BEV_heght (elevation) has a range of +/- 2 meters around the center of the elevation.
            BEV_segmt = bng.BEV_segmt
            BEV_path  = bng.BEV_path  # trail/roads

            intrinsic_data.append(state)
            color_data.append(BEV_color)
            elevt_data.append(BEV_heght)
            segmt_data.append(BEV_segmt)
            path_data.append(BEV_path)

            if time.time() - start > args.duration:
                print("Saving data...")
                np.save(output_path / "state.npy", np.array(intrinsic_data))
                np.save(output_path / "bev_color.npy", np.array(color_data))
                np.save(output_path / "bev_elevt.npy", np.array(elevt_data))
                np.save(output_path / "bev_segmt.npy", np.array(segmt_data))
                np.save(output_path / "bev_path.npy", np.array(path_data))
                break

        except Exception:
            print(traceback.format_exc())

    bng.bng.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, help='location to store test results')
    parser.add_argument('--start_pos', type=float, default=[-67, 336, 34.5], nargs=3, help='Starting position of the vehicle for tripped_flat on grimap_v2')
    parser.add_argument('--start_quat', type=float, default=[0, 0, 0.3826834, 0.9238795], nargs=4, help='Starting rotation (quat) of the vehicle.')
    parser.add_argument('--map_name', type=str, default='small_island', help='Map name.')
    parser.add_argument('--output_dir', type=str, default='manual_data')
    parser.add_argument('--duration', type=int, default=10)
    args = parser.parse_args()

    collect_data(args)