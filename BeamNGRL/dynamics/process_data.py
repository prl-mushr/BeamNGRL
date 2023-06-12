import argparse
import yaml
from pathlib import Path
import math
import tqdm
import multiprocessing
import gc
from utils.dataset_utils import *
from utils.vis_utils import *
from BeamNGRL import *
from collections import defaultdict
from BeamNGRL.BeamNG.beamng_interface import *
import os

BNG_HOME = os.environ.get('BNG_HOME')
bng = beamng_interface(BeamNG_path=BNG_HOME, use_beamng=False, shell_mode=True)

def calc_data_stats(dataset_path):

    out_dir = dataset_path.__str__()

    print(f'\nCalc. trajectory stats...')

    stats = {
        'mean:state': [],
        'mean:control': [],
        'std:state': [],
        'std:control': [],
    }

    traj_input_files = get_files(out_dir, "train/trajectories")
    for fn in tqdm.tqdm(traj_input_files):
        traj = np.load(fn, allow_pickle=True).item()

        state_traj, control_traj = get_full_traj(traj)
        stats['mean:state'].append(state_traj.mean(0))
        stats['std:state'].append(state_traj.std(0))
        stats['mean:control'].append(control_traj.mean(0))
        stats['std:control'].append(control_traj.std(0))

    stats = {k: np.array(v).mean(0) for k, v in stats.items()}
    for name, value in stats.items():
        print(f'{name} -- {value}')

    print(f'\nCalc. bev stats...')

    bev_stats = defaultdict(list)

    def get_bev_stats(type):
        bev_input_files = get_files(out_dir, f"train/bev_{type}")
        for fn in tqdm.tqdm(bev_input_files):
            bev_map = np.load(fn)
            bev_stats[f'mean:bev_{type}'].append(bev_map.mean())
            bev_stats[f'std:bev_{type}'].append(bev_map.std())
            # print(f'{type}', bev_map.mean(), bev_map.std())

    [get_bev_stats(t) for t in ['color', 'elev', 'normal']]

    bev_stats = {k: np.array(v).mean() for k, v in bev_stats.items()}
    for name, value in bev_stats.items():
        print(f'{name} -- {value}')

    stats.update(bev_stats)

    np.save(dataset_path / 'input_stats.npy', stats, allow_pickle=True)


def process_data(kwargs, save_vis=False):

    frame_idx = kwargs['frame_idx']
    cfg = kwargs['cfg']
    base_frame = kwargs['base_frame']
    bev_color = kwargs['bev_color']
    bev_elev = kwargs['bev_elev']
    bev_normal = kwargs['bev_normal']
    grid_size = kwargs['grid_size']

    past_len = cfg['past_traj_len']
    map_res = cfg['map']['resolution']

    curr_idx = past_len
    trajectory = {}

    # Current
    trajectory['timestamp'] = kwargs['traj_ts'][curr_idx]
    trajectory['state'] = kwargs['traj_states'][curr_idx]
    trajectory['control'] = kwargs['traj_controls'][curr_idx]

    # Past traj
    trajectory['past_timestamps'] = kwargs['traj_ts'][:curr_idx]
    trajectory['past_states'] = kwargs['traj_states'][:curr_idx]
    trajectory['past_controls'] = kwargs['traj_controls'][:curr_idx]

    # Future traj
    trajectory['future_timestamps'] = kwargs['traj_ts'][curr_idx+1:]
    trajectory['future_states'] = kwargs['traj_states'][curr_idx+1:]
    trajectory['future_controls'] = kwargs['traj_controls'][curr_idx+1:]

    # Elevation (Relative)
    vehicle_elev = trajectory['state'][2]
    bev_elev -= vehicle_elev

    vis_image = None
    if save_vis:
        vis_image = visualize_bev_traj(
            trajectory['state'],
            trajectory['future_states'],
            trajectory['past_states'],
            bev_color,
            map_res,
        )

    return {
        'frame_idx': frame_idx,
        'base_frame': base_frame,
        'trajectory': trajectory,
        'bev_color': bev_color,
        'bev_elev': bev_elev,
        'bev_normal': bev_normal,
        'vis_image': vis_image,
    }


def generate_dataset(args):

    n_workers = args.workers
    save_vis = args.save_vis
    cfg_path = DYN_DATA_CONFIG / '{}.yaml'.format(args.cfg)
    cfg = yaml.load(open(cfg_path).read(), Loader=yaml.SafeLoader)

    raw_data_dir = Path(cfg['raw_data_dir'])
    future_traj_len = cfg['future_traj_len']
    past_traj_len = cfg['past_traj_len']
    skip_frames = cfg['skip_frames']
    map_size = cfg['map']['width']
    map_res = cfg['map']['resolution']
    try:
        map_elev_range = cfg['map']['elevation_range'] ## defaults to 2.0
    except:
        map_elev_range = 2.0

    grid_size = int(map_size // map_res)

    bng.set_map_attributes(map_size=map_size, resolution=map_res, elevation_range=map_elev_range) ## technically should also take map name but we only have 1 map.

    output_path = DATASETS_PATH / args.output_dir
    output_path.mkdir(parents=True, exist_ok=True)

    # Save data config file to dataset path
    kwargs = vars(args)
    cfg.update(kwargs)
    with open(output_path / 'config.yaml', 'w') as outfile:
        yaml.dump(cfg, outfile, sort_keys=False)

    # Loop through raw data
    for split in cfg['split']:
        split_path = output_path / split
        split_path.mkdir(exist_ok=True)

        # Create output subdirs
        trajectory_path = split_path / 'trajectories'
        bev_color_path = split_path / 'bev_color'
        bev_elev_path = split_path / 'bev_elev'
        bev_normal_path = split_path / 'bev_normal'
        vis_path = split_path / 'vis_images'

        trajectory_path.mkdir(exist_ok=True)
        bev_color_path.mkdir(exist_ok=True)
        bev_elev_path.mkdir(exist_ok=True)
        bev_normal_path.mkdir(exist_ok=True)
        vis_path.mkdir(exist_ok=True)

        frame_idx = 0 # data element counter
        for sequence in cfg['split'][split]:

            print("\nProcessing split: %s, subdir: %s" % (split, sequence))

            # Raw data sequence path, files
            sequence_path = DATA_PATH / raw_data_dir / sequence

            # Gather data
            timestamps = load_timestamps('timestamps.npy', sequence_path)
            states_seq = get_state_trajectory('state.npy', sequence_path, timestamps)
            controls_seq = get_controls('state.npy', sequence_path)
            # bev_color_seq = load_bev_map('bev_color.npy', sequence_path)
            # bev_elev_seq = load_bev_map('bev_elev.npy', sequence_path)
            # bev_normal_seq = load_bev_map('bev_normal.npy', sequence_path)
            reset_seq = load_reset_data('reset.npy', sequence_path)

            # Define data elements for processing
            job_args = []
            num_frames = states_seq.shape[0]
            for i in range(0, num_frames, skip_frames):
                keyframe_idx = i
                start_idx = i - past_traj_len
                data_idxs = [start_idx + k for k in range(past_traj_len + future_traj_len + 1)]
                if data_idxs[0] < 0 or data_idxs[-1] >= num_frames:
                    # Skip frame if idxs are out of range
                    continue
                # Base w.r.t world coord.
                base_frame = states_seq[keyframe_idx, :6]

                ## generate the BEV_maps in-situ
                bng.pos = base_frame[:3] ## set bng pos
                bng.gen_BEVmap()
                bev_color = bng.BEV_color
                bev_segmt = bng.BEV_segmt
                bev_elev = bng.BEV_heght
                bev_normal = bng.BEV_normal

                traj_ts = timestamps[data_idxs]
                traj_controls = controls_seq[data_idxs]

                # Base-frame Trajectory
                traj_states = states_seq[data_idxs]
                traj_states[:, :3] = traj_states[:, :3] - base_frame[:3]

                # Verify trajectory does not exceed map limits
                # mode = 'default'
                mode = 'radius'
                traj_img_proj, in_range = project_traj_to_map(
                    traj_states, grid_size, map_res, mode)
                if any(~in_range):
                    print(f'\nTrajectory exceeds map limits. Skipping...')
                    continue

                job_args.append({
                    'cfg': cfg,
                    'grid_size': grid_size,
                    'frame_idx': frame_idx,
                    'base_frame': base_frame,
                    'traj_img_proj': traj_img_proj,
                    'traj_ts': traj_ts,
                    'traj_states': traj_states,
                    'traj_controls': traj_controls,
                    'bev_color': bev_color,
                    'bev_elev': bev_elev,
                    'bev_normal': bev_normal,
                })
                frame_idx += 1

                ## we check for reset between end of current episode and future_traj_len + skip frames to make sure the "next" frame does not have a reset flag
                reset_range = reset_seq[data_idxs[-1]: data_idxs[-1] + future_traj_len + skip_frames]
                if reset_range.any():
                    i = data_idxs[-1] + np.where(reset_range)[0].item() + past_traj_len + future_traj_len + skip_frames

            # Chunk jobs
            job_chunk_size = args.job_chunk_size
            if len(job_args) > job_chunk_size:
                jobs = []
                for i in range(math.ceil(len(job_args) / job_chunk_size)):
                    jobs.append(job_args[i*job_chunk_size:(i+1)*job_chunk_size])
            else:
                jobs = [job_args] # no chunking

            for job_chunk_args in jobs:
                manager = multiprocessing.Manager()
                ctx = multiprocessing.get_context('spawn')

                # Apply workers to a single chunk
                with ctx.Pool(n_workers) as pool:
                    gc.collect()
                    async_results = [pool.apply_async(process_data, (job, save_vis)) for job in job_chunk_args]

                    for future in tqdm.tqdm(async_results):
                        gc.collect()
                        ret = future.get()

                        # Save frame data
                        frame_idx = int(ret['frame_idx'])
                        trajectory = ret['trajectory']
                        bev_color = ret['bev_color']
                        bev_elev = ret['bev_elev']
                        bev_normal = ret['bev_normal']

                        np.save(trajectory_path / f'{frame_idx:05d}.npy', trajectory, allow_pickle=True)
                        np.save(bev_color_path / f'{frame_idx:05d}.npy', bev_color)
                        np.save(bev_elev_path / f'{frame_idx:05d}.npy', bev_elev)
                        np.save(bev_normal_path / f'{frame_idx:05d}.npy', bev_normal)

                        vis_image = ret['vis_image']
                        if vis_image is not None:
                            vis_image.save(vis_path / f'{frame_idx:05d}.png')

        if split == 'train':
            # Get standardization stats
            calc_data_stats(output_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--cfg', type=str, help='config file for dataset')
    parser.add_argument('--output_dir', type=str, help='output path')
    parser.add_argument('--workers', type=int, default=8, help='num_workers')
    parser.add_argument('--job_chunk_size', type=int, default=100)
    parser.add_argument('--save_vis', type=bool, default=False)

    args = parser.parse_args()

    ## Debug
    # args.cfg = 'small_island_manual'
    # args.output_dir = 'small_island_debug'
    # args.workers = 8
    # args.save_vis = True

    generate_dataset(args)
