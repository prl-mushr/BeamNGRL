import numpy as np
import torch
import os
from typing import Dict, Tuple, Union



def to_np(data: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    return data

def from_np(data, device = torch.device('cpu')):
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data).to(device)
    elif isinstance(data, torch.Tensor):
        data = data.to(device)
    return data

def recursive_glob(rootdir=".", suffix=""):
    return [
        os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames
        if filename.endswith(suffix)
    ]


def get_files(scan_dir, filter=None, suffix='.npy'):
    file_list = recursive_glob(rootdir=scan_dir, suffix=suffix)
    new_list = []
    for f in file_list:
        if filter is not None:
            if f.find(filter) == -1:
                continue
            new_list.append(f)
    return new_list


def get_full_traj(trajectory: Dict) -> Tuple[np.ndarray, np.ndarray]:
    state_traj = np.concatenate((
        trajectory['past_states'],
        trajectory['state'].reshape((1, -1)),
        trajectory['future_states'],
    ), axis=0)
    control_traj = np.concatenate((
        trajectory['past_controls'],
        trajectory['control'].reshape((1, -1)),
        trajectory['future_controls'],
    ), axis=0)
    return state_traj, control_traj


def load_timestamps(file_name: str, file_path: os.PathLike) -> np.ndarray:
    timestamp_arr = np.load(file_path / file_name, allow_pickle=True)
    timestamp_arr -= timestamp_arr[0] # Start at t = 0.
    return timestamp_arr


def get_kinematic_traj(file_name: str, file_path: os.PathLike,
                       timestamps: np.ndarray) -> np.ndarray:
    states_arr = np.load(file_path / file_name, allow_pickle=True)
    pos = states_arr[:, 0:3] # xyz world frame
    rot = states_arr[:, 3:6] # rpy world frame
    lin_vel = states_arr[:, 6:9] # xyz inertial frame
    lin_accel = states_arr[:, 9:12] # xyz inertial-frame

    # Finite-difference for angular vels (world frame)
    dt = timestamps[1:] - timestamps[:-1]
    ang_vel = (rot[1:] - rot[:-1]) / dt[:, None]
    ang_vel = np.concatenate((ang_vel, ang_vel[[-1], :]), axis=0) # Copy last value

    # Finite-difference for angular accels (world frame)
    # ang_accel = (ang_vel[1:] - ang_vel[:-1]) / dt[:, None]
    # ang_accel = np.concatenate((ang_accel, ang_accel[[-1], :]), axis=0) # Copy last value

    # World-frame trajectory
    trajectory = np.concatenate(
        (pos, rot, lin_vel, lin_accel, ang_vel),
        axis=-1)

    return trajectory



def get_controls(file_name: str, file_path: os.PathLike) -> np.ndarray:
    states_arr = np.load(file_path / file_name, allow_pickle=True)
    steer = states_arr[:, [15]]
    throttle = states_arr[:, [16]]
    controls = np.concatenate((steer, throttle), axis=-1)
    return controls


def load_bev_map(file_name: str, file_path: os.PathLike) -> np.ndarray:
    map = np.load(file_path / file_name, allow_pickle=True)
    return map


def project_traj_to_map(
        posns: np.ndarray, grid_size: int, resolution: float, mode: str = 'default',
) -> Tuple[np.ndarray, np.ndarray]:

        offsets = (posns[:, :2] / resolution).astype(np.int32)
        pixel_locs = offsets + np.array([grid_size // 2, grid_size // 2], np.int32)[None]

        if mode == 'clamp':
            traj = np.clip(pixel_locs, a_min=0, a_max=grid_size-1)
            in_range = None

        elif mode == 'radius':
            rad_max = grid_size // 2
            traj_rad = np.sqrt((offsets**2).sum(axis=1))
            in_range = traj_rad < rad_max
            indices = np.arange(len(pixel_locs))[in_range]
            traj = pixel_locs[indices]

        elif mode == 'default':
            in_range = (0 <= pixel_locs[:, 0]) & (pixel_locs[:, 0] < grid_size) & \
                       (0 <= pixel_locs[:, 1]) & (pixel_locs[:, 1] < grid_size)
            indices = np.arange(len(pixel_locs))[in_range]
            traj = pixel_locs[indices]
        else:
            raise IOError

        return traj, in_range


def crop_traj_within_grid(
        posns: np.ndarray, grid_size: int, resolution: float, past_traj: bool =False,
) -> Tuple[np.ndarray, np.ndarray]:

        offsets = (posns[:, :2] / resolution).astype(np.int32)
        pixel_locs = offsets + np.array([grid_size // 2, grid_size // 2], np.int32)[None]
        out_range = (pixel_locs[:, 0] < 0) | (grid_size <= pixel_locs[:, 0]) | \
                       (pixel_locs[:, 1] < 0) | (grid_size <= pixel_locs[:, 1])
        indices = np.arange(len(pixel_locs))[out_range]
        if indices.size > 0:
            indices = sorted(indices)
            if not past_traj:
                crop_idx = indices[0] # first violating idx
                posns_cropped = posns[:crop_idx]
            else:
                crop_idx = indices[-1] # last violating idx
                posns_cropped = posns[crop_idx+1:]
            return posns_cropped, crop_idx
        else:
            return posns, None
