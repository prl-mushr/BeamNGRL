import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import cv2
from glob import glob
from PIL import Image
from BeamNGRL import *
from utils.dataset_utils import from_np
from collections import defaultdict
from typing import List


def get_datasets(
    bs=32,
    dataset_path=None,
    shuffle=True,
    augment=False,
    map_cfg=None,
    state_feat: List = None,
    control_feat: List = None,
    state_input_key: str = None,
    ctrl_input_key: str = None,
    ctx_input_keys: List = None,
):

    # get dataset stats
    dataset_stats = np.load(dataset_path / 'input_stats.npy', allow_pickle=True)

    grid_size = map_cfg['width'] // map_cfg['resolution']

    train_ds = DynamicsDataset(
        dataset_path=dataset_path,
        split='train',
        grid_size=grid_size,
        aug=augment,
        state_feat=state_feat,
        control_feat=control_feat,
        state_input_key=state_input_key,
        ctrl_input_key=ctrl_input_key,
        ctx_input_keys=ctx_input_keys,
    )

    valid_ds = DynamicsDataset(
        dataset_path=dataset_path,
        split='valid',
        grid_size=grid_size,
        state_feat=state_feat,
        control_feat=control_feat,
        state_input_key=state_input_key,
        ctrl_input_key=ctrl_input_key,
        ctx_input_keys=ctx_input_keys,
    )

    train_loader = DataLoader(
            train_ds, batch_size=bs, num_workers=bs//4, shuffle=shuffle)

    valid_loader = DataLoader(
            valid_ds, batch_size=bs, num_workers=1, shuffle=shuffle)

    return train_loader, valid_loader, dataset_stats


class DynamicsDataset(Dataset):

    def __init__(
            self,
            dataset_path=None,
            split="train",
            grid_size=100,
            aug=False,
            state_feat: List = None,
            control_feat: List = None,
            state_input_key: str = None,
            ctrl_input_key: str = None,
            ctx_input_keys: List = None,
    ):

        super().__init__()

        self.split = split
        self.dataset_path = dataset_path
        self.bev_files = defaultdict(list)
        self.traj_files = []
        self.num_files = 0
        self.grid_size = grid_size
        self.aug = aug
        self.bev_list = []
        self.state_feat = state_feat
        self.control_feat = control_feat
        seq_path = dataset_path / split
        self.add_data(seq_path)
        self.num_files = len(self.traj_files)

        self.state_input_key = state_input_key
        self.ctrl_input_key = ctrl_input_key
        self.ctx_input_keys = ctx_input_keys


    def add_data(self, basepath):

        self.bev_list = [
            'bev_color',
            'bev_elev',
            'bev_normal',
        ]

        traj_path = basepath / 'trajectories'
        num_files = len(glob(traj_path.__str__() + "/*.npy"))

        for i in range(num_files):
            inp_name = f'{i:05d}.npy'
            for bev_type in self.bev_list:
                self.bev_files[bev_type].append(basepath / bev_type / inp_name)
            self.traj_files.append(traj_path / inp_name)

    def __len__(self):
        return int(self.num_files)

    def get_bevmap(self, bev_type, idx):
        bevmap = np.load(self.bev_files[f'{bev_type}'][idx])
        h, w = bevmap.shape[:2]
        return bevmap.reshape((h, w, -1))

    def rotate_trajectories(self, traj, angle):
        # traj: B, H, dim
        traj = np.copy(traj)
        c, s = np.cos(angle), np.sin(angle)
        R = np.array(((c, -s), (s, c)))
        traj[:, :, :2] = np.einsum('ij,klj->kli', R, traj[:, :, :2])
        traj[:, :, 2] += angle
        return traj

    def __getitem__(self, t):

        # Load input files
        traj_input = np.load(self.traj_files[t], allow_pickle=True).item()
        bev_input_dict = {bev_type: self.get_bevmap(bev_type, t) for bev_type in self.bev_list}

        curr_time = traj_input['timestamp']
        state = traj_input['state']
        control = traj_input['control']

        past_ts = traj_input['past_timestamps']
        past_states = traj_input['past_states']
        past_ctrls = traj_input['past_controls']

        future_ts = traj_input['future_timestamps']
        future_states = traj_input['future_states']
        future_ctrls = traj_input['future_controls']

        # Relative timestamps (to current time)
        past_ts -= curr_time
        future_ts -= curr_time
        curr_time = 0.

        ret = {}

        ret['state'] = state
        ret['control'] = control
        ret['curr_time'] = curr_time

        ret['past_times'] = past_ts
        ret['past_states'] = past_states
        ret['past_ctrls'] = past_ctrls

        ret['future_times'] = future_ts
        ret['future_states'] = future_states
        ret['future_ctrls'] = future_ctrls

        # BEV maps: reshape to (C, H, W)
        bev_input_dict = {k: v.transpose(2, 0, 1) for k, v in bev_input_dict.items()}
        ret.update(bev_input_dict)

        # Get specified inputs
        state_input = ret.get(self.state_input_key)
        ctrl_input = ret.get(self.ctrl_input_key)
        ctx_input_dict = {k: ret.get(k) for k in self.ctx_input_keys}

        return state_input, ctrl_input, ctx_input_dict
