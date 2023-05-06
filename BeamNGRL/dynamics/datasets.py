import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import cv2
from glob import glob
from PIL import Image
from BeamNGRL import *
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import random


def get_datasets(
    bs=32,
    dataset_path=None,
    shuffle=True,
    aug=False,
    map_cfg=None,
):

    # get dataset stats
    input_stats = np.load(dataset_path / 'input_stats.npy', allow_pickle=True)

    grid_size = map_cfg['width'] // map_cfg['resolution']

    train_ds = DynamicsDataset(
        dataset_path=dataset_path,
        split='train',
        grid_size=grid_size,
        aug=aug,
        stats=input_stats,
    )

    valid_ds = DynamicsDataset(
        dataset_path=dataset_path,
        split='valid',
        grid_size=grid_size,
        stats=input_stats,
    )

    train_loader = DataLoader(
            train_ds, batch_size=bs, num_workers=bs//4, shuffle=shuffle)

    valid_loader = DataLoader(
            valid_ds, batch_size=bs, num_workers=1, shuffle=shuffle)

    return train_loader, valid_loader


class DynamicsDataset(Dataset):

    def __init__(
            self,
            dataset_path=None,
            split="train",
            grid_size=100,
            aug=False,
            stats=None,
    ):

        super().__init__()

        self.split = split
        self.dataset_path = dataset_path
        self.bev_files = defaultdict(list)
        self.traj_files = []
        self.num_files = 0
        self.grid_size = grid_size
        self.aug = aug
        self.stats_dict = stats

        seq_path = dataset_path / split
        self.add_data(seq_path)
        self.num_files = len(self.traj_files)

    def add_data(self, basepath):

        bev_list = [
            'bev_color',
            'bev_elev',
            'bev_normal',
        ]

        traj_path = basepath / 'trajectories'
        num_files = len(glob(traj_path.__str__() + "/*.npy"))

        for i in range(num_files):
            inp_name = f'{i:05d}.npy'
            for bev_type in bev_list:
                self.bev_files[bev_type].append(basepath / bev_type / inp_name)
            self.traj_files.append(traj_path / inp_name)

    def __getitem__(self, t):
        raise NotImplementedError



