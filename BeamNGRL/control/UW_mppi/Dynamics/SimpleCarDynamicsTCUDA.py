import torch
from torch.utils.cpp_extension import load
import time
import numpy as np
import time
import os
import sys

class SimpleCarDynamics:
    """
    Class for Dynamics modelling
    """

    def __init__(
        self,
        Dynamics_config,
        Map_config,
        MPPI_config,
        dtype=torch.float32,
        device=torch.device("cuda"),
    ):
        self.dtype = dtype
        self.d = device

        self.throttle_to_wheelspeed = np.float32(Dynamics_config["throttle_to_wheelspeed"])
        self.steering_max = np.float32(Dynamics_config["steering_max"])

        self.dt_default = np.float32(Dynamics_config["dt"])
        self.dt = self.dt_default
        self.K = np.int32(MPPI_config["ROLLOUTS"])
        self.T = np.int32(MPPI_config["TIMESTEPS"])
        self.M = np.int32(MPPI_config["BINS"])
        self.NX = np.int32(17)
        self.NC = np.int32(2)

        self.BEVmap_size = np.float32(Map_config["map_size"])
        self.BEVmap_res = np.float32(Map_config["map_res"])
        self.BEVmap_size_px = np.int32(self.BEVmap_size / self.BEVmap_res)
        
        self.states = torch.zeros( (self.M, self.K, self.T, self.NX), dtype=self.dtype, device=self.d)

        ## pulled these values from: A Hybrid Hierarchical Rally Driver Model for Autonomous Vehicle Agile Maneuvering on Loose Surfaces
        self.D = np.float32(Dynamics_config["D"])
        self.B = np.float32(Dynamics_config["B"])
        self.C = np.float32(Dynamics_config["C"])
        self.lf = np.float32(Dynamics_config["lf"])
        self.lr = np.float32(Dynamics_config["lr"])
        self.Iz = np.float32(Dynamics_config["Iz"])
        self.LPF_tau = np.float32(Dynamics_config["LPF_tau"])
        self.LPF_st = np.float32(Dynamics_config["LPF_st"])
        self.LPF_th = np.float32(Dynamics_config["LPF_th"])
        self.res_coeff = np.float32(Dynamics_config["res_coeff"])
        self.drag_coeff = np.float32(Dynamics_config["drag_coeff"])

        self.car_l2 = np.float32(Dynamics_config["car_length"]/2)
        self.car_w2 = np.float32(Dynamics_config["car_width"]/2)
        self.cg_height = np.float32(Dynamics_config["cg_height"])

        # Set grid and block dimensions
        self.block_dim = 32 #min(MPPI_config["ROLLOUTS"], 1024) # use 32 for jetson, use 1024 for RTX GPUs
        self.grid_dim = int(np.ceil(self.K / self.block_dim))

        folder_name = 'BeamNGRL'

        # Check each directory in sys.path for the folder
        for path in sys.path:
            folder_path = os.path.join(path, folder_name)
            if os.path.exists(folder_path):
                break
        else:
            print("Did you forget to add BeamNGRL to your PYTHONPATH?")

        cpp_path = '{}/control/UW_mppi/Dynamics/analytical_bicycle.cpp'.format(folder_path)
        cuda_path = '{}/control/UW_mppi/Dynamics/analytical_bicycle.cu'.format(folder_path)
        # Compile and load the extension

        self.kernel = load(
            name="analytical_bicycle",
            sources=[cpp_path, cuda_path],
            verbose=False,
        )
        if Dynamics_config["type"] == "slip3d":
            self.model_forward = self.kernel.rollout_slip3d
        elif Dynamics_config["type"] == "noslip3d":
            self.model_forward = self.kernel.rollout_noslip3d

        self.BEVmap_height = torch.zeros((self.BEVmap_size_px, self.BEVmap_size_px), dtype=self.dtype, device=self.d)
        self.BEVmap_normal = torch.zeros((3, self.BEVmap_size_px, self.BEVmap_size_px), dtype=self.dtype, device=self.d)


    def set_BEV(self, BEVmap_height, BEVmap_normal):
        self.BEVmap_height = BEVmap_height
        self.BEVmap_normal = BEVmap_normal

    def get_states(self):
        return self.states

    def forward(self, state, controls):
        # Launch the CUDA kernel
        state_ = state.squeeze(0)
        controls_ = controls.squeeze(0)
        self.model_forward(state_, controls_, self.BEVmap_height, self.BEVmap_normal, self.dt, self.K, self.T, self.NX, self.NC,
                self.D, self. B, self.C, self.lf, self.lr, self.Iz, self.throttle_to_wheelspeed, self.steering_max,
                self.BEVmap_size_px, self.BEVmap_res, self.BEVmap_size, self.car_l2, self.car_w2, self.cg_height, self.LPF_tau, self.LPF_st, self.LPF_th, self.res_coeff, self.drag_coeff,
                self.block_dim, self.grid_dim)
        self.states = torch.clone(state_).unsqueeze(0)
        # this whole forward takes 0.1 ms for 24 steps.
        return self.states