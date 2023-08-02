import torch
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
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
        dtype=np.float32,
        device=cuda.Device(0),
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
        
        self.states = torch.zeros( (self.M, self.K, self.T, 17), dtype=torch.float32, device=torch.device('cuda:0'))

        ## pulled these values from: A Hybrid Hierarchical Rally Driver Model for Autonomous Vehicle Agile Maneuvering on Loose Surfaces
        self.D = np.float32(Dynamics_config["D"])
        self.B = np.float32(Dynamics_config["B"])
        self.C = np.float32(Dynamics_config["C"])
        self.lf = np.float32(Dynamics_config["lf"])
        self.lr = np.float32(Dynamics_config["lr"])
        self.Iz = np.float32(Dynamics_config["Iz"])
        self.LPF_tau = np.float32(Dynamics_config["LPF_tau"])
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

        file_path = '{}/control/UW_mppi/Dynamics/{}.cpp'.format(folder_path, Dynamics_config["type"])
       
        with open(file_path, 'r') as file:
            self.cuda_code = file.read()
        self.module = SourceModule(self.cuda_code)
        self.rollout = self.module.get_function("rollout")
        self.BEVmap_height = gpuarray.to_gpu(np.zeros((self.BEVmap_size_px, self.BEVmap_size_px), dtype=dtype) )
        self.BEVmap_normal = gpuarray.to_gpu(np.zeros((self.BEVmap_size_px, self.BEVmap_size_px, 3), dtype=dtype) )

    def set_BEV(self, BEVmap_height, BEVmap_normal):
        self.BEVmap_height = gpuarray.to_gpu(BEVmap_height.cpu().numpy())
        self.BEVmap_normal = gpuarray.to_gpu(BEVmap_normal.cpu().numpy())

    # faster and more memory efficient approach (Shaves off a whole 2 milliseconds on the jetson (which is 10% of 20 millisecond update cycle!))
    def set_BEV_numpy(self, BEVmap_height, BEVmap_normal):
        self.BEVmap_height = gpuarray.to_gpu(BEVmap_height)
        self.BEVmap_normal = gpuarray.to_gpu(BEVmap_normal)

    def get_states(self):
        return self.states

    def forward(self, state, controls):
        controls = gpuarray.to_gpu(controls.squeeze(0).cpu().numpy())
        state_ = gpuarray.to_gpu(state.squeeze(0).cpu().numpy())

        # Launch the CUDA kernel
        self.rollout(state_, controls, self.BEVmap_height, self.BEVmap_normal, self.dt, self.K, self.T, self.NX, self.NC,
                self.D, self. B, self.C, self.lf, self.lr, self.Iz, self.throttle_to_wheelspeed, self.steering_max,
                self.BEVmap_size_px, self.BEVmap_res, self.BEVmap_size, self.car_l2, self.car_w2, self.cg_height, self.LPF_tau, self.res_coeff, self.drag_coeff,
                block=(self.block_dim, 1, 1), grid=(self.grid_dim, 1))
        cuda.Context.synchronize()

        self.states  = torch.from_numpy(state_.get()).unsqueeze(0).to(torch.device('cuda'))
        return self.states
