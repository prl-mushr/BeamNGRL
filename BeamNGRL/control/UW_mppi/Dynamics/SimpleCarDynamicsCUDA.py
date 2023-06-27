import torch
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import numpy as np

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

        self.BEVmap_height = gpuarray.to_gpu(np.zeros((self.BEVmap_size_px, self.BEVmap_size_px), dtype=dtype) )
        self.BEVmap_normal = gpuarray.to_gpu(np.zeros((self.BEVmap_size_px, self.BEVmap_size_px, 3), dtype=dtype) )

        self.states = torch.zeros( (self.M, self.K, self.T, 17), dtype=torch.float32, device=torch.device('cuda:0'))

        ## pulled these values from: A Hybrid Hierarchical Rally Driver Model for Autonomous Vehicle Agile Maneuvering on Loose Surfaces
        self.D = np.float32(0.8)
        self.B = np.float32(6.8)
        self.C = np.float32(1.5)
        self.lf = np.float32(1.3)
        self.lr = np.float32(1.3)
        self.Iz = np.float32(1.0)

        self.car_l2 = np.float32(1.5)
        self.car_w2 = np.float32(0.75)
        self.cg_height = np.float32(0.5)

        # Set grid and block dimensions
        self.block_dim = MPPI_config["ROLLOUTS"]
        self.grid_dim = int(np.ceil(self.K / self.block_dim))

        file_path = '/home/stark/BeamNGRL/BeamNGRL/control/UW_mppi/Dynamics/cuda_kernel.cpp'

        with open(file_path, 'r') as file:
            self.cuda_code = file.read()
        self.module = SourceModule(self.cuda_code)
        self.rollout = self.module.get_function("rollout")


    def set_BEV(self, BEVmap_height, BEVmap_normal):
        self.BEVmap_height = gpuarray.to_gpu(BEVmap_height.cpu().numpy())
        self.BEVmap_normal = gpuarray.to_gpu(BEVmap_normal.cpu().numpy())

    def get_states(self):
        return self.states

    def forward(self, state, controls):
        controls = gpuarray.to_gpu(controls.squeeze(0).cpu().numpy())
        state_ = gpuarray.to_gpu(state.squeeze(0).cpu().numpy())

        # Launch the CUDA kernel
        self.rollout(state_, controls, self.BEVmap_height, self.BEVmap_normal, self.dt, self.K, self.T, self.NX, self.NC,
                self.D, self. B, self.C, self.lf, self.lr, self.Iz, self.throttle_to_wheelspeed, self.steering_max,
                self.BEVmap_size_px, self.BEVmap_res, self.BEVmap_size, self.car_l2, self.car_w2, self.cg_height,
                block=(self.block_dim, 1, 1), grid=(self.grid_dim, 1))
        # pack all values:
        self.states  = torch.from_numpy(state_.get()).unsqueeze(0).to(torch.device('cuda'))

        return self.states
