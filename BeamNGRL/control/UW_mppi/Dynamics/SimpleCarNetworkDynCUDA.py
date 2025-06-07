import torch
from BeamNGRL.dynamics.utils.exp_utils import build_nets
from BeamNGRL.dynamics.utils.network_utils import load_model
from typing import Dict
from torch.utils.cpp_extension import load
import numpy as np
import time
import os
import sys

torch.backends.cuda.matmul.allow_tf32 = True

class SimpleCarNetworkDyn:
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
        model_weights_path=None,
    ):
        self.dtype = dtype
        self.d = device
        self.tn_args = {'device': device, 'dtype': dtype}
        dyn_model = self.load_dyn_model(Dynamics_config, model_weights_path, self.tn_args)
        self.main =  torch.jit.script(dyn_model.main)
        self.CNN  =  dyn_model.CNN

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

        self.car_l2 = np.float32(Dynamics_config["car_length"]/2)
        self.car_w2 = np.float32(Dynamics_config["car_width"]/2)
        self.patch_size = Dynamics_config["patch_size"]
        self.patch_size_px = int(self.patch_size/self.BEVmap_res)
        self.bev_context = torch.zeros((self.K, self.patch_size_px, self.patch_size_px), dtype=self.dtype, device=self.d)
        self.sa = torch.zeros(self.K, 12, dtype=self.dtype, device=self.d)
        # Set grid and block dimensions
        # The number of parallel threads in total will be block_dim x grid_dim, 
        # so pick block dim such that grid dim is a nice round number
        self.block_dim = min(self.K, 32)
        self.grid_dim = int(np.ceil(self.K / self.block_dim))

        self.BEVmap_height = torch.zeros((self.BEVmap_size_px, self.BEVmap_size_px), dtype=self.dtype, device=self.d)
        self.BEVmap_normal = torch.zeros((3, self.BEVmap_size_px, self.BEVmap_size_px), dtype=self.dtype, device=self.d)
        self.mean_state = torch.zeros(self.NX, dtype=self.dtype, device =self.d)
        self.std_state = torch.zeros(self.NX, dtype=self.dtype, device =self.d)
        self.mean_state[:self.NX-self.NC] = torch.tensor(Dynamics_config["network"]["net_kwargs"]["mean_state"]).to(self.d)
        self.std_state[:self.NX-self.NC] = torch.tensor(Dynamics_config["network"]["net_kwargs"]["std_state"]).to(self.d)
        self.mean_state[self.NX-self.NC:] = torch.tensor(Dynamics_config["network"]["net_kwargs"]["mean_control"]).to(self.d)
        self.std_state[self.NX-self.NC:] = torch.tensor(Dynamics_config["network"]["net_kwargs"]["std_control"]).to(self.d)
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
        self.preprocess = self.kernel.rollout_preprocess
        self.euler_step = self.kernel.rollout_euler_step
        self.dt_avg = 0


    def load_dyn_model(self, config, weights_path, tn_args: Dict = None):
        net, _ = build_nets(config, tn_args, model_weight_file=weights_path)
        net.eval()
        return net

    def set_BEV(self, BEVmap_height, BEVmap_normal):
        self.BEVmap_height = BEVmap_height
        self.BEVmap_normal = BEVmap_normal

    def get_states(self):
        return self.states

    def forward(self, states, controls):
        # # Launch the CUDA kernel
        for i in range(0, int(self.T.item())):

            self.preprocess(states, controls, self.sa, self.bev_context, i, self.BEVmap_height, self.BEVmap_normal, 
                        self.BEVmap_size_px, self.BEVmap_res, self.BEVmap_size, self.K, self.T, 
                        self.NX, self.NC, self.car_l2, self.car_w2, self.std_state, self.mean_state, self.patch_size,
                        self.block_dim, self.grid_dim)

            context = self.CNN(self.bev_context.unsqueeze(0).transpose(0,1)) # raw: [1024, 12, 12]) expected: torch.Size([1024, 1, 12, 12]
            sac = torch.cat((self.sa, context), dim=-1)
            Ddot_q = self.main(sac)

            self.euler_step(states, controls, Ddot_q, i, self.BEVmap_height, self.BEVmap_normal, 
                            self.BEVmap_size_px, self.BEVmap_res, self.BEVmap_size, self.dt, self.K, self.T, 
                            self.NX, self.NC, self.car_l2, self.car_w2, self.std_state, self.mean_state,
                            self.block_dim, self.grid_dim)

        self.states = torch.clone(states)
        return self.states