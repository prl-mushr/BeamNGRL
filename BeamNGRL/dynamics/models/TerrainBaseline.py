import torch
import torch.nn as nn
import torch.nn.utils.spectral_norm as spnorm
from torch.utils.cpp_extension import load
from BeamNGRL.dynamics.models.base import DynamicsBase
from typing import Dict
import time
# from BeamNGRL.dynamics.utils.misc_utils import * ## uncomment on eval
import numpy as np
import time
import sys
import os


class SequentialContextMLP(DynamicsBase):

    def __init__(
            self,
            hidden_depth=2,
            hidden_dim=512,
            batch_norm=False,
            dt=0.1,
            wheelbase=2.6,
            trackwidth=1.5,
            BEVmap_size=128,
            BEVmap_res=0.25,
            patch_size=3.0, ## if this works, lets define the mean, std, std_err tensors like this as well.
            mean_state=None,
            std_state=None,
            std_state_err=None,
            mean_control=None,
            std_control=None,
            spectral_norm=False,
            **kwargs,
    ):

        super().__init__(**kwargs)

        self.dtype = torch.float
        self.d = torch.device('cuda')

        if mean_state is None or std_state is None or mean_control is None or std_control is None:
            print("=====================================================================")
            print("Please define the mean, std, std_err of the state in the config file.")
            print("=====================================================================")
            exit()
        
        self.NX = np.int32(17)
        self.NC = np.int32(2)

        self.mean_state = torch.zeros(self.NX, dtype=self.dtype, device =self.d)
        self.std_state = torch.zeros(self.NX, dtype=self.dtype, device =self.d)
        self.mean_state[:self.NX-self.NC] = torch.tensor(mean_state).to(self.d)
        self.std_state[:self.NX-self.NC] = torch.tensor(std_state).to(self.d)
        self.mean_state[self.NX-self.NC:] = torch.tensor(mean_control).to(self.d)
        self.std_state[self.NX-self.NC:] = torch.tensor(std_control).to(self.d)
        self.std_bev = torch.tensor(patch_size/2).to(self.d) ## expect the height in the crop patch to change around the center by at most patch_size/2, corresponding to a 45 degrees

        self.K = np.int32(1)
        self.T = np.int32(1)
        self.M = np.int32(1)
        ## pulled these values from: A Hybrid Hierarchical Rally Driver Model for Autonomous Vehicle Agile Maneuvering on Loose Surfaces
        self.car_l2 = np.float32(wheelbase/2)
        self.car_w2 = np.float32(trackwidth/2)

        self.BEVmap_size = torch.tensor(BEVmap_size, dtype=self.dtype, device=self.d)
        self.BEVmap_res = torch.tensor(BEVmap_res, dtype=self.dtype, device=self.d)
        self.BEVmap_size_px = (self.BEVmap_size/self.BEVmap_res).clone().detach().to(device=self.d, dtype=torch.int32)

        self.patch_size = patch_size
        self.patch_size_px = int(self.patch_size/self.BEVmap_res)

        self.BEVmap_height = torch.zeros((self.BEVmap_size_px, self.BEVmap_size_px), dtype=self.dtype, device=self.d)
        self.BEVmap_normal = torch.zeros((3, self.BEVmap_size_px, self.BEVmap_size_px), dtype=self.dtype, device=self.d) # this is a placeholder for now.

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

        # ================= DEFINE NETWORKS============================
        self.spectral_norm = spectral_norm
        self.state_dim = 10 ## vx, vy, vz, wx, wy, wz, cr, sr, cp, sp, st, th
        self.ctrl_dim = 2
        self.context_dim = 12
        self.input_dim = self.context_dim + self.state_dim + self.ctrl_dim
        self.output_dim = 6 ## dvx/dt, dvy/dt, dvz/dt, dwx/dt, dwy/dt, dwz/dt
        self.dt = dt

        self.kernel_size = 3
        self.stride = 1
        self.channels = 2
        K_pool = 3
        S_pool = 2
        L1 = int((self.patch_size_px - self.kernel_size)/self.stride) + 1
        L2 = int( (L1 - K_pool)/S_pool) + 1
        output_size = int( (L2 - self.kernel_size)/self.stride) + 1
        conv1 = nn.Conv2d(1, self.channels, kernel_size=self.kernel_size, stride=self.stride)
        maxpool = nn.MaxPool2d(kernel_size=K_pool, stride=S_pool)
        conv2 = nn.Conv2d(self.channels, self.channels, kernel_size=self.kernel_size, stride=self.stride)
        # use the image size, and formulas for CNN output size to compute the input size for the first FC layer
        fc1 = nn.Linear(output_size*output_size*self.channels, 16)
        fc2 = nn.Linear(16, self.context_dim)

        cnn_layers = [ conv1, nn.Tanh() ]
        cnn_layers += [maxpool]
        cnn_layers += [conv2, nn.Tanh() ]
        cnn_layers += [nn.Flatten()]
        cnn_layers += [ fc1, nn.Tanh() ]
        cnn_layers += [ fc2 ]
        self.CNN = nn.Sequential(*cnn_layers)

        if self.spectral_norm:
            fc_layers = [
                nn.Linear(self.input_dim, hidden_dim),
                nn.Tanh(),
            ]
            for _ in range(hidden_depth):
                fc_layers += [spnorm(nn.Linear(hidden_dim, hidden_dim))]
                fc_layers += [nn.Tanh()]
            fc_layers += [spnorm(nn.Linear(hidden_dim, self.output_dim))]
            fc_layers += [nn.Tanh()]
        
        else:
            fc_layers = [
                nn.Linear(self.input_dim, hidden_dim),
                nn.Tanh(),
            ]
            for _ in range(hidden_depth):
                fc_layers += [nn.Linear(hidden_dim, hidden_dim)]
                fc_layers += [nn.Tanh()]
            fc_layers += [nn.Linear(hidden_dim, self.output_dim)]
            fc_layers += [nn.Tanh()]


        self.main = nn.Sequential(*fc_layers)

    def _forward(
            self,
            states: torch.Tensor, # b, L, d
            controls: torch.Tensor,
            ctx_data: Dict,
            evaluation=False,
            count=0,
            dt=0.1
    ):
        states = states.clone().detach()
        ctrls = controls.clone().detach()

        states = torch.cat((states, ctrls), dim=-1) # hotfix babyy

        n = states.shape[-1]
        n_c = controls.shape[-1]
        t = states.shape[-2]
        batchsize = states.shape[-3]
        
        self.BEVmap_height = ctx_data['bev_elev']
        self.NX = np.int32(n)
        self.NC = np.int32(n_c)

        self.K = np.int32(t) # we flatten the time-series data and treat each time-step as IID for single-step prediction training.
        # this technically gives us a batchsize = t, rather than a batchsize of "1". 
        self.bev_context = torch.zeros((batchsize, self.K, self.patch_size_px, self.patch_size_px), dtype=self.dtype, device=self.d)
        self.sa = torch.zeros(batchsize, self.K, 12, dtype=self.dtype, device=self.d)
        # Set grid and block dimensions
        self.block_dim = 8 #min(MPPI_config["ROLLOUTS"], 1024) # use 32 for jetson, use 1024 for RTX GPUs
        self.grid_dim = int(np.ceil(self.K / self.block_dim))

        for i in range(batchsize):
            self.preprocess(states[i, ...].unsqueeze(1), ctrls[i, ...].unsqueeze(1), self.sa[i, ...], self.bev_context[i, ...], np.int32(0), 
                        self.BEVmap_height[i, ...].squeeze(0), self.BEVmap_normal,
                        self.BEVmap_size_px, self.BEVmap_res, self.BEVmap_size, self.K, self.T, 
                        self.NX, self.NC, self.car_l2, self.car_w2, self.std_state, self.mean_state, self.patch_size,
                        self.block_dim, self.grid_dim)

        bev_context = self.bev_context.reshape((batchsize * t, self.patch_size_px, self.patch_size_px))
        sa = self.sa.reshape((batchsize * t, 12))
        context = self.CNN(bev_context.unsqueeze(0).transpose(0,1))
        sac = torch.cat((sa, context), dim=-1)
        Ddot_q = self.main(sac)

        Ddot_q = Ddot_q.reshape((batchsize, t, self.output_dim))

        states[..., 6:9] = states[..., 6:9] + Ddot_q[..., 0:3]* self.std_state[9:12] * self.dt
        states[..., 12:15] = states[..., 12:15] + Ddot_q[..., 3:6]* self.std_state[12:15] * self.dt

        return states[..., : self.NX - self.NC] # because we're only expecting 15 back...

    def _rollout(
            self,
            states,
            controls,
            ctx_data,
            dt=0.08,
    ):

        horizon = states.shape[-2]
        for i in range(horizon - 1):
            states[..., [i+1], :] = self._forward(
                                    states[..., [i], :],
                                    controls[..., [i], :],
                                    ctx_data,
                                    evaluation=True,
                                    dt=dt,
                                    count = i,
                                )  # B x 1 x D
        return states

    def rollout(
            self,
            states_input,
            controls,
            ctx_data,
            dt = 0.08
    ):
        states = states_input.clone().detach()
        states[..., 1:, :] = states[...,[0],:]
        steer = controls[..., 0]
        throttle = controls[..., 1]
        with torch.no_grad():
            states_pred = self._rollout(states[...,:15], controls, ctx_data, dt=dt)

        x,y,z, roll, pitch, yaw, vx, vy, vz, ax, ay, az, wx, wy, wz = states_pred.split(1, dim=-1)
        ## squeeze all the singleton dimensions for all the states
        vx = vx.squeeze(-1) # + controls[..., 1]*20
        vy = vy.squeeze(-1)
        vz = vz.squeeze(-1)
        ax = ax.squeeze(-1)
        ay = ay.squeeze(-1)
        az = az.squeeze(-1)
        wx = wx.squeeze(-1)
        wy = wy.squeeze(-1)
        wz = wz.squeeze(-1) #vx*torch.tan(controls[..., 0] * 0.5)/2.6
        roll = roll.squeeze(-1)
        pitch = pitch.squeeze(-1)
        yaw = yaw.squeeze(-1)
        x = x.squeeze(-1)
        y = y.squeeze(-1)
        z = z.squeeze(-1)

        return torch.stack((x, y, z, roll, pitch, yaw, vx, vy, vz, ax, ay, az, wx, wy, wz, steer, throttle), dim=-1)
    
