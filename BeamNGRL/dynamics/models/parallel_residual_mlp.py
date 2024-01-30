import torch
import torch.nn as nn
from BeamNGRL.dynamics.models.base import DynamicsBase
from typing import Dict
from BeamNGRL.dynamics.utils.network_utils import get_feat_index_tn
from BeamNGRL.dynamics.utils.network_utils import get_state_features, get_ctrl_features
import time
import cv2
import numpy as np
import time


class ParallelContextMLP(DynamicsBase):

    def __init__(
            self,
            hidden_depth=2,
            hidden_dim=512,
            batch_norm=False, ## TODO: pass timesteps, dt, patch size through this interface. BEVmap size is useful for min-max normalization of the state
            dt=0.1,
            timesteps=40,
            BEVmap_size=128,
            BEVmap_res=0.25,
            patch_size=3.0, ## if this works, lets define the mean, std, std_err tensors like this as well.
            mean_state=None,
            std_state=None,
            std_state_err=None,
            mean_control=None,
            std_control=None,
            **kwargs,
    ):

        super().__init__(**kwargs)

        self.dtype = torch.float
        self.d = torch.device('cuda')

        self.BEVmap_size = torch.tensor(BEVmap_size, dtype=self.dtype, device=self.d)
        self.BEVmap_res = torch.tensor(BEVmap_res, dtype=self.dtype, device=self.d)
        self.crop_size = torch.tensor(patch_size/self.BEVmap_res, device=self.d, dtype=torch.long)
        self.bev_cent = int(self.crop_size/2)
        self.dt = dt

        self.state_dim = 15
        self.ctrl_dim = 2
        self.context_dim = self.crop_size ## square root of the number of pixels in the patch -- this is somewhat arbitrary but works as a good approximation.
        self.timesteps = timesteps ## TODO this should be provided during initialization
        self.normalized_input_dim = self.context_dim + self.state_dim + self.ctrl_dim
        self.input_dim = self.timesteps * self.normalized_input_dim
        self.output_dim = self.timesteps*self.state_dim*2

        # TODO: these should be loaded programmatically.
        if mean_state is None or std_state is None or std_state_err is None or mean_control is None or std_control is None:
            print("=====================================================================")
            print("Please define the mean, std, std_err of the state in the config file.")
            print("=====================================================================")
            exit()
        self.mean_state = torch.tensor(mean_state).to(self.d)
        self.std_state = torch.tensor(std_state).to(self.d)
        self.std_state_err = torch.Tensor(std_state_err).to(self.d)
        ## std_state err is the denormalization we will use for residuals
        self.mean_control = torch.tensor(mean_control).to(self.d)
        self.std_control = torch.tensor(std_control).to(self.d)
        self.std_bev = torch.tensor(patch_size/2).to(self.d) ## expect the height in the crop patch to change around the center by at most patch_size/2, corresponding to a 45 degrees

        self.execution_dt = 1e-3
        self.time_scaling = (torch.arange(1, self.timesteps+1, device=self.d, dtype=float)/self.timesteps).view(1,self.timesteps, 1)

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

        self.kernel_size = 3
        self.stride = 1
        self.channels = 2
        K_pool = 3
        S_pool = 2
        L1 = int((self.crop_size-self.kernel_size)/self.stride) + 1
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

    def _forward(
            self,
            states: torch.Tensor, # b, L, d
            controls: torch.Tensor,
            ctx_data: Dict,
            Evaluation=False
    ):
        n = states.shape[-1]
        n_c = controls.shape[-1]
        t = states.shape[-2]
        k = states.shape[-3]

        states_next = states.clone().detach()
        ctrls = controls.clone().detach()
        bev_input = ctx_data['rotate_crop'].clone().detach()
        vU = torch.zeros((k, t, self.state_dim + self.ctrl_dim), dtype=self.dtype, device=self.d)

        vU[..., :2]  = (states_next[..., :2]   - self.mean_state[:2]  )/(self.std_state[:2] * self.time_scaling)
        vU[..., 5:9] = (states_next[..., 5:9]  - self.mean_state[5:9] )/(self.std_state[5:9]* self.time_scaling)
        vU[..., 2:5] = (states_next[..., 2:5]  - self.mean_state[2:5] )/ self.std_state[2:5]
        vU[...,9:15] = (states_next[..., 9:15] - self.mean_state[9:15])/ self.std_state[9:15]
        vU[...,15:]  = (ctrls - self.mean_control)/self.std_control

        vU = vU.reshape(k*t, self.state_dim + self.ctrl_dim)

        bev_center = bev_input.clone()
        bev_center[...,:,:] = bev_input[...,self.bev_cent,self.bev_cent].unsqueeze(-1).unsqueeze(-1)
        bev_input = (bev_input - bev_center)/self.std_bev
        now = time.time()
        context = self.CNN(bev_input.unsqueeze(0).transpose(0,1))
        vUc = torch.cat((vU, context), dim=-1).reshape(k, t * self.normalized_input_dim)
        self.execution_dt = time.time() - now
        dV = self.main(vUc).reshape(k, t, self.output_dim//t)

        gain =  torch.sigmoid(dV[..., self.state_dim:])
        network_output = dV[..., :self.state_dim] * self.std_state_err * self.timesteps * self.dt
        network_output[..., :2] *= self.time_scaling
        network_output += self.mean_state
        out = states_next[..., :self.state_dim]*(1-gain) + gain*network_output

        return out

    def _rollout(
            self,
            states,
            controls,
            ctx_data,
            dt=0.02,
    ):
        states = self._forward(
                                states,
                                controls,
                                ctx_data,
                                dt=dt,
                            )  # B x 1 x D

        return states

    def rollout(
            self,
            states_input,
            controls,
            ctx_data,
            dt = 0.02,
    ):
        states = states_input.clone().detach()
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