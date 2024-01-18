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
            batch_norm=False,
            **kwargs,
    ):

        super().__init__(**kwargs)

        self.state_dim = 15
        self.ctrl_dim = 2
        self.context_dim = 6
        self.timesteps = 32
        self.normalized_input_dim = self.context_dim + self.state_dim + self.ctrl_dim
        self.input_dim = self.timesteps * self.normalized_input_dim ## pos, rpy, vel, acc, rot, st, th
        # self.input_dim = self.timesteps*(10 + 6) ## vx, vy, vz, wx, wy, wz, roll, pitch, st, th
        self.output_dim = self.timesteps*self.state_dim ## dvx/dt, dvy/dt, dvz/dt, dwx/dt, dwy/dt, dwz/dt, droll, dpitch

        fc_layers = [
            nn.Linear(self.input_dim, hidden_dim),
            nn.Tanh(),
        ]
        for _ in range(hidden_depth):
            fc_layers += [nn.utils.spectral_norm(nn.Linear(hidden_dim, hidden_dim))]
            fc_layers += [nn.Tanh()]
        fc_layers += [nn.Linear(hidden_dim, self.output_dim)]
        fc_layers += [nn.Tanh()]

        self.main = nn.Sequential(*fc_layers)

        self.dtype = torch.float
        self.d = torch.device('cuda')
        self.BEVmap_size = torch.tensor(64, dtype=self.dtype, device=self.d)
        self.BEVmap_res = torch.tensor(0.25, dtype=self.dtype, device=self.d)
        ## TODO: import map size and res from training data and throw error during forward pass if the numbers don't square tf up (in training, check on every pass, on inference just check outside for loop)

        self.BEVmap_size_px = torch.tensor((self.BEVmap_size/self.BEVmap_res), device=self.d, dtype=torch.int32)
        self.delta = torch.tensor(1.5/self.BEVmap_res, device=self.d, dtype=torch.long)
        self.wheelbase = torch.tensor(2.6/self.BEVmap_res, device=self.d, dtype=torch.long)
        self.trackwidth = torch.tensor(1.5/self.BEVmap_res, device=self.d, dtype=torch.long)

        conv1 = nn.Conv2d(1, 4, kernel_size=4, stride=1)
        maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        conv2 = nn.Conv2d(4, 4, kernel_size=4, stride=2)
        fc1 = nn.Linear(4, 4)
        fc2 = nn.Linear(4, 6)

        cnn_layers = [ conv1, nn.ReLU() ]
        cnn_layers += [maxpool]
        cnn_layers += [conv2, nn.ReLU() ]
        cnn_layers += [nn.Flatten()]
        cnn_layers += [ fc1, nn.ReLU() ]
        cnn_layers += [ fc2 ]
        self.CNN = nn.Sequential(*cnn_layers)
        # TODO: these should be loaded programmatically.
        self.mean_state = torch.tensor([0.0, 0.0 , 0.0, 0.0, 0.0,
                                       0.0, 7.5, 0.0, 0.0, 0.0,
                                       0.0, 9.8, 0.0, 0.0, 0.0]).to("cuda")
        self.std_state = torch.tensor([16.0, 16.0 , 4.0, 0.5, 0.5,
                                       1.0, 8.0, 4.0, 4.0, 12.5,
                                       12.5, 12.5, 1.0, 1.0, 1.0]).to("cuda")
        ## std_state is actually min/max of state around the mean so as to ensure all values we get are [-1,1]
        self.std_state_err = torch.Tensor([1.0, 1.0 , 0.5, 0.1, 0.1,
                0.2, 1.5, 1.5, 0.5, 3.0,
                3.0, 3.0, 1.0, 1.0, 0.3]).to("cuda")
        ## std_state err is the denormalization we will use for residuals
        self.mean_control = torch.tensor([0.0, 0.5]).to("cuda")
        self.std_control = torch.tensor([0.5, 0.2]).to("cuda")
        self.mean_bev = torch.tensor(0.0).to("cuda")
        self.std_bev = torch.tensor(2.0).to("cuda")

        self.GRAVITY = torch.tensor(9.81, dtype=self.dtype, device=self.d)
        self.dt = 1e-3

    def _forward(
            self,
            states: torch.Tensor, # b, L, d
            controls: torch.Tensor,
            ctx_data: Dict,
            dt = 0.1,
    ):
        n = states.shape[-1]
        n_c = controls.shape[-1]
        t = states.shape[-2]
        k = states.shape[-3]

        states_next = states.clone().detach()
        ctrls = controls.clone().detach()
        bev_input = ctx_data['rotate_crop']

        '''
        context data contains BEV hght map --
        I get k bevs of shape k x 1 x bevshape x bevshape
        we don't rotate the image, but we do provide the yaw angle of the vehicle I assume relative to the start?
        '''
        now = time.time()
        states_next = states_next.reshape((k*t, n))
        ctrls = ctrls.reshape((k*t, n_c))


        vU = torch.zeros((k*t, self.state_dim + self.ctrl_dim), dtype=self.dtype, device=self.d)

        vU[...,:15] = (states_next[..., :15] - self.mean_state)/self.std_state
        vU[...,15:] = (ctrls - self.mean_control)/self.std_control
        bev_center = bev_input.clone()
        bev_center[...,:,:] = bev_input[...,6,6].unsqueeze(-1).unsqueeze(-1)
        bev_input = (bev_input[...,:,:] - bev_center)/self.std_bev

        context = self.CNN(bev_input.unsqueeze(0).transpose(0,1))
        vUc = torch.cat((vU, context), dim=-1).reshape(k, t * self.normalized_input_dim)
        
        dV = self.main(vUc).reshape(k*t, self.state_dim)
        states_next[..., 1:, :15] = states_next[..., 1:, :15] + (dV[..., 1:, :] * self.std_state_err * self.timesteps * dt)

        states_next = states_next.reshape((k,t,n))

        return states_next

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