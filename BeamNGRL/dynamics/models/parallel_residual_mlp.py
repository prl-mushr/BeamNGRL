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

        self.state_dim = 11 # dynamics only depends on: r,p,vx,vy,vz,wx,wy,wz,ax,ay,az
        self.ctrl_dim = 2
        self.predict_dim = 12 # V, dV, w
        self.context_dim = self.crop_size ## square root of the number of pixels in the patch -- this is somewhat arbitrary but works as a good approximation.
        self.timesteps = timesteps ## TODO this should be provided during initialization
        self.normalized_input_dim = self.context_dim + self.state_dim + self.ctrl_dim
        self.input_dim = self.timesteps * self.normalized_input_dim
        self.output_dim = self.timesteps* self.predict_dim
        self.GRAVITY = torch.tensor(9.81, dtype=self.dtype, device=self.d)
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
        self.channels = 4
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

        vU[..., :2] = (states_next[..., 3:5]  - self.mean_state[3:5] )/ self.std_state[3:5] ## rp
        vU[..., 2:11] = (states_next[..., 6:15]  - self.mean_state[6:15] )/(self.std_state[6:15]) #v,a,w
        vU[...,11:]  = (ctrls - self.mean_control)/self.std_control

        vU = vU.reshape(k*t, self.state_dim + self.ctrl_dim)

        bev_center = bev_input.clone()
        bev_center[...,:,:] = bev_input[...,self.bev_cent,self.bev_cent].unsqueeze(-1).unsqueeze(-1)
        bev_input = (bev_input - bev_center)/self.std_bev
        now = time.time()
        context = self.CNN(bev_input.unsqueeze(0).transpose(0,1))
        vUc = torch.cat((vU, context), dim=-1).reshape(k, t * self.normalized_input_dim)
        dV = self.main(vUc).reshape(k, t, self.output_dim//t)

        # predict only dV, dW:
        states_next[..., 6:9]   = states_next[...,[0], 6:9]   + self.dt * torch.cumsum(dV[...,0:3], dim=-2) * self.std_state[9:12] 
        states_next[..., 12:15] = states_next[...,[0], 12:15] + self.dt * torch.cumsum(dV[...,3:6], dim=-2) * self.std_state[12:15]
        # states_next[..., 3:5]   = self.mean_state[3:5]   + dV[..., 6:8]  * self.std_state[3:5]
        # states_next[..., 9:12]  = states_next[...,[0],  9:12]  + dV[..., 9:12] * self.std_state[9:12]
        self.execution_dt = time.time() - now
        # cr = torch.cos(states_next[..., 3])
        # sr = torch.sin(states_next[..., 3])
        # cp = torch.cos(states_next[..., 4])
        # sp = torch.sin(states_next[..., 4])

        # states_next[..., 9]  = dV[..., 0]*self.std_state[9]  - (states_next[..., 7]*states_next[..., 14] - states_next[..., 8]*states_next[..., 13] + sp*self.GRAVITY)
        # states_next[..., 10] = dV[..., 1]*self.std_state[10] - (-states_next[..., 6]*states_next[..., 14] + states_next[..., 8]*states_next[..., 12] - sr*cp*self.GRAVITY)
        # states_next[..., 11] = dV[..., 2]*self.std_state[11] - (states_next[..., 6]*states_next[..., 13] - states_next[..., 7]*states_next[..., 12] - cp*cr*self.GRAVITY)


        # with torch.no_grad():
        # cr = torch.cos(states_next[..., 0, 3])
        # sr = torch.sin(states_next[..., 0, 3])
        # cp = torch.cos(states_next[..., 0, 4])
        # sp = torch.sin(states_next[..., 0, 4])
        # cy = torch.cos(states_next[..., 0, 5])
        # sy = torch.sin(states_next[..., 0, 5])
        # for i in range(1, t):
        #     wx = states_next[..., i, 12]
        #     wy = states_next[..., i, 13]
        #     wz = states_next[..., i, 14]
        #     # states_next[..., i, 3] = states_next[..., i-1, 3] + self.dt*( wx*1 + wy*(sr*sp/cp) + wz*(sp*cr/cp) )
        #     # states_next[..., i, 4] = states_next[..., i-1, 4] + self.dt*( wx*0 + wy*cr         + wz*(-sr)      )
        #     states_next[..., i, 5] = states_next[..., i-1, 5] + self.dt*( wx*0 + wy*(sr/cp)    + wz*(cr/cp)    )

        #     cr = torch.cos(states_next[..., i, 3])
        #     sr = torch.sin(states_next[..., i, 3])
        #     cp = torch.cos(states_next[..., i, 4])
        #     sp = torch.sin(states_next[..., i, 4])
        #     cy = torch.cos(states_next[..., i, 5])
        #     sy = torch.sin(states_next[..., i, 5])

        #     states_next[..., i, 9]  = dV[..., i, 0]*self.std_state[9]  - (states_next[..., i, 7]*states_next[..., i, 14] - states_next[..., i, 8]*states_next[..., i, 13] + sp*self.GRAVITY)
        #     states_next[..., i, 10] = dV[..., i, 1]*self.std_state[10] - (-states_next[..., i, 6]*states_next[..., i, 14] + states_next[..., i, 8]*states_next[..., i, 12] - sr*cp*self.GRAVITY)
        #     states_next[..., i, 11] = dV[..., i, 2]*self.std_state[11] - (states_next[..., i, 6]*states_next[..., i, 13] - states_next[..., i, 7]*states_next[..., i, 12] - cp*cr*self.GRAVITY)

        #     # vx = states_next[..., i-1, 6]
        #     # vy = states_next[..., i-1, 7]
        #     # vz = states_next[..., i-1, 8]

        #     # ## using the same rotation matrix, just do G*R where G is [0,0,9.8] and then subtract.
        #     # ## Details in "Vehicle Models and Optimal Control on a Nonplanar Surface"
        #     # ax = states_next[..., i, 9]  + vy*wz - vz*wy + sp*9.8
        #     # ay = states_next[..., i, 10] - vx*wz + vz*wx - sr*cp*9.8
        #     # az = states_next[..., i, 11] + vx*wy - vy*wx - cp*cr*9.8

        #     # states_next[..., i, 6] = states_next[..., i-1, 6] + ax * self.dt;
        #     # states_next[..., i, 7] = states_next[..., i-1, 7] + ay * self.dt;
        #     # states_next[..., i, 8] = states_next[..., i-1, 8] + az * self.dt;
        #     states_next[..., i, 0] = states_next[..., i-1, 0] + self.dt*( states_next[..., i, 6]*cp*cy + states_next[..., i, 7]*(sr*sp*cy - cr*sy) + states_next[..., i, 8]*(cr*sp*cy + sr*sy) )
        #     states_next[..., i, 1] = states_next[..., i-1, 1] + self.dt*( states_next[..., i, 6]*cp*sy + states_next[..., i, 7]*(sr*sp*sy + cr*cy) + states_next[..., i, 8]*(cr*sp*sy - sr*cy) )
        #     states_next[..., i, 2] = states_next[..., i-1, 2] + self.dt*( states_next[..., i, 6]*(-sp) + states_next[..., i, 7]*(sr*cp)            + states_next[..., i, 8]*(cr*cp)            )
                
        
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