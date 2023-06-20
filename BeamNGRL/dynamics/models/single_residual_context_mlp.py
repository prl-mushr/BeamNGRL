import torch
import torch.nn as nn
from torchvision.transforms.functional import rotate
from BeamNGRL.dynamics.models.base import DynamicsBase
from typing import Dict
from BeamNGRL.dynamics.utils.network_utils import get_feat_index_tn
from BeamNGRL.dynamics.utils.network_utils import get_state_features, get_ctrl_features
import time
from BeamNGRL.dynamics.utils.misc_utils import * ## uncomment on eval


class ContextMLP(DynamicsBase):

    def __init__(
            self,
            hidden_depth=2,
            hidden_dim=512,
            # dt=0.1,
            dt=0.02,
            batch_norm=False,
            **kwargs,
    ):

        super().__init__(**kwargs)

        self.dt = dt
        input_dim = 10 + 30 ## vx, vy, vz, wx, wy, wz, roll, pitch, st, th
        output_dim = 6 ## dvx/dt, dvy/dt, dvz/dt, dwx/dt, dwy/dt, dwz/dt

        fc_layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
        ]
        for _ in range(hidden_depth):
            fc_layers += [nn.Linear(hidden_dim, hidden_dim)]
            # if batch_norm:
            #     fc_layers += [nn.BatchNorm1d(hidden_dim)]
            fc_layers += [nn.Tanh()]
        fc_layers += [nn.Linear(hidden_dim, output_dim)]
        fc_layers += [nn.Tanh()]

        self.main = nn.Sequential(*fc_layers)

        self.dtype = torch.float
        self.d = torch.device('cuda')
        self.BEVmap_size = torch.tensor(64, dtype=self.dtype, device=self.d)
        self.BEVmap_res = torch.tensor(0.25, dtype=self.dtype, device=self.d)
        ## TODO: import map size and res from training data and throw error during forward pass if the numbers don't square tf up (in training, check on every pass, on inference just check outside for loop)

        self.BEVmap_size_px = torch.tensor((self.BEVmap_size/self.BEVmap_res), device=self.d, dtype=torch.int32)
        self.delta = torch.tensor(3/self.BEVmap_res, device=self.d, dtype=torch.long)

        conv1 = nn.Conv2d(1, 4, kernel_size=4, stride=1)
        maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        conv2 = nn.Conv2d(4, 4, kernel_size=4, stride=2)
        fc1 = nn.Linear(64, 48)
        fc2 = nn.Linear(48, 30)

        cnn_layers = [ conv1, nn.ReLU() ]
        cnn_layers += [maxpool]
        cnn_layers += [conv2, nn.ReLU() ]
        cnn_layers += [nn.Flatten()]
        cnn_layers += [ fc1, nn.ReLU() ]
        cnn_layers += [ fc2 ]
        self.CNN = nn.Sequential(*cnn_layers)

        self.mean = torch.ones(11, dtype=self.dtype, device=self.d)
        self.std = torch.ones(11, dtype=self.dtype, device=self.d)
        # vx, vy, vz, wx, wy, wz, roll, pitch
        self.mean[0] *= 7.68207746
        self.mean[1] *= 0.0
        self.mean[2] *= 0.0
        self.mean[3] *= 0.0 
        self.mean[4] *= 0.0 
        self.mean[5] *= 0.0
        self.mean[6] *= 0.0 
        self.mean[7] *= 0.0
        #  st, th
        self.mean[8] *= 0.01554329
        self.mean[9] *= 0.54285316
        # bev_elev:
        self.mean[10] *= -0.10409253

        self.std[0] *= 1.40012654
        self.std[1] *= 1.18910109
        self.std[2] *= 0.33882454
        self.std[3] *= 0.32675986
        self.std[4] *= 0.25927919
        self.std[5] *= 0.45646246       
        self.std[6] *= 0.08
        self.std[7] *= 0.08

        self.std[8] *= 0.50912194 
        self.std[9] *= 0.16651182
        # bev elev:
        self.std[10] *= 2.2274804

    def _forward(
            self,
            states: torch.Tensor, # b, L, d
            controls: torch.Tensor,
            ctx_data: Dict,
            evaluation=False,
    ):
        n = states.shape[-1]
        n_c = controls.shape[-1]
        t = states.shape[-2]
        k = states.shape[-3]

        states_next = states.clone().detach()
        ctrls = controls.clone().detach()

        dt = self.dt

        '''
        context data contains BEV hght map --
        I get k bevs of shape k x 1 x bevshape x bevshape
        we don't rotate the image, but we do provide the yaw angle of the vehicle I assume relative to the start?
        '''
        # with torch.no_grad():
        bev = ctx_data['bev_elev']
        if evaluation:
            bev_input = torch.zeros((k, self.delta*2, self.delta*2), dtype=self.dtype, device=self.d) ## a lot of compute time is wasted producing this "empty" array every "timestep"
            center = torch.clamp( ((states_next[..., :2] + self.BEVmap_size*0.5) / self.BEVmap_res).to(dtype=torch.long, device=self.d), 0 + self.delta, self.BEVmap_size_px - 1 - self.delta)
            angle = -states_next[..., 5] ## the map rotates in the opposite direction to the car!
            bev_input = crop_rotate_batch(bev, bev_input, center, angle) ## the order of center coordinates is x,y as opposed to that used in manual cropping which is y,x
        else:
            bev_input = torch.zeros((k, t, self.delta*2, self.delta*2), dtype=self.dtype, device=self.d) ## a lot of compute time is wasted producing this "empty" array every "timestep"
            center = torch.clamp( ((states_next[..., :2] + self.BEVmap_size*0.5) / self.BEVmap_res).to(dtype=torch.long, device=self.d), 0 + self.delta, self.BEVmap_size_px - 1 - self.delta)
            angle = -states_next[..., 5] ## the map rotates in the opposite direction to the car!
            for i in range(k):
                bev_input[i,...] = crop_rotate_batch(bev[i,...], bev_input[i,...], center[i,...], angle[i,...])## the order of center coordinates is x,y as opposed to that used in manual cropping which is y,x


        states_next = states_next.reshape((k*t, n))
        ctrls = ctrls.reshape((k*t, n_c))
        
        vU = torch.zeros((k*t, 10), dtype=self.dtype, device=self.d)
        
        vU[..., 0] = (states_next[..., 6] - self.mean[0])/self.std[0]  # vx
        vU[..., 1] = (states_next[..., 7] - self.mean[1])/self.std[1]  # vy
        vU[..., 2] = (states_next[..., 8] - self.mean[2])/self.std[2]  # vz
        
        vU[..., 3] = (states_next[..., 12] - self.mean[3])/self.std[3]  # wx
        vU[..., 4] = (states_next[..., 13] - self.mean[4])/self.std[4]  # wy
        vU[..., 5] = (states_next[..., 14] - self.mean[5])/self.std[5]  # wz

        vU[..., 6] = (states_next[..., 3] - self.mean[6])/self.std[6] # roll
        vU[..., 7] = (states_next[..., 4] - self.mean[7])/self.std[8] # pitch

        vU[..., 8] = (ctrls[..., 0] - self.mean[8])/self.std[8] # steering
        vU[..., 9] = (ctrls[..., 1] - self.mean[9])/self.std[9] # wheelspeed

        bev_input = (bev_input.reshape((k*t, self.delta*2, self.delta*2)) - self.mean[10])/self.std[10]

        context = self.CNN(bev_input.unsqueeze(0).transpose(0,1))

        vUc = torch.cat((vU, context), dim=-1)

        dV = self.main(vUc)

        states_next[..., 6] = states_next[..., 6] + dV[..., 0]*12.5 * dt
        states_next[..., 7] = states_next[..., 7] + dV[..., 1]*12.5 * dt
        states_next[..., 8] = states_next[..., 8] + dV[..., 2]*12.5 * dt

        states_next[..., 12] = states_next[..., 12] + dV[..., 3]*15 * dt
        states_next[..., 13] = states_next[..., 13] + dV[..., 4]*15 * dt
        states_next[..., 14] = states_next[..., 14] + dV[..., 5]*15 * dt

        states_next[..., 3:6] = states_next[..., 3:6] + states_next[..., 12:15] * dt ## aggregate roll, pitch errors 

        states_next = states_next.reshape((k,t,n))

        return states_next

    def _rollout(
            self,
            states,
            controls,
            ctx_data,
    ):

        horizon = states.shape[-2]
        for i in range(horizon - 1):
            states[..., [i+1], :] = self._forward(
                                    states[..., [i], :],
                                    controls[..., [i], :],
                                    ctx_data,
                                    evaluation=True,
                                )  # B x 1 x D
        return states

    def rollout(
            self,
            states,
            controls,
            ctx_data,
    ):

        x = states[..., 0]
        y = states[..., 1]
        z = states[..., 2]
        roll = states[..., 3]
        pitch = states[..., 4]
        yaw = states[..., 5]
        vx = states[..., 6]
        vy = states[..., 7]
        vz = states[..., 8]
        ax = states[..., 9]
        ay = states[..., 10]
        az = states[..., 11]
        wx = states[..., 12]
        wy = states[..., 13]
        wz = states[..., 14]

        steer = controls[..., 0]
        throttle = controls[..., 1]

        with torch.no_grad():
            states_pred = self._rollout(states, controls, ctx_data)

        _,_,_,roll,pitch,_,vx, vy, vz, ax, ay, az, wx, wy, wz  = states_pred.split(1, dim=-1)

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
        # roll = roll + torch.cumsum(wx*self.dt, dim=-1)
        # pitch = pitch + torch.cumsum(wy*self.dt, dim=-1)
        yaw = yaw + torch.cumsum(wz*self.dt, dim=-1)

        cy = torch.cos(yaw)
        sy = torch.sin(yaw)
        cp = torch.cos(pitch)
        sp = torch.sin(pitch)
        cr = torch.cos(roll)
        sr = torch.sin(roll)
        ct = torch.sqrt(cp*cp + cr*cr)

        x = x + self.dt*torch.cumsum(( vx*cp*cy + vy*(sr*sp*cy - cr*sy) + vz*(cr*sp*cy + sr*sy) ), dim=-1)
        y = y + self.dt*torch.cumsum(( vx*cp*sy + vy*(sr*sp*sy + cr*cy) + vz*(cr*sp*sy - sr*cy) ), dim=-1)
        z = z + self.dt*torch.cumsum(( vx*(-sp) + vy*(sr*cp)            + vz*(cr*cp)            ), dim=-1)

        return torch.stack((x, y, z, roll, pitch, yaw, vx, vy, vz, ax, ay, az, wx, wy, wz, steer, throttle), dim=-1)

