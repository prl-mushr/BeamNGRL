import torch
import torch.nn as nn
from torchvision.transforms.functional import rotate
from BeamNGRL.dynamics.models.base import DynamicsBase
from typing import Dict
from BeamNGRL.dynamics.utils.network_utils import get_feat_index_tn
from BeamNGRL.dynamics.utils.network_utils import get_state_features, get_ctrl_features
import time
from BeamNGRL.dynamics.utils.misc_utils import * ## uncomment on eval
import cv2

class ContextMLP(DynamicsBase):

    def __init__(
            self,
            hidden_depth=2,
            hidden_dim=512,
            batch_norm=False,
            **kwargs,
    ):

        super().__init__(**kwargs)

        input_dim = 10 + 6 ## vx, vy, vz, wx, wy, wz, roll, pitch, st, th
        output_dim = 8 ## dvx/dt, dvy/dt, dvz/dt, dwx/dt, dwy/dt, dwz/dt, droll, dpitch

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
        self.delta = torch.tensor(1.5/self.BEVmap_res, device=self.d, dtype=torch.long)
        self.wheelbase = torch.tensor(2.6/self.BEVmap_res, device=self.d, dtype=torch.long)
        self.trackwidth = torch.tensor(1.5/self.BEVmap_res, device=self.d, dtype=torch.long)
        self.flx = self.delta + self.wheelbase//2 # mid plus half of wheelbase
        self.fly = self.delta + self.trackwidth//2 # mid minus half of trackwidth
        self.brx = self.delta - self.wheelbase//2 # mid minus half of wheelbase
        self.bry = self.delta - self.trackwidth//2 # mid plus half of trackwidth
        self.frx = self.flx
        self.fry = self.bry
        self.blx = self.brx
        self.bly = self.fly

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
        self.std[6] *= torch.sin(torch.tensor(0.5))
        self.std[7] *= torch.sin(torch.tensor(0.5))

        self.std[8] *= 0.50912194 
        self.std[9] *= 0.16651182
        # bev elev:
        self.std[10] *= 2.2274804
        self.GRAVITY = torch.tensor(9.81, dtype=self.dtype, device=self.d)

    def _forward(
            self,
            states: torch.Tensor, # b, L, d
            controls: torch.Tensor,
            ctx_data: Dict,
            evaluation=False,
            dt = 0.1,
            count=0,
    ):
        n = states.shape[-1]
        n_c = controls.shape[-1]
        t = states.shape[-2]
        k = states.shape[-3]

        states_next = states.clone().detach()
        ctrls = controls.clone().detach()

        '''
        context data contains BEV hght map --
        I get k bevs of shape k x 1 x bevshape x bevshape
        we don't rotate the image, but we do provide the yaw angle of the vehicle I assume relative to the start?
        '''
        bev = ctx_data['bev_elev']
        if evaluation:
            bev_input = torch.zeros((k, self.delta*2, self.delta*2), dtype=self.dtype, device=self.d) ## a lot of compute time is wasted producing this "empty" array every "timestep"
            center = torch.clamp( ((states_next[..., :2] + self.BEVmap_size*0.5) / self.BEVmap_res).to(dtype=torch.long, device=self.d), 0 + self.delta, self.BEVmap_size_px - 1 - self.delta)
            angle = states_next[..., 5]
            bev_input = crop_rotate_batch(bev, self.delta.item()*2, self.delta.item()*2, center, angle) ## the order of center coordinates is x,y as opposed to that used in manual cropping which is y,x
            fl = torch.zeros(k, dtype=self.dtype, device=self.d)
            fr = torch.zeros(k, dtype=self.dtype, device=self.d)
            bl = torch.zeros(k, dtype=self.dtype, device=self.d)
            br = torch.zeros(k, dtype=self.dtype, device=self.d)
            fl = bev_input[ :, self.fly, self.flx]
            fr = bev_input[ :, self.fry, self.frx]
            bl = bev_input[ :, self.bly, self.blx]
            br = bev_input[ :, self.bry, self.brx]
        else:
            bev_input = torch.zeros((k, t, self.delta*2, self.delta*2), dtype=self.dtype, device=self.d) ## a lot of compute time is wasted producing this "empty" array every "timestep"
            center = torch.clamp( ((states_next[..., :2] + self.BEVmap_size*0.5) / self.BEVmap_res).to(dtype=torch.long, device=self.d), 0 + self.delta, self.BEVmap_size_px - 1 - self.delta)
            angle = states_next[..., 5] ## the map rotates in the opposite direction to the car!
            fl = torch.zeros((k,t), dtype=self.dtype, device=self.d)
            fr = torch.zeros((k,t), dtype=self.dtype, device=self.d)
            bl = torch.zeros((k,t), dtype=self.dtype, device=self.d)
            br = torch.zeros((k,t), dtype=self.dtype, device=self.d)
            for i in range(k):
                bev_input[i,...] = crop_rotate_batch(bev[i,...], self.delta.item()*2, self.delta.item()*2, center[i,...], angle[i,...])## the order of center coordinates is x,y as opposed to that used in manual cropping which is y,x
                fl[i, :] = bev_input[i, :, self.fly, self.flx]
                fr[i, :] = bev_input[i, :, self.fry, self.frx]
                bl[i, :] = bev_input[i, :, self.bly, self.blx]
                br[i, :] = bev_input[i, :, self.bry, self.brx]

        roll = (torch.atan( ((fl + bl) - (fr + br))/(2*self.trackwidth*self.BEVmap_res))).reshape((k*t))
        pitch = (torch.atan( ((bl + br) - (fl + fr))/(2*self.wheelbase*self.BEVmap_res))).reshape((k*t))

        states_next = states_next.reshape((k*t, n))
        ctrls = ctrls.reshape((k*t, n_c))
        
        vU = torch.zeros((k*t, 10), dtype=self.dtype, device=self.d)
        
        vU[..., 0] = (states_next[..., 6] - self.mean[0])/self.std[0]  # vx
        vU[..., 1] = (states_next[..., 7] - self.mean[1])/self.std[1]  # vy
        vU[..., 2] = (states_next[..., 8] - self.mean[2])/self.std[2]  # vz
        
        vU[..., 3] = (states_next[..., 12] - self.mean[3])/self.std[3]  # wx
        vU[..., 4] = (states_next[..., 13] - self.mean[4])/self.std[4]  # wy
        vU[..., 5] = (states_next[..., 14] - self.mean[5])/self.std[5]  # wz

        # terrain derived roll/pitch
        vU[..., 6] = torch.sin(roll - self.mean[6])/self.std[6] # roll
        vU[..., 7] = torch.sin(pitch - self.mean[7])/self.std[7] # pitch

        vU[..., 8] = (ctrls[..., 0] - self.mean[8])/self.std[8] # steering
        vU[..., 9] = (ctrls[..., 1] - self.mean[9])/self.std[9] # wheelspeed

        bev_input = (bev_input.reshape((k*t, self.delta*2, self.delta*2)) - self.mean[10])/self.std[10]

        context = self.CNN(bev_input.unsqueeze(0).transpose(0,1))

        vUc = torch.cat((vU, context), dim=-1)

        dV = self.main(vUc)

        states_next[..., 6] = states_next[..., 6] + dV[..., 0]*12.5 * dt
        states_next[..., 7] = states_next[..., 7] + dV[..., 1]*12.5 * dt
        states_next[..., 8] = states_next[..., 8] + dV[..., 2]*12.5 * dt

        states_next[..., 12] = states_next[..., 12] + dV[..., 3]* 6.0 * dt
        states_next[..., 13] = states_next[..., 13] + dV[..., 4]* 6.0 * dt
        states_next[..., 14] = states_next[..., 14] + dV[..., 5]* 2.0 * dt

        # learn the residual for roll and pitch
        states_next[..., 3] = roll + dV[..., 6]/self.std[6]
        states_next[..., 4] = pitch + dV[..., 7]/self.std[7]
        states_next[..., 3:6] = states_next[..., 3:6] + states_next[..., 12:15] * dt

        with torch.no_grad():
            cr = torch.cos(states_next[..., 3])
            sr = torch.sin(states_next[..., 3])
            cp = torch.cos(states_next[..., 4])
            sp = torch.sin(states_next[..., 4])
            cy = torch.cos(states_next[..., 5])
            sy = torch.sin(states_next[..., 5])
            ct = torch.sqrt(torch.clamp(1 - sp**2 - sr**2,0,1))

            states_next[..., 9] = dV[..., 0]*12.5  - states_next[..., 7]*states_next[..., 14]
            states_next[..., 10] = dV[..., 1]*12.5 + states_next[..., 6]*states_next[..., 14]
            states_next[..., 11] = dV[..., 2]*12.5 + self.GRAVITY*ct 
            states_next[..., 0] = states_next[..., 0] + dt*( states_next[..., 6]*cp*cy + states_next[..., 7]*(sr*sp*cy - cr*sy) + states_next[..., 8]*(cr*sp*cy + sr*sy) )
            states_next[..., 1] = states_next[..., 1] + dt*( states_next[..., 6]*cp*sy + states_next[..., 7]*(sr*sp*sy + cr*cy) + states_next[..., 8]*(cr*sp*sy - sr*cy) )
            states_next[..., 2] = states_next[..., 2] + dt*( states_next[..., 6]*(-sp) + states_next[..., 7]*(sr*cp)            + states_next[..., 8]*(cr*cp)            )

        states_next = states_next.reshape((k,t,n))

        return states_next

    def _rollout(
            self,
            states,
            controls,
            ctx_data,
            dt=0.1,
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
            dt = 0.02
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
    
    