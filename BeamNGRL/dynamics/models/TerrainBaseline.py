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
import time

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
            **kwargs,
    ):

        super().__init__(**kwargs)

        self.dtype = torch.float
        self.d = torch.device('cuda')

        self.state_dim = 10 ## vx, vy, vz, wx, wy, wz, cr, sr, cp, sp, st, th
        self.ctrl_dim = 2
        self.context_dim = 12
        self.input_dim = self.context_dim + self.state_dim + self.ctrl_dim
        self.output_dim = 6 ## dvx/dt, dvy/dt, dvz/dt, dwx/dt, dwy/dt, dwz/dt
        self.dt = dt

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

        self.BEVmap_size = torch.tensor(BEVmap_size, dtype=self.dtype, device=self.d)
        self.BEVmap_res = torch.tensor(BEVmap_res, dtype=self.dtype, device=self.d)
        ## TODO: import map size and res from training data and throw error during forward pass if the numbers don't square tf up (in training, check on every pass, on inference just check outside for loop)

        self.BEVmap_size_px = torch.tensor((self.BEVmap_size/self.BEVmap_res), device=self.d, dtype=torch.int32)
        self.delta = torch.tensor(0.5*patch_size/self.BEVmap_res, device=self.d, dtype=torch.long)
        self.wheelbase = torch.tensor(wheelbase/self.BEVmap_res, device=self.d, dtype=torch.long)
        self.trackwidth = torch.tensor(trackwidth/self.BEVmap_res, device=self.d, dtype=torch.long)
        self.flx = self.delta + self.wheelbase//2 # mid plus half of wheelbase
        self.fly = self.delta + self.trackwidth//2 # mid minus half of trackwidth
        self.brx = self.delta - self.wheelbase//2 # mid minus half of wheelbase
        self.bry = self.delta - self.trackwidth//2 # mid plus half of trackwidth
        self.frx = self.flx
        self.fry = self.bry
        self.blx = self.brx
        self.bly = self.fly

        self.cr_mean = torch.cos(self.mean_state[3])
        self.sr_mean = torch.sin(self.mean_state[3])
        self.cr_std = torch.cos(self.std_state[3])
        self.sr_std = torch.sin(self.std_state[3])
        self.cp_mean = torch.cos(self.mean_state[4])
        self.sp_mean = torch.sin(self.mean_state[4])
        self.cp_std = torch.cos(self.std_state[4])
        self.sp_std = torch.sin(self.std_state[4])

        self.GRAVITY = torch.tensor(9.81, dtype=self.dtype, device=self.d)

        # ================= DEFINE NETWORKS============================

        self.kernel_size = 4
        self.stride = 1
        output_size = int((int((int(((self.delta*2)-self.kernel_size)/self.stride)+1)/2)-self.kernel_size)/self.stride+1)
        conv1 = nn.Conv2d(1, 4, kernel_size=self.kernel_size, stride=self.stride)
        maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        conv2 = nn.Conv2d(4, 4, kernel_size=self.kernel_size, stride=self.stride)
        # use the image size, and formulas for CNN output size to compute the input size for the first FC layer
        fc1 = nn.Linear(output_size*output_size*4, 16)
        fc2 = nn.Linear(16, self.context_dim)

        cnn_layers = [ conv1, nn.Tanh() ]
        cnn_layers += [maxpool]
        cnn_layers += [conv2, nn.Tanh() ]
        cnn_layers += [nn.Flatten()]
        cnn_layers += [ fc1, nn.Tanh() ]
        cnn_layers += [ fc2 ]
        self.CNN = nn.Sequential(*cnn_layers)
        self.CNN = nn.Sequential(*cnn_layers)

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
            bev_input -= bev_input[..., self.delta.item(), self.delta.item()].clone().unsqueeze(-1).unsqueeze(-1)
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
            bev_input -= bev_input[..., self.delta.item(), self.delta.item()].clone().unsqueeze(-1).unsqueeze(-1)

        roll = (torch.atan( ((fl + bl) - (fr + br))/(2*self.trackwidth*self.BEVmap_res))).reshape((k*t))
        pitch = (torch.atan( ((bl + br) - (fl + fr))/(2*self.wheelbase*self.BEVmap_res))).reshape((k*t))

        states_next = states_next.reshape((k*t, n))
        ctrls = ctrls.reshape((k*t, n_c))
        
        vU = torch.zeros((k*t, self.state_dim + self.ctrl_dim), dtype=self.dtype, device=self.d)
        
        vU[..., 0:3] = (states_next[..., 6:9] - self.mean_state[6:9])/self.std_state[6:9]  # vels
        vU[..., 3:6] = (states_next[..., 12:15] - self.mean_state[12:15])/self.std_state[12:15]  # rates
        # terrain derived roll/pitch
        cr = torch.cos(roll)
        sr = torch.sin(roll)
        cp = torch.cos(pitch)
        sp = torch.sin(pitch)
        cy = torch.cos(states_next[..., 5])
        sy = torch.sin(states_next[..., 5])
        ct = torch.sqrt(torch.clamp(1 - sp**2 - sr**2,1e-2,1))
        
        vU[..., 6] = (cr - self.cr_mean)/self.cr_std
        vU[..., 7] = (sr - self.sr_mean)/self.sr_std
        vU[..., 8] = (cp - self.cp_mean)/self.cp_std
        vU[..., 9] = (sp - self.sp_mean)/self.sp_std
        vU[..., 10:12] = (ctrls - self.mean_control)/self.std_control # controls

        bev_input = bev_input.reshape((k*t, self.delta*2, self.delta*2))/self.std_bev

        context = self.CNN(bev_input.unsqueeze(0).transpose(0,1))

        vUc = torch.cat((vU, context), dim=-1)
        dV = self.main(vUc)

        states_next[..., 6:9] = states_next[..., 6:9] + dV[..., 0:3]* self.std_state[9:12] * self.dt

        states_next[..., 12:15] = states_next[..., 12:15] + dV[..., 3:6]* self.std_state[12:15] * self.dt

        with torch.no_grad():
            # this is just to remove non inertial forces (so that we get what the IMU would tell us)
            states_next[..., 9]  = dV[..., 0]*self.std_state[9]  - states_next[..., 7]*states_next[..., 14]
            states_next[..., 10] = dV[..., 1]*self.std_state[10] + states_next[..., 6]*states_next[..., 14]
            states_next[..., 11] = dV[..., 2]*self.std_state[11] + self.GRAVITY*ct

            states_next[..., 3] = roll
            states_next[..., 4] = pitch
            states_next[..., 5] = states_next[..., 5] + self.dt*( states_next[..., 12]*0    + states_next[..., 13]*(sr/cp)           + states_next[..., 14]*(cr/cp)           )

            states_next[..., 0] = states_next[..., 0] + self.dt*( states_next[..., 6]*cp*cy + states_next[..., 7]*(sr*sp*cy - cr*sy) + states_next[..., 8]*(cr*sp*cy + sr*sy) )
            states_next[..., 1] = states_next[..., 1] + self.dt*( states_next[..., 6]*cp*sy + states_next[..., 7]*(sr*sp*sy + cr*cy) + states_next[..., 8]*(cr*sp*sy - sr*cy) )
            states_next[..., 2] = states_next[..., 2] + self.dt*( states_next[..., 6]*(-sp) + states_next[..., 7]*(sr*cp)            + states_next[..., 8]*(cr*cp)            )

        states_next = states_next.reshape((k,t,n))

        return states_next

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
    
    