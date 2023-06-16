import torch
import torch.nn as nn
from torchvision.transforms.functional import rotate
from BeamNGRL.dynamics.models.base import DynamicsBase
from typing import Dict
from BeamNGRL.dynamics.utils.network_utils import get_feat_index_tn
from BeamNGRL.dynamics.utils.network_utils import get_state_features, get_ctrl_features
import time
from BeamNGRL.dynamics.utils.misc_utils import *
class ContextMLP(DynamicsBase):

    def __init__(
            self,
            hidden_depth=2,
            hidden_dim=512,
            batch_norm=False,
            **kwargs,
    ):

        super().__init__(**kwargs)

        feat_idx_tn = get_feat_index_tn(self.state_feat_list)

        self.register_buffer('state_feat_idx', feat_idx_tn)

        input_dim = 7 + 30 ## vx, vy, wz, roll, pitch, st, th
        output_dim = 7 ## dvx, dvy, ax, ay, dr, dp, dwz

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
        self.BEVmap_size = torch.tensor(32, dtype=self.dtype, device=self.d)
        self.BEVmap_res = torch.tensor(0.25, dtype=self.dtype, device=self.d)

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

    def _forward(
            self,
            states: torch.Tensor, # b, L, d
            controls: torch.Tensor,
            ctx_data: Dict,
            evaluation=False
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
            center = torch.clamp( ((states_next[..., 0] + self.BEVmap_size*0.5) / self.BEVmap_res).to(dtype=torch.long, device=self.d), 0 + self.delta, self.BEVmap_size_px - 1 - self.delta)
            angle = states_next[..., 5]
            bev_input = crop_rotate_batch(bev, bev_input, center, -angle)
        else:
            bev_input = torch.zeros((k, t, self.delta*2, self.delta*2), dtype=self.dtype, device=self.d)
            c_X = torch.clamp( ((states_next[..., 0] + self.BEVmap_size*0.5) / self.BEVmap_res).to(dtype=torch.long, device=self.d), 0 + self.delta, self.BEVmap_size_px - 1 - self.delta)
            c_Y = torch.clamp( ((states_next[..., 1] + self.BEVmap_size*0.5) / self.BEVmap_res).to(dtype=torch.long, device=self.d), 0 + self.delta, self.BEVmap_size_px - 1 - self.delta)
            y_min = c_Y - self.delta
            y_max = c_Y + self.delta
            x_min = c_X - self.delta
            x_max = c_X + self.delta
            for i in range(k):
                for j in range(t):
                    X = bev[i, 0, y_min[i, j]: y_max[i, j], x_min[i, j]: x_max[i, j]]
                    X = X.unsqueeze(0)
                    X = rotate(X, -states_next[i, j , 5].item()*57.3)
                    bev_input[i, j, :, :] = ( X.squeeze(0) - 0.53 ) / 1.43 ## subtract mean, normalize.


        mean_state = torch.ones_like(states_next[0,0,6:12])
        mean_state[..., 0] *= 9.16903211
        mean_state[..., 1] *= 0.0
        mean_state[..., 2] *= 0.0
        mean_state[..., 3] *= 0.0 ## roll
        mean_state[..., 4] *= 0.0 ## pitch
        mean_state[..., 5] *= 0.0 ## yaw
        std_state = torch.ones_like(mean_state)
        std_state[..., 0] *= 1.40012654
        std_state[..., 1] *= 0.61146516
        std_state[..., 2] *= 0.34990806
        std_state[..., 3] *= 0.08
        std_state[..., 4] *= 0.08
        std_state[..., 5] *= 0.32

        mean_ctrl = torch.ones_like(ctrls[0,0,:])
        mean_ctrl[..., 0] *= -0.011925
        mean_ctrl[..., 1] *= 0.36558446
        std_ctrl = torch.ones_like(mean_ctrl)
        std_ctrl[..., 0] *= 0.32298747
        std_ctrl[..., 1] *= 0.36558446

        states_next = states_next.reshape((k*t, n))
        ctrls = ctrls.reshape((k*t, n_c))
        bev_input = bev_input.reshape((k*t, self.delta*2, self.delta*2))

        vU = torch.zeros((k*t, 7), dtype=self.dtype, device=self.d)

        vU[..., 0] = (states_next[..., 6] - mean_state[..., 0])/std_state[..., 0]
        vU[..., 1] = (states_next[..., 7] - mean_state[..., 1])/std_state[..., 1]
        vU[..., 2] = (states_next[..., 14] - mean_state[..., 2])/std_state[..., 2]
        
        vU[..., 3] = (states_next[..., 3] - mean_state[..., 3])/std_state[..., 3]
        vU[..., 4] = (states_next[..., 4] - mean_state[..., 4])/std_state[..., 4]
        # vU[..., 5] = (states_next[..., 5] - mean_state[..., 5])/std_state[..., 5]

        vU[..., 5] = (ctrls[..., 0] - mean_ctrl[..., 0])/std_ctrl[..., 0]
        vU[..., 6] = (ctrls[..., 1] - mean_ctrl[..., 1])/std_ctrl[..., 1]

        context = self.CNN(bev_input.unsqueeze(0).transpose(0,1))

        vUc = torch.concatenate((vU, context), dim=-1)

        dV = self.main(vUc)

        states_next[..., 6] = states_next[..., 6] + dV[..., 0]*0.25*5
        states_next[..., 7] = states_next[..., 7] + dV[..., 1]*0.25*5
        states_next[..., 14] = states_next[..., 14] + dV[..., 2]*0.2*5
        states_next[..., 9] = dV[..., 5]*10.0
        states_next[..., 10] = dV[..., 6]*10.0
        states_next[..., 3] = states_next[..., 3] + dV[..., 3]*0.01*5
        states_next[..., 4] = states_next[..., 4] + dV[..., 4]*0.01*5

        states_next = states_next.reshape((k,t,n))

        return states_next

    def _rollout(
            self,
            states,
            controls,
            ctx_data,
    ):
        '''
        so let me get this straight. We have a dynamics class that has a "forward" method,
        which internally calls a dynamics model that has a rollout method
        which internall calls a "forward" method, that internall calls the "main" on a sequential NN.
        The inception is strong with this one.
        '''
        horizon = states.shape[-2]
        for i in range(horizon - 1):
            states[0, :, [i+1], :] = self._forward(
                                    states[0, :, [i], :],
                                    controls[0, :, [i], :],
                                    ctx_data,
                                    evaluation=True
                                )  # B x 1 x D
        return states