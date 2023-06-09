import torch
import torch.nn as nn
from BeamNGRL.dynamics.models.base import DynamicsBase
from typing import Dict
from BeamNGRL.dynamics.utils.network_utils import get_feat_index_tn
from BeamNGRL.dynamics.utils.network_utils import get_state_features, get_ctrl_features

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

        input_dim = 7 ## vx, vy, wz, roll, pitch, st, th
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
        so let me get this straight,
        you're getting k x t samples
        for each "k" you have t samples
        for each "t" sample you need to crop-rotate the image (prep)
        then each image is passed into the network as context.
        '''
        if not evaluation:
            bev = ctx_data['bev_elev']
            bev_k = torch.zeros((k, t, bev.shape[0], bev.shape[0]))
            c_X = torch.clamp( ((states_next[..., 0] + self.BEVmap_size*0.5) / self.BEVmap_res).to(dtype=torch.long, device=self.d), 0, self.BEVmap_size_px - 1)
            c_Y = torch.clamp( ((states_next[..., 1] + self.BEVmap_size*0.5) / self.BEVmap_res).to(dtype=torch.long, device=self.d), 0, self.BEVmap_size_px - 1)
            y_min = c_Y - self.delta
            y_max = c_Y + self.delta
            x_min = c_X - self.delta
            x_max = c_X + self.delta
            print(x_min, x_max, y_min, y_max)
            for i in range(k):
                for j in range(t):
                    bev_new = bev[i, 0, x_min[i, j]: x_max[i, j], y_min[i, j]: y_max[i, j]]

            # print(bev_k.shape)


        mean_state = torch.ones_like(states_next[0,0,6:11])
        mean_state[..., 0] *= 4.18
        mean_state[..., 1] *= 0.0
        mean_state[..., 2] *= 0.0
        mean_state[..., 3] *= 0.0 ## roll
        mean_state[..., 4] *= 0.0 ## pitch
        std_state = torch.ones_like(mean_state)
        std_state[..., 0] *= 1.26483479
        std_state[..., 1] *= 0.28369839
        std_state[..., 2] *= 0.25
        std_state[..., 3] *= 0.08
        std_state[..., 4] *= 0.08

        mean_ctrl = torch.ones_like(ctrls[0,0,:])
        mean_ctrl[..., 0] *= 0.01391617 
        mean_ctrl[..., 1] *= 0.16625684
        std_ctrl = torch.ones_like(mean_ctrl)
        std_ctrl[..., 0] *= 0.40226127
        std_ctrl[..., 1] *= 0.58021804

        states_next = states_next.reshape((k*t, n))
        ctrls = ctrls.reshape((k*t, n_c))

        vx = (states_next[..., 6] - mean_state[..., 0])/std_state[..., 0]
        vy = (states_next[..., 7] - mean_state[..., 1])/std_state[..., 1]
        wz = (states_next[..., 14] - mean_state[..., 2])/std_state[..., 2]
        wz = (states_next[..., 14] - mean_state[..., 2])/std_state[..., 2]
        
        rl = (states_next[..., 3] - mean_state[..., 3])/std_state[..., 3]
        pt = (states_next[..., 4] - mean_state[..., 4])/std_state[..., 4]

        st = (ctrls[..., 0] - mean_ctrl[..., 0])/std_ctrl[..., 0]
        th = (ctrls[..., 1] - mean_ctrl[..., 1])/std_ctrl[..., 1]
        vU = torch.stack((vx, vy, wz, rl, pt, st, th), dim=-1)

        dV = self.main(vU)
        states_next[..., 6] = states_next[..., 6] + dV[..., 0]*0.25
        states_next[..., 7] = states_next[..., 7] + dV[..., 1]*0.25
        states_next[..., 14] = states_next[..., 14] + dV[..., 2]*0.2
        states_next[..., 9] = dV[..., 5]*10.0
        states_next[..., 10] = dV[..., 6]*10.0
        states_next[..., 3] = states_next[..., 3] + dV[..., 3]*0.01
        states_next[..., 4] = states_next[..., 4] + dV[..., 4]*0.01

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
                                )  # B x 1 x D
        return states