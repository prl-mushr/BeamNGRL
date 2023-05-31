import torch
import torch.nn as nn
from BeamNGRL.dynamics.models.base import DynamicsBase
from typing import Dict
from BeamNGRL.dynamics.utils.network_utils import get_feat_index_tn
from BeamNGRL.dynamics.utils.network_utils import get_state_features, get_ctrl_features


class ResidualMLP(DynamicsBase):

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

        input_dim = 5 ## vx, vy, wz , st, th
        output_dim = 5 ## dvx, dvy, ax, ay, dwz

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

    def _forward(
            self,
            states: torch.Tensor, # b, L, d
            controls: torch.Tensor,
            ctx_data: Dict,
    ):
        t = states.shape[-2]
        states_next = states.clone().detach()
        ctrls = controls.clone().detach()
        mean_vel = torch.ones_like(states_next[0,0,6:9])
        mean_vel[..., 0] *= 4.18
        mean_vel[..., 1] *= 0.0
        mean_vel[..., 2] *= 0.0
        std_vel = torch.ones_like(mean_vel)
        std_vel[..., 0] *= 1.26483479
        std_vel[..., 1] *= 0.28369839
        std_vel[..., 2] *= 0.25
        mean_ctrl = torch.ones_like(ctrls[0,0,:])
        mean_ctrl[..., 0] *= 0.01391617 
        mean_ctrl[..., 1] *= 0.16625684
        std_ctrl = torch.ones_like(mean_ctrl)
        std_ctrl[..., 0] *= 0.40226127
        std_ctrl[..., 1] *= 0.58021804


        for i in range(t - 1):
            vx = (states_next[...,i, 6] - mean_vel[..., 0])/std_vel[..., 0]
            vy = (states_next[...,i, 7] - mean_vel[..., 1])/std_vel[..., 1]
            wz = (states_next[...,i, 14] - mean_vel[..., 2])/std_vel[..., 2]
            st = (ctrls[..., i, 0] - mean_ctrl[..., 0])/std_ctrl[..., 0]
            th = (ctrls[..., i, 1] - mean_ctrl[..., 1])/std_ctrl[..., 1]
            vU = torch.stack((vx, vy, wz, st, th), dim=-1)

            dV = self.main(vU)
            states_next[..., i+1, 6] = states_next[..., i, 6] + dV[..., 0]*0.25
            states_next[..., i+1, 7] = states_next[..., i, 7] + dV[..., 1]*0.25
            states_next[..., i+1, 14] = states_next[..., i, 14] + dV[..., 2]*0.2
            states_next[..., i+1, 9] = dV[..., 0]*10.0
            states_next[..., i+1, 10] = dV[..., 1]*10.0

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
        # for t in range(horizon - 1):
        states = self._forward(
                                states[0,...],
                                controls[0,...],
                                ctx_data,
                            )  # B x 1 x D
        return states.unsqueeze(0)

