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

        input_dim = self.state_dim + self.ctrl_dim
        output_dim = self.state_dim

        fc_layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        ]
        for _ in range(hidden_depth):
            fc_layers += [nn.Linear(hidden_dim, hidden_dim)]
            if batch_norm:
                fc_layers += [nn.BatchNorm1d(hidden_dim)]
            fc_layers += [nn.ReLU()]
        fc_layers += [nn.Linear(hidden_dim, output_dim)]
        fc_layers += [nn.ReLU()]

        self.main = nn.Sequential(*fc_layers)

    def _forward(
            self,
            states: torch.Tensor, # b, L, d
            controls: torch.Tensor,
            ctx_data: Dict,
    ):

        _, h, _ = states.shape
        if self.normalizer:
            states[...,:15], controls = self.normalizer(states[...,:15], controls)

        states_next = torch.zeros_like(states)
        states_next[...,0,:] = states[...,0,:]
        for i in range(h - 1):
            vawU = torch.cat((states_next[..., i,6:15], controls[..., i,:]), dim=-1)
            states_next[..., i+1, 6:15] = self.main(vawU) # vaw_next = f(v,a,w,U)
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
        b, n, horizon, d = states.shape
        _, _, _,       d_c = controls.shape

        states = states.view(-1, horizon, d)
        controls = controls.view(-1, horizon, d_c)

        pred_states = self._forward(
            states,
            controls,
            ctx_data,
        )  # B x T x D
        pred_states = pred_states.reshape(b, n, horizon, d)

        return pred_states[...,:15]

