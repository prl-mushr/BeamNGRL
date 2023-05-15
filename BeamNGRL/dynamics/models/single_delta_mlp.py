import torch
import torch.nn as nn
from BeamNGRL.dynamics.models.base import DynamicsBase
from typing import Dict
from BeamNGRL.dynamics.utils.network_utils import get_feat_index_tn


class DeltaMLP(DynamicsBase):

    def __init__(
            self,
            hidden_depth=2,
            hidden_dim=512,
            batch_norm=False,
            dt=None,
            **kwargs,
    ):

        self.dt = dt

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

        b, h, _ = states.shape

        # Get input features
        state_feats, ctrl_feats = self.process_input(states, controls)
        state_feats = state_feats.view(-1, self.state_dim)
        ctrl_feats = ctrl_feats.view(-1, self.ctrl_dim)

        x = torch.cat((state_feats, ctrl_feats), dim=-1)

        x_out = self.main(x)

        x_out = x_out.reshape(b, h, -1)

        # Populate delta tensor
        delta_states = torch.zeros_like(states)
        delta_states = delta_states.scatter_(-1, self.state_feat_idx, x_out)

        states_next = states + delta_states

        return states_next


    def _rollout(
            self,
            states,
            controls,
            ctx_data,
    ):

        b, n, horizon, d = states.shape
        b, n, horizon, d_c = controls.shape

        states = states.view(-1, horizon, d)
        controls = controls.view(-1, horizon, d_c)

        horizon = controls.size(1)

        for t in range(horizon - 1):
            next_state = self._forward(
                states[:, [t]],
                controls[:, [t]],
                ctx_data,
            )  # B x 1 x D

            next_state = next_state.detach()
            states[:, [t+1]] = next_state

        pred_states = states
        pred_states = pred_states.reshape(b, n, horizon, d)

        return pred_states

