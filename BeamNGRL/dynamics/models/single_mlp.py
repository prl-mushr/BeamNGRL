import torch
import torch.nn as nn
from BeamNGRL.dynamics.models.base import DynamicsBase
from typing import Dict


class BasicMLP(DynamicsBase):

    def __init__(
            self,
            hidden_depth=2,
            hidden_dim=512,
            batch_norm=False,
            **kwargs,
    ):
        super().__init__(**kwargs)

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

        states = states.view(-1, self.state_dim)
        controls = controls.view(-1, self.ctrl_dim)

        x = torch.cat((states, controls), dim=-1)
        states_next = self.main(x)

        states_next = states_next.reshape(b, h, -1)
        return states_next

    def _rollout(
            self,
            state,
            input_ctrl_seq,
            ctx_data,
    ):

        horizon = input_ctrl_seq.size(1)

        # Initialize input state sequence
        b = state.size(0)
        device = state.device
        input_state_seq = torch.zeros((b, horizon + 1, self.state_dim)).to(device)
        input_state_seq[:, 0] = state

        for t in range(horizon):
            next_state = self._forward(
                input_state_seq[:, [t]],
                input_ctrl_seq[:, [t]],
                ctx_data,
            )  # B x 1 x D
            next_state = next_state.detach()
            input_state_seq[:, t+1] = next_state

        pred_seq = input_state_seq[:, 1:]   # next-state predictions only
        return pred_seq
