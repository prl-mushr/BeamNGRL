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

    def kinematic(x,u):


    def _forward(
            self,
            states: torch.Tensor, # b, L, d
            controls: torch.Tensor,
            ctx_data: Dict,
    ):

        b, h, _ = states.shape

        states = states.view(-1, self.state_dim)
        controls = controls.view(-1, self.ctrl_dim)

        ['x', 'y', 'th', 'vx', 'wy', 'ay', 'az', 'st', 'th']

        K = torch.tan(controls[..., 0] * 0.5) / 2.6
        
        states[..., 3] = controls[..., 1] * 20.0

        dS = states[..., 3] * 0.025

        wz = states[..., 3] * K

        states[..., 2] += 0.025 * wz # this is what the yaw will become
        
        cy = torch.cos(states[..., 2])
        sy = torch.sin(states[..., 2])

        states[..., 0] += dS * cy
        states[..., 1] += dS * sy

        states[..., 5] = (states[..., 3] * wz) ## this is the Y acceleration in the inertial frame as would be reported by an accelerometer
        states[..., 6] = (-states[..., 3] * wy) + self.GRAVITY ## this is the Z acc in the inertial frame as reported by an IMU

        x = torch.cat((states, controls), dim=-1)
        states_next = states + self.main(x) ## learn the residue

        states_next = states_next.reshape(b, h, -1)
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
