import torch
import torch.nn as nn
from BeamNGRL.dynamics.models.base import DynamicsBase
from typing import Dict
from BeamNGRL.dynamics.utils.network_utils import get_feat_index_tn
from BeamNGRL.dynamics.utils.network_utils import get_state_features, get_ctrl_features
from .normalizers import FeatureNormalizer


class BaselineMLP(DynamicsBase):

    def __init__(
            self,
            hidden_depth=2,
            hidden_dim=512,
            batch_norm=False,
            dt=0.02,
            past_len=None,
            input_stats: Dict=None,
            use_normalizer=False,
            activation='Tanh',
            **kwargs,

    ):

        super().__init__(**kwargs)

        self.dt = dt
        self.past_len = past_len

        input_dim = (self.state_input_dim + self.ctrl_dim) * (past_len + 1)
        output_dim = self.state_output_dim

        self.normalizer = None
        if use_normalizer:
            self.normalizer = FeatureNormalizer(
                self.state_input_feat_list,
                self.state_output_feat_list,
                self.ctrl_feat_list,
                input_stats,
            )

        if activation == 'Tanh':
            activation_fn = nn.Tanh
        elif activation == 'ReLU':
            activation_fn = nn.ReLU
        else:
            raise NotImplementedError

        fc_layers = [
            nn.Linear(input_dim, hidden_dim),
            activation_fn(),
        ]

        for _ in range(hidden_depth):
            fc_layers += [nn.Linear(hidden_dim, hidden_dim)]
            if batch_norm:
                fc_layers += [nn.BatchNorm1d(hidden_dim)]
            fc_layers += [activation_fn()]
        fc_layers += [nn.Linear(hidden_dim, output_dim)]
        fc_layers += [activation_fn()]

        self.main = nn.Sequential(*fc_layers)

    def _step(self, curr_state, curr_ctrl, past_states, past_ctrls, ctx_data=None):

        b_size = curr_state.size(0)

        # Combine state, ctrl inputs
        state_inputs = torch.cat((past_states, curr_state), dim=1)
        ctrl_inputs = torch.cat((past_ctrls, curr_ctrl), dim=1)

        input_feat = torch.cat((state_inputs, ctrl_inputs), dim=2) # B x L x (s_dim + c_dim)
        input_feat = input_feat.flatten(1)

        # Prevent BTT for now
        input_feat = input_feat.detach()

        # Delta
        delta_state = self.main(input_feat)
        delta_state = delta_state.reshape(b_size, 1, self.state_output_dim)

        # Update state
        next_state = curr_state + delta_state

        return next_state

    def _forward(
            self,
            future_states: torch.Tensor, # b, L, d
            future_ctrls: torch.Tensor,
            ctx_data: Dict,
    ):

        b, horizon, _ = future_ctrls.shape
        horizon += 1 # include current ctrl input

        # Append past states, ctrls
        curr_state = ctx_data['state'].view(b, 1, -1)
        curr_ctrl = ctx_data['control'].view(b, 1, -1)
        past_states = ctx_data['past_states'][:, -self.past_len:]
        past_ctrls = ctx_data['past_ctrls'][:, -self.past_len:]

        full_input_states = torch.cat((past_states, curr_state, future_states), dim=1)
        full_input_ctrls = torch.cat((past_ctrls, curr_ctrl, future_ctrls), dim=1)

        bev_state = ctx_data['state']

        # Get input features (w/ normalization if specified)
        state_seq, ctrl_seq = self.process_input(full_input_states, full_input_ctrls)

        print(f'\nPred horizon: {horizon}')
        # Sliding window across full-length trajectory
        pred_states = []
        for t in range(self.past_len, self.past_len + horizon):
            curr_state = state_seq[:, [t]]
            past_states = state_seq[:, t - self.past_len: t]

            curr_ctrl = ctrl_seq[:, [t]]
            past_ctrls = ctrl_seq[:, t - self.past_len: t]

            next_state = self._step(
                curr_state,
                curr_ctrl,
                past_states,
                past_ctrls,
                ctx_data,
            )  # B x 1 x D

            # Unnormalize
            # next_state_feat = self.process_output(next_state)

            pred_states += [next_state]
            if t < self.past_len + horizon - 1:
                state_seq[:, [t+1]] = next_state.clone()

        pred_states = torch.cat(pred_states, dim=1) # B x horizon x s_dim
        print(f'\npred_states shape {pred_states.shape}')
        return pred_states

    def _rollout(
            self,
            states,
            controls,
            ctx_data,
    ):

        states_shape_orig = states.shape
        horizon = states.shape[-2]
        d = states.shape[-1]
        d_c = controls.shape[-1]

        states = states.view(-1, horizon, d) # current state, repeated over horizon
        controls = controls.view(-1, horizon, d_c)

        # Omit current state/ctrl for rollout
        future_states = states[:, 1:].clone()
        future_ctrls = controls[:, 1:].clone()

        # Predict horizon
        rollout_states = self._forward(future_states, future_ctrls, ctx_data)

        # Unnormalize
        rollout_states = self.process_output(rollout_states)

        # Assign preds. state features to full state tensor
        v_t = rollout_states[..., :3]
        w_t = rollout_states[..., 3:6]

        future_states[..., 6:9] = v_t
        future_states[..., 12:15] = w_t

        pred_states = torch.cat((states[:, [0], :], future_states), dim=1)
        return pred_states.reshape(states_shape_orig)

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

