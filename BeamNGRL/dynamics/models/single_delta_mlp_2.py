import torch
import torch.nn as nn
from BeamNGRL.dynamics.models.base import DynamicsBase
from typing import Dict
from BeamNGRL.dynamics.utils.network_utils import get_feat_index_tn
from BeamNGRL.dynamics.utils.network_utils import get_state_features, get_ctrl_features
from .normalizers import FeatureNormalizer


class DeltaMLP2(DynamicsBase):

    def __init__(
            self,
            hidden_depth=2,
            hidden_dim=512,
            batch_norm=False,
            dt=0.02,
            input_stats: Dict=None,
            use_normalizer=True,
            **kwargs,
    ):

        super().__init__(**kwargs)

        self.dt = dt

        input_dim = self.state_input_dim + self.ctrl_dim
        output_dim = self.state_output_dim

        self.normalizer = None
        if use_normalizer:
            self.normalizer = FeatureNormalizer(
                self.state_input_feat_list,
                self.state_output_feat_list,
                self.ctrl_feat_list,
                input_stats,
            )

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

        x = torch.cat((state_feats.view(-1, self.state_input_dim),
                       ctrl_feats.view(-1, self.ctrl_dim)),
                      dim=-1)

        x_out = self.main(x)

        delta_state_feat = x_out.reshape(b, h, self.state_output_dim)

        state_feats_next = state_feats.clone().detach()

        state_feats_next += delta_state_feat

        return state_feats_next

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
            next_state_feat = self._forward(
                states[:, [t]],
                controls[:, [t]],
                ctx_data,
            )  # B x 1 x D

            # Unnormalize
            next_state_feat = self.process_output(next_state_feat)

            v_t = next_state_feat[..., :3]
            w_t = next_state_feat[..., 3:6]

            states[:, [t+1], 6:9] = v_t
            states[:, [t+1], 12:15] = w_t

        pred_states = states
        pred_states = pred_states.reshape(b, n, horizon, d)

        return pred_states
    #
    # def process_targets(self, states: torch.Tensor):
    #     state_feats = get_state_features(states, self.state_feat_list)
    #     if self.normalizer:
    #         state_feats = self.normalizer.normalize_state(state_feats)
    #     return state_feats
    #
    # def process_input(self, states: torch.Tensor, controls: torch.Tensor):
    #     state_feats = get_state_features(states, self.state_feat_list)
    #     ctrl_feats = get_ctrl_features(controls, self.ctrl_feat_list)
    #     if self.normalizer:
    #         state_feats, ctrl_feats = self.normalizer(state_feats, ctrl_feats)
    #     return state_feats, ctrl_feats
    #
    # def process_output(self, state_feats: torch.Tensor):
    #     if self.normalizer:
    #         state_feats = self.normalizer.unnormalize_state(state_feats)
    #     return state_feats
