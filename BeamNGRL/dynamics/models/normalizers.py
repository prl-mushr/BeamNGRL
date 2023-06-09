import torch
import torch.nn as nn
from typing import Dict, List
from BeamNGRL.dynamics.utils.network_utils import get_state_features, get_ctrl_features


class FeatureNormalizer(nn.Module):

    def __init__(
            self,
            state_input_feat: List,
            state_output_feat: List,
            ctrl_feat: List,
            input_stats: Dict = None,
    ):
        super().__init__()

        self.state_input_feat = state_input_feat
        self.state_output_feat = state_output_feat
        self.ctrl_feats = ctrl_feat

        self.register_buffer('state_input_mean', torch.zeros(len(state_input_feat)))
        self.register_buffer('state_input_std', torch.ones(len(state_input_feat)))
        self.register_buffer('state_output_mean', torch.zeros(len(state_output_feat)))
        self.register_buffer('state_output_std', torch.ones(len(state_output_feat)))
        self.register_buffer('ctrl_mean', torch.zeros(len(ctrl_feat)))
        self.register_buffer('ctrl_std', torch.ones(len(ctrl_feat)))

        if input_stats is not None:
            (self.state_input_mean,
             self.state_input_std,
             self.ctrl_mean,
             self.ctrl_std) = self.get_stats(input_stats, state_input_feat, ctrl_feat)

            # (self.state_output_mean,
            #  self.state_output_std, _, _) = self.get_stats(input_stats, state_output_feat)

    def get_stats(self, stats: Dict, state_feats: List, ctrl_feats=None):
        state_mean = get_state_features(stats['mean:state'], state_feats)
        state_std = get_state_features(stats['std:state'], state_feats)

        ctrl_mean, ctrl_std = None, None
        if ctrl_feats:
            ctrl_mean = get_ctrl_features(stats['mean:control'], ctrl_feats)
            ctrl_std = get_ctrl_features(stats['std:control'], ctrl_feats)

        # Handle features not in data stats
        for f in ['sin_th', 'cos_th']:
            if f in state_feats:
                idx = state_feats.index(f)
                state_mean[..., idx] = 0.
                state_std[..., idx] = 1.
        for f in ['dvx_dt', 'dvx_dt']:
            if f in state_feats:
                idx = state_feats.index(f)
                state_mean[..., idx] /= self.dt
                state_std[..., idx] /= self.dt

        return state_mean, state_std, ctrl_mean, ctrl_std

    def normalize_state_input(self, states: torch.Tensor):
        return (states - self.state_input_mean) / self.state_input_std

    def unnormalize_state_input(self, states: torch.Tensor):
        return states * self.state_input_std + self.state_input_mean

    def normalize_state_output(self, states: torch.Tensor):
        return (states - self.state_output_mean) / self.state_output_std

    def unnormalize_state_output(self, states: torch.Tensor):
        return states * self.state_output_std + self.state_output_mean

    def normalize_ctrl(self, ctrls: torch.Tensor):
        return (ctrls - self.ctrl_mean) / self.ctrl_std

    def unnormalize_ctrl(self, ctrls: torch.Tensor):
        return ctrls * self.ctrl_std + self.ctrl_mean

    def forward(self, states, controls):
        states = self.normalize_state_input(states)
        controls = self.normalize_ctrl(controls)
        return states, controls


# class StateNormalizer(nn.Module):
#
#     def __init__(
#             self,
#             input_stats: Dict = None,
#     ):
#         super().__init__()
#
#         state_feats = ['x', 'y', 'z',
#                        'r', 'p', 'th',
#                        'vx', 'vy', 'vz',
#                        'ax', 'ay', 'az',
#                        'wx', 'wy', 'wz',]
#
#         ctrl_feats = ['steer', 'throttle']
#
#         self.state_feats = state_feats
#         self.ctrl_feats = ctrl_feats
#
#         self.register_buffer('state_mean', torch.zeros(len(state_feats)))
#         self.register_buffer('state_std', torch.ones(len(state_feats)))
#         self.register_buffer('ctrl_mean', torch.zeros(len(ctrl_feats)))
#         self.register_buffer('ctrl_std', torch.ones(len(ctrl_feats)))
#
#         if input_stats is not None:
#             self.get_stats(input_stats, state_feats, ctrl_feats)
#
#     def get_stats(self, input_stats, state_feats, ctrl_feats):
#         self.state_mean = get_state_features(input_stats['mean:state'], state_feats)
#         self.state_std = get_state_features(input_stats['std:state'], state_feats)
#
#         self.ctrl_mean = get_ctrl_features(input_stats['mean:control'], ctrl_feats)
#         self.ctrl_std = get_ctrl_features(input_stats['std:control'], ctrl_feats)
#
#         # Handle features not in data stats
#         for f in ['sin_th', 'cos_th']:
#             if f in state_feats:
#                 idx = state_feats.index(f)
#                 self.state_mean[..., idx] = 0.
#                 self.state_std[..., idx] = 1.
#
#         for f in ['dvx_dt', 'dvx_dt']:
#             if f in state_feats:
#                 idx = state_feats.index(f)
#                 self.state_mean[..., idx] /= self.dt
#                 self.state_std[..., idx] /= self.dt
#
#     def normalize_state(self, states: torch.Tensor):
#         return (states - self.state_mean) / self.state_std
#
#     def unnormalize_state(self, states: torch.Tensor):
#         return states * self.state_std + self.state_mean
#
#     def normalize_ctrl(self, ctrls: torch.Tensor):
#         return (ctrls - self.ctrl_mean) / self.ctrl_std
#
#     def unnormalize_ctrl(self, ctrls: torch.Tensor):
#         return ctrls * self.ctrl_std + self.ctrl_mean
#
#     def forward(self, states, controls):
#         states = self.normalize_state(states)
#         controls = self.normalize_ctrl(controls)
#         return states, controls
#
