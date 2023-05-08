import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Dict, List
from BeamNGRL.dynamics.utils.network_utils import get_state_features, get_ctrl_features


class Standardizer(nn.Module):

    def __init__(
            self,
            state_feats: List,
            ctrl_feats: List,
            input_stats: Dict = None,
    ):
        super().__init__()

        self.state_feats = state_feats
        self.ctrl_feats = ctrl_feats

        state_mean, state_std, ctrl_mean, ctrl_std = None, None, None, None
        if input_stats is not None:
            (state_mean,
             state_std,
             ctrl_mean,
             ctrl_std,) = self.get_stats(input_stats, state_feats, ctrl_feats)

        self.register_buffer('state_mean', state_mean)
        self.register_buffer('state_std', state_std)
        self.register_buffer('ctrl_mean', ctrl_mean)
        self.register_buffer('ctrl_std', ctrl_std)

    def get_stats(self, input_stats, state_feats, ctrl_feats):
        state_mean = get_state_features(input_stats['mean:state'], state_feats)
        state_std = get_state_features(input_stats['std:state'], state_feats)

        ctrl_mean = get_ctrl_features(input_stats['mean:control'], ctrl_feats)
        ctrl_std = get_ctrl_features(input_stats['std:control'], ctrl_feats)

        # Handle feats not in stats
        for f in ['sin_th', 'cos_th']:
            if f in state_feats:
                idx = state_feats.index(f)
                state_mean[..., idx] = 0.
                state_std[..., idx] = 1.

        return state_mean, state_std, ctrl_mean, ctrl_std

    def normalize_state(self, states: torch.Tensor):
        return (states - self.state_mean) / self.state_std

    def unnormalize_state(self, states: torch.Tensor):
        return states * self.state_std + self.state_mean

    def normalize_ctrl(self, ctrls: torch.Tensor):
        return (ctrls - self.ctrl_mean) / self.ctrl_std

    def unnormalize_ctrl(self, ctrls: torch.Tensor):
        return ctrls * self.ctrl_std + self.ctrl_mean

    def forward(self, states, controls):
        states = self.normalize_state(states)
        controls = self.normalize_ctrl(controls)
        return states, controls


class DynamicsBase(ABC, nn.Module):

    def __init__(
            self,
            state_feat: List,
            ctrl_feat: List,
            input_stats: Dict,
            **kwargs,
    ):
        super().__init__()

        self.state_dim = len(state_feat)
        self.ctrl_dim = len(ctrl_feat)

        self.state_feat_list = state_feat
        self.ctrl_feat_list = ctrl_feat

        self.standardizer = Standardizer(state_feat, ctrl_feat, input_stats)

    def process_targets(self, states: torch.Tensor):
        states = get_state_features(states, self.state_feat_list)
        states = self.standardizer.normalize_state(states)
        return states

    def process_input(self, states: torch.Tensor, controls: torch.Tensor):
        states = get_state_features(states, self.state_feat_list)
        controls = get_ctrl_features(controls, self.ctrl_feat_list)

        states, controls = self.standardizer(states, controls)

        return states, controls

    def forward(
            self,
            states: torch.Tensor,
            controls: torch.Tensor,
            ctx_data: Dict,
    ):

        states, controls = self.process_input(states, controls)

        state_preds = self._forward(
            states,
            controls,
            ctx_data,
        )

        return state_preds

    def rollout(
            self,
            state_init: torch.Tensor,
            control_seq: torch.Tensor,
            ctx_data: Dict,
    ):

        assert state_init.size(0) == control_seq.size(0)
        state_init, control_seq = self.process_input(state_init, control_seq)

        state_seq = self._rollout(state_init, control_seq, ctx_data)

        return state_seq

    @abstractmethod
    def _forward(
            self,
            input_state_seq,
            input_ctrl_seq,
            ctx_data,
    ):
        pass

    @abstractmethod
    def _rollout(
            self,
            state,
            control_seq,
            ctx_data,
    ):
        pass