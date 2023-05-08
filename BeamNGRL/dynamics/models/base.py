import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Dict, List
from BeamNGRL.dynamics.utils.network_utils import get_state_features, get_ctrl_features


class Standardizer(nn.Module):

    def __init__(
            self,
            input_stats: Dict,
            state_feats: List,
            ctrl_feats: List,
    ):
        super().__init__()

        self.state_feats = state_feats
        self.ctrl_feats = ctrl_feats

        self.state_mean = None
        self.state_std = None
        self.ctrl_mean = None
        self.ctrl_std = None

        self.init_stats(input_stats)

    def init_stats(self, input_stats, state_feats, ctrl_feats):
        self.state_mean = get_state_features(input_stats['mean:state'], state_feats)
        self.state_std = get_state_features(input_stats['std:state'], state_feats)

        self.ctrl_mean = get_ctrl_features(input_stats['mean:ctrl'], ctrl_feats)
        self.ctrl_std = get_ctrl_features(input_stats['std:ctrl'], ctrl_feats)

        # Handle feats not in stats
        for f in ['sin_th', 'cos_th']:
            if f in state_feats:
                idx = state_feats.index(f)
                self.state_mean[..., idx] = 0.
                self.state_std[..., idx] = 1.

    def normalize_state(self, states: torch.Tensor):
        return (states - self.state_mean) / self.state_std

    def unnormalize_state(self, states: torch.Tensor):
        return states * self.state_std + self.state_mean

    def normalize_ctrl(self, ctrls: torch.Tensor):
        return (ctrls - self.ctrl_mean) / self.ctrl_std

    def unnormalize_ctrl(self, ctrls: torch.Tensor):
        return ctrls * self.ctrl_std + self.ctrl_mean


class DynamicsBase(ABC, nn.Module):

    def __init__(
            self,
            state_feats: List,
            ctrl_feats: List,
            input_stats: Dict,
    ):
        super().__init__()

        self.state_dim = len(state_feats)
        self.ctrl_dim = len(ctrl_feats)

        self.state_feat_list = state_feats
        self.ctrl_feat_list = ctrl_feats

        self.standardizer = Standardizer(input_stats, state_feats, ctrl_feats)

    def process_targets(self, states: torch.Tensor):
        states = get_state_features(states, self.state_feat_list)
        states = self.standardizer.normalize_state(states)
        return states

    def process_input(self, states: torch.Tensor, controls: torch.Tensor):
        states = get_state_features(states, self.state_feat_list)
        controls = get_ctrl_features(controls, self.ctrl_feat_list)

        states = self.standardizer.normalize_state(states)
        controls = self.standardizer.normalize_ctrl(controls)

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