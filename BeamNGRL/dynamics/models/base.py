import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Dict, List
from BeamNGRL.dynamics.utils.network_utils import get_state_features, get_ctrl_features
from .normalizers import FeatureNormalizer, StateNormalizer


class DynamicsBase(ABC, nn.Module):

    def __init__(
            self,
            state_feat: List,
            ctrl_feat: List,
            **kwargs,
    ):
        super().__init__()

        self.state_dim = len(state_feat)
        self.ctrl_dim = len(ctrl_feat)

        self.state_feat_list = state_feat
        self.ctrl_feat_list = ctrl_feat

    def process_targets(self, states: torch.Tensor):
        states = get_state_features(states, self.state_feat_list)
        if self.normalizer:
            states = self.normalizer.normalize_state(states)
        return states

    def process_input(self, states: torch.Tensor, controls: torch.Tensor):
        states = get_state_features(states, self.state_feat_list)
        controls = get_ctrl_features(controls, self.ctrl_feat_list)
        if self.normalizer:
            states, controls = self.normalizer(states, controls)
        return states, controls

    def process_output(self, states: torch.Tensor):
        if self.normalizer:
            states = self.normalizer.unnormalize_state(states)
        return states

    def forward(
            self,
            states: torch.Tensor,
            controls: torch.Tensor,
            ctx_data: Dict,
    ):

        state_feat_preds = self._forward(
            states,
            controls,
            ctx_data,
        )

        return state_feat_preds

    def rollout(
            self,
            states_init: torch.Tensor,
            control_seq: torch.Tensor,
            ctx_data: Dict,
    ):

        assert states_init.size(0) == control_seq.size(0)

        states_pred = self._rollout(states_init, control_seq, ctx_data)

        return states_pred

    @abstractmethod
    def _forward(
            self,
            state_feats,
            ctrl_feats,
            ctx_data,
    ):
        pass

    @abstractmethod
    def _rollout(
            self,
            state_feat,
            ctrl_feats,
            ctx_data,
    ):
        pass