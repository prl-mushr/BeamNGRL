import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod


class Loss(ABC):

    def __call__(self, outputs, targets):
        return self.loss(outputs, targets)

    @abstractmethod
    def loss(self, outputs, targets):
        raise NotImplementedError


class StatePredMSE(Loss):

    def loss(self, state_preds, state_targets):
        mse = F.mse_loss(state_preds, state_targets)
        return mse


class NextStatePredMSE(Loss):

    def loss(self, next_state_preds, next_state_targets):
        next_state_preds = next_state_preds[:, :-1] # no label for last prediction
        next_state_targets = next_state_targets[:, 1:] # first entry is an input.
        mse = F.mse_loss(next_state_preds, next_state_targets)
        return mse

class AggregatedMSE(Loss):

    def loss(self, next_state_preds, next_state_targets):
        pred = next_state_preds[...,6:15].clone()
        targ = next_state_targets[...,6:15].clone()
        next_state_preds *= 0
        next_state_targets *= 0
        next_state_preds[...,6:15] = pred
        next_state_targets[...,6:15] = targ
        next_state_preds = next_state_preds[:, :-1] # no label for last prediction
        next_state_targets = next_state_targets[:, 1:] # first entry is an input.
        mse = F.mse_loss(next_state_preds, next_state_targets)
        return mse

class UH1(Loss):
    def loss(self, next_state_preds, next_state_targets):
      target = next_state_targets[:, 25]
      return F.mse_loss(next_state_preds, target)

