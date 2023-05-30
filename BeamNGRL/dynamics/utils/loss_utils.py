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

    def loss(self, next_state_preds, next_state_targets, step=5):
        next_state_preds = next_state_preds[:, :-step] # no label for last prediction
        next_state_targets = torch.roll(next_state_targets, dims=1, shifts=step)[:, :-step]# first entry is an input.
        mse = F.mse_loss(next_state_preds, next_state_targets)
        return mse

class AggregatedMSE(Loss):

    def loss(self, next_state_preds, next_state_targets, step=1):
        next_state_preds = next_state_preds[:, :-step] # no label for last prediction
        next_state_targets = torch.roll(next_state_targets, dims=1, shifts=step)[:, :-step]# first entry is an input.
        vel_loss = F.mse_loss(next_state_preds[..., 6:8], next_state_targets[..., 6:8])
        acc_loss = F.mse_loss(next_state_preds[..., 9:11], next_state_targets[..., 9:11])
        yaw_loss = F.mse_loss(next_state_preds[..., 14], next_state_targets[..., 14])
        mse = vel_loss + yaw_loss
        return mse