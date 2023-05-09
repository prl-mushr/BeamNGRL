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
