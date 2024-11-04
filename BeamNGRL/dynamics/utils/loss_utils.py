import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod

class Loss(ABC):
    def __call__(self, outputs, targets, conf=None, loss_multiplier=None):
        return self.loss(outputs, targets, conf, loss_multiplier)

    @abstractmethod
    def loss(self, outputs, targets, conf, loss_multiplier):
        raise NotImplementedError


class BaselineAggregatedMSE_dV(Loss):

    def loss(self, next_state_preds, next_state_targets, conf=None, loss_multiplier=None):
        if conf is None:
            std = torch.ones_like(next_state_preds.shape[-1]).to("cuda") # it is better to take the normalized loss, so please provide that in the config file.
        else:
            std = torch.Tensor(conf["loss_std"]).to("cuda")
        next_state_preds = next_state_preds[:, :-1] # no label for last prediction
        next_state_targets = torch.roll(next_state_targets, dims=1, shifts=-1)[:, :-1] # generating sp from s

        vel_loss = F.mse_loss(next_state_preds[...,6:9]/std[6:9], next_state_targets[...,6:9]/std[6:9])
        rate_loss = F.mse_loss(next_state_preds[...,12:15]/std[12:15], next_state_targets[...,12:15]/std[12:15])

        mse = vel_loss + rate_loss
        return mse