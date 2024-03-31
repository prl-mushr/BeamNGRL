import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod

class Loss(ABC):
    def __call__(self, outputs, targets, conf=None):
        return self.loss(outputs, targets, conf)

    @abstractmethod
    def loss(self, outputs, targets, conf):
        raise NotImplementedError


class BaselineAggregatedMSE_dV(Loss):

    def loss(self, next_state_preds, next_state_targets, conf=None):
        if conf is None:
            print("please provide a std for normalizing the error")
            exit()
        std = torch.Tensor(conf["loss_std"]).to("cuda")
        next_state_preds = next_state_preds[:, :-1] # no label for last prediction
        next_state_targets = torch.roll(next_state_targets, dims=1, shifts=-1)[:, :-1]# first entry is an input.
        vel_loss = F.mse_loss(next_state_preds[...,6:9]/std[6:9], next_state_targets[...,6:9]/std[6:9])
        rate_loss = F.mse_loss(next_state_preds[...,12:15]/std[12:15], next_state_targets[...,12:15]/std[12:15])
        mse = vel_loss + rate_loss
        return mse

class ResidualMSE_dV(Loss):

    def loss(self, next_state_preds, next_state_targets, conf=None):
        if conf is None:
            print("config not provided")
            exit()
        std = torch.Tensor(conf["loss_std"]).to("cuda")
        rate_loss = F.mse_loss(next_state_preds[..., 12:15]/std[12:15], next_state_targets[..., 12:15]/std[12:15])
        vel_loss =  F.mse_loss(next_state_preds[...,6:9]/std[6:9], next_state_targets[...,6:9]/std[6:9])
        return rate_loss + vel_loss