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
        state_loss = F.mse_loss(next_state_preds[..., 1:, :5]/std[:5], next_state_targets[..., 1:, :5]/std[:5])
        c_yaw_loss = F.mse_loss(torch.cos(next_state_preds[..., 1:, 5])/torch.cos(std[5]), torch.cos(next_state_targets[..., 1:, 5])/torch.cos(std[5]))
        s_yaw_loss = F.mse_loss(torch.sin(next_state_preds[..., 1:, 5])/torch.sin(std[5]), torch.sin(next_state_targets[..., 1:, 5])/torch.sin(std[5]))
        dyn_loss = F.mse_loss(next_state_preds[..., 1:, 6:15]/std[6:15], next_state_targets[..., 1:, 6:15]/std[6:15])

        dt = conf["loss_dt"]
        pos_diff = torch.zeros_like(next_state_preds[...,  :3])
        pos_diff_min = torch.zeros_like(next_state_targets[..., :3])

        cr = torch.cos(next_state_preds[..., 3])
        sr = torch.sin(next_state_preds[..., 3])
        cp = torch.cos(next_state_preds[..., 4])
        sp = torch.sin(next_state_preds[..., 4])
        cy = torch.cos(next_state_preds[..., 5])
        sy = torch.sin(next_state_preds[..., 5])

        pos_diff[..., 0] = dt*( next_state_preds[..., 6]*cp*cy + next_state_preds[..., 7]*(sr*sp*cy - cr*sy) + next_state_preds[..., 8]*(cr*sp*cy + sr*sy) )
        pos_diff[..., 1] = dt*( next_state_preds[..., 6]*cp*sy + next_state_preds[..., 7]*(sr*sp*sy + cr*cy) + next_state_preds[..., 8]*(cr*sp*sy - sr*cy) )
        pos_diff[..., 2] = dt*( next_state_preds[..., 6]*(-sp) + next_state_preds[..., 7]*(sr*cp)            + next_state_preds[..., 8]*(cr*cp)            )

        pos_pred = torch.cumsum(pos_diff, dim=-2)
        # compare the predicted positions and the positions you would have gotten if you had integrated your velocities and orientations -- this is a "physics inspired loss"
        pos_consistency = F.mse_loss(pos_pred[...,1:,:]/std[:3], next_state_preds[..., 1:, :3]/std[:3])

        return state_loss + c_yaw_loss + s_yaw_loss + dyn_loss# + 0.1*dt*dt*pos_consistency