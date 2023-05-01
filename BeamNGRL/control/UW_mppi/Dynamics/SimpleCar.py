import torch
import torch.nn as nn


class SimpleCar(nn.Module):
    """
    Class for Dynamics modelling
    """
    def __init__(
        self,
        wheelbase=1,
        wheelspeed_max=25.,
        steering_max=0.611, # 35/57.3
        dt=0.04,
        dtype=torch.float32,
        device=torch.device("cuda"),
    ):

        super().__init__()

        self.tn_args = {'dtype': dtype, 'device': device}

        self.wheelbase = torch.tensor(wheelbase).to(**self.tn_args)
        self.wheelspeed_max = torch.tensor(wheelspeed_max).to(**self.tn_args)
        self.steering_max = torch.tensor(steering_max).to(**self.tn_args)
        self.lf = torch.tensor(1.3).to(**self.tn_args)
        self.lr = torch.tensor(1.3).to(**self.tn_args)
        self.dt = torch.tensor(dt).to(**self.tn_args)
        self.POS = slice(0, 3)
        self.YAW = slice(5, 6)
        self.VEL = slice(6, 7)
        self.ACCEL = slice(9, 10)
        self.STEER = slice(15, 16)
        self.THROTTLE = slice(16, 17)
        self.dtype = torch.float

        ## always have NU and NX!
        self.NU = max(self.STEER.stop, self.THROTTLE.stop)
        self.NX = max(self.POS.stop, self.YAW.stop, self.VEL.stop, self.STEER.stop, self.THROTTLE.stop)

        # self.throttle_to_accel = torch.tensor(10, device=device, dtype=dtype)
        self.throttle_to_accel = torch.tensor(1, device=device, dtype=dtype)
        self.curvature_max = torch.tensor(1, device=device, dtype=dtype)

    def forward(self, state, controls):
        pos = state[..., self.POS]
        x = pos[..., [0]]
        y = pos[..., [1]]
        yaw = state[..., self.YAW]
        vel = state[..., self.VEL]
        # accel = state[..., self.ACCEL]

        delta_steer = torch.cumsum(controls[..., [0]].unsqueeze(dim=0) * self.dt, dim=2)
        delta_throttle = torch.cumsum(controls[..., [1]].unsqueeze(dim=0) * self.dt, dim=2)
        steer = state[..., self.STEER] + delta_steer
        throttle = state[..., self.THROTTLE] + delta_throttle
        curvature = torch.tan(steer * self.steering_max) / (self.lf + self.lr)
        accel_x = throttle * self.wheelspeed_max
        vel = vel + torch.cumsum(accel_x * self.dt, dim=2)
        delta = vel * self.dt

        gz = vel * curvature
        accel_y = vel * gz

        yaw = yaw + torch.cumsum(delta * curvature, dim=2)  # this is what the yaw will become
        x = x + torch.cumsum(delta * torch.cos(yaw), dim=2)
        y = y + torch.cumsum(delta * torch.sin(yaw), dim=2)

        states = torch.cat((x, y, yaw, vel, steer, accel_x, accel_y), dim=-1).to(**self.tn_args)
        return states