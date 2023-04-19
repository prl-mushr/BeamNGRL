import torch


class SimpleCar(torch.nn.Module):
    """
	Class for Dynamics modelling
    """
    def __init__(
        self,
        wheelbase = 1,
        speed_max = 1,
       	steering_max = 1,
       	dt = 0.05,
        dtype=torch.float32,
        device=torch.device("cuda"),
    ):

        super(SimpleCar, self).__init__()

        self.wheelbase = torch.tensor(wheelbase, device=device, dtype=dtype)
        self.speed_max = torch.tensor(speed_max, device=device, dtype=dtype)
        self.steering_max = torch.tensor(steering_max, device=device, dtype=dtype)
        self.dt = torch.tensor(dt, device=device, dtype=dtype)
        self.POS = slice(0, 2)
        self.YAW = slice(2, 3)
        self.VEL = slice(3, 4)
        self.CURVATURE = slice(4, 5)
        self.ACCEL = slice(5, 6)
        self.dtype = torch.float

        self.STEER = slice(0, 1)
        self.THROTTLE = slice(1, 2)
        ## always have NU and NX!
        self.NU = max(self.STEER.stop, self.THROTTLE.stop)
        self.NX = max(self.POS.stop, self.YAW.stop, self.VEL.stop, self.STEER.stop, self.THROTTLE.stop)

        self.throttle_to_accel = torch.tensor(10, device=device, dtype=dtype)
        self.curvature_max = torch.tensor(1, device=device, dtype=dtype)

    def forward(self, state, perturbed_actions):
        pos = state[..., self.POS]
        x = pos[..., [0]]
        y = pos[..., [1]]
        yaw = state[..., self.YAW]
        vel = state[..., self.VEL]

        steer = state[..., self.CURVATURE] + torch.cumsum(perturbed_actions[..., self.STEER].unsqueeze(dim=0) * self.dt, dim=-1)
        accel = state[..., self.ACCEL] + torch.cumsum(perturbed_actions[..., self.THROTTLE].unsqueeze(dim=0) * self.dt, dim=-1)

        K = steer * self.curvature_max  # this is just a placeholder for curvature since steering correlates to curvature
        
        vel = vel + torch.cumsum(accel * self.throttle_to_accel * self.dt, dim=-1)
        delta = vel * self.dt

        yaw = yaw + torch.cumsum(delta * K, dim=-1)  # this is what the yaw will become
        x = x + torch.cumsum(delta * torch.cos(yaw), dim=-1)
        y = y + torch.cumsum(delta * torch.sin(yaw), dim=-1)

        return torch.stack((x, y, yaw, vel, steer, accel), dim=3).to(dtype=self.dtype)
    