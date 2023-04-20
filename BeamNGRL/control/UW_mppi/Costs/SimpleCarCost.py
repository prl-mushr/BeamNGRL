import torch
import torch.nn as nn


class SimpleCarCost(nn.Module):
    """
    Class for Cost modelling
    """
    def __init__(
        self,
        max_speed=10.,
        state_dim=6,
        map_size=32,
        map_res=0.25,
        device=torch.device("cuda"),
        dtype=torch.float32,
    ):

        super().__init__()

        self.device = device
        self.dtype = dtype
        self.tn_args = {'device': device, 'dtype': dtype}

        self.state_dim = state_dim

        self.goal_state = torch.zeros(2).to(device)  # possible goal state
        self.max_speed = torch.tensor(max_speed, device=device, dtype=dtype)

        assert map_res > 0

        self.map_size = map_size
        self.map_res = map_res
        self.cell_size = int(map_size // map_res)
        self.map_center = map_size * 0.5
        self.map_res_inv = 1. / map_res

        self.bev_path = torch.zeros(map_size, map_size).to(device, dtype)
        self.bev_elev = torch.zeros_like(self.bev_path)
        self.bev_segmt = torch.zeros_like(self.bev_path)
        self.bev_color = torch.zeros_like(self.bev_path)
        self.bev_normal = torch.zeros_like(self.bev_path)

    @torch.jit.export
    def update_maps(self, bev_path, bev_elev, bev_segmt, bev_color, bev_normal):
        self.bev_path = bev_path
        self.bev_elev = bev_elev
        self.bev_segmt = bev_segmt
        self.bev_color = bev_color
        self.bev_normal = bev_normal

    def set_goal(self, goal_state):
        self.goal_state = goal_state

    def forward(self, states, controls):
        """
        State is a tensor of shape
        (bins, samples, horizon, state_dim). If you're not using bins, you can take mean over the first dimension or just use the first bin.
        """

        assert states.size(-1) == self.state_dim

        x, y, yaw, vel, steer, accel = states.split(1, dim=-1)
        
        img_X = ((x + self.map_center) / self.map_res).to(self.device, torch.long)
        img_Y = ((y + self.map_center) / self.map_res).to(self.device, torch.long)

        b, n_ctrl_samples, horizon, _ = states.shape

        running_cost = torch.zeros(b, n_ctrl_samples, horizon).to(**self.tn_args)
        terminal_cost = torch.zeros(b, n_ctrl_samples).to(**self.tn_args)

        # Path cost
        path_cost = self.bev_path[img_Y, img_X][..., 0] / 255.
        # path_cost = path_cost**2
        # path_cost = path_cost * 2.
        running_cost += path_cost.squeeze(-1)

        # Velocity
        vel_cost = torch.abs(self.max_speed - vel)/self.max_speed
        vel_cost = torch.sqrt(vel_cost)
        vel_cost = 1.5 * vel_cost
        running_cost += vel_cost.squeeze(-1)

        # Acceleration
        ay = vel*yaw
        # accel_cost = ay*ay
        accel_cost = 0*ay
        condition = accel_cost > 6
        accel_cost = torch.masked_fill(
            accel_cost, condition, torch.tensor(1000., dtype=self.dtype))
        running_cost += accel_cost.squeeze(-1)

        # Goal distance (terminal)
        xy_final = states[...,-1, :2]
        goal_cost = torch.linalg.norm(xy_final - self.goal_state.view(1, 1, 2), dim=-1)
        terminal_cost += goal_cost

        # Sum across horizon
        total_cost = running_cost.sum(dim=-1)

        # Add Terminal costs
        total_cost += terminal_cost

        return total_cost
    