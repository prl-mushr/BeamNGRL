import torch

class SimpleCarCost(torch.nn.Module):
    """
	Class for Dynamics modelling
    """
    def __init__(
        self,
        BEVmap_size=32,
        BEVmap_res=0.25,
        device=torch.device("cuda"),
    ):

        super(SimpleCarCost, self).__init__()

        self.POS = slice(0, 2)
        self.YAW = slice(2, 3)
        self.VEL = slice(3, 4)
        self.CURVATURE = slice(4, 5)
        self.ACCEL = slice(5, 6)

        self.STEER = slice(0, 1)
        self.THROTTLE = slice(1, 2)
        ## always have NU and NX!
        self.NU = max(self.STEER.stop, self.THROTTLE.stop)
        self.NX = max(self.POS.stop, self.YAW.stop, self.VEL.stop, self.STEER.stop, self.THROTTLE.stop)
        self.d = device

        self.goal_state = torch.zeros(2).to(self.d)  # possible goal state
        self.BEVmap_size = torch.tensor(BEVmap_size).to(self.d)
        self.BEVmap_res = torch.tensor(BEVmap_res).to(self.d)
        assert self.BEVmap_res > 0
        self.BEVmap_size_px = torch.tensor((self.BEVmap_size/self.BEVmap_res), device=self.d, dtype=torch.int32)
        self.BEVmap = torch.zeros((self.BEVmap_size_px.item(), self.BEVmap_size_px.item() )).to(self.d)
        self.BEVmap_heght = torch.zeros_like(self.BEVmap)
        self.BEVmap_segmt = torch.zeros_like(self.BEVmap)
        self.BEVmap_color = torch.zeros_like(self.BEVmap)
        self.BEVmap_normal = torch.zeros_like(self.BEVmap)
        self.BEVmap_center = self.BEVmap_size * 0.5
        self.BEVmap_res_inv = 1.0 / self.BEVmap_res

    @torch.jit.export
    def update_maps(self, BEVmap, BEVmap_heght, BEVmap_segmt, BEVmap_color, BEVmap_normal):
        self.BEVmap = BEVmap
        self.BEVmap_heght = BEVmap_heght
        self.BEVmap_segmt = BEVmap_segmt
        self.BEVmap_color = BEVmap_color
        self.BEVmap_normal = BEVmap_normal

    def set_goal(self, goal_state):
        self.goal_state = goal_state

    def forward(self, state):
        ## State is a tensor of shape
        # (bins, samples, horizon, state_dim). If you're not using bins, you can take mean over the first dimension or just use the first bin.

        # print(f'\nCost: state shape: {state.shape}')
        pos = state[..., self.POS]
        x = pos[..., [0]]
        y = pos[..., [1]]
        yaw = state[..., self.YAW]
        vel = state[..., self.VEL]
        
        img_X = ((x + self.BEVmap_center) / self.BEVmap_res_inv).to(dtype=torch.long, device=self.d)
        img_Y = ((y + self.BEVmap_center) / self.BEVmap_res_inv).to(dtype=torch.long, device=self.d)
        
        state_cost = self.BEVmap[img_Y, img_X]
        state_cost *= state_cost
        condition = state_cost >= 0.9  # Boolean mask
        state_cost = torch.masked_fill(state_cost, condition, torch.tensor(100.0, dtype=self.dtype))

        vel_cost = torch.abs(self.max_speed - vel)/self.max_speed
        vel_cost = torch.sqrt(vel_cost)

        ay = vel*yaw
        accel_cost = ay*ay
        condition = accel_cost > 25
        accel_cost = torch.masked_fill(accel_cost, condition, torch.tensor(100.0, dtype=self.dtype))

        cost_to_come = 1.5*vel_cost + state_cost + torch.tensor(0.01,dtype=self.dtype)*accel_cost
        cost_to_come = cost_to_come.mean(dim=0).sum(dim = -1)

        cost_to_go = torch.linalg.norm(state[...,-1,:2] - self.goal_state.unsqueeze(dim=0), dim=2)
        cost_to_go = cost_to_go.mean(dim=0)

        return cost_to_come + cost_to_go
    