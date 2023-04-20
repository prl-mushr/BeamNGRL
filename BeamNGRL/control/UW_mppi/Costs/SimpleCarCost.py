import torch

class SimpleCarCost(torch.nn.Module):
    """
    Class for Dynamics modelling
    """
    def __init__(
        self,
        goal_w = 1,
        speed_w = 1,
        roll_w = 1,
        lethal_w = 1,
        speed_target = 10,
        critical_z = 0.5,
        critical_FA = 0.5,
        BEVmap_size=64,
        BEVmap_res=0.25,
        dtype=torch.float32,
        device=torch.device("cuda"),
    ):

        super(SimpleCarCost, self).__init__()
        self.dtype = dtype
        self.d = device
        ## I feel like "state" definitions like these should be shared across dynamics and cost, perhaps something to think about
        ## like a class "CAR" that has all the state definitions and dynamics and cost functions as methods plus the bevmap stuff? then we can just pass that in?

        self.NX = 17

        self.BEVmap_size = torch.tensor(BEVmap_size).to(self.d)
        self.BEVmap_res = torch.tensor(BEVmap_res).to(self.d)
        self.BEVmap_size_px = torch.tensor((self.BEVmap_size/self.BEVmap_res), device=self.d, dtype=torch.int32)
        self.BEVmap = torch.zeros((self.BEVmap_size_px.item(), self.BEVmap_size_px.item() )).to(self.d)
        self.BEVmap_height = torch.zeros_like(self.BEVmap)
        self.BEVmap_normal = torch.zeros((self.BEVmap_size_px.item(), self.BEVmap_size_px.item(), 3), dtype=self.dtype).to(self.d)
        self.BEVmap_center = torch.zeros(3, dtype=self.dtype).to(self.d)

        self.GRAVITY = torch.tensor(9.8, dtype=self.dtype).to(self.d)

        self.critical_z = torch.tensor(critical_z, dtype=self.dtype).to(self.d)
        self.speed_target = torch.tensor(speed_target, dtype=self.dtype).to(self.d)
        self.critical_FA = torch.tensor(critical_FA, dtype=self.dtype).to(self.d)

        self.lethal_w = torch.tensor(lethal_w, dtype=self.dtype).to(self.d)
        self.goal_w = torch.tensor(goal_w, dtype=self.dtype).to(self.d)
        self.speed_w = torch.tensor(speed_w, dtype=self.dtype).to(self.d)
        self.roll_w = torch.tensor(roll_w, dtype=self.dtype).to(self.d)

        self.goal_state = torch.zeros(2, device = self.d, dtype=self.dtype)

    @torch.jit.export
    def set_BEV(self, BEVmap_height, BEVmap_normal, BEV_center):
        '''
        BEVmap_height, BEVmap_normal are robot-centric elevation and normal maps.
        BEV_center is the x,y,z coordinate at the center of the map. Technically this could just be x,y, but its easier to just remove it from all dims at once.
        '''
        assert BEVmap_height.shape[0] == self.BEVmap_size_px
        self.BEVmap_height = BEVmap_height
        self.BEVmap_normal = BEVmap_normal
        self.BEVmap_center = BEV_center  # translate the state into the center of the costmap.

    @torch.jit.export
    def set_goal(self, goal_state):
        self.goal_state = goal_state[:2]

    def forward(self, state):
        # unpack all values we can remove the stuff we don't need later
        x = state[..., 0] 
        y = state[..., 1]
        z = state[..., 2]
        roll = state[..., 3]
        pitch = state[..., 4]
        yaw = state[..., 5]
        vx = state[...,6]
        vy = state[...,7]
        vz = state[...,8]
        ax = state[...,9]
        ay = state[...,10]
        az = state[...,11]
        wx = state[...,12]
        wy = state[...,13]
        wz = state[...,14]

        state_cost = torch.square(1/torch.cos(roll)) + torch.square(1/torch.cos(pitch))

        vel_cost = torch.square( (self.speed_target - vx)/self.speed_target )

        force_angle = torch.atan2(ay, az)
        condition = torch.abs(force_angle) > self.critical_FA
        roll_cost = torch.masked_fill(force_angle, condition, torch.tensor(1000.0, dtype=self.dtype))

        terminal_cost = torch.linalg.norm(state[:,:,-1,:2] - self.goal_state.unsqueeze(dim=0), dim=-1)

        running_cost = self.lethal_w * state_cost + self.roll_w * roll_cost + self.speed_w * vel_cost
        cost_to_go = self.goal_w * terminal_cost

        ## for running cost mean over the 0th dimension (bins), which results in a KxT tensor. Then sum over the 1st dimension (time), which results in a [K] tensor.
        ## for terminal cost, just mean over the 0th dimension (bins), which results in a [K] tensor.
        return (running_cost.mean(dim=0)).sum(dim=1) + cost_to_go.mean(dim=0)