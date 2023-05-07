import torch

class SimpleCarCost(torch.nn.Module):
    """
    Class for Dynamics modelling
    """
    def __init__(
        self,
        Cost_config,
        Map_config,
        dtype=torch.float32,
        device=torch.device("cuda"),
    ):

        super(SimpleCarCost, self).__init__()
        self.dtype = dtype
        self.d = device

        self.critical_SA = torch.tensor(Cost_config["critical_SA"], dtype=self.dtype, device=self.d)
        self.speed_target = torch.tensor(Cost_config["speed_target"], dtype=self.dtype, device=self.d)
        self.critical_RI = torch.tensor(Cost_config["critical_RI"], dtype=self.dtype, device=self.d)
        self.lethal_w = torch.tensor(Cost_config["lethal_w"], dtype=self.dtype, device=self.d)
        self.goal_w = torch.tensor(Cost_config["goal_w"], dtype=self.dtype, device=self.d)
        self.speed_w = torch.tensor(Cost_config["speed_w"], dtype=self.dtype, device=self.d)
        self.roll_w = torch.tensor(Cost_config["roll_w"], dtype=self.dtype, device=self.d)

        self.BEVmap_size = torch.tensor(Map_config["map_size"], dtype=self.dtype, device=self.d)
        self.BEVmap_res = torch.tensor(Map_config["map_res"], dtype=self.dtype, device=self.d)

        self.BEVmap_size_px = torch.tensor((self.BEVmap_size/self.BEVmap_res), device=self.d, dtype=torch.int32)
        self.BEVmap = torch.zeros((self.BEVmap_size_px.item(), self.BEVmap_size_px.item() )).to(self.d)
        self.BEVmap_height = torch.zeros_like(self.BEVmap)
        self.BEVmap_normal = torch.zeros((self.BEVmap_size_px.item(), self.BEVmap_size_px.item(), 3), dtype=self.dtype, device=self.d)
        self.BEVmap_center = torch.zeros(3, dtype=self.dtype, device=self.d)
        self.BEVmap_path = torch.zeros_like(self.BEVmap)

        self.GRAVITY = torch.tensor(9.8, dtype=self.dtype, device=self.d)

        self.goal_state = torch.zeros(2, device = self.d, dtype=self.dtype)

    @torch.jit.export
    def set_BEV(self, BEVmap_height, BEVmap_normal, BEV_path):
        '''
        BEVmap_height, BEVmap_normal are robot-centric elevation and normal maps.
        BEV_path is the x,y,z coordinate at the center of the map. Technically this could just be x,y, but its easier to just remove it from all dims at once.
        '''
        assert BEVmap_height.shape[0] == self.BEVmap_size_px
        self.BEVmap_height = BEVmap_height
        self.BEVmap_normal = BEVmap_normal
        self.BEVmap_path = BEV_path  # translate the state into the center of the costmap.

    @torch.jit.export
    def set_goal(self, goal_state):
        self.goal_state = goal_state[:2]

    def forward(self, state, controls):
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
        
        normalizer = 1/torch.tensor(float(state.shape[-2]), device = self.d, dtype = self.dtype)
        
        img_X = torch.clamp( ((x + self.BEVmap_size*0.5) / self.BEVmap_res).to(dtype=torch.long, device=self.d), 0, self.BEVmap_size_px - 1)
        img_Y = torch.clamp( ((y + self.BEVmap_size*0.5) / self.BEVmap_res).to(dtype=torch.long, device=self.d), 0, self.BEVmap_size_px - 1)

        # state_cost = torch.square(torch.clamp( ( (1/self.BEVmap_normal[img_Y, img_X, 2]) / (self.critical_SA)) - 1, 0, 10) ) ## don't go fast over uneven terrain
        # state_cost = state_cost + torch.clamp(torch.square(self.BEVmap_height[img_Y, img_X]) - 0.09, 0, 10)
        state_cost = torch.square(self.BEVmap_path[img_Y, img_X,0])
        vel_cost = torch.square((self.speed_target - vx)/self.speed_target)

        roll_cost = torch.clamp(torch.square((ay/az) / self.critical_RI ) - 1, 0, 1)
        
        terminal_cost = torch.linalg.norm(state[:,:,-1,:2] - self.goal_state.unsqueeze(dim=0), dim=-1)

        running_cost = normalizer *( self.lethal_w * state_cost + self.roll_w * roll_cost + self.speed_w * vel_cost )
        cost_to_go = self.goal_w * terminal_cost

        ## for running cost mean over the 0th dimension (bins), which results in a KxT tensor. Then sum over the 1st dimension (time), which results in a [K] tensor.
        ## for terminal cost, just mean over the 0th dimension (bins), which results in a [K] tensor.
        return (running_cost.mean(dim=0)).sum(dim=1) + cost_to_go.mean(dim=0)