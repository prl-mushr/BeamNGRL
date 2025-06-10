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

        # self.critical_SA = torch.tensor(Cost_config["critical_SA"], dtype=self.dtype, device=self.d)
        # self.speed_target = torch.tensor(Cost_config["speed_target"], dtype=self.dtype, device=self.d)
        self.critical_RI = torch.tensor(Cost_config["critical_RI"], dtype=self.dtype, device=self.d)
        self.lethal_w = torch.tensor(Cost_config["lethal_w"], dtype=self.dtype, device=self.d)
        # self.stop_w   = torch.tensor(Cost_config["stop_w"], dtype=self.dtype, device=self.d)
        self.critical_vert_acc = torch.tensor(Cost_config["critical_vert_acc"], dtype=self.dtype, device=self.d)
        # self.critical_vert_spd = torch.tensor(Cost_config["critical_vert_spd"], dtype=self.dtype, device=self.d)
        self.pos_w = torch.tensor(Cost_config["pos_w"], dtype=self.dtype, device=self.d)
        self.roll_ditch_w = torch.tensor(Cost_config["roll_ditch_w"], dtype=self.dtype, device=self.d)
        self.speed_w = torch.tensor(Cost_config["speed_w"], dtype=self.dtype, device=self.d)
        self.heading_w = torch.tensor(Cost_config["heading_w"], dtype=self.dtype, device=self.d)
        self.scaling_factor = torch.tensor(Cost_config["scaling_factor"], dtype=self.dtype, device=self.d)
        self.scaling = None
        self.BEVmap_size = torch.tensor(Map_config["map_size"], dtype=self.dtype, device=self.d)
        self.BEVmap_res = torch.tensor(Map_config["map_res"], dtype=self.dtype, device=self.d)

        self.BEVmap_size_px = torch.tensor((self.BEVmap_size/self.BEVmap_res), device=self.d, dtype=torch.int32)
        self.BEVmap = torch.zeros((self.BEVmap_size_px.item(), self.BEVmap_size_px.item() )).to(self.d)
        self.BEVmap_height = torch.zeros_like(self.BEVmap)
        self.BEVmap_normal = torch.zeros((self.BEVmap_size_px.item(), self.BEVmap_size_px.item(), 3), dtype=self.dtype, device=self.d)
        self.BEVmap_center = torch.zeros(3, dtype=self.dtype, device=self.d)
        self.BEVmap_cost = torch.zeros_like(self.BEVmap_height)

        self.GRAVITY = torch.tensor(9.8, dtype=self.dtype, device=self.d)

        self.goal_state = torch.zeros(2, device = self.d, dtype=self.dtype)

        self.car_w2 = torch.tensor(Cost_config["car_bb_width"]/2, dtype=self.dtype, device=self.d)
        self.car_l2 = torch.tensor(Cost_config["car_bb_length"]/2, dtype=self.dtype, device=self.d)
        # self.step_cost = torch.tensor(0,dtype=self.dtype, device=self.d)
        # self.roll_cost = torch.tensor(0,dtype=self.dtype, device=self.d)
        self.bad_physics = False

    @torch.jit.export
    def set_BEV(self, BEVmap_height, BEVmap_normal, BEVmap_cost):
        '''
        BEVmap_height, BEVmap_normal are robot-centric elevation and normal maps.
        BEV_path is the x,y,z coordinate at the center of the map. Technically this could just be x,y, but its easier to just remove it from all dims at once.
        '''
        self.BEVmap_height = BEVmap_height
        self.BEVmap_normal = BEVmap_normal
        self.BEVmap_cost = (255 - BEVmap_cost)/255

    @torch.jit.export
    def set_goal(self, goal_state):
        self.goal_state = goal_state[:2]

    def set_path(self, path):
        self.path = torch.tensor(path, dtype=self.dtype, device=self.d)

    @torch.jit.export
    def set_speed_limit(self, speed_lim):
        self.speed_target = torch.tensor(speed_lim, dtype=self.dtype, device=self.d)

    def meters_to_px(self, meters):
        return torch.clamp( ((meters + self.BEVmap_size*0.5) / self.BEVmap_res).to(dtype=torch.long, device=self.d), 0, self.BEVmap_size_px - 1)

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
                
        img_X = self.meters_to_px(x)
        img_Y = self.meters_to_px(y)
        
        beta = torch.atan2(vy, vx)
        # cy = torch.cos(beta)
        # sy = torch.sin(beta)
        cy = torch.cos(yaw)
        sy = torch.sin(yaw)
        V = torch.sqrt(vx**2 + vy**2) * torch.sign(vx)
        flx = x + self.car_l2*cy - self.car_w2*sy
        fly = y + self.car_l2*sy + self.car_w2*cy
        frx = x + self.car_l2*cy + self.car_w2*sy
        fry = y + self.car_l2*sy - self.car_w2*cy
        blx = x - self.car_l2*cy - self.car_w2*sy
        bly = y - self.car_l2*sy + self.car_w2*cy
        brx = x - self.car_l2*cy + self.car_w2*sy
        bry = y - self.car_l2*sy - self.car_w2*cy

        flx_px = self.meters_to_px(flx)
        fly_px = self.meters_to_px(fly)
        frx_px = self.meters_to_px(frx)
        fry_px = self.meters_to_px(fry)
        blx_px = self.meters_to_px(blx)
        bly_px = self.meters_to_px(bly)
        brx_px = self.meters_to_px(brx)
        bry_px = self.meters_to_px(bry)
        
        # # state_cost = state_cost + torch.clamp(torch.square(self.BEVmap_height[img_Y, img_X]) - 0.09, 0, 10)
        # # state_cost = torch.square(self.BEVmap_cost[img_Y, img_X,0])
        # # evaluate state cost using footprint
        # # state cost is the maximum state cost of all the footprint points
        state_cost = torch.zeros_like(x)
        state_cost = torch.max(state_cost, torch.square(self.BEVmap_cost[fly_px, flx_px]))
        state_cost = torch.max(state_cost, torch.square(self.BEVmap_cost[fry_px, frx_px]))
        state_cost = torch.max(state_cost, torch.square(self.BEVmap_cost[bly_px, blx_px]))
        state_cost = torch.max(state_cost, torch.square(self.BEVmap_cost[bry_px, brx_px]))

        cr = torch.cos(roll)
        cp = torch.cos(pitch)
        beta = torch.atan2(vy, vx)**2
        # now we compute running cost as a weighted sum of the xy, cy-sy, and velocity errors
        roll_ditch_cost = (
            + torch.clamp(torch.abs(az - self.GRAVITY*cr*cp) - self.critical_vert_acc, 0, 10.0)/10.0
            + torch.clamp(torch.abs(ay/az) - self.critical_RI, 0, 1)
            ) *self.roll_ditch_w
        constraint_cost = self.lethal_w * state_cost

        x_err = x - self.path[:,0]
        y_err = y - self.path[:,1]
        # cy, sy error
        cy_err = cy - torch.cos(self.path[:,2])
        sy_err = sy - torch.sin(self.path[:,2])
        yaw_err = cy_err**2 + sy_err**2
        # velocity error
        vel_err = (V - self.path[:,3])**2

        pos_err = x_err**2 + y_err**2
        running_cost = self.pos_w * pos_err + self.heading_w * yaw_err + self.speed_w * vel_err + beta*1.5
        if self.scaling is None:
            steps = running_cost.shape[-1]
            self.scaling = torch.linspace(0.1, self.scaling_factor, steps, device=self.d)
        running_cost = running_cost * self.scaling + roll_ditch_cost
        constraint_cost[torch.where(pos_err < 1)] = 0
        # roll_ditch_cost[torch.where(pos_err < 1)] = 0

        constraint_cost = constraint_cost.mean(dim=0).sum(dim=1)
        if torch.all(constraint_cost > self.lethal_w):
            self.bad_physics = True
        else:
            self.bad_physics = False

        ## for running cost mean over the 0th dimension (bins), which results in a KxT tensor. Then sum over the 1st dimension (time), which results in a [K] tensor.
        ## for terminal cost, just mean over the 0th dimension (bins), which results in a [K] tensor.
        return (running_cost.mean(dim=0)).sum(dim=1) + constraint_cost