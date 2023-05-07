import torch


class SimpleCarDynamics(torch.nn.Module):
    """
    Class for Dynamics modelling
    """
    def __init__(
        self,
        Dynamics_config,
        Map_config,
        MPPI_config,
        dtype=torch.float32,
        device=torch.device("cuda"),
    ):

        super(SimpleCarDynamics, self).__init__()
        self.dtype = dtype
        self.d = device

        self.wheelbase = torch.tensor(Dynamics_config["wheelbase"], device=self.d, dtype=self.dtype)
        self.throttle_to_wheelspeed = torch.tensor(Dynamics_config["throttle_to_wheelspeed"], device=self.d, dtype=self.dtype)
        self.steering_max = torch.tensor(Dynamics_config["steering_max"], device=self.d, dtype=self.dtype)
        self.dt = torch.tensor(Dynamics_config["dt"], device=self.d, dtype=self.dtype)
        self.K = MPPI_config["ROLLOUTS"]
        self.T = MPPI_config["TIMESTEPS"]
        self.M = MPPI_config["BINS"]
        self.BEVmap_size = torch.tensor(Map_config["map_size"], dtype=self.dtype, device=self.d)
        self.BEVmap_res = torch.tensor(Map_config["map_res"], dtype=self.dtype, device=self.d)

        self.curvature_max = torch.tensor(self.steering_max / self.wheelbase, device=self.d, dtype=self.dtype)

        self.BEVmap_size_px = torch.tensor((self.BEVmap_size/self.BEVmap_res), device=self.d, dtype=torch.int32)
        self.BEVmap = torch.zeros((self.BEVmap_size_px.item(), self.BEVmap_size_px.item() )).to(self.d)
        self.BEVmap_height = torch.zeros_like(self.BEVmap)
        self.BEVmap_normal = torch.zeros((self.BEVmap_size_px.item(), self.BEVmap_size_px.item(), 3), dtype=self.dtype).to(self.d)

        self.GRAVITY = torch.tensor(9.8, dtype=self.dtype).to(self.d)
        
        self.NX = 17
        
        self.states = torch.zeros((self.M, self.K, self.T, self.NX), dtype=self.dtype).to(self.d)


    @torch.jit.export
    def set_BEV(self, BEVmap_height, BEVmap_normal):
        '''
        BEVmap_height, BEVmap_normal are robot-centric elevation and normal maps.
        BEV_center is the x,y,z coordinate at the center of the map. Technically this could just be x,y, but its easier to just remove it from all dims at once.
        '''
        assert BEVmap_height.shape[0] == self.BEVmap_size_px
        self.BEVmap_height = BEVmap_height
        self.BEVmap_normal = BEVmap_normal

    @torch.jit.export
    def get_states(self):
        return self.states

    ## remember, this function is called only once! If you have a single-step dynamics function, you will need to roll it out inside this function.
    def forward(self, state, controls):
        # unpack all values:
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

        steer = controls[...,0]
        throttle = controls[...,1]

        K = torch.tan(steer * self.steering_max)/self.wheelbase  # this is just a placeholder for curvature since steering correlates to curvature
        
        vx = throttle * self.throttle_to_wheelspeed

        dS = vx * self.dt

        wz = vx * K
        ay = (vx * wz)
        ay = torch.clamp(ay, -14, 14)
        wz = ay/torch.clamp(vx,1,25)

        yaw = yaw + self.dt * torch.cumsum(wz, dim=2)  # this is what the yaw will become
        
        cy = torch.cos(yaw)
        sy = torch.sin(yaw)

        x = x + torch.cumsum(dS * cy, dim=-1)
        y = y + torch.cumsum(dS * sy, dim=-1)

        img_X = torch.clamp( ((x + self.BEVmap_size*0.5) / self.BEVmap_res).to(dtype=torch.long, device=self.d), 0, self.BEVmap_size_px - 1)
        img_Y = torch.clamp( ((y + self.BEVmap_size*0.5) / self.BEVmap_res).to(dtype=torch.long, device=self.d), 0, self.BEVmap_size_px - 1)
        
        z = self.BEVmap_height[img_Y, img_X]
        normal = self.BEVmap_normal[img_Y, img_X] ## normal is a unit vector

        heading = torch.stack([cy, sy, torch.zeros_like(yaw)], dim=3) ## heading is a unit vector --ergo, all cross products will be unit vectors and don't need normalization

        # Calculate the cross product of the heading and normal vectors to get the vector perpendicular to both
        left = torch.cross(normal, heading)
        # Calculate the cross product of the right and normal vectors to get the vector perpendicular to both and facing upwards
        forward = torch.cross(left, normal)
        # Calculate the roll angle (rotation around the forward axis)
        roll = torch.asin(left[...,2])
        # Calculate the pitch angle (rotation around the right axis)
        pitch = -torch.asin(forward[...,2])

        wx[...,1:] = torch.diff(roll, dim=-1)/self.dt
        wy[...,1:] = torch.diff(pitch, dim=-1)/self.dt

        vy = torch.zeros_like(vx)
        vz = torch.zeros_like(vx)

        # ax[...,:-1] = torch.diff(vx, dim=-1)/self.dt
        ay = (vx * wz) + self.GRAVITY * torch.sin(roll) ## this is the Y acceleration in the inertial frame as would be reported by an accelerometer
        az = (-vx * wy) + self.GRAVITY * normal[...,2] ## this is the Z acc in the inertial frame as reported by an IMU
        # print(roll[0,0,0]*57.3, pitch[0,0,0]*57.3, normal[0,0,0])
        # pack all values: 
        self.states = torch.stack((x, y, z, roll, pitch, yaw, vx, vy, vz, ax, ay, az, wx, wy, wz, steer, throttle), dim=3)
        return self.states