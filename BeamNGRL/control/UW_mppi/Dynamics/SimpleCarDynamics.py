import torch
import torch.nn as nn

class SimpleCarDynamics(torch.nn.Module):
    """
    Class for Dynamics modelling
    """
    def __init__(
        self,
        wheelbase = 0.5,
        speed_max = 17,
        steering_max = 1,
        dt = 0.02,
        BEVmap_size=64,
        BEVmap_res=0.25,
        ROLLOUTS=512,
        TIMESTEPS=32,
        BINS=1,
        BW = 4.0,
        dtype=torch.float32,
        device=torch.device("cuda"),
    ):

        super(SimpleCarDynamics, self).__init__()
        self.dtype = dtype
        self.d = device
        self.wheelbase = torch.tensor(wheelbase, device=self.d, dtype=self.dtype)
        self.speed_max = torch.tensor(speed_max, device=self.d, dtype=self.dtype)
        self.steering_max = torch.tensor(steering_max, device=self.d, dtype=self.dtype)
        self.dt = torch.tensor(dt, device=self.d, dtype=self.dtype)
        self.NU = 2
        self.NX = 17
        self.BW = torch.tensor(BW, device=self.d, dtype=self.dtype)

        self.throttle_to_wheelspeed = torch.tensor(self.speed_max, device=self.d, dtype=self.dtype)
        self.curvature_max = torch.tensor(self.steering_max / self.wheelbase, device=self.d, dtype=self.dtype)

        self.BEVmap_size = torch.tensor(BEVmap_size).to(self.d)
        self.BEVmap_res = torch.tensor(BEVmap_res).to(self.d)
        self.BEVmap_size_px = torch.tensor((self.BEVmap_size/self.BEVmap_res), device=self.d, dtype=torch.int32)
        self.BEVmap = torch.zeros((self.BEVmap_size_px.item(), self.BEVmap_size_px.item() )).to(self.d)
        self.BEVmap_height = torch.zeros_like(self.BEVmap)
        self.BEVmap_normal = torch.zeros((self.BEVmap_size_px.item(), self.BEVmap_size_px.item(), 3), dtype=self.dtype).to(self.d)
        self.BEVmap_center = torch.zeros(3, dtype=self.dtype).to(self.d)

        self.GRAVITY = torch.tensor(9.8, dtype=self.dtype).to(self.d)
        
        self.K = ROLLOUTS
        self.T = TIMESTEPS
        self.M = BINS
        
        self.states = torch.zeros((self.M, self.K, self.T, self.NX), dtype=self.dtype).to(self.d)

        sigma = torch.tensor(1, dtype=self.dtype, device=self.d)
        filter_size = 7
        kernel = torch.zeros(filter_size, dtype=self.dtype, device=self.d)
        m = filter_size//2
        for x in range(-m, m+1):
            X = torch.tensor(x, dtype=self.dtype)
            x1 = 2*torch.pi*(sigma**2)
            x2 = torch.exp(-(X**2)/(2* sigma**2))
            kernel[x+m] = (1/x1)*x2

        self.smoothing = nn.Conv1d(self.K, self.K, kernel_size=filter_size, padding=filter_size//2, bias=False).to(self.d)
        self.smoothing.weight.data[:, :, :] = kernel.view(1, 1, -1).repeat(self.K, self.K, 1).to(self.d)


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
    def get_states(self):
        return self.states

    ## remember, this function is called only once! If you have a single-step dynamics function, you will need to roll it out inside this function.
    def forward(self, state, perturbed_actions):
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

        controls = torch.clamp(state[..., 15:17] + self.BW*torch.cumsum(perturbed_actions.unsqueeze(dim=0) * self.dt, dim=-2), -1, 1) # last dimension is the NU channel!

        controls[...,1] = torch.clamp(controls[...,1], 0,0.5) ## car can't go in reverse

        perturbed_actions[:,1:,:] = torch.diff(controls - state[...,15:17], dim=-2).squeeze(dim=0)/(self.dt * self.BW)

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

        # ax[...,:-1] = self.smoothing(torch.diff(vx, dim=-1)/self.dt)
        ay = (vx * wz) + self.GRAVITY * torch.sin(roll) ## this is the Y acceleration in the inertial frame as would be reported by an accelerometer
        az = (-vx * wy) + self.GRAVITY * normal[...,2] ## this is the Z acc in the inertial frame as reported by an IMU
        # print(roll[0,0,0]*57.3, pitch[0,0,0]*57.3, normal[0,0,0])
        # pack all values: 
        self.states = torch.stack((x, y, z, roll, pitch, yaw, vx, vy, vz, ax, ay, az, wx, wy, wz, steer, throttle), dim=3)
        return self.states, perturbed_actions