import logging
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
import time

## this is a generalized MPPI module, with K control trajectories, M rollouts per control trajectory and T timesteps
## if you don't need M rollouts per traj, the code can be made more efficient.
## in general, making the code more efficient involves getting rid of flow-control (for loops, if/else conditions)
## if conditions create branches; if you already know that you're gonna be running in a particular setting,
## you should modify the module and "hardcode" that branch into the code, for instance, if you will need M rollouts, 
## remove all if conditions where it checks for them and just write what was in that block
## if you don't have a terminal cost, just comment out the block that checks for it, otherwise, inline it and so on

class MPPI(torch.nn.Module):
    """
    Model Predictive Path Integral control
    This implementation batch samples the trajectories and so scales well with the number of samples K.

    Implemented according to algorithm 2 in Williams et al., 2017
    'Information Theoretic MPC for Model-Based Reinforcement Learning',
    based off of https://github.com/ferreirafabio/mppi_pendulum
    """

    def __init__(self, nx, noise_sigma, num_samples=100, horizon=15, device="cuda:0",
                 terminal_state_cost=None,
                 lambda_=1.,
                 noise_mu=None,
                 u_min=None,
                 u_max=None,
                 u_init=None,
                 U_init=None,
                 u_scale=1,
                 u_per_command=1,
                 step_dependent_dynamics=False,
                 rollout_samples=1,
                 rollout_var_cost=0,
                 rollout_var_discount=0.95,
                 sample_null_action=False,
                 noise_abs_cost=False,
                 BEVmap_size=20,
                 BEVmap_res=0.4):
        """
        :param dynamics: function(state, action) -> next_state (K x nx) taking in batch state (K x nx) and action (K x nu)
        :param running_cost: function(state, action) -> cost (K) taking in batch state and action (same as dynamics)
        :param nx: state dimension
        :param noise_sigma: (nu x nu) control noise covariance (assume v_t ~ N(u_t, noise_sigma))
        :param num_samples: K, number of trajectories to sample
        :param horizon: T, length of each trajectory
        :param device: pytorch device
        :param terminal_state_cost: function(state) -> cost (K x 1) taking in batch state
        :param lambda_: temperature, positive scalar where larger values will allow more exploration
        :param noise_mu: (nu) control noise mean (used to bias control samples); defaults to zero mean
        :param u_min: (nu) minimum values for each dimension of control to pass into dynamics
        :param u_max: (nu) maximum values for each dimension of control to pass into dynamics
        :param u_init: (nu) what to initialize new end of trajectory control to be; defeaults to zero
        :param U_init: (T x nu) initial control sequence; defaults to noise
        :param step_dependent_dynamics: whether the passed in dynamics needs horizon step passed in (as 3rd arg)
        :param rollout_samples: M, number of state trajectories to rollout for each control trajectory
            (should be 1 for deterministic dynamics and more for networks that output a distribution)
        :param rollout_var_cost: Cost attached to the variance of costs across trajectory rollouts
        :param rollout_var_discount: Discount of variance cost over control horizon
        :param sample_null_action: Whether to explicitly sample a null action (bad for starting in a local minima)
        :param noise_abs_cost: Whether to use the absolute value of the action noise to avoid bias when all states have the same cost
        """
        super(MPPI, self).__init__()
        self.d = device
        self.dtype = torch.float
        self.K = num_samples  # N_SAMPLES
        self.T = horizon  # TIMESTEPS
        # dimensions of state and control
        self.nx = nx
        self.nu = 1 if len(noise_sigma.shape) == 0 else noise_sigma.shape[0]
        self.lambda_ = torch.tensor(lambda_).to(self.d)
        if noise_mu is None:
            noise_mu = torch.zeros(self.nu, dtype=self.dtype)

        if u_init is None:
            u_init = torch.zeros_like(noise_mu)

        # handle 1D edge case
        if self.nu == 1:
            noise_mu = noise_mu.view(-1)
            noise_sigma = noise_sigma.view(-1, 1)

        self.noise_mu = noise_mu.to(self.d)
        self.noise_sigma = noise_sigma.to(self.d)
        self.noise_sigma_inv = torch.inverse(self.noise_sigma).to(self.d)
        self.noise = torch.matmul(torch.randn((self.K, self.T, self.nu), device = self.d), self.noise_sigma) + self.noise_mu  # scale and add mean
        self.noise_dist = MultivariateNormal(self.noise_mu, covariance_matrix=self.noise_sigma)

        self.u_per_command = u_per_command
        # T x nu control sequence
        self.U = U_init
        self.u_init = u_init.to(self.d)

        if self.U is None:
            self.U = torch.randn((self.T, self.nu)).to(self.d)

        self.terminal_state_cost = terminal_state_cost
        self.sample_null_action = sample_null_action
        self.noise_abs_cost = noise_abs_cost

        # handling dynamics networks that output a distribution (take multiple trajectory samples)
        self.M = rollout_samples
        self.rollout_var_cost = rollout_var_cost
        self.rollout_var_discount = rollout_var_discount

        # sampled results from last command
        self.cost_total = torch.zeros(self.K, device = self.d)
        self.cost_total_non_zero = torch.zeros(self.K, device = self.d)
        self.omega = torch.zeros(self.K, device = self.d)
        self.states = torch.zeros((self.M, self.K, self.T, self.nx), device = self.d)
        self.actions = torch.zeros((1, self.K, self.T, self.nu), device = self.d)
        t = torch.arange(0,self.T)
        self.rollout_discount = (self.rollout_var_discount ** t).to(self.d)

        ## extra variables that are specific to your problem statement:
        self.goal_state = torch.zeros(self.nx).to(self.d)  # possible goal state
        self.BEVmap_size = torch.tensor(BEVmap_size).to(self.d)
        self.BEVmap_res = torch.tensor(BEVmap_res).to(self.d)
        assert self.BEVmap_res > 0
        self.BEVmap_size_px = torch.tensor((self.BEVmap_size/self.BEVmap_res), device=self.d, dtype=torch.int32)
        self.BEVmap = torch.zeros((self.BEVmap_size_px.item(), self.BEVmap_size_px.item() )).to(self.d)
        self.BEVmap_center = torch.zeros(2).to(self.d)
        ## dynamics model-specific variables:
        self.B = torch.tensor(2.58).to(self.d)
        self.C = torch.tensor(1.2).to(self.d)
        self.D = torch.tensor(9.8 * 0.9).to(self.d)
        self.lf = torch.tensor(1.4).to(self.d)
        self.lr = torch.tensor(1.4).to(self.d)
        self.Iz = torch.tensor(1).to(self.d)

        self.steering_max = torch.tensor(25/57.3).to(self.d)
        self.wheelspeed_max = torch.tensor(17.0).to(self.d)
        self.dt_var = torch.tensor(0.05).to(self.d)#torch.arange(0.02, 0.1, 0.08/self.T).to(self.d)
        self.dt = torch.tensor(0.05).to(self.d)
        self.gravity = torch.tensor(9.8).to(self.d)
        self.last_action = torch.zeros((self.u_per_command, 2)).to(self.d)
        self.curvature_max = torch.tan(self.steering_max)/ (self.lf + self.lr)
        self.max_speed = torch.tensor(20.0).to(self.d)

    def reset(self):
        """
        Clear controller state after finishing a trial
        """
        self.U = torch.randn((self.T, self.nu))
        self.last_action = torch.zeros((self.u_per_command, 2)).to(self.d)

    def forward(self, state):
        """
        :param state: (nx) or (K x nx) current state, or samples of states (for propagating a distribution of states)
        :returns action: (nu) best action
        """
        # shift command 1 time step
        self.U = torch.roll(self.U, -1, dims=0)
        self.U[-1] = self.u_init
        state[15:17] = self.last_action[0]
        self.dt_var = self.dt# * torch.tensor(10.0)/ torch.clamp(state[6], 8, 10)
        delta_action = self._command(state)
        action = delta_action*self.dt + self.last_action
        action = torch.clamp(action, -1, 1)
        self.last_action = torch.clone(action)
        return action

    def _command(self, state):
        cost_total = self._compute_total_cost_batch(state)

        beta = torch.min(cost_total)
        self.cost_total_non_zero = torch.exp((-1/self.lambda_) * (cost_total - beta))

        eta = torch.sum(self.cost_total_non_zero)
        self.omega = (1. / eta) * self.cost_total_non_zero

        for t in range(self.T):
            self.U[t] = self.U[t] + torch.sum(self.omega.view(-1, 1) * self.noise[:, t], dim=0)
        return self.U[:self.u_per_command]

    def _compute_total_cost_batch(self, state):
        # parallelize sampling across trajectories
        # resample noise each time we take an action
        # self.noise = torch.matmul(torch.randn((self.K, self.T, self.nu), device = self.d), self.noise_sigma) + self.noise_mu  # scale and add mean
        self.noise = self.noise_dist.sample((self.K, self.T))
        # broadcast own control to noise over samples; now it's K x T x nu
        perturbed_action = self.U + self.noise

        action_cost = self.lambda_ * torch.matmul(self.noise, self.noise_sigma_inv)

        self._compute_rollout_costs(state, perturbed_action)

        # action perturbation cost
        perturbation_cost = torch.sum(self.U * action_cost, dim=(1, 2))

        self.cost_total = self.cost_total + perturbation_cost.to(self.d)
        return self.cost_total

    def _compute_rollout_costs(self, _state, perturbed_actions):
        ## M bins per control traj, K rollouts.

        # state = _state.view(1, -1).repeat(self.M, self.K, 1)
        # for t in range(self.T):
        #     ## M possible bins for controls
        #     u = perturbed_actions[:, t].repeat(self.M, 1, 1).to(self.dtype)
        #     state = self.dynamics(state, u, t)
        #     self.states[:,:,t,:] = state
        #     cost_samples += self.running_cost(state)
        # terminal_cost = self.terminal_cost(self.states)

        self.states = _state.view(1, -1).repeat(self.M, self.K, self.T, 1)
        self.states = self.vectorized_dynamics(self.states, perturbed_actions)
        cost_samples, terminal_cost = self.cost_vectorized(self.states)
        ## terminal cost is not really needed since you could just do that in the running-cost cost function!
        ## dimension 0 is self.M !
        # print(cost_samples.var(dim=0).shape)
        self.cost_total = torch.sum(cost_samples.mean(dim=0), dim=1) + terminal_cost.mean(dim=0)

    @torch.jit.export
    def set_goal(self, goal_state):
        self.goal_state = goal_state[:2]

    @torch.jit.export
    def set_BEV(self, BEV_color, BEV_heght, BEV_segmt, BEV_path, BEV_center):
        assert BEV_color.shape[0] == self.BEVmap_size_px
        self.BEVmap = torch.tensor(BEV_path[:,:,0], dtype=self.dtype).to(self.d)/255.0
        self.BEVmap_center = BEV_center  # translate the state into the center of the costmap.

    @torch.jit.export
    def get_states(self):
        return self.states

    def dynamics(self, state, perturbed_actions, t:int):
        x = state[:, :, 0]
        y = state[:, :, 1]
        z = state[:, :, 2]
        roll = state[:, :, 3]
        pitch = state[:, :, 4]
        yaw = state[:, :, 5]
        vx = state[:, :,6]
        vy = state[:, :,7]
        vz = state[:, :,8]
        ax = state[:, :,9]
        ay = state[:, :,10]
        az = state[:, :,11]
        gx = state[:, :,12]
        gy = state[:, :,13]
        gz = state[:, :,14]

        u0 = torch.clamp(state[:, :, 15] + perturbed_actions[:,:,0] * self.dt_var, -1, 1)
        u1 = torch.clamp(state[:, :, 16] + perturbed_actions[:,:,1] * self.dt_var, -1, 1)
        
        v = torch.sqrt(vx**2 + vy**2)
        Frx = u1*torch.tensor(5)
        pos_index = torch.where(Frx>0)
        Frx[pos_index] = torch.clamp(Frx[pos_index]*5/torch.clamp(v[pos_index],5,30),0,5)

        delta = u0*self.steering_max
        alphaf = delta - torch.atan2(gz*self.lf + vy, vx) 
        alphar = torch.atan2(gz*self.lr - vy, vx)
        Fry = self.D*torch.sin(self.C*torch.atan(self.B*alphar))
        Ffy = self.D*torch.sin(self.C*torch.atan(self.B*alphaf))
        ax = (Frx - Ffy*torch.sin(delta) + vy*gz) + self.gravity*torch.sin(pitch)
        ay = (Fry + Ffy*torch.cos(delta) - vx*gz) - self.gravity*torch.sin(roll)
        vx = vx + ax*self.dt_var
        vy = vy + ay*self.dt_var
        gz = gz + self.dt_var*(Ffy*self.lf*torch.cos(delta) - Fry*self.lr)/self.Iz
        x = x + (torch.cos(yaw)*vx - torch.sin(yaw)*vy)*self.dt_var
        y = y + (torch.sin(yaw)*vx + torch.cos(yaw)*vy)*self.dt_var
        yaw = yaw + self.dt_var*gz

        return torch.stack((x, y, z, roll, pitch, yaw, vx, vy, vz, ax, ay, az, gx, gy, gz, u0, u1), dim=2)

    def vectorized_dynamics(self, state, perturbed_actions):
        x = state[:, :, :, 0]
        y = state[:, :, :, 1]
        z = state[:, :, :, 2]

        roll = state[:, :, :, 3]
        pitch = state[:, :, :, 4]
        yaw = state[:, :, :, 5]

        vx = state[:, :, :,6]
        vy = state[:, :, :,7]
        vz = state[:, :, :,8]

        ax = state[:, :, :,9]
        ay = state[:, :, :,10]
        az = state[:, :, :,11]

        gx = state[:, :, :,12]
        gy = state[:, :, :,13]
        gz = state[:, :, :,14]

        u0 = state[:, :, :, 15] + torch.cumsum(perturbed_actions[:,:, 0].unsqueeze(dim=0) * self.dt, dim=2)
        u1 = state[:, :, :, 16] + torch.cumsum(perturbed_actions[:,:, 1].unsqueeze(dim=0) * self.dt, dim=2)

        curvature = u0 * self.curvature_max  # this is just a placeholder for curvature since steering correlates to curvature
        
        pos_index = torch.where(u1 > 0)
        ax = u1
        ax[pos_index] = torch.clamp(ax[pos_index]*5/torch.abs(vx[pos_index]),0,5)
        
        vx = vx + torch.cumsum(ax * self.dt, dim=2)
        gz = vx * curvature

        ay = vx * gz

        yaw = yaw + torch.cumsum(gz * self.dt, dim=2)  # this is what the yaw will become
        x = x + torch.cumsum(vx * torch.cos(yaw) * self.dt, dim=2)
        y = y + torch.cumsum(vx * torch.sin(yaw) * self.dt, dim=2)

        return torch.stack(
            (x, y, z,
             roll, pitch, yaw,
             vx, vy, vz,
             ax, ay, az,
             gx, gy, gz,
             u0, u1),
            dim=3).to(dtype=self.dtype)

    def running_cost(self, state):
        x = state[:, :, 0]
        y = state[:, :, 1]
        vx =state[:, :, 6]
        ay =state[:, :, 9]

        img_X = ((x + self.BEVmap_size*0.5) / self.BEVmap_res).to(dtype=torch.long, device=self.d)
        img_Y = ((y + self.BEVmap_size*0.5) / self.BEVmap_res).to(dtype=torch.long, device=self.d)
        state_cost = self.BEVmap[img_Y, img_X]
        state_cost *= state_cost
        state_cost[torch.where(state_cost>=0.9)] = 100
        vel_cost = torch.abs(self.max_speed - vx)/self.max_speed
        vel_cost = torch.sqrt(vel_cost)
        accel_cost = (ay*0.1)**2
        accel_cost[torch.where(torch.abs(ay) > 5)] = 100
        return 0.05*vel_cost + state_cost + accel_cost

    def cost_vectorized(self, state):
        x = state[:, :, :, 0]
        y = state[:, :, :, 1]
        vx =state[:, :, :, 6]
        ay =state[:, :, :, 9]

        img_X = ((x + self.BEVmap_size*0.5) / self.BEVmap_res).to(dtype=torch.long, device=self.d)
        img_Y = ((y + self.BEVmap_size*0.5) / self.BEVmap_res).to(dtype=torch.long, device=self.d)
        state_cost = self.BEVmap[img_Y, img_X]
        state_cost *= state_cost
        state_cost[torch.where(state_cost>=0.9)] = torch.tensor(1000.0, dtype=self.dtype)
        vel_cost = torch.abs(self.max_speed - vx)/self.max_speed
        vel_cost = torch.sqrt(vel_cost)
        accel_cost = (ay*0.1)**2
        accel_cost[torch.where(torch.abs(ay) > 5)] = torch.tensor(100.0, dtype=self.dtype)
        terminal_cost = torch.linalg.norm(state[:,:,-1,:2] - self.goal_state.unsqueeze(dim=0), dim=2)
        return 0.5*vel_cost + state_cost + accel_cost, terminal_cost

    def terminal_cost(self, state):
        return torch.linalg.norm(state[0,:,-1,:2] - self.goal_state, dim=1)



if __name__ == '__main__':
    dtype = torch.float
    d = torch.device("cuda")
    ns = torch.zeros((2,2), device=d, dtype=dtype)
    ns[0,0] = 0.05
    ns[1,1] = 0.2

    BEVmap = torch.zeros((50, 50)).to(d)
    BEVmap_center = torch.zeros(2)

    state = torch.zeros(17).to(d)
    goal_state = torch.zeros(2).to(d)

    ## make sure state is a torch tensor on the gpu!
    with torch.no_grad():
        my_module = MPPI(nx=17, noise_sigma=ns, num_samples=512, horizon=16, lambda_=0.1, rollout_samples = 1)

        sm = torch.jit.script(my_module)
        sm.save("MPPI.pt")

        sm.eval()
        sm.set_goal(goal_state)
        # sm.set_BEVmap(BEVmap, BEVmap_center)

        for i in range(int(1e2)):
            output = sm(state)
        print("begin")
        now = time.time()
        for i in range(int(1e2)):
            output = sm(state)
        dt = (time.time() - now)*1e-2
        print("Dt:", dt)