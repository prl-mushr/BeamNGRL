import torch
import time

'''
1) points: use lowercase
2) explicit variable names
'''
class MPPI(torch.nn.Module):
    """
    Model Predictive Path Integral control
    Implemented according to algorithm 2 in Williams et al., 2017
    'Information Theoretic MPC for Model-Based Reinforcement Learning',
    """

    def __init__(
        self,
        Dynamics,
        Costs,
        CTRL_NOISE,
        CTRL_NOISE_MU=None,
        device="cuda:0",
        lambda_=1.0,
        u_final=None,
        u_per_command=1,
        num_optimizations=1,
        Rollout_bins=1,
    ):
        """
        :param Dynamics: nn module (object) that provides a forward function for propagating the dynamics forward in time
        :param Costs: nn module (object) that provides a forward function for evaluating the costs of state trajectories
        :param NX: state dimension
        :param CTRL_NOISE: (nu x nu) control noise covariance (assume v_t ~ N(u_t, CTRL_NOISE))
        :param N_SAMPLES: K, number of trajectories to sample
        :param TIMESTEPS: T, length of each trajectory
        :param device: pytorch device
        :param lambda_: temperature, positive scalar where larger values will allow more exploration
        :param u_init: (nu) what to initialize new end of trajectory control to be; defeaults to zero
        """
        super(MPPI, self).__init__()
        self.d = device
        self.dtype = torch.float

        self.Dynamics = Dynamics
        self.Costs = Costs
        
        self.K = self.Dynamics.K
        self.T = self.Dynamics.T
        self.M = self.Dynamics.M
        # dimensions of state and control
        self.nu = 1 if len(CTRL_NOISE.shape) == 0 else CTRL_NOISE.shape[0]
        self.lambda_ = torch.tensor(lambda_).to(self.d).to(dtype=self.dtype)
        if CTRL_NOISE_MU is None:
            CTRL_NOISE_MU = torch.zeros(self.nu, dtype=self.dtype)

        self.CTRL_NOISE_MU = CTRL_NOISE_MU.to(self.d)
        self.CTRL_NOISE = CTRL_NOISE.to(self.d).to(dtype=self.dtype)
        self.CTRL_NOISE_inv = torch.inverse(self.CTRL_NOISE).to(self.d)
        ## for torchscript we have to initialize these things to same shape and size as what we'll use later
        self.noise = (
            torch.matmul(
                torch.randn((self.K, self.T, self.nu), device=self.d, dtype=self.dtype),
                self.CTRL_NOISE,
            )
            + self.CTRL_NOISE_MU
        )  # scale and add mean

        self.u_per_command = u_per_command
        # T x nu control sequence
        if u_final is None:
            u_final = torch.zeros(self.nu, dtype=self.dtype)
        self.u_final = u_final.to(self.d)

        self.U = torch.zeros((self.T, self.nu), dtype=self.dtype).to(self.d)
        self.perturbed_actions = torch.zeros(self.K, self.T, self.nu, device=self.d, dtype=self.dtype)
        # handling dynamics models that output a distribution (take multiple trajectory samples)

        # sampled results from last command
        self.cost_total = torch.zeros(self.K, device=self.d, dtype=self.dtype)
        self.cost_total_non_zero = torch.zeros(self.K, device=self.d, dtype=self.dtype)
        self.omega = torch.zeros(self.K, device=self.d, dtype=self.dtype)
        self.num_optimizations = num_optimizations


    @torch.jit.export
    def reset(self, U=None):
        """
        Clear controller state after finishing a trial
        """
        if U is not None:
            self.U = U.to(device=self.d).to(dtype=self.dtype)
        else:
            self.U = torch.randn((self.T, self.nu), dtype=self.dtype, device=self.d)

    def forward(self, state):
        """
        :param: state
        :returns: best actions
        """
        ## shift command 1 time step
        self.U = torch.roll(self.U, self.u_per_command, dims=0)
        self.U[-self.u_per_command : , :] = self.U[-self.u_per_command,:] # repeat last control
        # for i in range(self.num_optimizations):
        self.optimize(state)
        return self.U[: self.u_per_command]

    def optimize(self, state):
        """
        :param: state
        :returns: best set of actions
        """
        self.compute_total_cost_batch(state)

        beta = torch.min(self.cost_total)
        self.cost_total_non_zero = torch.exp((-1 / self.lambda_) * (self.cost_total - beta))

        eta = torch.sum(self.cost_total_non_zero)
        self.omega = (1.0 / eta) * self.cost_total_non_zero

        self.U = self.U + (self.omega.view(-1, 1, 1) * self.noise).sum(dim=0)

    def compute_total_cost_batch(self, state):
        # parallelize sampling across trajectories
        # resample noise each time we take an action
        self.noise = (
            torch.matmul(
                torch.randn((self.K, self.T, self.nu), device=self.d, dtype=self.dtype), self.CTRL_NOISE
            )
            + self.CTRL_NOISE_MU
        )  # scale and add mean
        # broadcast own control to noise over samples; now it's K x T x nu
        perturbed_action = self.U + self.noise
        ## compute the costs:
        self.compute_rollout_costs(state, perturbed_action)
        # # the dynamics propagation may have imposed constraints on the controls, so we find the actual noise used
        self.noise = self.perturbed_actions - self.U

        action_cost = self.lambda_ * torch.matmul(self.noise, self.CTRL_NOISE_inv)
        ## action perturbation cost
        perturbation_cost = torch.sum(self.U * action_cost, dim=(1, 2))
        ## Evaluate total cost:
        self.cost_total = self.cost_total + perturbation_cost.to(self.d)

    def compute_rollout_costs(self, _state, perturbed_actions):
        ## All the states are initialized as copies of the current state
        ## M bins per control traj, K control trajectories, T timesteps, NX states
        states = _state.view(1, -1).repeat(self.M, self.K, self.T, 1)
        ## update all the states using the dynamics function
        states, self.perturbed_actions = self.Dynamics.forward(states, perturbed_actions)
        ## Evaluate costs on STATES with dimensions M x K x T x NX.
        ## Including the terminal costs in here is YOUR own responsibility!
        self.cost_total = self.Costs.forward(states)
