import torch
import torch.nn as nn


class MPPI(nn.Module):
    """
    Model Predictive Path Integral control
    Implemented according to algorithm 2 in Williams et al., 2017
    'Information Theoretic MPC for Model-Based Reinforcement Learning',
    """

    def __init__(
        self,
        dynamics_func,
        cost_func,
        ctrl_sigma,
        ctrl_mean=None,
        n_ctrl_samples=128,
        horizon=16,
        lambda_=1.0,
        dt=0.04,
        u_final=None,
        u_per_command=1,
        num_optimizations=1,
        n_state_samples=1,
        device="cuda:0",
        dtype=torch.float32,
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
        super().__init__()

        tn_args = {'device': device, 'dtype': dtype}
        self.tn_args = tn_args
        self.device = device
        self.dtype = dtype

        self.num_optimizations = num_optimizations
        self.n_ctrl_samples = n_ctrl_samples
        self.n_state_samples = n_state_samples
        self.horiz = horizon
        self.dt = dt

        self.state_dim = dynamics_func.NX
        self.ctrl_dim = 1 if len(ctrl_sigma.shape) == 0 else ctrl_sigma.shape[0]

        self.lambda_ = torch.tensor(lambda_).to(**tn_args)
        if ctrl_mean is None:
            ctrl_mean = torch.zeros(self.ctrl_dim, dtype=self.dtype)

        self.ctrl_mean = ctrl_mean.to(device)
        self.ctrl_sigma = ctrl_sigma.to(**tn_args)
        self.ctrl_sigma_inv = torch.inverse(self.ctrl_sigma)

        ## for torchscript we have to initialize these things to same shape and size as what we'll use later
        self.ctrl_eps = (
            torch.matmul(
                torch.randn(n_ctrl_samples, horizon, self.ctrl_dim).to(**tn_args),
                self.ctrl_sigma_inv,
            )
            + self.ctrl_mean
        )

        self.u_per_command = u_per_command
        if u_final is None:
            u_final = torch.zeros(self.ctrl_dim, dtype=self.dtype)
        self.u_final = u_final.to(device)

        self.U = torch.randn((horizon, self.ctrl_dim), dtype=dtype).to(device)
        self.last_action = torch.zeros((u_per_command, 2)).to(**tn_args)

        # sampled results from last command
        self.cost_total = torch.zeros(n_ctrl_samples).to(**tn_args)
        self.cost_total_non_zero = torch.zeros(n_ctrl_samples).to(**tn_args)
        self.omega = torch.zeros(n_ctrl_samples).to(**tn_args)

        self.dynamics_func = dynamics_func
        self.cost_func = cost_func

    @torch.jit.export
    def reset(self, U=None):
        """
        Clear controller state after finishing a trial
        """
        if U is not None:
            self.U = U.to(**self.tn_args)
        else:
            self.U = torch.randn((self.horiz, self.ctrl_dim),
                                 dtype=self.dtype, device=self.device)

    def forward(self, state):
        """
        :param: state
        :returns: best actions
        """

        # FIXME: shift operation should be after action selection
        self.U = torch.roll(self.U, -1, dims=0)
        self.U[-1] = self.u_final

        for i in range(self.num_optimizations):
            delta_action = self.optimize(state)

        action = delta_action*self.dt + self.last_action
        action = torch.clamp(action, -1, 1)
        self.last_action = torch.clone(action)
        return action

    def optimize(self, state):
        """
        :param: state
        :returns: best set of actions
        """
        # Sample
        ctrl_samples = self.sample_controls()

        # Rollout
        states = self.rollout_dynamics(state, ctrl_samples)

        # Cost eval.
        cost_total = self.compute_total_cost(states, ctrl_samples)

        # Update ctrl param.
        self.U = self.update_ctrl_params(cost_total)

        return self.U[: self.u_per_command]

    def sample_controls(self):
        self.ctrl_eps = (
            torch.matmul(
                torch.randn(
                    (self.n_ctrl_samples, self.horiz, self.ctrl_dim),
                    device=self.device), self.ctrl_sigma
            )
            + self.ctrl_mean
        )
        ctrl_samples = self.U + self.ctrl_eps
        return ctrl_samples

    def rollout_dynamics(self, state, controls):
        # FIXME: why repeat across horizon?
        states = state.view(1, -1).repeat(
            self.n_state_samples,
            self.n_ctrl_samples,
            self.horiz,
            1,
        )
        states = self.dynamics_func(states, controls)
        self.set_rollouts(states, controls)
        return states

    def set_rollouts(self, states, controls):
        self.rollout_states = states
        self.rollout_controls = controls

    def compute_total_cost(self, states, ctrl_samples):
        rollout_costs = self.compute_rollout_costs(states, ctrl_samples)

        # FIXME: these should be separate from cost
        action_cost = self.lambda_ * torch.matmul(ctrl_samples - self.U, self.ctrl_sigma_inv)
        perturbation_cost = torch.sum(self.U * action_cost, dim=(1, 2))

        cost_total = rollout_costs + perturbation_cost.to(self.device)
        return cost_total

    def compute_rollout_costs(self, states, controls):
        cost_total = self.cost_func(states, controls)
        return cost_total

    def update_ctrl_params(self, costs):
        beta = torch.min(costs)
        cost_total_non_zero = torch.exp((-1 / self.lambda_) * (costs - beta))
        eta = torch.sum(cost_total_non_zero)
        omega = (1. / eta) * cost_total_non_zero
        return self.U + (omega.view(-1, 1, 1) * self.ctrl_eps).sum(dim=0)