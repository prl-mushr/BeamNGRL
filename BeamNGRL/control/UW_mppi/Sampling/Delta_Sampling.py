import torch
import torch.nn as nn


class Delta_Sampling(torch.nn.Module):
    """
    Class for Dynamics modelling
    """
    def __init__(
        self,
        sampling_config,
        MPPI_config,
        dtype=torch.float32,
        device=torch.device("cuda"),
    ):
        super(Delta_Sampling, self).__init__()
        self.dtype = dtype
        self.d = device

        self.nu = sampling_config["control_dim"]
        self.K = MPPI_config["ROLLOUTS"]
        self.T = MPPI_config["TIMESTEPS"]
        self.M = MPPI_config["BINS"]

        self.temperature = torch.tensor(sampling_config["temperature"], dtype=self.dtype, device = self.d)
        self.scaled_dt = torch.tensor(sampling_config["scaled_dt"], dtype=self.dtype, device = self.d)

        self.CTRL_NOISE = torch.zeros((self.nu, self.nu), device=self.d, dtype=self.dtype)
        self.CTRL_NOISE[0,0] = float(sampling_config["noise_0"])
        self.CTRL_NOISE[1,1] = float(sampling_config["noise_1"])

        self.CTRL_NOISE_inv = torch.inverse(self.CTRL_NOISE)
        self.CTRL_NOISE_MU = torch.zeros(self.nu, dtype=self.dtype, device=self.d)

        ## for torchscript we have to initialize these things to same shape and size as what we'll use later
        torch.manual_seed(0)
        self.noise = (
            torch.matmul(
                torch.randn((self.K, self.T, self.nu), device=self.d, dtype=self.dtype),
                self.CTRL_NOISE,
            )
            + self.CTRL_NOISE_MU
        )  # scale and add mean

        self.max_thr = torch.tensor(sampling_config["max_thr"], dtype=self.dtype, device = self.d)
        self.min_thr = torch.tensor(sampling_config["min_thr"], dtype=self.dtype, device = self.d)

    def sample(self, state, U):
        '''
        sampling is done in the delta control space
        add this to the previous delta_controls
        integrate delta_controls and add previous controls to get controls
        find new noise after clamping
        find perturbation cost
        return controls, perturbation cost
        '''
        self.noise = (
            torch.matmul(
                torch.randn((self.K, self.T, self.nu), device=self.d, dtype=self.dtype), self.CTRL_NOISE
            )
            + self.CTRL_NOISE_MU
        )  # scale and add mean

        perturbed_actions = U + self.noise

        controls = torch.clamp(state[..., 15:17] + (self.scaled_dt)*torch.cumsum(perturbed_actions.unsqueeze(dim=0), dim=-2), -1, 1)
        controls[...,1] = torch.clamp(controls[...,1], self.min_thr, self.max_thr) ## car can't go in reverse, can't have more than 50 % speed

        perturbed_actions[:,1:,:] = torch.diff(controls - state[...,15:17], dim=-2).squeeze(dim=0)/(self.scaled_dt)

        self.noise = perturbed_actions - U

        action_cost = self.temperature * torch.matmul(self.noise, self.CTRL_NOISE_inv)
        perturbation_cost = torch.sum(U * action_cost, dim=(1, 2))

        return controls, perturbation_cost

    def update_control(self, cost_total, U, state):
        '''
        find total cost such that the minimum of total cost is not 0
        find the weighting for all the K samples
        update the delta controls (weighted average)
        integrate delta controls and add previous controls to obtain the applied controls
        return controls, delta_controls
        '''
        beta = torch.min(cost_total)
        cost_total_non_zero = torch.exp((-1 / self.temperature) * (cost_total - beta))

        eta = torch.sum(cost_total_non_zero)
        omega = (1.0 / eta) * cost_total_non_zero

        U = U + (omega.view(-1, 1, 1) * self.noise).sum(dim=0)
        ## you'd return just "U" if this was a standard control space sampling scheme
        controls = torch.clamp(state[15:17] + self.scaled_dt*torch.cumsum(U, dim=-2), -1, 1)
        # controls[1] = torch.clamp(controls[1], 0, 0.5)
        return controls, U
