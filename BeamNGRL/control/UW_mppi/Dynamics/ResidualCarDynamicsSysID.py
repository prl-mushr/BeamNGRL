import torch
# import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import numpy as np
import time
import os
import sys
from BeamNGRL.dynamics.utils.exp_utils import build_nets
from BeamNGRL.dynamics.utils.network_utils import load_model
from typing import Dict
import numba
from numba import jit
import atexit

def load_dyn_model(config, weights_path, tn_args: Dict = None):
    net, _ = build_nets(config, tn_args, model_weight_file=weights_path)
    net.eval()
    return net


## photogrammetry. Yes I'm old school.
@jit(nopython=True)
def sys_resampling(bins, cdf, perturb_out, perturb_in):
    cnt = 0
    idx = 0
    PARAM_K = perturb_out.shape[0]
    while idx < PARAM_K:
        if bins[idx] < cdf[cnt]:
            perturb_out[idx] = perturb_in[cnt]  # assign the particle
            idx += 1
        else:
            cnt += 1
    return perturb_out


class ResidualCarDynamics:
    """
    Class for Dynamics modelling
    """

    def __init__(
        self,
        Dynamics_config,
        Map_config,
        MPPI_config,
        model_weights_path=None,
        dtype=np.float32,
        device=cuda.Device(0),
    ):
        self.tn_args = {'device': torch.device("cuda"), 'dtype': torch.float32}

        self.throttle_to_wheelspeed = np.float32(Dynamics_config["throttle_to_wheelspeed"])
        self.steering_max = np.float32(Dynamics_config["steering_max"])

        self.dt_default = np.float32(Dynamics_config["dt"])
        self.dt = self.dt_default
        self.K = np.int32(MPPI_config["ROLLOUTS"])
        self.T = np.int32(MPPI_config["TIMESTEPS"])
        self.M = np.int32(MPPI_config["BINS"])
        self.NX = np.int32(17)
        self.NC = np.int32(2)

        self.BEVmap_size = np.float32(Map_config["map_size"])
        self.BEVmap_res = np.float32(Map_config["map_res"])
        self.BEVmap_size_px = np.int32(self.BEVmap_size / self.BEVmap_res)
        
        self.states = torch.zeros( (self.M, self.K, self.T, 17), dtype=torch.float32, device=torch.device('cuda:0'))

        ## pulled these values from: A Hybrid Hierarchical Rally Driver Model for Autonomous Vehicle Agile Maneuvering on Loose Surfaces
        self.D = np.float32(Dynamics_config["D"])
        self.B = np.float32(Dynamics_config["B"])
        self.C = np.float32(Dynamics_config["C"])
        self.lf = np.float32(Dynamics_config["lf"])
        self.lr = np.float32(Dynamics_config["lr"])
        self.Iz = np.float32(Dynamics_config["Iz"])
        self.LPF_tau = np.float32(Dynamics_config["LPF_tau"])
        self.LPF_st = np.float32(Dynamics_config["LPF_st"])
        self.LPF_th = np.float32(Dynamics_config["LPF_th"])
        self.res_coeff = np.float32(Dynamics_config["res_coeff"])
        self.drag_coeff = np.float32(Dynamics_config["drag_coeff"])

        self.car_l2 = np.float32(Dynamics_config["car_length"]/2)
        self.car_w2 = np.float32(Dynamics_config["car_width"]/2)
        self.cg_height = np.float32(Dynamics_config["cg_height"])

        self.dt_avg = 1e-3

        self.PARAM_NX = np.int32(15)
        self.PARAM_K =  np.int32(1024)
        self.NPM = 7 # D, res, drag, LPF_rp, LPF_st, LPF_th, Iz
        self.cost_total = torch.zeros(self.PARAM_K).to(device = self.tn_args['device'], dtype=self.tn_args['dtype'])
        self.traj_num_static = 200
        self.traj_num_dyn = 5
        self.static_decay_rate = 0.8
        self.param_cnt = 0
        self.min_D = 0.8
        self.ice_D = 0.3
        self.param_temp = torch.tensor(0.5).to(device=self.tn_args['device'], dtype=self.tn_args['dtype'])
        self.PARAM_NOISE_MU = torch.zeros(self.NPM).to(device=self.tn_args['device'], dtype=self.tn_args['dtype'])
        self.PARAM = torch.ones(self.NPM).to(device=self.tn_args['device'], dtype=self.tn_args['dtype'])
        self.PARAM[0] *= self.D
        self.PARAM[1] *= self.res_coeff
        self.PARAM[2] *= self.drag_coeff
        self.PARAM[3] *= self.LPF_tau
        self.PARAM[4] *= self.LPF_st
        self.PARAM[5] *= self.LPF_th
        self.PARAM[6] *= self.Iz
        self.avg_excitation = 0

        folder_name = 'BeamNGRL'

        # Check each directory in sys.path for the folder
        for path in sys.path:
            folder_path = os.path.join(path, folder_name)
            if os.path.exists(folder_path):
                break
        else:
            print("Did you forget to add BeamNGRL to your PYTHONPATH?")

        self.dtype = dtype
        self.device = cuda.Device(0)
        self.pycuda_ctx = self.device.retain_primary_context()
        self.pycuda_ctx.push()

        # Set grid and block dimensions
        self.block_dim = 32 #min(MPPI_config["ROLLOUTS"], 1024) # use 32 for jetson, use 1024 for RTX GPUs
        self.grid_dim = int(np.ceil(self.K / self.block_dim))

        file_path = '{}/control/UW_mppi/Dynamics/{}.cpp'.format(folder_path, Dynamics_config["type"])
        with open(file_path, 'r') as file:
            cuda_code = file.read()
        module = SourceModule(cuda_code)
        self.rollout = module.get_function("rollout")

        file_path = '{}/control/UW_mppi/Dynamics/{}.cpp'.format(folder_path, "param_est_slip3d")
        with open(file_path, 'r') as file:
            cuda_code = file.read()
        module = SourceModule(cuda_code)
        self.param_cost = module.get_function("param_cost")

        self.BEVmap_height = gpuarray.to_gpu(np.zeros((self.BEVmap_size_px, self.BEVmap_size_px), dtype=dtype) )
        self.BEVmap_normal = gpuarray.to_gpu(np.zeros((self.BEVmap_size_px, self.BEVmap_size_px, 3), dtype=dtype) )

        self.pycuda_ctx.pop()
        atexit.register(self.cleanup)
        self.mode = "static"

    def cleanup(self):
        try:
            current_context = cuda.Context.get_current()
            if current_context is not None:
                current_context.pop()
                print("PyCUDA context popped on exit.")
        except cuda.LogicError:
            pass

    def setup(self, mode):
        self.mode = mode
        self.PARAM_NOISE = torch.eye(self.NPM).to(device=self.tn_args['device'], dtype=self.tn_args['dtype'])
        if mode == "static":
            # self.PARAM_NOISE[3:, 3:] *= 0.5
            self.perturb = torch.zeros(self.PARAM_K, self.NPM).to(device=self.tn_args['device'], dtype=self.tn_args['dtype'])
            # sample uniformly from (0,1)*scale + shift
            self.perturb[:, 3:] = (
                torch.matmul(
                    torch.randn((self.PARAM_K, 4)).to(device=self.tn_args['device'], dtype=self.tn_args['dtype']), self.PARAM_NOISE[3:, 3:]
                ) 
            )  # scale and add mean
            self.perturb += self.PARAM_NOISE_MU
        elif mode == "dynamic":
            # self.PARAM_NOISE[0,0] *= 0.4
            # self.PARAM_NOISE[1,1] *= 0.05
            # self.PARAM_NOISE[2,2] *= 0.05
            self.perturb = torch.zeros(self.PARAM_K, self.NPM).to(device=self.tn_args['device'], dtype=self.tn_args['dtype'])
            # sample uniformly from (0,1)*scale + shift
            self.perturb[:, :3] = (
                torch.matmul(
                    torch.randn((self.PARAM_K, 3)).to(device=self.tn_args['device'], dtype=self.tn_args['dtype']), self.PARAM_NOISE[:3, :3]
                ) 
            )  # scale and add mean
            self.perturb += self.PARAM_NOISE_MU
            # self.PARAM_NOISE[0,0] *= 0.2
            # self.PARAM_NOISE[1,1] *= 0.1
            # self.PARAM_NOISE[2,2] *= 0.1
            ## PF vars:
            self.avg = 1.0 / self.PARAM_K
            self.weights = self.avg*torch.ones(self.PARAM_K).to(device=self.tn_args['device'], dtype=self.tn_args['dtype'])
            self.bins = np.arange(self.PARAM_K, dtype=np.float32)*self.avg

        self.param_cnt = 0
        self.PARAM_NOISE_inv = torch.inverse(self.PARAM_NOISE)

    ## TODO: these should be moved to utils if we don't end up switching to torch cpp extension later
    def tensor_to_gpuarray(self, tensor, dtype):
        return gpuarray.GPUArray(tensor.shape, dtype, gpudata = tensor.data_ptr())

    def gpuarray_to_tensor(self, gpa, out, dtype, npdtype):
        ## note: this function assumes you already have the location where you would like to copy the data.
        gpa_copy = self.tensor_to_gpuarray(out, dtype=npdtype)
        byte_size = gpa.itemsize * gpa.size
        cuda.memcpy_dtod(gpa_copy.gpudata, gpa.gpudata, byte_size)

    def set_BEV(self, BEVmap_height, BEVmap_normal):
        self.pycuda_ctx.push()
        self.BEVmap_height = self.tensor_to_gpuarray(BEVmap_height, np.float32)
        self.BEVmap_normal = self.tensor_to_gpuarray(BEVmap_normal, np.float32)
        cuda.Context.synchronize()
        self.pycuda_ctx.pop()

    def set_GT(self, gt_states, gt_controls):
        self.gt_states_torch = gt_states.clone()
        self.pycuda_ctx.push()
        self.gt_states = self.tensor_to_gpuarray(gt_states, np.float32) ## expecting array of shape TxNX
        self.gt_controls = self.tensor_to_gpuarray(gt_controls, np.float32)
        cuda.Context.synchronize()
        self.pycuda_ctx.pop()

    def get_states(self):
        return self.states

    def sample_params_static(self):
        self.perturb[:, :3] *= 0
        self.perturb[:,3:] =  torch.matmul(
                torch.randn((self.PARAM_K, 4)).to(device=self.tn_args['device'], dtype=self.tn_args['dtype']), self.PARAM_NOISE[3:, 3:]
            )          
        self.perturb += self.PARAM_NOISE_MU
        perturbed_params = self.PARAM + self.perturb ## perturbed params of shape K x NPM
        perturbed_params[:, 3:6] = torch.clamp(perturbed_params[:, 3:6], 0.05, 0.9)
        perturbed_params[:, 6] = torch.clamp(perturbed_params[:, 6], 0.01, 2.0)
        self.perturb = perturbed_params - self.PARAM
        action_cost = torch.abs(self.param_temp * torch.matmul(self.perturb, self.PARAM_NOISE_inv)) ## action cost of shape PARAM_K x NPM
        self.cost_total = torch.sum(action_cost, dim=1) # cost of shape PARAM_K. This is the perturbation cost actually, but I am re-using the total-cost variable.
        return perturbed_params

    def sample_params_dyn(self):
        self.perturb[:, 3:] *= 0
        self.perturb[:,:3] =  torch.matmul(
                torch.randn((self.PARAM_K, 3)).to(device=self.tn_args['device'], dtype=self.tn_args['dtype']), self.PARAM_NOISE[:3, :3]
            )          
        self.perturb += self.PARAM_NOISE_MU
        perturbed_params = self.PARAM + self.perturb ## perturbed params of shape K x NPM
        perturbed_params[:, 0] = torch.clamp(perturbed_params[:, 0], self.ice_D, 1.5) ## prevent parameters from taking nonsensical values
        perturbed_params[:, 1:3] = torch.clamp(perturbed_params[:, 1:3], 0.0, 0.1)
        self.perturb = perturbed_params - self.PARAM
        action_cost = torch.abs(self.param_temp * torch.matmul(self.perturb, self.PARAM_NOISE_inv)) ## action cost of shape PARAM_K x NPM
        self.cost_total = torch.sum(action_cost, dim=1) # cost of shape PARAM_K. This is the perturbation cost actually, but I am re-using the total-cost variable.
        return perturbed_params

    def update_params(self):
        if torch.any(torch.isnan(self.cost_total)):
            print("NAAAAN")
            return
        beta = torch.min(self.cost_total)
        cost_total_non_zero = torch.exp((-1 / self.param_temp) * (self.cost_total - beta))
        eta = torch.sum(cost_total_non_zero)
        omega = (1.0 / eta) * cost_total_non_zero ## this is just normalization
        self.PARAM = self.PARAM + (self.perturb * omega.view(-1, 1)).sum(dim=0)

    def sample_params_PF(self):
        cdf = torch.cumsum(self.weights, dim=0).cpu().numpy()
        r = np.random.uniform(0, self.avg)
        bins = self.bins * r
        perturb = torch.zeros_like(self.perturb).cpu().numpy()
        perturb = sys_resampling(bins, cdf, perturb, self.perturb.cpu().numpy())
        self.perturb = torch.from_numpy(perturb).to(device=self.tn_args['device'], dtype=self.tn_args['dtype'])
        self.perturb[:,:3] +=  torch.matmul(
                torch.randn((self.PARAM_K, 3)).to(device=self.tn_args['device'], dtype=self.tn_args['dtype']), self.PARAM_NOISE[:3, :3]
            )
        perturbed_params = self.PARAM + self.perturb ## perturbed params of shape K x NPM
        perturbed_params[:, 0] = torch.clamp(perturbed_params[:, 0], self.ice_D, 1.5) ## prevent parameters from taking nonsensical values
        perturbed_params[:, 1:3] = torch.clamp(perturbed_params[:, 1:3], 0.0, 0.1)
        self.perturb = perturbed_params - self.PARAM ## clipping
        return perturbed_params

    def update_params_PF(self):
        if torch.any(torch.isnan(self.cost_total)):
            # print("NAAAAN")
            return
        beta = torch.min(self.cost_total)
        cost_total_non_zero = torch.exp((-1 / self.param_temp) * (self.cost_total - beta))
        eta = torch.sum(cost_total_non_zero)
        self.weights = (1.0 / eta) * cost_total_non_zero ## this is just normalization
        delta_param = (self.perturb * self.weights.view(-1, 1)).sum(dim=0)
        self.perturb -= delta_param
        self.PARAM += delta_param

    def set_params(self):
        self.D = self.PARAM[0].cpu().numpy()
        self.drag_coeff = self.PARAM[1].cpu().numpy()
        self.res_coeff = self.PARAM[2].cpu().numpy()
        self.LPF_tau = self.PARAM[3].cpu().numpy()
        self.LPF_st = self.PARAM[4].cpu().numpy()
        self.LPF_th = self.PARAM[5].cpu().numpy()
        self.Iz = self.PARAM[6].cpu().numpy()

    def forward(self, state, controls):
        now = time.time()
        
        if self.mode == "static":
            if self.param_cnt % self.traj_num_static == 0:
                self.update_params()
                self.perturbed_params = self.sample_params_static()
                self.set_params()
                self.param_cnt = 0
                self.PARAM_NOISE *= self.static_decay_rate
            self.param_cnt += 1
        if self.mode == "dynamic":
            self.avg_excitation += torch.mean(torch.abs(self.gt_states_torch[...,14]*self.gt_states_torch[..., 6])/self.traj_num_dyn)
            if self.param_cnt % self.traj_num_dyn == 0:
                if self.avg_excitation > self.ice_D*9.81:
                    # self.update_params_PF()
                    self.update_params()
                self.avg_excitation = 0
                # self.perturbed_params = self.sample_params_PF()
                self.perturbed_params = self.sample_params_dyn()
                self.set_params()
                self.param_cnt = 0
            self.param_cnt += 1
        else:
            pass

        # begin pycuda context:
        self.pycuda_ctx.push()
        # transfer data from torch to pycuda arrays for forward rollout
        controls_ = self.tensor_to_gpuarray(controls.squeeze(0), np.float32)
        state_ = self.tensor_to_gpuarray(state.squeeze(0), np.float32)

        perturbed_params_ = self.tensor_to_gpuarray(self.perturbed_params, np.float32)
        cost_total_ = self.tensor_to_gpuarray(self.cost_total, np.float32)

        # sync context before we start kernels
        cuda.Context.synchronize()
        # Launch the sys-ID CUDA kernel
        self.param_cost(self.gt_states, self.gt_controls, self.BEVmap_height, self.BEVmap_normal, self.dt, self.PARAM_K, self.T, self.PARAM_NX, self.NC,
                self.D, self. B, self.C, self.lf, self.lr, self.Iz, self.throttle_to_wheelspeed, self.steering_max,
                self.BEVmap_size_px, self.BEVmap_res, self.BEVmap_size, self.car_l2, self.car_w2, self.cg_height, self.LPF_tau, self.res_coeff, self.drag_coeff,
                perturbed_params_, cost_total_, np.int32(self.NPM),
                block=(self.block_dim, 1, 1), grid=(self.grid_dim, 1))
        
        # Launch the forward dynamics CUDA kernel
        self.rollout(state_, controls_, self.BEVmap_height, self.BEVmap_normal, self.dt, self.K, self.T, self.NX, self.NC,
                self.D, self. B, self.C, self.lf, self.lr, self.Iz, self.throttle_to_wheelspeed, self.steering_max,
                self.BEVmap_size_px, self.BEVmap_res, self.BEVmap_size, self.car_l2, self.car_w2, self.cg_height, self.LPF_tau, self.LPF_st, self.LPF_th, self.res_coeff, self.drag_coeff,
                block=(self.block_dim, 1, 1), grid=(self.grid_dim, 1))

        cuda.Context.synchronize()

        self.gpuarray_to_tensor(cost_total_, self.cost_total, torch.float32, np.float32)
        self.gpuarray_to_tensor(state_, self.states, torch.float32, np.float32)
        # end pycuda context. Run pytorch model OUTSIDE this context
        self.pycuda_ctx.pop()
        dt = time.time() - now
        self.dt_avg = self.dt_avg*0.9 + dt*0.1
        # print(self.dt_avg*1e3)
        return self.states

    def forward_fixed(self, state, controls):
        now = time.time()

        # begin pycuda context:
        self.pycuda_ctx.push()
        # transfer data from torch to pycuda arrays for forward rollout
        controls_ = self.tensor_to_gpuarray(controls.squeeze(0), np.float32)
        state_ = self.tensor_to_gpuarray(state.squeeze(0), np.float32)

        cuda.Context.synchronize()
        # Launch the forward dynamics CUDA kernel
        self.rollout(state_, controls_, self.BEVmap_height, self.BEVmap_normal, self.dt, self.K, self.T, self.NX, self.NC,
                self.D, self. B, self.C, self.lf, self.lr, self.Iz, self.throttle_to_wheelspeed, self.steering_max,
                self.BEVmap_size_px, self.BEVmap_res, self.BEVmap_size, self.car_l2, self.car_w2, self.cg_height, self.LPF_tau, self.LPF_st, self.LPF_th, self.res_coeff, self.drag_coeff,
                block=(self.block_dim, 1, 1), grid=(self.grid_dim, 1))

        cuda.Context.synchronize()

        self.gpuarray_to_tensor(state_, self.states, torch.float32, np.float32)
        # end pycuda context. Run pytorch model OUTSIDE this context
        self.pycuda_ctx.pop()
        dt = time.time() - now
        self.dt_avg = self.dt_avg*0.9 + dt*0.1
        # print(self.dt_avg*1e3)
        return self.states