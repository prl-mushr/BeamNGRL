import torch
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
import atexit

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
        if model_weights_path is not None:
            self.dyn_model = self.load_dyn_model(Dynamics_config, model_weights_path, self.tn_args)

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


        self.crop_size = int(Dynamics_config["patch_size"]/self.BEVmap_res)
        self.states = torch.zeros( (self.M, self.K, self.T, 17), dtype=torch.float32, device=torch.device('cuda:0'))
        self.bev_input = torch.zeros((self.K*self.T, self.crop_size, self.crop_size)).to(device=torch.device("cuda"), dtype=torch.float32) ## a lot of compute time is wasted producing this "empty" array every "timestep"
        self.dummy_input = torch.zeros((self.crop_size,self.crop_size,self.K*self.T)).to(device=torch.device("cuda"), dtype=torch.float32)

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

        x = self.device.L2_CACHE_SIZE/(self.crop_size*self.crop_size*4) # 4 corresponds to 4 bytes in float32
        self.max_threads = 4096 # int(pow(2, np.ceil(np.log(x)/np.log(2)))) ## round up to nearest power of 2. 10000 --> 16384, 3000 --> 4096. Exceeding the cache limit by a small margin seems to be "fine"
        self.crop_block_size = (self.crop_size, self.crop_size, 1)
        self.output_images = gpuarray.zeros((self.crop_size, self.crop_size, self.K*self.T), dtype=np.float32)
        self.crop_grid_size = (int((self.output_images.shape[0] + self.crop_block_size[0] - 1) // self.crop_block_size[0]), 
                               int((self.output_images.shape[1] + self.crop_block_size[1] - 1) // self.crop_block_size[1]), 
                               int(self.max_threads))
        self.batch_size = np.int32(self.K*self.T/self.max_threads)

        file_path = '{}/control/UW_mppi/Dynamics/{}.cpp'.format(folder_path, Dynamics_config["type"])
        print("loading physics model: ", Dynamics_config["type"])
        with open(file_path, 'r') as file:
            cuda_code = file.read()
        module = SourceModule(cuda_code)
        self.rollout = module.get_function("rollout")

        file_path = '{}/control/UW_mppi/Dynamics/{}.cpp'.format(folder_path, 'rotate_crop')
        with open(file_path, 'r') as file:
            cuda_code = file.read()
        module = SourceModule(cuda_code)
        self.rotate_crop = module.get_function("rotate_crop")

        self.BEVmap_height = gpuarray.to_gpu(np.zeros((self.BEVmap_size_px, self.BEVmap_size_px), dtype=dtype) )
        self.BEVmap_normal = gpuarray.to_gpu(np.zeros((self.BEVmap_size_px, self.BEVmap_size_px, 3), dtype=dtype) )

        self.pycuda_ctx.pop()
        atexit.register(self.cleanup)

    def cleanup(self):
        try:
            if self.pycuda_ctx is not None:
                self.pycuda_ctx.pop()
                print("PyCUDA context popped on exit.")
        except:
            pass

    def load_dyn_model(self, config, weights_path, tn_args: Dict = None):
        net, _ = build_nets(config, tn_args, model_weight_file=weights_path)
        net.eval()
        return net

    ## TODO: these should be moved to utils if we don't end up switching to torch cpp extension later
    def tensor_to_gpuarray(self, tensor, dtype):
        return gpuarray.GPUArray(tensor.shape, dtype, gpudata = tensor.data_ptr())

    def gpuarray_to_tensor(self, gpa, out, dtype, npdtype):
        ## note: this function assumes you already have the location where you would like to copy the data.
        gpa_copy = self.tensor_to_gpuarray(out, dtype=npdtype)
        byte_size = gpa.itemsize * gpa.size
        cuda.memcpy_dtod(gpa_copy.gpudata, gpa.gpudata, byte_size)

    ## TODO: remove
    def set_BEV(self, BEVmap_height, BEVmap_normal):
        self.pycuda_ctx.push()
        cuda.Context.synchronize()
        self.BEVmap_height = self.tensor_to_gpuarray(BEVmap_height, np.float32)
        self.BEVmap_normal = self.tensor_to_gpuarray(BEVmap_normal, np.float32)
        cuda.Context.synchronize()
        self.pycuda_ctx.pop()

    # TODO: remove
    def set_BEV_numpy(self, BEVmap_height, BEVmap_normal):
        self.BEVmap_height = gpuarray.to_gpu(BEVmap_height)
        self.BEVmap_normal = gpuarray.to_gpu(BEVmap_normal)

    def get_states(self):
        return self.states

    def forward(self, state, controls): #, BEVmap_height, BEVmap_normal, gt_states=None, gt_controls=None):
        now = time.time()

        # begin pycuda context:
        self.pycuda_ctx.push()
        cuda.Context.synchronize()
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
        cuda.Context.synchronize()
        self.pycuda_ctx.pop()
        center = torch.clamp( ((self.states[..., :2] + self.BEVmap_size*0.5) / self.BEVmap_res).to(dtype=torch.int32, device=torch.device("cuda")), self.bev_input.shape[2], self.BEVmap_size_px - 1 - self.bev_input.shape[1]).squeeze(0)
        center = center.reshape((self.K*self.T, 2))
        angle_torch = self.states[..., 5].reshape((self.K * self.T))
        
        self.pycuda_ctx.push()
        
        center = self.tensor_to_gpuarray(center, np.int32)
        angle  = self.tensor_to_gpuarray(angle_torch,np.float32)
        block_size = (self.crop_size, self.crop_size, 1)
        grid_size = ((self.bev_input.shape[2] + block_size[0] - 1) // block_size[0], (self.bev_input.shape[1] + block_size[1] - 1) // block_size[1], np.int32(self.K*self.T))

        cuda.Context.synchronize()
        for i in range(self.batch_size):
            self.rotate_crop(self.BEVmap_height, self.output_images[..., self.max_threads*i:self.max_threads*(i+1)],
                             angle[self.max_threads*i: self.max_threads*(i+1)], center[self.max_threads*i: self.max_threads*(i+1),...], 
                             np.int32(self.BEVmap_height.shape[1]), np.int32(self.BEVmap_height.shape[1]), np.int32(self.bev_input.shape[1]),
                             np.int32(self.bev_input.shape[2]), np.int32(self.max_threads), block=(self.crop_size,self.crop_size,1), grid=(1,1,self.max_threads))
            cuda.Context.synchronize()

        # self.rotate_crop(self.BEVmap_height, self.output_images, angle, center, 
        #                  np.int32(self.BEVmap_height.shape[1]), np.int32(self.BEVmap_height.shape[1]), np.int32(self.bev_input.shape[1]),
        #                  np.int32(self.bev_input.shape[2]), np.int32(self.K*self.T), block=(self.crop_size,self.crop_size,1), grid=(1,1,int(self.K*self.T)))
        # cuda.Context.synchronize()
        
        self.gpuarray_to_tensor(self.output_images, self.dummy_input, dtype=torch.float32, npdtype=np.float32)
        cuda.Context.synchronize()
        # end pycuda context. Run pytorch model OUTSIDE this context
        self.pycuda_ctx.pop()

        self.bev_input = self.dummy_input.permute(2,0,1)

        yaw = self.states[0,0,0,5]
        residual_input_states = self.rotate_traj(self.states.clone(), rotation_angle= -yaw) ## remove yaw component
        residual_corrected_states = self.dyn_model._forward(residual_input_states.squeeze(0), controls.squeeze(0), ctx_data={'rotate_crop': self.bev_input}, Evaluation=True)
        states_new = self.rotate_traj(residual_corrected_states, rotation_angle=yaw)
        self.states[..., 1:, :15] = states_new[..., 1:, :15].unsqueeze(0)
        
        dt = time.time() - now

        self.dt_avg = self.dt_avg*0.8 + dt*0.2

        return self.states

    def forward_train(self, state, controls, BEVmap_height, BEVmap_normal, print_something=None):
        # begin pycuda context:
        '''
        this function is to train the res model in a batch-wise fashion. As such
        the input states are KxTxNx, where we get K different starting points.
        the controls are KxTxNu where we get K different control trajectories
        instead of getting 1 BEV we get K BEVs.
        We launch the rollout kernel with size 1 instead of 1024, and launch K such kernels.
        We do the same for the crop-rotation.
        '''
        batch_size = state.shape[1]
        timesteps = state.shape[2]
        states_out = torch.zeros_like(state).squeeze(0)
        self.pycuda_ctx.push()
        # transfer data from torch to pycuda arrays for forward rollout
        controls_ = self.tensor_to_gpuarray(controls.squeeze(0), np.float32)
        state_ = self.tensor_to_gpuarray(state.squeeze(0), np.float32)

        BEVmap_height = self.tensor_to_gpuarray(BEVmap_height, np.float32)
        BEVmap_normal = self.tensor_to_gpuarray(BEVmap_normal, np.float32)
        # sync context before we start kernels
        cuda.Context.synchronize()
        for i in range(batch_size):
            # Launch the forward dynamics CUDA kernel
            self.rollout(state_[i,...], controls_[i, ...], BEVmap_height[i,...], BEVmap_normal[i,...], self.dt, np.int32(1), np.int32(timesteps), self.NX, self.NC,
                    self.D, self. B, self.C, self.lf, self.lr, self.Iz, self.throttle_to_wheelspeed, self.steering_max,
                    self.BEVmap_size_px, self.BEVmap_res, self.BEVmap_size, self.car_l2, self.car_w2, self.cg_height, self.LPF_tau, self.LPF_st, self.LPF_th, self.res_coeff, self.drag_coeff,
                    block=(1, 1, 1), grid=(1, 1))

        cuda.Context.synchronize()
        self.gpuarray_to_tensor(state_, states_out, torch.float32, np.float32)
        cuda.Context.synchronize()
        if print_something:
            print(print_something)

        center = torch.clamp( ((states_out[..., :2] + self.BEVmap_size*0.5) / self.BEVmap_res).to(dtype=torch.int32, device=torch.device("cuda")), self.bev_input.shape[2], self.BEVmap_size_px - 1 - self.bev_input.shape[1]).squeeze(0)
        angle = states_out[..., 5].squeeze(0)
        
        dummy_input = torch.zeros((self.crop_size,self.crop_size,batch_size*timesteps)).to(device=torch.device("cuda"), dtype=torch.float32)
        
        output_images = gpuarray.zeros((self.crop_size, self.crop_size, batch_size*timesteps), dtype=np.float32)
        center = self.tensor_to_gpuarray(center, np.int32)
        angle = self.tensor_to_gpuarray(angle, np.float32)
        cuda.Context.synchronize()

        grid_size = ((self.bev_input.shape[2] + self.crop_block_size[0] - 1) // self.crop_block_size[0], (self.bev_input.shape[1] + self.crop_block_size[1] - 1) // self.crop_block_size[1], timesteps)
        for i in range(batch_size):
            # for each trajectory, we have self.T number of BEV maps.
            # instead of having K*T x out x out, we have T * out,
            self.rotate_crop(BEVmap_height[i,...], output_images[..., timesteps*i: timesteps*(i+1)], angle[i,...], center[i,...], np.int32(BEVmap_height.shape[1]), np.int32(BEVmap_height.shape[1]), np.int32(self.bev_input.shape[1]),
                             np.int32(self.bev_input.shape[2]), np.int32(timesteps), block=self.crop_block_size, grid=grid_size )

        self.gpuarray_to_tensor(output_images, dummy_input, dtype=torch.float32, npdtype=np.float32)
        cuda.Context.synchronize()
        # end pycuda context. Run pytorch model OUTSIDE this context
        self.pycuda_ctx.pop()


        self.bev_input_train = dummy_input.permute(2,0,1).reshape(batch_size*timesteps, self.crop_size, self.crop_size)

        return states_out.unsqueeze(0)

    ## TODO: this should happen inside the residual network module. This is being done outside to avoid running into issues with pytorch autograd
    def rotate_traj(self, states, rotation_angle = 0):
        X = states[..., 0]
        Y = states[..., 1]
        yaw = states[..., 5]
        ct = torch.cos(rotation_angle)
        st = torch.sin(rotation_angle)
        x = ct*X - st*Y
        y = ct*Y + st*X
        yaw += rotation_angle
        states[..., 0] = x
        states[..., 1] = y
        states[..., 5] = yaw
        return states