import torch
from BeamNGRL.dynamics.utils.exp_utils import build_nets
from BeamNGRL.dynamics.utils.network_utils import load_model
from typing import Dict
import time

def load_dyn_model(config, weights_path, tn_args: Dict = None):
    net, _ = build_nets(config, tn_args, model_weight_file=weights_path)
    net.eval()
    return net


class SimpleCarNetworkDyn(torch.nn.Module):
    """
    Class for Dynamics modelling
    """
    def __init__(
        self,
        Dynamics_config,
        Map_config,
        MPPI_config,
        model_weights_path=None,
        dtype=torch.float32,
        device=torch.device("cuda"),
    ):

        super().__init__()
        self.dtype = dtype
        self.d = device
        self.tn_args = {'device': device, 'dtype': dtype}
        self.dyn_model = load_dyn_model(Dynamics_config, model_weights_path, self.tn_args)

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
        now = time.time()

        states_pred = self.dyn_model.rollout(state.squeeze(0), controls.squeeze(0), ctx_data={'bev_elev':self.BEVmap_height, 'bev_normal':self.BEVmap_normal}, dt =self.dt)
        dt = time.time() - now
        # print(dt)
        self.states = states_pred.unsqueeze(0)

        return self.states
