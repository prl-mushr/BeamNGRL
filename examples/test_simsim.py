"""
Simple Sim

Use MPPI's dynamics model as a simulator.
"""

import time
import torch
import numpy as np
import os
import yaml
from pathlib import Path

from BeamNGRL.control.UW_mppi.MPPI import MPPI
from BeamNGRL.control.UW_mppi.Dynamics.SimpleCarDynamics import SimpleCarDynamics
from BeamNGRL.control.UW_mppi.Costs.SimpleCarCost import SimpleCarCost
from BeamNGRL.control.UW_mppi.Sampling.Delta_Sampling import Delta_Sampling
from BeamNGRL.utils.keygrabber import KeyGrabber
from BeamNGRL.utils.visualisation import Vis


class DynSim:
    def __init__(self, dyn):
        self.dyn = dyn
        self.state = torch.zeros(17, dtype=dyn.dtype, device=dyn.d)

    def step(self, action):
        offset = torch.clone(self.state[:3])
        self.state[:3] = 0
        padded_state = self.state[None, None, None, :]
        padded_action = action[None, None, None, :]
        self.state = self.dyn.forward(padded_state, padded_action)
        self.state = self.state.squeeze()
        self.state[:3] += offset

if __name__ == "__main__":

    config_path = str(Path(os.getcwd()).parent.absolute()) + "/BeamNGRL/control/UW_mppi/Configs/"

    with open(config_path + 'MPPI_config.yaml') as f:
        MPPI_config = yaml.safe_load(f)

    with open(config_path + 'Dynamics_config.yaml') as f:
        Dynamics_config = yaml.safe_load(f)

    with open(config_path + 'Cost_config.yaml') as f:
        Cost_config = yaml.safe_load(f)

    with open(config_path + 'Sampling_config.yaml') as f:
        Sampling_config = yaml.safe_load(f)

    with open(config_path + 'Map_config.yaml') as f:
        Map_config = yaml.safe_load(f)

    map_res = Map_config["map_res"]
    map_size = Map_config["map_size"]

    dtype = torch.float
    d = torch.device("cuda")

    dynamics = SimpleCarDynamics(Dynamics_config, Map_config, MPPI_config)
    costs = SimpleCarCost(Cost_config, Map_config)
    sampling = Delta_Sampling(Sampling_config, MPPI_config)

    controller = MPPI(
        dynamics,
        costs,
        sampling,
        MPPI_config
    )
    controller.reset()

    map_size_px = int(map_size/map_res)

    BEV_heght = torch.zeros((map_size_px, map_size_px), device=d, dtype=dtype)
    BEV_normal = torch.zeros((map_size_px, map_size_px,3), device=d, dtype=dtype)
    BEV_normal[:,:,2] = 1.0 ## Z axis = 1
    BEV_path = torch.zeros((map_size_px, map_size_px,3), device=d, dtype=dtype)
    ## these are just for compatibility, they won't (or shouldn't) actually do anything:
    controller.Dynamics.set_BEV(BEV_heght, BEV_normal)
    controller.Costs.set_BEV(BEV_heght, BEV_normal, BEV_path)

    sim = DynSim(dynamics)
    vis = Vis()
    kg = KeyGrabber()
    goal = torch.tensor((10, 0.0))
    action_tensor = torch.zeros(2, device=d, dtype=dtype)

    print("WASD to move goal")
    realtime = True

    while True:
        # Only apply one action in the sequence for now. (u_per_command = 1)
        now = time.time()

        controller.Costs.set_goal( (torch.from_numpy(np.copy(goal))).to(device=d, dtype=dtype) - sim.state[:2] )# you can also do this asynchronously
        state = torch.clone(sim.state)
        state[:3] = 0

        # we use our previous control output as input for next cycle!
        state[15:17] = action_tensor ## adhoc wheelspeed.
        action_tensor = controller.forward(state)[0]
        sim.step(action_tensor)

        vis_state = sim.state.cpu().numpy()
        vis.setcar(pos=vis_state[:3], rpy=vis_state[3:6])
        vis.setgoal(goal)

        chars = kg.read()

        for c in chars:
            if c in 'wW':
                goal[0] += 1.0
            elif c in 'aA':
                goal[1] += 1.0
            elif c in 'sS':
                goal[0] -= 1.0
            elif c in 'dD':
                goal[1] -= 1.0
            else:
                print("Unmapped char", c)

        while time.time() - now < Dynamics_config["dt"] and realtime:
            time.sleep(0.001)