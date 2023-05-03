"""
Simple Sim

Use MPPI's dynamics model as a simulator.
"""

import time
import torch

from BeamNGRL.control.UW_mppi.MPPI import MPPI, Config as MPPIConfig
from BeamNGRL.control.UW_mppi.systems.bicycle_pos import Config, State, Dynamics, Cost
from BeamNGRL.control.UW_mppi.util.keygrabber import KeyGrabber
from BeamNGRL.control.UW_mppi.vis.meshcat import Vis

class DynSim:
  def __init__(self, dyn):
    self.state = State().vec()
    self.dyn = dyn

  def step(self, action):
    padded_state = self.state[None, None, None, :]
    padded_action = action[None, None, :]
    self.state, postaction = self.dyn.forward(padded_state, padded_action)

    self.state = self.state.squeeze()

if __name__ == "__main__":
  device = 'cpu'
  ctrl_noise = torch.tensor(((0.2, 0), (0, 0.02)))

  config = dict(
    n_rollouts=256,
    n_timesteps=70,
    n_bins=1,
    dt=0.1,
    temperature=0.01,
  )
  mppiconfig = MPPIConfig(**config)

  goal = torch.tensor((10, 0.0))
  speed_target = 7.0
  config = Config(**config)

  cost = Cost(goal, speed_target, config)
  ctrl_dynamics = Dynamics(config)
  sim_dynamics = Dynamics(config)

  mppi = MPPI(ctrl_dynamics, cost, mppiconfig, ctrl_noise, device=device)
  mppi.reset()

  sim = DynSim(sim_dynamics)
  vis = Vis()
  kg = KeyGrabber()

  print("WASD to move goal")

  while 1:
    # Only apply one action in the sequence for now. (u_per_command = 1)
    action = mppi.forward(sim.state)[0]
    sim.step(action)

    vis.setcar(pos=sim.state[:2], yaw=sim.state[2])
    vis.setgoal(goal)
    cost.setgoal(goal)

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

    #time.sleep(config.dt / 5.0)
