import time
import torch

from UW_mppi.Dynamics.SimpleCar import SimpleCar, CarState
from UW_mppi.Costs.SimpleCarCost import SimpleCarCost
from UW_mppi.MPPI  import MPPI

class DynSim:
  def __init__(self, dyn):
    self.state = CarState().vec()
    self.dyn = dyn

  def step(self, action):
    padded_state = self.state.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    padded_action = action.unsqueeze(0).unsqueeze(0)
    self.state = self.dyn.forward(padded_state, padded_action).squeeze()

if __name__ == "__main__":
  device = 'cpu'
  ctrl_noise = torch.tensor(((0.02, 0), (0, 0.03)))

  n_rollouts = 512
  timesteps = 128
  dt = 0.02

  costs = SimpleCarCost(device=device)
  dynamics = SimpleCar(dt=dt, device=device)
  mppi = MPPI(dynamics, costs, ctrl_noise, N_SAMPLES=n_rollouts, TIMESTEPS=64, device=device)

  mppi.reset()

  sim = DynSim(dynamics)

  while 1:
    # Only apply one action in the sequence for now. (u_per_command = 1)
    action = mppi.forward(sim.state)[0]
    print(sim.state)
    print(action)

    sim.step(action)
