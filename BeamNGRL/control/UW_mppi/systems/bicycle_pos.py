import torch

"""
A standard "kinematic" bicycle model.

State is
  2D pos     (R^2)
  yaw        (R)

Control is
  velocity  (R)
  curvature (R)
"""

class Config:
  def __init__(self, **kwargs):
    for k, v in kwargs.items():
      setattr(self, k, v)

    # TODO Pull all config from outside (loaded from separate files)
    self.w_goal = 10.0
    self.w_vel = 1.0

class State:
  def __init__(self, pos=torch.zeros(2), yaw=0.0):
    self.pos = pos
    self.yaw = torch.tensor(yaw)

  def vec(self):
    return torch.hstack((self.pos, self.yaw))

class Dynamics:
  def __init__(self, config):
    self.dt = config.dt

  def forward(self, state, controls):
    """
      args:
        state:
          torch.tensor with size = (..X.., H, 3)
          Probably makes more sense to not have the input state include the horizon dimension, but let's leave for now.
        control:
          torch.tensor with size = (..X.., H, 2)

      return:
        states:
          torch.tensor with size = (..X.., H, 3)
    """
    x = state[..., 0]
    y = state[..., 1]
    yaw = state[..., 2]

    vel = controls[..., 0]
    curv = controls[..., 1]

    dS = vel * self.dt
    dyaw  = vel * curv * self.dt

    # Does these change the original state vector?
    yaw += torch.cumsum(dyaw, dim=-1)
    x += torch.cumsum(dS * torch.cos(yaw), dim=-1)
    y += torch.cumsum(dS * torch.sin(yaw), dim=-1)

    return torch.stack((x, y, yaw), dim=-1), controls

class Cost:
  def __init__(self, goal, speed_target, config):
    self.speed_target = speed_target

    self.w_vel = config.w_vel
    self.w_goal = config.w_goal

    self.setgoal(goal)

  def setgoal(self, goal):
    self.goal = goal

  def compute(self, state, control):
    """ Time axis (horizon) is assumed to be just prior to state dimension axis.
      i.e. state is   [..., H, 3]
           control is [..., H, 2]
    """
    pos = state[..., 0:2]
    vx = control[..., 0]

    vel_err = torch.square(torch.abs(vx - self.speed_target))
    goal_dist = torch.linalg.norm(pos - self.goal, dim=-1)

    cost = self.w_vel * vel_err + self.w_goal + goal_dist
    return cost.mean(dim=(0, 2))
