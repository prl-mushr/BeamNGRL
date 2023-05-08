import gym
from gym.envs.registration import register
from BeamNGRL.gym.envs import OffroadSmallIsland

register(
    id='offroad-small-island-v0',
    entry_point='BeamNGRL.gym.envs:OffroadSmallIsland',
    kwargs={}
)