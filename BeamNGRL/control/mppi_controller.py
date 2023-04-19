import numpy as np
import torch
from pytorch_mppi import mppi
import torch
import cv2
import time
import pandas as pd
import os
from pathlib import Path
import BeamNGRL

ROOT_PATH = Path(BeamNGRL.__file__).parent
CRL_PATH = ROOT_PATH / 'control'


class control_system:

	def __init__(self, N_SAMPLES=256, TIMESTEPS=30, lambda_= 0.1, max_speed=15, BEVmap_size = 16, BEVmap_res = 0.25):
		nx = 17
		d = torch.device("cpu") if torch.cuda.is_available() else torch.device("cpu")
		self.d = torch.device("cpu")
		dtype = torch.float

		## extra variables that are specific to your problem statement:
		self.goal_state = torch.zeros(2).to(self.d)  # possible goal state
		self.BEVmap_size = torch.tensor(BEVmap_size).to(self.d)
		self.BEVmap_res = torch.tensor(BEVmap_res).to(self.d)
		assert self.BEVmap_res > 0
		self.BEVmap_size_px = torch.tensor((self.BEVmap_size/self.BEVmap_res), device=self.d, dtype=torch.int32)
		self.BEVmap = torch.zeros((self.BEVmap_size_px.item(), self.BEVmap_size_px.item() )).to(self.d)
		self.BEVmap_heght = torch.zeros_like(self.BEVmap)
		self.BEVmap_segmt = torch.zeros_like(self.BEVmap)
		self.BEVmap_color = torch.zeros_like(self.BEVmap)
		self.BEVmap_normal = torch.zeros_like(self.BEVmap)

		self.max_speed = max_speed
		self.steering_max = torch.tensor(0.5).to(self.d)
		self.noise_sigma = torch.zeros((2,2), device=d, dtype=dtype)
		self.noise_sigma[0,0] = 0.5
		self.noise_sigma[1,1] = 0.5
		self.dt = 0.05
		self.now = time.time()
		self.mppi = mppi.MPPI(self.dynamics, self.running_cost, nx, self.noise_sigma, num_samples=N_SAMPLES, horizon=TIMESTEPS, lambda_=lambda_, terminal_state_cost = self.terminal_cost)
		self.last_U = torch.zeros(2, device=d)

		print(f'\nCRL_PATH: {CRL_PATH}')
		self.dyn_csts = pd.read_json(CRL_PATH / 'bicycle_model.json', typ='series')
		self.Br, self.Cr, self.Dr, self.Bf, self.Cf, self.Df,\
		self.m, self.Iz, self.lf, self.lr = [self.dyn_csts[key] for key in ['Br', 'Cr', 'Dr', 'Bf', 'Cf', 'Df',
														'm', 'Iz', 'lf', 'lr']]
		self.Iz /= self.m
		self.Df *= 9.8
		self.Dr *= 9.8

	def forward(self, data):
		x = data[0]
		y = data[1]
		self.x = x
		self.y = y
		state = np.hstack((data, self.last_U.cpu()))
		action = self.mppi.command(state)*self.dt + self.last_U.cpu()
		self.now = time.time()
		action = torch.clamp(action, -1, 1)
		self.last_U = action
		return action

	def set_goal(self, goal_state):
		self.goal_state = goal_state[:2]

	def set_BEV(self, BEV_color, BEV_heght, BEV_segmt, BEV_path, BEV_normal, BEV_center):
		assert BEV_color.shape[0] == self.BEVmap_size_px
		self.BEVmap = torch.tensor(BEV_path[:,:,0], dtype=torch.float).to(self.d)/255.0
		self.BEVmap_heght = BEV_heght
		self.BEVmap_segmt = BEV_segmt
		self.BEVmap_color = BEV_color
		self.BEVmap_normal = BEV_normal
		self.BEVmap_center = BEV_center  # translate the state into the center of the costmap.

	def get_states(self):
		return self.mppi.print_states

	def dynamics(self, state, perturbed_action):
		x = state[:, 0].view(-1, 1)
		y = state[:, 1].view(-1, 1)
		z = state[:, 2].view(-1, 1)
		roll = state[:, 3].view(-1, 1)
		pitch = state[:, 4].view(-1, 1)
		yaw = state[:, 5].view(-1, 1)
		vx = state[:,6].view(-1, 1)
		vy = state[:,7].view(-1, 1)
		vz = state[:,8].view(-1, 1)
		ax = state[:,9].view(-1, 1)
		ay = state[:,10].view(-1, 1)
		az = state[:,11].view(-1, 1)
		gx = state[:,12].view(-1, 1)
		gy = state[:,13].view(-1, 1)
		gz = state[:,14].view(-1, 1)

		state[:,15] += perturbed_action[:,0]*self.dt
		state[:,16] += perturbed_action[:,1]*self.dt

		u = torch.clamp(state[:,15:17], -1, 1)
		v = torch.sqrt(vx**2 + vy**2)
		accel = 5*u[:,1].view(-1,1)
		pos_index = torch.where(accel>0)
		accel[pos_index] = torch.clamp(accel[pos_index]*25/torch.clamp(v[pos_index],5,30),0,5)

		delta = u[:,0].view(-1,1)*self.steering_max
		alphaf = delta - torch.atan2(gz*self.lf + vy, vx) 
		alphar = torch.atan2(gz*self.lr - vy, vx)
		Fry = self.Dr*torch.sin(self.Cr*torch.atan(self.Br*alphar))
		Ffy = self.Df*torch.sin(self.Cf*torch.atan(self.Bf*alphaf))
		Frx = accel
		ax = (Frx - Ffy*torch.sin(delta) + vy*gz) + 9.8*torch.sin(pitch)
		ay = (Fry + Ffy*torch.cos(delta) - vx*gz) - 9.8*torch.sin(roll)
		vx += ax*self.dt
		vy += ay*self.dt
		gz += self.dt*(Ffy*self.lf*torch.cos(delta) - Fry*self.lr)/self.Iz
		x += (torch.cos(yaw)*vx - torch.sin(yaw)*vy)*self.dt
		y += (torch.sin(yaw)*vx + torch.cos(yaw)*vy)*self.dt
		yaw += self.dt*gz

		img_X = ((x + self.BEVmap_size*0.5) / self.BEVmap_res).to(dtype=torch.long, device=self.d)
		img_Y = ((y + self.BEVmap_size*0.5) / self.BEVmap_res).to(dtype=torch.long, device=self.d)
		
		z = self.BEVmap_heght[img_Y, img_X]
		normal = self.BEVmap_normal[img_Y, img_X]

		heading = torch.stack([torch.cos(yaw), torch.sin(yaw), torch.zeros_like(yaw)], dim=2)
		# Calculate the cross product of the heading and normal vectors to get the vector perpendicular to both
		left = torch.cross(normal, heading)
		# Calculate the cross product of the right and normal vectors to get the vector perpendicular to both and facing upwards
		forward = torch.cross(left, normal)
		# Calculate the roll angle (rotation around the forward axis)
		roll = torch.asin(left[:,:,2])
		# Calculate the pitch angle (rotation around the right axis)
		pitch = torch.asin(forward[:,:,2])
		
		state = torch.cat((x, y, z, roll, pitch, yaw, vx, vy, vz, ax, ay, az, gx, gy, gz, state[:,15].view(-1,1), state[:, 16].view(-1,1)), dim=1)
		return state

	def running_cost(self, state, action):
		x = state[:, 0]
		y = state[:, 1]
		z = state[:, 2]
		roll = state[:, 3]
		pitch = state[:, 4]
		yaw = state[:, 5]
		vx = state[:,6]
		vy = state[:,7]
		vz = state[:,8]
		ax = state[:,9]
		ay = state[:,10]
		az = state[:,11]
		gx = state[:,12]
		gy = state[:,13]
		gz = state[:,14]

		## get the location within the truncated costmap
		img_X = ((x + self.BEVmap_size*0.5) / self.BEVmap_res).to(dtype=torch.long, device=self.d)
		img_Y = ((y + self.BEVmap_size*0.5) / self.BEVmap_res).to(dtype=torch.long, device=self.d)
		state_cost = self.BEVmap[img_Y, img_X]
		state_cost *= state_cost
		state_cost[np.where(state_cost>=0.9)] = 100
		vel_cost = torch.abs(self.max_speed - vx)/self.max_speed
		vel_cost = torch.sqrt(vel_cost)
		accel = ay*0.1
		accel_cost = accel**2
		accel_cost[torch.where(torch.abs(accel) > 0.5)] = 100
		return 0.05*vel_cost + state_cost + accel_cost

	def terminal_cost(self, state, action):
		return torch.linalg.norm(state[0,:,-1,:2] - self.goal_state, dim=1)