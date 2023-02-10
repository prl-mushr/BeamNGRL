import numpy as np
import torch
from pytorch_mppi import mppi
import torch
import cv2
import time
import pandas as pd
import os

class control_system:

	def __init__(self,trajectory, N_SAMPLES=256, TIMESTEPS=30, lambda_= 0.1, costmap_resolution = 0.1, max_speed=25, track_width = 1):
		nx = 15
		d = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
		self.device = torch.device("cuda")
		dtype = torch.float
		self.costmap_resolution = costmap_resolution
		self.costmap_resolution_inv = 1/self.costmap_resolution
		self.track_width = track_width
		self.create_costmap(trajectory, costmap_resolution = self.costmap_resolution, track_width = self.track_width)
		print("got costmap")
		self.max_speed = max_speed
		self.wheelbase = torch.tensor(2.6)
		self.steering_max = torch.tensor(0.5)
		self.noise_sigma = torch.zeros((2,2), device=d, dtype=dtype)
		self.noise_sigma[0,0] = 0.05
		self.noise_sigma[1,1] = 0.2
		self.dt = 0.05
		self.now = time.time()
		self.map_size = 40  # half map size
		self.mppi = mppi.MPPI(self.dynamics, self.running_cost, nx, self.noise_sigma, num_samples=N_SAMPLES, horizon=TIMESTEPS, lambda_=lambda_, num_optimizations = 1)
		self.train_data = []
		self.train_data_iter = int(60/self.dt)
		self.last_U = torch.zeros(2, device=d, dtype=dtype)
		self.use_model = True
		model_name = 'dynamics_full.h5'
		checkpoint = torch.load(model_name) #uncomment this and the line below to train an existing model
		self.network = checkpoint['net']
		for param in self.network.parameters():
			param.requires_grad = False
		self.error = None
		self.dyn_csts = pd.read_json(os.path.join('ilqc/envs/bicycle_model.json'), typ='series')
		self.Br, self.Cr, self.Dr, self.Bf, self.Cf, self.Df,\
		self.m, self.Iz, self.lf, self.lr = [self.dyn_csts[key] for key in ['Br', 'Cr', 'Dr', 'Bf', 'Cf', 'Df',
														'm', 'Iz', 'lf', 'lr']]
		self.Iz /= self.m
		self.Df *= 9.8
		self.Dr *= 9.8

	def update(self, data):
		x = data[0]
		y = data[1]
		self.x = x
		self.y = y
		self.img_X = np.array(x*self.costmap_resolution_inv + self.shift_X, dtype=np.int32)
		self.img_Y = np.array(y*self.costmap_resolution_inv + self.shift_Y, dtype=np.int32)
		state = data
		# if(self.train_data_iter > 0):
		# 	# note that xyz positions don't matter for dynamics prediction.
		# 	self.train_data.append( np.hstack( (data, self.last_U.cpu().numpy() ) ) )
		# 	self.train_data_iter -= 1
		# else:
		# 	self.train_data = np.array(self.train_data)
		# 	np.save("train_data.npy", self.train_data)
		# 	print("done")
		# 	exit()
		self.noise_sigma[0,0] = 1.0/max(1,abs(data[6]) )
		self.noise_sigma[1,1] = 1.0/max(1,max(abs(data[6]/2.5), 4) )
		self.mppi.update_noise_sigma(self.noise_sigma)
		self.create_costmap_truncated()
		# self.show_location_on_map(state)
		action = self.mppi.command(state)
		# self.dt = time.time() - self.now
		self.now = time.time()
		# print("dt: ", dt*1000)
		action[0] = action[0]*0.5 + self.last_U[0]*0.5
		action[1] = action[1]*0.2 + self.last_U[1]*0.8 - np.sin(data[4])/(np.pi)
		action = torch.clamp(action, -1, 1)

		# phi_ref = np.linalg.norm(data[6:8]) * torch.tan(action[0]*self.steering_max) / self.wheelbase
		# phi_real = data[14]
		# feedback = (phi_ref - phi_real)
		# action[0] += 0.1*feedback		
		self.last_U = action
		return action

	def show_location_on_map(self, state):
		X = int((state[0] - self.x + self.map_size)*self.costmap_resolution_inv)
		Y = int((state[1] - self.y + self.map_size)*self.costmap_resolution_inv)
		costmap = np.copy(self.costmap)
		# cv2.circle(costmap, (X,Y), int(self.costmap_resolution_inv), 1, -1)
		if(self.mppi.print_states is not None):
			print_states = self.mppi.print_states
			x = print_states[:,:,:,0].flatten().numpy()
			y = print_states[:,:,:,1].flatten().numpy()
			X = np.array((x - self.x + self.map_size)*self.costmap_resolution_inv, dtype=np.int32)
			Y = np.array((y - self.y + self.map_size)*self.costmap_resolution_inv, dtype=np.int32)
			costmap[Y,X] = 0

		costmap = cv2.flip(costmap, 0)
		cv2.imshow("map", costmap)
		cv2.waitKey(1)

	def create_costmap_truncated(self):
		Y_min = int(self.img_Y - self.map_size*self.costmap_resolution_inv)
		Y_max = int(self.img_Y + self.map_size*self.costmap_resolution_inv)

		X_min = int(self.img_X - self.map_size*self.costmap_resolution_inv)
		X_max = int(self.img_X + self.map_size*self.costmap_resolution_inv)
		self.costmap = self.costmap_full[Y_min:Y_max,X_min:X_max]

	def create_costmap(self, trajectory, costmap_resolution, track_width):
		max_x = np.max(trajectory[:,0]) * self.costmap_resolution_inv
		min_x = np.min(trajectory[:,0]) * self.costmap_resolution_inv
		max_y = np.max(trajectory[:,1]) * self.costmap_resolution_inv
		min_y = np.min(trajectory[:,1]) * self.costmap_resolution_inv

		width = int(3*(max_x - min_x))
		height = int(3*(max_y - min_y))
		print(width, height)
		# conversion = x,y -> x * self.costmap_resolution_inv + (width*0.5 - min_x), y * self.costmap_resolution_inv + (height*0.5 - min_y)
		costmap = np.ones((height,width), np.float32)
		self.shift_X = (width*0.33 - min_x)
		self.shift_Y = (height*0.33 - min_y)
		for i in range(len(trajectory)):
			X = int(trajectory[i,0] * self.costmap_resolution_inv + self.shift_X)
			Y = int(trajectory[i,1] * self.costmap_resolution_inv + self.shift_Y)
			cv2.circle(costmap, (X,Y), int(track_width * self.costmap_resolution_inv), 0, -1)
		k = int(2*track_width * self.costmap_resolution_inv)
		costmap = cv2.blur(costmap, (k, k))

		self.costmap_full = torch.from_numpy(costmap) 

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

		u = perturbed_action
		u = torch.clamp(u, -1, 1)
		v = torch.sqrt(vx**2 + vy**2)
		accel = 5*u[:,1].view(-1, 1)
		pos_index = torch.where(accel>0)
		accel[pos_index] = torch.clamp(accel[pos_index]*25/torch.clamp(v[pos_index],5,30),0,5)
		# ax = accel
		# v = v + 5*accel*self.dt  # acceleration
		# omega = v * torch.tan(u[:,0].view(-1, 1)*self.steering_max) / self.wheelbase
		# ay = v*omega
		delta = u[:,0].view(-1, 1)*self.steering_max
		alphaf = delta - torch.atan2(gz*self.lf + vy, vx) 
		alphar = torch.atan2(gz*self.lr - vy, vx)
		Fry = self.Dr*torch.sin(self.Cr*torch.atan(self.Br*alphar))
		Ffy = self.Df*torch.sin(self.Cf*torch.atan(self.Bf*alphaf))
		Frx = accel
		ax = (Frx - Ffy*torch.sin(delta) + vy*gz) + 9.8*torch.sin(pitch)
		ay = (Fry + Ffy*torch.cos(delta) - vx*gz) - 9.8*torch.sin(pitch)
		vx += ax*self.dt
		vy += ay*self.dt
		gz += self.dt*(Ffy*self.lf*torch.cos(delta) - Fry*self.lr)/self.Iz
		x += (torch.cos(yaw)*vx - torch.sin(yaw)*vy)*self.dt
		y += (torch.sin(yaw)*vx + torch.cos(yaw)*vy)*self.dt
		yaw += self.dt*gz


		state = torch.cat((x, y, z, roll, pitch, yaw, vx, vy, vz, ax, ay, az, gx, gy, gz), dim=1)
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

		K = torch.tan(action[:,0]*self.steering_max) / self.wheelbase
		## get the location within the truncated costmap
		img_X = torch.tensor((x - self.x + self.map_size) * self.costmap_resolution_inv, dtype=torch.long)
		img_Y = torch.tensor((y - self.y + self.map_size) * self.costmap_resolution_inv, dtype=torch.long)
		state_cost = self.costmap[img_Y, img_X]
		state_cost *= state_cost
		state_cost[np.where(state_cost>=0.9)] = 100
		vel_cost = torch.abs(self.max_speed - vx)/self.max_speed
		vel_cost = torch.sqrt(vel_cost)
		accel = ay*0.1
		accel_cost = accel**2
		# accel_cost[np.where(vx < 15)] *= 0.01

		return 1.5*vel_cost + state_cost + 0.04*vx[0]*accel_cost