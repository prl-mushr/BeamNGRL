import numpy as np
import torch
import torch
import cv2
import time

from ilqr import iLQR
from ilqr.cost import QRCost
from ilqr.dynamics import constrain

class control_system:

	def __init__(self,trajectory, N_SAMPLES=1000, TIMESTEPS=40, lambda_=0.01, costmap_resolution = 0.5, max_speed=20, track_width = 2):
		nx = 4
		d = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
		dtype = torch.float
		self.costmap_resolution = costmap_resolution
		self.costmap_resolution_inv = 1/self.costmap_resolution
		self.track_width = track_width
		self.create_costmap(trajectory, costmap_resolution = self.costmap_resolution, track_width = self.track_width)
		print("got costmap")
		self.max_speed = max_speed
		self.wheelbase = torch.tensor(2.6)
		self.steering_max = torch.tensor(0.5)
		noise_sigma = torch.zeros((2,2), device=d, dtype=dtype)
		noise_sigma[0,0] = 0.1
		noise_sigma[1,1] = 1.0
		self.dt = 0.05
		self.mppi = mppi.MPPI(self.dynamics, self.running_cost, nx, noise_sigma, num_samples=N_SAMPLES, horizon=TIMESTEPS, lambda_=lambda_)

	def update(self, data):
		x = data[0]
		y = data[1]
		# self.x = x
		# self.y = y
		# self.img_X = np.array(x*self.costmap_resolution_inv + self.shift_X, dtype=np.int32)
		# self.img_Y = np.array(y*self.costmap_resolution_inv + self.shift_Y, dtype=np.int32)
		theta = data[5]
		speed = np.linalg.norm(data[6:8])
		state = np.array([x, y, theta, speed])
		self.show_location_on_map(state)
		now = time.time()
		action = self.mppi.command(state)
		dt = time.time() - now
		# print("dt: ", dt*1000)
		action = torch.clamp(action, -1, 1)
		return action

	def show_location_on_map(self, state):
		X = int(state[0]*self.costmap_resolution_inv + self.shift_X)
		Y = int(state[1]*self.costmap_resolution_inv + self.shift_Y)
		costmap = np.copy(self.costmap)
		cv2.circle(costmap, (X,Y), int(0.2*self.track_width*self.costmap_resolution_inv), 1, -1)
		if(self.mppi.print_states is not None):
			print_states = self.mppi.print_states
			x = print_states[:,:,:,0].flatten().numpy()
			y = print_states[:,:,:,1].flatten().numpy()
			X = np.array(x*self.costmap_resolution_inv + self.shift_X, dtype=np.int32)
			Y = np.array(y*self.costmap_resolution_inv + self.shift_Y, dtype=np.int32)
			costmap[Y,X] = 0

		costmap = cv2.flip(costmap, 0)
		cv2.imshow("map", costmap)
		cv2.waitKey(1)

	def create_costmap(self, trajectory, costmap_resolution, track_width):
		max_x = np.max(trajectory[:,0]) * self.costmap_resolution_inv
		min_x = np.min(trajectory[:,0]) * self.costmap_resolution_inv
		max_y = np.max(trajectory[:,1]) * self.costmap_resolution_inv
		min_y = np.min(trajectory[:,1]) * self.costmap_resolution_inv

		width = int(3*(max_x - min_x))
		height = int(1.5*(max_y - min_y))
		print(width, height)
		# conversion = x,y -> x * self.costmap_resolution_inv + (width*0.5 - min_x), y * self.costmap_resolution_inv + (height*0.5 - min_y)
		costmap = np.ones((height,width), np.float32)
		self.shift_X = (width*0.33 - min_x)
		self.shift_Y = (height*0.16 - min_y)
		for i in range(len(trajectory)):
			X = int(trajectory[i,0] * self.costmap_resolution_inv + self.shift_X)
			Y = int(trajectory[i,1] * self.costmap_resolution_inv + self.shift_Y)
			cv2.circle(costmap, (X,Y), int(track_width * self.costmap_resolution_inv), 0, -1)
		k = int(2*track_width * self.costmap_resolution_inv)
		costmap = cv2.blur(costmap, (k, k))

		self.costmap = torch.from_numpy(costmap) 


	def dynamics(self, state, perturbed_action):
		x = state[:, 0].view(-1, 1)
		y = state[:, 1].view(-1, 1)
		theta = state[:,2].view(-1,1)
		speed = state[:,3].view(-1,1)

		u = perturbed_action
		u = torch.clamp(u, -1, 1)
		dt = 0.05

		v = speed + u[:,1].view(-1,1)*dt  # acceleration
		omega = speed * torch.tan(u[:,0].view(-1,1)*self.steering_max) / self.wheelbase

		x += dt*speed*torch.cos(theta)
		y += dt*speed*torch.sin(theta)
		theta += dt*omega
		speed = v
		state = torch.cat((x, y, theta, speed), dim=1)
		return state

	def running_cost(self, state, action):
		x = state[:, 0]
		y = state[:, 1]
		theta = state[:,2]
		speed = state[:,3]
		K = torch.tan(action[:,0]*self.steering_max) / self.wheelbase
		img_X = torch.tensor(x * self.costmap_resolution_inv + self.shift_X, dtype=torch.long)
		img_Y = torch.tensor(y * self.costmap_resolution_inv + self.shift_Y, dtype=torch.long)
		state_cost = self.costmap[img_Y, img_X]
		state_cost *= state_cost
		state_cost[np.where(state_cost>=0.9)] = 1000
		vel_cost = torch.abs(self.max_speed - speed)/self.max_speed
		turn_cost = (torch.abs(speed * speed * K)/15)
		turn_cost[np.where(turn_cost > 1)] = 100
		turn_cost[np.where(turn_cost < 1)] = 0

		return vel_cost + state_cost + turn_cost

