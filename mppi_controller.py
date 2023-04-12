import numpy as np
import torch
from pytorch_mppi import mppi
import torch
import cv2
import time
import pandas as pd
import os

class control_system():
	def __init__(self, N_SAMPLES=512, TIMESTEPS=50, lambda_= 0.1, costmap_resolution = 0.1, max_speed=10, track_width = 1):
		nx = 15
		d = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
		self.device = torch.device("cpu")
		dtype = torch.float
		self.costmap_resolution = costmap_resolution
		self.costmap_resolution_inv = 1/self.costmap_resolution
		self.track_width = track_width
		self.max_speed = max_speed
		self.wheelbase = torch.tensor(2.6)
		self.steering_max = torch.tensor(0.5)
		self.noise_sigma = torch.zeros((2,2), device=d, dtype=dtype)
		self.noise_sigma[0,0] = 0.5
		self.noise_sigma[1,1] = 0.5
		self.dt = 0.05
		self.now = time.time()
		self.mppi = mppi.MPPI(self.dynamics, self.running_cost, nx, self.noise_sigma, num_samples=N_SAMPLES, horizon=TIMESTEPS, lambda_=lambda_, num_optimizations = 1, device=self.device)
		self.map_size = 40  # half map size
		self.last_U = torch.zeros(2, device=self.device)

		self.dyn_csts = pd.read_json(os.path.join('ilqc/envs/bicycle_model.json'), typ='series')
		self.Br, self.Cr, self.Dr, self.Bf, self.Cf, self.Df,\
		self.m, self.Iz, self.lf, self.lr = [self.dyn_csts[key] for key in ['Br', 'Cr', 'Dr', 'Bf', 'Cf', 'Df',
														'm', 'Iz', 'lf', 'lr']]										
		self.Iz /= self.m*5
		self.lf /= 5
		self.lr /= 5
		self.Df *= 9.8
		self.Dr *= 9.8
		self.x = torch.tensor(0, device=self.device, dtype=dtype)
		self.y = torch.tensor(0, device=self.device, dtype=dtype)

	def update(self, data, costmap, target_wp):
		self.x = data[0]
		self.y = data[1]
		self.costmap = costmap.to(self.device)
		self.target_wp = torch.from_numpy(target_wp).to(self.device)
		state = torch.from_numpy( np.hstack((data, self.last_U.cpu())) ).to(self.device)
		self.show_location_on_map()
		action = self.mppi.command(state)*self.dt + self.last_U 
		action = torch.clamp(action, -1, 1)
		# action[1] = torch.clamp(action[1], -1, 0.5)
		self.last_U = action
		return action

	def show_location_on_map(self):
		X = int((self.map_size)*self.costmap_resolution_inv)
		Y = int((self.map_size)*self.costmap_resolution_inv)
		costmap = np.copy(self.costmap)
		cv2.circle(costmap, (X,Y), int(self.costmap_resolution_inv), 1, -1)
		goal_X = int((self.target_wp[0] - self.x + self.map_size)*self.costmap_resolution_inv)
		goal_Y = int((self.target_wp[1] - self.y + self.map_size)*self.costmap_resolution_inv)
		cv2.line(costmap, (X, Y), (goal_X, goal_Y), (0,1,0), 2)
		cv2.circle(costmap, (goal_X, goal_Y), int(self.costmap_resolution_inv), (1,0,0), -1)

		if(self.mppi.print_states is not None):
			print_states = self.mppi.print_states
			x = print_states[:,:,:,0].flatten().numpy()
			y = print_states[:,:,:,1].flatten().numpy()
			X = np.array((x - self.x + self.map_size)*self.costmap_resolution_inv, dtype=np.int32)
			Y = np.array((y - self.y + self.map_size)*self.costmap_resolution_inv, dtype=np.int32)
			costmap[Y,X] = 1

		costmap = cv2.flip(costmap, 0)
		cv2.imshow("map", costmap)
		cv2.waitKey(1)

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

		img_X = torch.tensor((x - self.x + self.map_size) * self.costmap_resolution_inv, dtype=torch.long).to(self.device)
		img_Y = torch.tensor((y - self.y + self.map_size) * self.costmap_resolution_inv, dtype=torch.long).to(self.device)
		speed_dist = 1 - self.costmap[img_Y, img_X,1]

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
		ay = (Fry + Ffy*torch.cos(delta) - vx*gz) - 9.8*torch.sin(pitch)
		vx += ax*self.dt
		vx *= speed_dist
		vy += ay*self.dt
		gz += self.dt*(Ffy*self.lf*torch.cos(delta) - Fry*self.lr)/self.Iz
		x += (torch.cos(yaw)*vx - torch.sin(yaw)*vy)*self.dt
		y += (torch.sin(yaw)*vx + torch.cos(yaw)*vy)*self.dt
		yaw += self.dt*gz


		state = torch.cat((x, y, z, roll, pitch, yaw, vx, vy, vz, ax, ay, az, gx, gy, gz, state[:,15].view(-1,1), state[:, 16].view(-1,1)), dim=1)
		return state

	def running_cost(self, state, action):
		xyz = state[:, :3]
		roll = state[:, 3]
		pitch = state[:, 4]
		yaw = state[:, 5]
		vx = state[:,6]
		vy = state[:,7]
		vz = state[:,8]
		vhat = state[:,6:9]
		ax = state[:,9]
		ay = state[:,10]
		az = state[:,11]
		gx = state[:,12]
		gy = state[:,13]
		gz = state[:,14]

		K = torch.tan(action[:,0]*self.steering_max) / self.wheelbase
		## get the location within the truncated costmap
		img_X = torch.tensor((xyz[:,0] - self.x + self.map_size) * self.costmap_resolution_inv, dtype=torch.long).to(self.device)
		img_Y = torch.tensor((xyz[:,1] - self.y + self.map_size) * self.costmap_resolution_inv, dtype=torch.long).to(self.device)
		state_cost = self.costmap[img_Y, img_X,0] ## lethal cost is on channel 0
		# state_cost *= state_cost
		# state_cost[torch.where(state_cost>=0.9)] = 100
		state_cost += torch.linalg.norm(xyz - self.target_wp, dim = 1)
		# state_cost += self.costmap[img_Y, img_X,1]*10
		# print(state_cost.shape)
		vel_cost = (vx - self.max_speed)/self.max_speed
		vel_cost[torch.where(vel_cost < 0)] = 0
		vel_cost = torch.sqrt(vel_cost)*100
		accel = ay*0.1
		accel_cost = accel**2
		# accel_cost[np.where(vx < 15)] *= 0.01

		return 10*vel_cost + state_cost + vx*accel_cost + 1*torch.abs(vy)


class costmap_handler():
	def __init__(self,trajectory, costmap_resolution = 0.1, track_width = 3, make_arena = False):
		d = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
		self.device = torch.device("cpu")
		dtype = torch.float
		self.costmap_resolution = costmap_resolution
		self.costmap_resolution_inv = 1/self.costmap_resolution
		self.track_width = track_width
		self.map_size = 40  # half map size
		self.create_costmap(trajectory, costmap_resolution = self.costmap_resolution, track_width = self.track_width, make_arena=make_arena)

	def create_costmap_truncated(self, data):
		x = data[0]
		y = data[1]
		self.x = x
		self.y = y
		self.img_X = np.array(x*self.costmap_resolution_inv + self.shift_X, dtype=np.int32)
		self.img_Y = np.array(y*self.costmap_resolution_inv + self.shift_Y, dtype=np.int32)
		Y_min = int(self.img_Y - self.map_size*self.costmap_resolution_inv)
		Y_max = int(self.img_Y + self.map_size*self.costmap_resolution_inv)

		X_min = int(self.img_X - self.map_size*self.costmap_resolution_inv)
		X_max = int(self.img_X + self.map_size*self.costmap_resolution_inv)
		self.costmap = self.costmap_full[Y_min:Y_max,X_min:X_max].to(self.device)

	def create_costmap(self, trajectory, costmap_resolution, track_width, make_arena=False):

		max_x = np.max(trajectory[:,0]) * self.costmap_resolution_inv
		min_x = np.min(trajectory[:,0]) * self.costmap_resolution_inv
		max_y = np.max(trajectory[:,1]) * self.costmap_resolution_inv
		min_y = np.min(trajectory[:,1]) * self.costmap_resolution_inv

		width = int(3*(max_x - min_x))
		height = int(3*(max_y - min_y))
		print(width, height)
		# conversion = x,y -> x * self.costmap_resolution_inv + (width*0.5 - min_x), y * self.costmap_resolution_inv + (height*0.5 - min_y)
		costmap = np.ones((height,width,3), np.float32)
		costmap[:,:,0] = 100  # 0th channel is lethal cost
		print(costmap.shape)
		self.shift_X = (width*0.33 - min_x)
		self.shift_Y = (height*0.33 - min_y)
		for i in range(len(trajectory)):
			X = int(trajectory[i,0] * self.costmap_resolution_inv + self.shift_X)
			Y = int(trajectory[i,1] * self.costmap_resolution_inv + self.shift_Y)
			cv2.circle(costmap, (X,Y), int(track_width * self.costmap_resolution_inv), (0,0,0), -1)
			cv2.circle(costmap, (X,Y), int(track_width * self.costmap_resolution_inv), (0,0.2,0), -1)
		for i in range(len(trajectory)):
			X = int(trajectory[i,0] * self.costmap_resolution_inv + self.shift_X)
			Y = int(trajectory[i,1] * self.costmap_resolution_inv + self.shift_Y)
			cv2.circle(costmap, (X,Y), int(track_width*0.5 * self.costmap_resolution_inv), (0,0.1,0), -1)
		# k = int(2*track_width * self.costmap_resolution_inv)
		# costmap = cv2.blur(costmap, (k, k))
		if(make_arena):
			costmap[:,:,:] = 0 # reset everthing
			self.create_arena(costmap)
		self.costmap_full = torch.from_numpy(costmap).to(self.device)

	def draw_rect(self, start, end, costmap, color):
		start_X = int(start[0] * self.costmap_resolution_inv + self.shift_X)
		start_Y = int(start[1] * self.costmap_resolution_inv + self.shift_Y)

		end_X = int(end[0] * self.costmap_resolution_inv + self.shift_X)
		end_Y = int(end[1] * self.costmap_resolution_inv + self.shift_Y)

		cv2.rectangle(costmap, (start_X,start_Y), (end_X, end_Y), color, -1)
		return costmap

	def create_arena(self, costmap):
		arena_start  = np.array([-340, -272])
		arena_end = np.array([-262, -376])

		costmap = self.draw_rect(arena_start, arena_end, costmap, (0,0,0))  # make a completely 0 cost arena in a white area

		mud_start = np.array([-336.5, -275.5])
		mud_end = np.array([-303.5, -372.5])
		costmap = self.draw_rect(mud_start, mud_end, costmap, (0,0.2,0))  # green means mud

		sand_start = np.array([-300, -276])
		sand_end = np.array([-268, -372])

		costmap = self.draw_rect(sand_start, sand_end, costmap, (0,0.1,0)) # red means sand

