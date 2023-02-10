import numpy as np
import torch
from pytorch_mppi import mppi
import torch
import cv2
import time

class control_system:

	def __init__(self,trajectory, N_SAMPLES=512, TIMESTEPS=30, lambda_= 0.0, costmap_resolution = 0.1, max_speed=20, track_width = 5, num_optimizations=1, noise_scale=1):
		nx = 15
		d = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
		self.device = torch.device("cuda:0")
		dtype = torch.float
		self.costmap_resolution = costmap_resolution
		self.costmap_resolution_inv = 1/self.costmap_resolution
		self.track_width = track_width
		self.create_costmap(trajectory, costmap_resolution = self.costmap_resolution, track_width = self.track_width)
		# print("got costmap")
		self.max_speed = max_speed
		self.wheelbase = torch.tensor(2.6)
		self.steering_max = torch.tensor(0.5)
		self.noise_sigma = torch.zeros((2,2), device=d, dtype=dtype)
		self.noise_sigma[0,0] = 0.1*noise_scale
		self.noise_sigma[1,1] = 0.2*noise_scale
		self.dt = 0.05
		self.now = time.time()
		self.map_size = 40  # half map size
		self.mppi = mppi.MPPI(self.dynamics, self.running_cost, nx, self.noise_sigma, num_samples=N_SAMPLES, horizon=TIMESTEPS, lambda_=lambda_, num_optimizations = num_optimizations, percent_elites=0.2)
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
		self.elu = torch.nn.ELU()

	def update(self, data):
		x = data[0]
		y = data[1]
		self.x = x
		self.y = y
		self.img_X = np.array(x*self.costmap_resolution_inv + self.shift_X, dtype=np.int32)
		self.img_Y = np.array(y*self.costmap_resolution_inv + self.shift_Y, dtype=np.int32)
		state = data
		# self.noise_sigma[0,0] = 0.1/max(1,abs(data[6]/5) )
		# self.noise_sigma[1,1] = 0.4/max(1,abs(data[6]/5) )
		# self.mppi.update_noise_sigma(self.noise_sigma)
		self.create_costmap_truncated()
		# self.show_location_on_map(state)
		action = self.mppi.command(state)
		# self.dt = time.time() - self.now  # for comparing output quality, we assume dt is "constant" regardless of actual dt. Will change to different dt when we want to show perf advantage.
		self.now = time.time()
		# print("dt: ", dt*1000)
		action = torch.clamp(action, -1, 1)

		# action[0] = action[0]*0.5 + self.last_U[0]*0.5
		# action[1] = action[1]*0.1 + self.last_U[1]*0.9
		# phi_ref = np.linalg.norm(data[6:8]) * torch.tan(action[0]*self.steering_max) / self.wheelbase
		# phi_real = data[14]
		# feedback = (phi_ref - phi_real)
		# action[0] += 0.1*feedback		
		# self.last_U = action
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
			costmap[Y,X] = 1

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
		# print(width, height)
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
		costmap = self.add_obstacles(costmap)
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
		accel = u[:,1].view(-1, 1)
		pos_index = torch.where(accel>0)
		accel[pos_index] = torch.clamp(accel[pos_index]*25/v[pos_index],0,5)
		ax = accel
		v = v + 5*accel*self.dt  # acceleration
		omega = v * torch.tan(u[:,0].view(-1, 1)*self.steering_max) / self.wheelbase
		ay = v*omega

		x += self.dt*(vx*torch.cos(yaw))# - vy*torch.sin(yaw))
		y += self.dt*(vx*torch.sin(yaw))# + vy*torch.cos(yaw))
		yaw += self.dt*omega
		vx = v
		state = torch.cat((x, y, z, roll, pitch, yaw, vx, vy, vz, ax, ay, az, gx, gy, gz), dim=1)
		return state

	def dynamics_single(self, state, perturbed_action):
		x = state[ 0]
		y = state[ 1]
		z = state[ 2]
		roll = state[ 3]
		pitch = state[ 4]
		yaw = state[ 5]
		vx = state[6]
		vy = state[7]
		vz = state[8]
		ax = state[9]
		ay = state[10]
		az = state[11]
		gx = state[12]
		gy = state[13]
		gz = state[14]

		u = perturbed_action
		u = torch.clamp(u, -1, 1)
		v = torch.sqrt(vx**2 + vy**2)
		accel = u[1]
		if(accel > 0):
			accel = torch.clamp(accel*25/v,0,5)
		else:
			accel *= 5
		ax = accel
		v = v + 5*accel*self.dt  # acceleration
		omega = v * torch.tan(u[0]*self.steering_max) / self.wheelbase
		ay = v*omega

		x += self.dt*(vx*torch.cos(yaw))# - vy*torch.sin(yaw))
		y += self.dt*(vx*torch.sin(yaw))# + vy*torch.cos(yaw))
		yaw += self.dt*omega
		vx = v
		state = torch.hstack((x, y, z, roll, pitch, yaw, vx, vy, vz, ax, ay, az, gx, gy, gz))
		return state.numpy()

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
		state_cost[np.where(state_cost>=0.9)] = 1000
		vel_cost = torch.abs(self.max_speed - vx)/self.max_speed
		vel_cost = torch.sqrt(vel_cost)
		accel = (ax**2 + ay**2)*0.01
		accel_cost = accel
		accel_cost[np.where(accel < 0.9)] = 0

		return 2*vel_cost + state_cost + 0.1*accel_cost

	def running_cost_single(self, state, action):
		x = state[ 0]
		y = state[ 1]
		z = state[ 2]
		roll = state[ 3]
		pitch = state[ 4]
		yaw = state[ 5]
		vx = state[6]
		vy = state[7]
		vz = state[8]
		ax = state[9]
		ay = state[10]
		az = state[11]
		gx = state[12]
		gy = state[13]
		gz = state[14]

		K = torch.tan(action[0]*self.steering_max) / self.wheelbase
		## get the location within the truncated costmap
		img_X = torch.tensor((x - self.x + self.map_size) * self.costmap_resolution_inv, dtype=torch.long)
		img_Y = torch.tensor((y - self.y + self.map_size) * self.costmap_resolution_inv, dtype=torch.long)
		state_cost = self.costmap[img_Y, img_X]
		state_cost *= state_cost
		if(state_cost > 0.9):
			state_cost = 1000
		vel_cost = torch.abs(self.max_speed - vx)/self.max_speed
		vel_cost = torch.sqrt(vel_cost)
		accel = (ax**2 + ay**2)*0.01
		accel_cost = accel
		if(accel < 0.9):
			accel_cost = 0

		return (2*vel_cost + state_cost + 0.1*accel_cost).numpy()

	def add_obstacles(self, costmap):
		max_shape = np.max(costmap.shape)
		points = np.random.randint(0,max_shape, size=(1600,2))
		for i in range(len(points)):
			X = points[i,0]
			Y = points[i,1]
			cv2.circle(costmap, (X,Y), int(2 * self.costmap_resolution_inv), 1, -1)
		return costmap