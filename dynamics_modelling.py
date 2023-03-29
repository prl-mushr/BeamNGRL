import numpy as np
import torch
from pytorch_mppi import mppi
import torch
import cv2
import time

class control_system:

	def __init__(self,trajectory, N_SAMPLES=256, TIMESTEPS=50, lambda_= 0.1, costmap_resolution = 0.1, max_speed=20, track_width = 2):
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
		self.mppi = mppi.MPPI(self.dynamics, self.running_cost, nx, self.noise_sigma, num_samples=N_SAMPLES, horizon=TIMESTEPS, lambda_=lambda_, num_optimizations = 2)
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
		self.create_costmap_truncated()
		self.show_location_on_map(state)
		self.costmap = self.costmap.to(device=self.device)
		self.dt = torch.tensor( (time.time() - self.now), device=self.device, dtype=torch.float)
		self.now = time.time()
		action = self.mppi.command(state)

		print(self.error.mean())
		# print("dt: ", dt*1000)
		action = torch.clamp(action, -1, 1)
		action[0] = action[0]*0.9 + self.last_U[0]*0.1
		action[1] = action[1]*0.1 + self.last_U[1]*0.9
		
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

	# @torch.jit.export
	def dynamics(self, state, perturbed_action):
		x = state[:, 0].view(-1, 1).to(device=self.device)
		y = state[:, 1].view(-1, 1).to(device=self.device)
		z = state[:, 2].view(-1, 1).to(device=self.device)
		roll = state[:, 3].view(-1, 1).to(device=self.device)
		pitch = state[:, 4].view(-1, 1).to(device=self.device)
		yaw = state[:, 5].view(-1, 1).to(device=self.device)
		vx = state[:,6].view(-1, 1).to(device=self.device)
		vy = state[:,7].view(-1, 1).to(device=self.device)
		vz = state[:,8].view(-1, 1).to(device=self.device)
		ax = state[:,9].view(-1, 1).to(device=self.device)
		ay = state[:,10].view(-1, 1).to(device=self.device)
		az = state[:,11].view(-1, 1).to(device=self.device)
		gx = state[:,12].view(-1, 1).to(device=self.device)
		gy = state[:,13].view(-1, 1).to(device=self.device)
		gz = state[:,14].view(-1, 1).to(device=self.device)

		u = perturbed_action
		u = torch.clamp(u, -1, 1).to(device=self.device)
		state = state.to(device=self.device)
		XU = torch.hstack( (state[:,3:], u) ).to(device = self.device)
		XU[:,5] = 0
		
		# if(self.use_model):
		delta = self.network(XU)  #  vel, A, G,
		ax = delta[:,3].view(-1, 1)
		ay = delta[:,4].view(-1, 1)
		az = delta[:,5].view(-1, 1)
		gx = delta[:,6].view(-1, 1)
		gy = delta[:,7].view(-1, 1)
		gz = delta[:,8].view(-1, 1)
		vx += self.dt*ax
		vy += self.dt*ay
		# vz += self.dt*az
		yaw += self.dt*gz
		roll += self.dt*gx
		pitch += self.dt*gy
		# v = vx + 5*u[:,1].view(-1, 1)*self.dt  # acceleration
		# omega = vx * torch.tan(u[:,0].view(-1, 1)*self.steering_max) / self.wheelbase

		x += self.dt*(vx*torch.cos(yaw) - vy*torch.sin(yaw))
		y += self.dt*(vx*torch.sin(yaw) + vy*torch.cos(yaw))
		# yaw += self.dt*omega
		# vx = v
		state = torch.cat((x, y, z, roll, pitch, yaw, vx, vy, vz, ax, ay, az, gx, gy, gz), dim=1)
		return state

	# @torch.jit.export
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
		state_cost[torch.where(state_cost>=0.9)] = 1000
		vel_cost = torch.abs(self.max_speed - vx)/self.max_speed
		vel_cost = torch.sqrt(vel_cost)
		
		# lateral_accel = vx * vx * K
		accel = (ax**2 + ay**2)*0.01
		accel_cost = (accel - 1)**2
		# turn_cost = (torch.abs(speed * speed * K)/7)
		# turn_cost[np.where(turn_cost < 1)] = 0

		return (2*vel_cost + state_cost + 0.03*accel_cost).cpu()

    # def train(self, new_data):
  
    #     # not normalized inside the simulator
    #     new_data[:, 0] = angle_normalize(new_data[:, 0])
    #     if not torch.is_tensor(new_data):
    #         new_data = torch.from_numpy(new_data)
    #     # clamp actions
    #     new_data[:, -1] = torch.clamp(new_data[:, -1], ACTION_LOW, ACTION_HIGH)
    #     new_data = new_data.to(device=d)
    #     # append data to whole dataset
    #     if dataset is None:
    #         dataset = new_data
    #     else:
    #         dataset = torch.cat((dataset, new_data), dim=0)

    #     # train on the whole dataset (assume small enough we can train on all together)
    #     XU = dataset
    #     dtheta = angular_diff_batch(XU[1:, 0], XU[:-1, 0])
    #     dtheta_dt = XU[1:, 1] - XU[:-1, 1]
    #     Y = torch.cat((dtheta.view(-1, 1), dtheta_dt.view(-1, 1)), dim=1)  # x' - x residual
    #     XU = XU[:-1]  # make same size as Y

    #     # thaw network
    #     for param in network.parameters():
    #         param.requires_grad = True

    #     optimizer = torch.optim.Adam(network.parameters())
    #     for epoch in range(TRAIN_EPOCH):
    #         optimizer.zero_grad()
    #         # MSE loss
    #         Yhat = network(XU)
    #         loss = (Y - Yhat).norm(2, dim=1) ** 2
    #         loss.mean().backward()
    #         optimizer.step()
    #         logger.debug("ds %d epoch %d loss %f", dataset.shape[0], epoch, loss.mean().item())

    #     # freeze network
    #     for param in network.parameters():
    #         param.requires_grad = False

    #     # evaluate network against true dynamics
    #     yt = true_dynamics(statev, actionv)
    #     yp = dynamics(statev, actionv)
    #     dtheta = angular_diff_batch(yp[:, 0], yt[:, 0])
    #     dtheta_dt = yp[:, 1] - yt[:, 1]
    #     E = torch.cat((dtheta.view(-1, 1), dtheta_dt.view(-1, 1)), dim=1).norm(dim=1)
    #     logger.info("Error with true dynamics theta %f theta_dt %f norm %f", dtheta.abs().mean(),
    #                 dtheta_dt.abs().mean(), E.mean())
    #     logger.debug("Start next collection sequence")