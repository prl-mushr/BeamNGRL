import torch
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
import numpy as np
import logging
import matplotlib.pyplot as plt
import time

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format='[%(levelname)s %(asctime)s %(pathname)s:%(lineno)d] %(message)s',
                    datefmt='%m-%d %H:%M:%S')

d = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
dtype = torch.float
H_UNITS = 32
TRAIN_EPOCH = 20000
BOOT_STRAP_ITER = 100

nx = 12  # not considering the first 3 states
nu = 2
npred = 9
# network output is state residual
network = torch.nn.Sequential(
	torch.nn.Linear(nx + nu, H_UNITS),
	torch.nn.Tanh(),
	torch.nn.Linear(H_UNITS, H_UNITS),
	torch.nn.Tanh(),
	torch.nn.Linear(H_UNITS, npred)
).float().to(device=d)

model_name = 'dynamics_full.h5'

checkpoint = torch.load(model_name) #uncomment this and the line below to train an existing model
network = checkpoint['net']

def train():
	global model_name
	new_data = np.load('train_data.npy')
	# not normalized inside the simulator
	if not torch.is_tensor(new_data):
		new_data = torch.from_numpy(new_data)
		new_data = torch.tensor(new_data, dtype=torch.float)
	# clamp actions
	new_data = new_data.to(device=d)
	# append data to whole dataset
	# if dataset is None:
	#     dataset = new_data
	# else:
	#     dataset = torch.cat((dataset, new_data), dim=0)
	XU = new_data[:-1,3:]  # given previous rpy, vel, A, G, U
	XU[:,5] = 0  # I don't want it to do anything with the yaw
	Y = new_data[1:,6:15]  # predict new vel_body, A, G
	# train on the whole dataset (assume small enough we can train on all together)
	# thaw network
	for param in network.parameters():
		param.requires_grad = True

	logger_loss = []
	optimizer = torch.optim.Adam(network.parameters())

	now = time.time()
	for epoch in range(TRAIN_EPOCH):
		optimizer.zero_grad()
		# MSE loss
		x = new_data[:-1, 0]
		y = new_data[:-1, 1]
		z = new_data[:-1, 2]
		Yhat = network(XU)
		# roll, pitch, yaw = delta[:, 0], delta[:, 1], delta[:, 2]
		# vx, vy, vz = delta[:, 3], delta[:, 4], delta[:, 5]
		# ax, ay, az = delta[:, 3 + 1], delta[:, 4 + 1], delta[:, 5 + 1]
		# gx, gy, gz = delta[:, 3 + 2], delta[:, 4 + 2], delta[:, 5 + 2]
		# sy = torch.sin(yaw)
		# cy = torch.cos(yaw)
		# cp = torch.cos(pitch)
		# sp = torch.sin(pitch)
		# cr = torch.cos(roll)
		# sr = torch.sin(roll)
		# x = x + (vx*cp*cy + vy*cr*sy)*0.025
		# y = y + (vy*cr*cy - vx*cp*sy)*0.025
		# z = z + (vz*cp*cr)*0.025 
		# Yhat = torch.cat((x, y, z, roll, pitch, yaw, vx, vy, vz, ax, ay, az, gx, gy, gz), dim=1)
		loss = (Y - Yhat).norm(2, dim=1) ** 2
		loss.mean().backward()
		optimizer.step()
		logger_loss.append(loss.mean().item())
	dt = time.time() - now
	print("per epoch time:", dt/TRAIN_EPOCH)
	# freeze network
	for param in network.parameters():
		param.requires_grad = False

	state = {'net':network}
	torch.save(state,model_name)

	plt.plot(np.arange(len(logger_loss)), np.array(logger_loss) )
	plt.show()

if __name__ == '__main__':
	train()