# BeamNGRL

## Installation:
1) Download the map folder and BeamNG folder from [here](https://drive.google.com/drive/folders/1sZ8aDUqtnTomdXn6bxoryJ8X4yT7RS06). Extract the 'BeamNG' outside the BeamNGRL folder(usually `/home/username/` directory, for instance


2) Environment setup:
put this in your bash:
```bash
export BNG_HOME=/absolute/path/to/BeamNG/BeamNG/
export PYTHONPATH="${PYTHONPATH}:/absolute/path/to/BeamNGRL"
```

## Running minimal interface example:
### Setting up the paths:
You will need to specify the path to BeamNG and the map folder when running the interface:
```python
def main(map_name, start_point, start_quat, BeamNG_dir='/home/stark/'):  ## --> BeamNG_dir is the directory in which you kept the BeamNG folder.
    map_res = 0.05
    map_size = 16 # 16 x 16 map

    bng_interface = beamng_interface(BeamNG_dir = BeamNG_dir)
    ## the car is called "RACER" only because it was modded to mimic the dynamics of a 3000 pound off-road vehicle like the rzr
    bng_interface.load_scenario(scenario_name=map_name, car_make='sunburst', car_model='RACER',
                                start_pos=start_point, start_rot=start_quat)
    bng_interface.set_map_attributes(map_size = map_size, resolution=map_res, path_to_maps='/home/stark/')  ## --> path_to_maps is the directory where you extracted the "map" folder
```

### Executing minimal example
```bash
cd ~/BeamNGRL
python3 beamng_interface_minimal.py
```
### Minimal example explained
```python
    map_res = 0.05 ## resolution in meters. 0.05 meters per pixel
    map_size = 16 # 16 x 16 map
	## this is how you instantiate the interface (Start game)
    bng_interface = beamng_interface(BeamNG_dir = BeamNG_dir)  
    ## this loads a specific scenario(map) with a specific vehicle at a specific position and rotation
    ## start point is numpy array, start_quat is also a numpy array representing the orientation quaternion.
    bng_interface.load_scenario(scenario_name=map_name, car_make='sunburst', car_model='RACER',
                                start_pos=start_point, start_rot=start_quat)

    ## this sets the BEV map generation attributes. Right now we only have the maps for "small-island"
    bng_interface.set_map_attributes(map_size = map_size, resolution=map_res, path_to_maps='/home/stark/')  ## --> path_to_maps is the directory where you extracted the "map" folder
```
#### Running the simulator with "lock-step" (similar to gym's env.step()):
```python
    ## this runs the simulator in lock-step, meaning that the simulator will pause while you do your calculations and only run when a control command is sent
    bng_interface.set_lockstep(True)

```
The game will take a while to load for the first time, if prompted by the OS to wait/kill program, chose "wait". This only happens the first time you run the game.

#### Getting the state:
`state_poll()` causes the interface to poll the pose, twist, accelerations as well as generate the corresponding BEV maps. We don't get camera images or lidar data right now as BeamNG doesn't support it on Ubuntu (yet). We have the code for polling that as well, we're just waiting on BeamNG devs. The state information follows the [ROS REP103 standards](https://www.ros.org/reps/rep-0103.html). Note that twists (linear and angular) are in body frame

```python
state = bng_interface.state_poll() ## returns a numpy array: np.hstack((pos, rpy, vel, A, G, st, th/br))
```
You can also query individual states after calling this function:
```python
state = bng_interface.state_poll() ## this basically updates the internal states
quat = bng_interface.quat  ## orientation quaternion if you wanted that
pos = bng_interface.pos # world frame position
vel = bng_interface.vel ## body frame velocity vector
vel_wf = bng_interface.vel_wf  ## world frame velocity
accel = bng_interface.A # body frame acceleration (x,y,z)
gyration = bng_interface.G # body frame rotation
```

#### BEV maps:
Once the map is loaded in the game, you should see 3 BEV images pop up on the screen. These BEV images correspond to the BEV-map around the car. The interface can provide BEV images for:
1) Elevation
2) Color
3) Semantics
4) Paths (trails)

In the following lines, we get the BEV maps, resize and display them. Note that elevation map has floating point data type, and is body-centric in altitude (meaning the center of the map is at 0 height) with a ceiling of 2 meters and a floor of -2 meters.
```python
            
            ## get robot_centric BEV (not rotated into robot frame)
            BEV_color = bng_interface.BEV_color # color map
            BEV_heght = (bng_interface.BEV_heght + 2.0)/4.0  # note that BEV_heght (elevation) has a range of +/- 2 meters around the center of the elevation.
            BEV_segmt = bng_interface.BEV_segmt # segmented map
            BEV_path  = bng_interface.BEV_path  # trail/roads
            BEV_normal = bng_interface.BEV_normal # surface normals
            ## displaying BEV for visualization:
            BEV = cv2.resize(BEV_color, (500,500), interpolation= cv2.INTER_AREA)
            cv2.imshow('color', BEV)
            BEV = cv2.resize(BEV_heght, (500,500), interpolation= cv2.INTER_AREA)
            cv2.imshow('height', BEV)
            BEV = cv2.resize(BEV_segmt, (500,500), interpolation= cv2.INTER_AREA)
            cv2.imshow('segment', BEV)
            cv2.waitKey(1)
```
The images are robot-centric, with the X/Y axes aligned with the NE axes of the map (as in, the map does not rotate with the car). To get the body-frame map (as in, the map rotates around the car), you can set the rotation attribute to True:
```python
bng_interface.set_map_attributes(map_size = map_size, resolution=map_res, path_to_maps='/home/stark/', rotate=True) # --> rotate is False by default, set to True to get body-frame rotated map
```

#### Sending control commands:
There are two controls: steering(0) and throttle/brake (1). We can modify this in the future if you wish to have throttle and brake as separate
```python
            action = np.ones(2, dtype=np.float64)  # has to be numpy array. The inputs are always between (-1.0, 1.0) (for both throttle and steering)
            ## steering = action[0], throttle/brake = action[1]. Turning left = positive steering.
            bng_interface.send_ctrl(action) # this sends the commands to the simulator.
```

#### Resetting the vehicle:
You can check if the vehicle is flipped over and reset it to the "start_point" location as follows:
```python
            if(bng_interface.flipped_over):
                bng_interface.reset()
```

You can reset it to an arbitrary location for an arbitrary condition as follows:
```python
            if(arbitrary_condition):
                bng_interface.reset(numpy_array_representing_arbitrary_location)
```
Note that we can't change the rotation of the vehicle in this method, only position.


## Running default MPPI:
Assuming you've got the pytorch_mppi submodule, you can run:
```bash
python test_mppi_pytorch_default.py
```
### Code explained:
Setting up the 
```python

def main(map_name, start_point, start_quat, BeamNG_dir='/home/stark/', target_WP=None):
    map_res = 0.1
    map_size = 64 # 16 x 16 map

    bng_interface = beamng_interface(BeamNG_dir = BeamNG_dir)
    bng_interface.load_scenario(scenario_name=map_name, car_make='sunburst', car_model='RACER',
                                start_pos=start_point, start_rot=start_quat)
    bng_interface.set_map_attributes(map_size = map_size, resolution=map_res, path_to_maps='/home/stark/')

    # bng_interface.set_lockstep(True)
    dtype = torch.float
    d = torch.device("cpu")

```
Initialize control_system (from mppi_controller.py)
```python
    controller = control_system(BEVmap_size = map_size, BEVmap_res = map_res)
    current_wp_index = 0 # initialize waypoint index with 0
    goal = None
    action = np.zeros(2)
```
Main loop:
```python
    while True:
        try:
            bng_interface.state_poll()
            # state is np.hstack((pos, rpy, vel, A, G, st, th/br)) ## note that velocity is in the body-frame
            state =  bng_interface.state
            pos = np.copy(state[:2])  # example of how to get car position in world frame. All data points except for dt are 3 dimensional.
```
Generate goal position from a numpy array of goal positions at some "lookahead" distance. Use lookahead distance of "20" meters.
```python
            goal, terminate, current_wp_index = update_goal(goal, pos, target_WP, current_wp_index, 20)

            if(terminate):
                print("done!")
                bng_interface.send_ctrl(np.zeros(2))
                time.sleep(5)
                exit()
            ## get robot_centric BEV (not rotated into robot frame)
            BEV_color = torch.from_numpy(bng_interface.BEV_color).to(d)
            BEV_heght = torch.from_numpy(bng_interface.BEV_heght).to(d)
            BEV_segmt = torch.from_numpy(bng_interface.BEV_segmt).to(d)
            BEV_path  = torch.from_numpy(bng_interface.BEV_path).to(d)  # trail/roads
            BEV_normal = torch.from_numpy(bng_interface.BEV_normal).to(d)
            BEV_center = torch.from_numpy(pos).to(d)  # added normal map
```
Set the BEVs and goal:
```python
            controller.set_BEV(BEV_color, BEV_heght, BEV_segmt, BEV_path, BEV_normal, BEV_center)
            controller.set_goal(torch.from_numpy(np.copy(goal) - np.copy(pos)).to(d)) # you can also do this asynchronously

            state[:3] = np.zeros(3)  # this is for the MPPI: technically this should be state[:3] -= BEV_center

```
Call the forward function on the controller:
```python

            action = np.array(controller.forward(torch.from_numpy(state).to(d)).cpu().numpy(), dtype=np.float64)
            ## visualization:
            visualization(controller.get_states().cpu().numpy(), pos, np.copy(goal), np.copy(BEV_path.cpu().numpy()), 1/map_res)
            ## sending controls:
            bng_interface.send_ctrl(action)

        except Exception:
            print(traceback.format_exc())
    bng_interface.bng.close()


```

## MPPI controller overview (mppi_controller.py):
Class init function:
```python
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

```
This is where the MPPI object gets created (using pytorch_mppi's backend):
```python
        self.mppi = mppi.MPPI(self.dynamics,  # dynamics function
                              self.running_cost,  # running cost -- does not include terminal cost 
                              nx,
                              self.noise_sigma, 
                              num_samples=N_SAMPLES,
                              horizon=TIMESTEPS,
                              lambda_=lambda_,
                              terminal_state_cost = self.terminal_cost  ## pass in the terminal cost
                              )
        self.last_U = torch.zeros(2, device=d)
```
Dynamics model parameters:
```python
        self.dyn_csts = pd.read_json(os.path.join('bicycle_model.json'), typ='series')
        self.Br, self.Cr, self.Dr, self.Bf, self.Cf, self.Df,\
        self.m, self.Iz, self.lf, self.lr = [self.dyn_csts[key] for key in ['Br', 'Cr', 'Dr', 'Bf', 'Cf', 'Df',
                                                        'm', 'Iz', 'lf', 'lr']]
        self.Iz /= self.m
        self.Df *= 9.8
        self.Dr *= 9.8

```

Forward function:
```python
    def forward(self, data):
        x = data[0]
        y = data[1]
        self.x = x
        self.y = y
        state = np.hstack((data, self.last_U.cpu())) ## append controls at the end of state
        action = self.mppi.command(state)*self.dt + self.last_U.cpu() ## we sample in the delta control space, thus the output is "delta" control, not control
        self.now = time.time()
        action = torch.clamp(action, -1, 1)
        self.last_U = action
        return action
```

Dynamics function:
```python
    def dynamics(self, state, perturbed_action):
        ### you have to "view" the state like this: It is a K x nx array (K is number of samples)
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

        ## Sampling in delta control sapce means new control += delta_control * dt
        ## Idea finessed from: Kim, Taekyung, Gyuhyun Park, Kiho Kwak, Jihwan Bae, and Wonsuk Lee. "Smooth model predictive path integral control without smoothing." IEEE Robotics and Automation Letters 7, no. 4 (2022): 10406-10413.
        state[:,15] += perturbed_action[:,0]*self.dt
        state[:,16] += perturbed_action[:,1]*self.dt

        u = torch.clamp(state[:,15:17], -1, 1)
        v = torch.sqrt(vx**2 + vy**2)
        ## wheel force is approximated as:
        accel = 5*u[:,1].view(-1,1)
        pos_index = torch.where(accel>0)
        accel[pos_index] = torch.clamp(accel[pos_index]*25/torch.clamp(v[pos_index],5,30),0,5)

        ## this dynamic function was finessed from: Liniger, Alexander, Alexander Domahidi, and Manfred Morari. "Optimization‐based autonomous racing of 1: 43 scale RC cars." Optimal Control Applications and Methods 36, no. 5 (2015): 628-647.
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

        ## getting the position on map from state:
        img_X = ((x + self.BEVmap_size*0.5) / self.BEVmap_res).to(dtype=torch.long, device=self.d)
        img_Y = ((y + self.BEVmap_size*0.5) / self.BEVmap_res).to(dtype=torch.long, device=self.d)
        ## referencing position:
        z = self.BEVmap_heght[img_Y, img_X] # project to elevation map
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
```

Running cost:
```python
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
        ## I use the path map to get the "ground truth" on state cost:
        state_cost = self.BEVmap[img_Y, img_X]
        state_cost *= state_cost
        state_cost[np.where(state_cost>=0.9)] = 100 ## lethal cost
        ## velocity cost:
        vel_cost = torch.abs(self.max_speed - vx)/self.max_speed
        vel_cost = torch.sqrt(vel_cost)
        accel = ay*0.1
        accel_cost = accel**2
        accel_cost[torch.where(accel > 0.5)] = 100
        return 0.05*vel_cost + state_cost + accel_cost
```

Terminal cost:
Currently using Euclidean distance:
```python
    def terminal_cost(self, state, action):
        return torch.linalg.norm(state[0,:,-1,:2] - self.goal_state, dim=1)
```


## Additional information for map generation:
Collecting map images (requires windows OS). This is currently only configured for the small-island (off-road) map, may be updated in the future to automatically collect images for any map.
```bash
python BEV_map_generator.py
```
This script captures color, depth, segmentation images along with the position of the camera (x,y,z). The camera's Y axis is aligned with the map's North-south axis with the north being positive. The system takes photos on a grid across the map. We first take a low-resolution photo to determine terrain height at the image center, then adjust the Z height such that the terrain depth at the center of the image is always 50 meters. This allows us to get higher resolution photos (which requires low altitudes relative to terrain) without clipping mountaineous regions. 


The images and position data will be placed in a folder called "map_data_binary_50". The following script then performs photogrammetry to extract the maps from this. Additionally, it also uses "meta_data" to give us the map that includes trails and roads (paths.png in the map folder).
```bash
python Extract_map.py
```
The output of this script is already available in the map folder.

