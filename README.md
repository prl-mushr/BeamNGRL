# BeamNGRL

## Installation:
1) Download the map folder and BeamNG folder from [here](https://drive.google.com/drive/folders/1sZ8aDUqtnTomdXn6bxoryJ8X4yT7RS06). Extract the 'BeamNG' and 'map' folder outside the BeamNGRL folder(usually `/home/username/` directory, for instance, I keep it in `/home/stark/`)


2) inside the BeamNGRL folder, run:

```
git submodule update --init --recursive
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
            BEV_color = bng_interface.BEV_color
            BEV_heght = (bng_interface.BEV_heght + 2.0)/4.0  # note that BEV_heght (elevation) has a range of +/- 2 meters around the center of the elevation.
            BEV_segmt = bng_interface.BEV_segmt
            BEV_path  = bng_interface.BEV_path  # trail/roads
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

