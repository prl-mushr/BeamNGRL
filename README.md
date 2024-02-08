# BeamNGRL

## Getting BeamNG:
1) Apply for a BeamNG academic license [here](https://register.beamng.tech/) (use your institute email ID. They usually respond quickly)
2) Download version BeamNG.tech version 0.26 (for now we only support 0.26, will support newer versions soon) and follow their instructions on adding the key/licenses

## Installation:
1) Download the [map folder,](https://drive.google.com/drive/folders/1T4XLVU1tnPTZpCAUkvObrbnER6I_AnQW?usp=drive_link) and the [content folder](https://drive.google.com/drive/folders/1qTb1biMKTuAfmcfRviB0IcYbDuUO4Nqt?usp=sharing).
2) If you are following the HOUND installation instructions, the install scripts will automatically pull this repository for you. You will however still need to pull this repository on your windows system.
```bash
cd path/to/python_installation/
git clone https://github.com/sidtalia/BeamNGRL.git
cd BeamNGRL/
pip install -r requirements.txt
```

4) Put the map_data folder in a directory called "BeamNGRL/data" (you would have to make this directory for now).
```bash
cd ~/BeamNGRL
mkdir data
```
The file structure should look something like:
```bash
BeamNGRL
├── data
    ├── map_data
        ├── elevation_map.npy
        ├── paths.png
        ├── color_map.png
        └── segmt_map.png
```
3) The content folder contains 3 folders: levels, vehicles, and a userfolder. The levels and vehicles folder contain "new" levels and vehicles. The userfolder contains a "mods" folder and a "vehicles" folder that contain modified versions of existing vehicles offered by the simulator. Place the zip files inside the levels and vehicles folder in the main (content) folder inside, you guessed it, the "content" folder in BeamNG.
```bash
BeamNG
├── content
    ├── vehicles
    |   ├── car1.zip
    |   ├── ....
    |   ├── savage_low_f.zip ---> place the folders in here.
    |   ├── savage_normal.zip
    |   └── savage.zip
    └── levels
        ├── small_island.zip
        ├── ....
        └── custom_level.zip ---> place custom levels here
```
Similarly, put the contents of the userfolder into the "userfolder" inside BeamNG. Note that the "userfolder" will not exist until the first time you boot the simulator, so this step requires you to boot the simulator first.
```bash
BeamNG
├── userfolder
    ├── 0.26
    |   ├── mods
    |   ├── vehicles --> place the mods and vehicles folders here
    |   ├── ...
    └──  research_helper.txt
```

Put the content/vehicles/ files in the BeamNG/BeamNG/content/vehicles/ folder. Do the same for "levels" if that folder isn't empty on the google drive.

2) Environment setup:
put this in your bash:
```bash
export BNG_HOME=/absolute/path/to/BeamNG/BeamNG/
export PYTHONPATH="${PYTHONPATH}:/absolute/path/to/BeamNGRL"
```
On Windows, you may need to add the above paths to your "PATH" variable.

### Connecting your Windows and Ubuntu machine via an ethernet cable ([source](https://unix.stackexchange.com/questions/251057/can-i-connect-a-ubuntu-linux-laptop-to-a-windows-10-laptop-via-ethernet-cable)):
1) On the Windows computer: check the current IP by running ipconfig in a terminal/commandline and note the the current IP(s) to compare later
2) On both computers: attach the full-duplex ethernet cable to both machines
3) On the Windows machine: run ipconfig again in the terminal/commandline and compare with previously obtained IPs. It may resemble: 169.254.216.9
4) On the Ubuntu machine: Go to Settings > Network > Edit Connections > select the wired type > create a new wired connection > name it. Copy the following  settings:
```
Method: Manual.
address: 169.254.216.11 (change the last two digits to make sure you don't have the same IP address as the windows machine)
netmask: 255.255.0.0
gateway: leave this blank
```
Save the settings.

5) Check if the connection works: From the ubuntu computer:
```bash
ping <IP_ADDRESS_OF_WINDOWS_MACHINE>
```
You should observe a ping of less than or equal to 1 millisecond (or in that ballpark).

## Running minimal interface example:

### Executing minimal example:
Confirm that the Windows and Ubuntu machine can ping each other. 

On the Windows machine:
```bash
cd path/to/BeamNGRL/examples/
python boot_beamng.py
```
This should start the simulator (you can potentially also just use this to play around in the simulator and explore it)

On the Ubuntu machine:
```bash
cd ~/BeamNGRL/examples
python3 beamng_interface_minimal.py
```

### Minimal example explained
You need to provide a start pos/quat for the vehicle. This would require knowing the exact height where the car needs to be placed. If you place the car at the wrong "height" it will either drop from the sky or drop below the map. This is something I'm working on fixing, such that in the future the correct Z height is extracted from the BEV map itself.
```python
    start_pos = np.array([-86.5, 322.26, 35.5]) ## start pose of the vehicle
    start_quat = np.array([0, 0, 0, 1])
```

The interface spoofs the birds-eye-view elevation map, semantic map, color map, and path map. Currently we only support map-spoofing for the small-island map. The smallgrid map is an empty map, so the elevation map, color map, semantic map and path map are always blank for it. We currently only support the small-island and smallgrid map for BEV access (other maps in progress).
```python
    Map_config = dict()
    Map_config = {
        "map_name": "small_island",
        "map_size": 64, ## this is in meters, and corresponds to the body-centric map's size. This is NOT the size of the full map you will be using.
        "map_res": 0.25, ## this is the resolution of the map in meters/pixel.
        "map_res_hitl": 0.25, ## used by BeamNG_ros for hitl with ROS
        "elevation_range": 4.0, ## BEV_heightmap = clamp(raw_heightmap, raw_heightmap_center_z - 4.0, raw_heightmap_center_z - 4.0)
        "layers": { ## used by BeamNG_ros for hitl with ROS
            "color": 3,
            "elevation": 1,
            "semantics": 3,
            "costmap": 1
        },
        "topic_name": "/grid_map_occlusion_inpainting/all_grid_map" ## used by BeamNG_ROS
    }
```

Sensor configuration. Currently, the camera and lidar are only supported on the windows system. If you run the simulator on the same system as your main code, expect the overall execution to be slow (BeamNG is CPU dependent for all the physics calculations).
The sensor configurations here are defined as dictionaries, but as you will see in other implementations, we usually load them from a yaml file (recommended)
```python
    camera_config = dict()
    lidar_config = dict()
    IMU_config = dict()
    camera_config = {
        "enable": False, ## do you want the camera or not.
        "width": 640,
        "height": 480,
        "fps": 30,
        "fov": 87.0,
        "pos": [0.15, 0.047, 0.02],
        "dir": [0, -1, 0],
        "up": [0, 0, 1],
        "rot": [0, 0, 0, 1],
        "color_optical_frame": "camera_color_optical_frame",
        "depth_optical_frame": "camera_depth_optical_frame",
        "depth_frame": "camera_depth_frame",
        "camera_color_topic": "/camera/color/image_raw",
        "camera_depth_topic": "/camera/depth/image_rect_raw",
        "camera_color_info_topic": "/camera/color/camera_info",
        "camera_depth_info_topic": "/camera/depth/camera_info",
        "monitor_topic": "/camera/depth/image_rect_raw",
        "annotation": False
    }

    lidar_config = {
        "enable": False,
        "rays_per_second_per_scan": 5000,
        "channels": 3,
        "fps": 10,
        "vertical_angle": 26.9,
        "pos": [0.04, 0, 0.07],
        "rot": [0, 0, 0, 1],
        "dir": [0, -1, 0],
        "up": [0, 0, 1],
        "frame": "laser_frame",
        "max_distance": 10.0,
        "scan_topic": "/scan",
        "monitor_topic": "/scan",
        "pc_topic": "converted_pc"
    }

    IMU_config = {
        "pos": [0, 0, 0.1], ## IMU position relative to body-center of the vehicle. All other sensors are in reference to the IMU!
        "fps": 50,
        "monitor_topic": "/mavros/imu/data_raw",
        "pose_topic": "/mavros/local_position/pose",
        "odom_topic": "/mavros/local_position/odom",
        "state_topic": "/mavros/state",
        "gps_topic": "/mavros/gpsstatus/gps1/raw",
        "notification_topic": "/mavros/play_tune",
        "channel_topic": "mavros/rc/in",
        "raw_input_topic": '/mavros/manual_control/send',
        "frame": "base_link",
        "failure_action": "rosrun mavros mavsys rate --all 50"
    }

```

Launching the interface
```python
    bng_interface = get_beamng_default(
        car_model="offroad",
        start_pos=start_pos, ## start position in ENU (east north up). Center of the map is usually 0,0, height is terrain dependent. TODO: use the map model to estimate terrain height.
        start_quat=start_quat, ## start quaternion -- TODO: there should be a ROS to BeamNG to ROS conversion system for reference frames.
        car_make="sunburst", ## car make (company/manufacturer)
        map_config=Map_config, ## Map config; this is "necessary"
        remote=args.remote, ## are you running the simulator remotely (on a separate computer or on the same computer but outside the docker)? 
        host_IP=args.host_IP, ## if using a remote connection (usually the case when running sim on a separate computer)
        camera_config=camera_config, ## currently, camera only works on windows, so you can only use this if you have the sim running remotely or you're using windows as the host
        lidar_config=lidar_config, ## currently, lidar only works on windows, so you can only use this if the sim is running remotely or you're using a windows host
        accel_config=IMU_config, ## IMU config. if left blank, a default config is used.
        burn_time=0.02, ## step or dt time
        run_lockstep=False ## whether the simulator waits for control input to move forward in time. Set to true to have a gym "step" like functionality
    )
```
#### Running the simulator with "lock-step" (similar to gym's env.step()):
```python
    ## this runs the simulator in lock-step, meaning that the simulator will pause while you do your calculations and only run when a control command is sent
    bng_interface.set_lockstep(True)

```

The simulator will take a while to load for the first time, if prompted by the OS to wait/kill program, chose "wait". This only happens the first time you run the game.

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
The images are robot-centric, with the X/Y axes aligned with the NE axes of the map (as in, the map does not rotate with the car). To get the body-frame map (as in, the map rotates around the car), you can set the map config as:
```python
    Map_config = dict()
    Map_config = {
        "map_name": "small_island",
        "map_size": 64, ## this is in meters, and corresponds to the body-centric map's size. This is NOT the size of the full map you will be using.
        "map_res": 0.25, ## this is the resolution of the map in meters/pixel.
        "map_res_hitl": 0.25, ## used by BeamNG_ros for hitl with ROS
        "elevation_range": 4.0, ## BEV_heightmap = clamp(raw_heightmap, raw_heightmap_center_z - 4.0, raw_heightmap_center_z - 4.0)
        "layers": { ## used by BeamNG_ros for hitl with ROS
            "color": 3,
            "elevation": 1,
            "semantics": 3,
            "costmap": 1
        },
        "topic_name": "/grid_map_occlusion_inpainting/all_grid_map", ## used by BeamNG_ROS
        "rotate": True ## ===============================>>>>>>>>>>>>>>>>> this
    }
```

#### Sending control commands:
There are two controls: steering(0) and throttle/brake (1). We can modify this in the future if you wish to have throttle and brake as separate
```python
            action = np.ones(2, dtype=np.float64)  # has to be numpy array. The inputs are always between (-1.0, 1.0) (for both throttle and steering)
            ## steering = action[0], throttle/brake = action[1]. Turning left = positive steering.
            bng_interface.send_ctrl(action) # this sends the commands to the simulator.
```

To use built-in PID-FF wheelspeed controller use:
```python
            bng_interface.send_ctrl(action, speed_ctrl=True, speed_max = max_speed_val, Kp=2, Ki=0.05, Kd=0.0, FF_gain=0.0)
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



## Additional information for map generation (WIP!!!! DO NOT USE):
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

