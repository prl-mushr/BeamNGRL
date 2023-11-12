import time
from beamngpy import BeamNGpy, Scenario, Vehicle
from beamngpy.sensors import Camera
import cv2
import numpy as np
import os
from pathlib import Path
import BeamNGRL
import argparse
import yaml
import math as m
import traceback
import numba
from numba import jit
import json
import BeamNGRL
import zipfile
import faulthandler

faulthandler.enable()

ROOT_PATH = Path(BeamNGRL.__file__).parent
DATA_PATH = ROOT_PATH.parent / 'data'
BNG_HOME = os.environ.get('BNG_HOME')

## photogrammetry. Yes I'm old school.
@jit(nopython=True)
def photogrammetry(depth_image, elevation_map, pos, color_image, color_map, segment_image, segment_map, resolution, F_inv):
    CX_DEPTH = depth_image.shape[0]/2
    CY_DEPTH = depth_image.shape[1]/2
    ## camera position in BEV image space(meters -> pixels):
    position = np.array([pos[0]/resolution + elevation_map.shape[0]//2, pos[1]/resolution + elevation_map.shape[1]//2], dtype=np.int32)
    width = depth_image.shape[0]
    height = depth_image.shape[1]
    for u in range(0, depth_image.shape[0], 1):  # change the "1" to 5 to produce a quick preview before you go for full resolution
        for v in range(0, depth_image.shape[1], 1):
            z = depth_image[u,v]
            y = ((v - CY_DEPTH)*z*F_inv)
            x = ((u - CX_DEPTH)*z*F_inv)
            U = -int(y/resolution) + position[1]
            V = -int(x/resolution) + position[0]
            U = max(0, min(U, elevation_map.shape[0]-1))
            V = max(0, min(V, elevation_map.shape[1]-1))
            elevation_map[U,V] = pos[2] - z
            color_map[U,V] = color_image[u,v]
            segment_map[U,V] = segment_image[u,v]

    return elevation_map, color_map, segment_map

def read_json(zip_file_path, json_file_path_within_zip):
    if os.path.exists(zip_file_path):
        # Extract the JSON file from the zip
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            if json_file_path_within_zip in zip_ref.namelist():
                with zip_ref.open(json_file_path_within_zip) as json_file:
                    output = json_file.readlines()
            else:
                print(f"The file does not exist in the zip.")
                return None
    else:
        print(f"The zip file does not exist.")
        return None
    return output

def main(config_path, args):
    with open(config_path) as f:
        Config = yaml.safe_load(f)

    cam_height = Config["cam_height"]
    step_size = Config["step_size"]
    map_data_path = str(DATA_PATH / Config["output_dir"])
    fov_degrees = Config["fov"]
    fov = fov_degrees/57.3
    image_size = Config["image_size"]
    map_exclude_list = Config["map_exclude_list"]
    F_inv = m.tan(fov*0.5)/(image_size/2)
    resolution = Config["map_res"] ## this is like 0.1 by default
    maps_file_path =  str(DATA_PATH) + '/content/levels/'
    file_names = [f for f in os.listdir(maps_file_path) if os.path.isfile(os.path.join(maps_file_path, f))]
    map_names = [os.path.splitext(f)[0] for f in file_names]
    print("map list:", map_names)
    print("maps_file_path: ", maps_file_path)

    if args.remote ==True and args.host_IP is not None:
        bng = BeamNGpy(args.host_IP, 64256, remote=True)
        bng.open(launch=False, deploy=False)
        shmem = False
    elif args.remote  == True and args.host_IP is None:
        print("~Ara Ara! Trying to run BeamNG remotely without providing any host IP?")
        exit()
    else:
        bng = BeamNGpy(host, port, home=BeamNG_path, user=BeamNG_path + '/userfolder')
        bng.open()
        shmem = True

    for map_name in map_names:
        if any(map_name == s for s in map_exclude_list):
            print("Skipping {} because it is not actually meant to be used for sim".format(map_name))
            continue
        zip_file_path =  str(DATA_PATH) + '/content/levels/{}.zip'.format(map_name)
        road_file = 'levels/{}/main/MissionGroup/roads/items.level.json'.format(map_name)
        alt_road_file = 'levels/{}/main/MissionGroup/Decal_roads/items.level.json'.format(map_name)
        alt_2_road_file = 'levels/{}/main/MissionGroup/ai_paths/items.level.json'.format(map_name)
        info_file = 'levels/{}/info.json'.format(map_name)
        map_dir_name = map_data_path + "/" + map_name
        if(not os.path.isdir(map_dir_name)):
            os.makedirs(map_dir_name)
        else:
            if os.path.exists(map_dir_name + '/color_map.png'):
                print("map for {} exists, skipping...".format(map_name))
                continue

        trail_data = read_json(zip_file_path, road_file)
        if trail_data == None:
            trail_data = read_json(zip_file_path, alt_road_file)
        if trail_data == None:
            trail_data = read_json(zip_file_path, alt_2_road_file)
        if trail_data == None:
            print("trail data does not exist for map_name:", map_name)
        info_data = read_json(zip_file_path, info_file)
        if info_data == None:
            print("no info data available for map_name {}, skipping..".format(map_name))
            continue
        json_str = b''.join(info_data).decode('utf-8')
        data = json.loads(json_str)
        map_size = np.array(data.get("size", []))
        ## keep constant map size because we don't want to blow up the memory.
        if map_size[0] == -1:
            map_size = np.array([1024, 1024])

        NE = np.array(map_size)//2
        SW = -NE
        positions = np.mgrid[SW[0] + step_size : NE[0]:step_size, SW[1] + step_size: NE[1] : step_size].reshape(2,-1).T

        elevation_map_size = map_size/resolution  # pixels needed for the whole map
        elevation_map = np.zeros((int(elevation_map_size[0]) , int(elevation_map_size[1])), dtype=np.float32)  
        color_map = np.zeros((int(elevation_map_size[0]) , int(elevation_map_size[1]), 3), dtype=np.uint8)
        segment_map = np.zeros((int(elevation_map_size[0]) , int(elevation_map_size[1]), 4), dtype=np.uint8)

        scenario = Scenario(map_name, name="map_generate")
        vehicle = Vehicle('ego_vehicle', model='sunburst', partConfig='vehicles/sunburst/offroad.pc')        
        scenario.add_vehicle(vehicle, pos= (NE[0], NE[1], 100),
                             rot_quat=(0, 0, 0, 1))
        bng.set_tod(0.5)
        scenario.make(bng)
        bng.load_scenario(scenario)
        bng.start_scenario()
        time.sleep(2)
        bng.set_deterministic()
        print("pausing before starting")
        print("map_name: ", map_name)
        bng.set_steps_per_second(50)
        bng.pause()

        for i in range(len(positions)):
            x = positions[i][0]
            y = positions[i][1]
            hit = False
            camera = None
            cam_height = Config["cam_height"]
            bng_steps = 5
            while not hit:
                try:
                    now = time.time()
                    camera = Camera('camera', bng, vehicle, pos=(int(x), int(y), cam_height), dir=(0,0,-1), field_of_view_y=fov_degrees, resolution=(100, 100), update_priority=1,
                                    is_render_colours=True, is_render_depth=True, is_render_annotations=True,is_visualised=False,
                                    requested_update_time=0.01, near_far_planes=(0.1, 2000.0), is_using_shared_memory=shmem, is_static=True) # get camera data
                    # time.sleep(sleep_time)
                    bng.step(bng_steps)
                    camera_readings = camera.poll()
                    depth = camera_readings['depth']
                    center_depth = depth[depth.shape[0]//2, depth.shape[1]//2]
                    adjusted_height = cam_height + (step_size - center_depth)
                    camera.remove()
                    # time.sleep(sleep_time)
                    bng.step(bng_steps)
                    camera = Camera('camera', bng, vehicle, pos=(int(x), int(y), adjusted_height), dir=(0,0,-1), field_of_view_y=fov_degrees, resolution=(image_size, image_size), update_priority=1,
                                    is_render_colours=True, is_render_depth=True, is_render_annotations=True,is_visualised=False,
                                    requested_update_time=0.01, near_far_planes=(0.1, 2000.0), is_using_shared_memory=shmem, is_static=True) # get camera data
                    # time.sleep(sleep_time)
                    bng.step(bng_steps)
                    camera_readings = camera.poll()
                    color = camera_readings['colour']
                    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
                    depth = camera_readings['depth']
                    segment = camera_readings['annotation']
                    camera.remove()
                    # time.sleep(sleep_time)
                    bng.step(bng_steps)
                    hit = True
                    pose = np.array([x,y,adjusted_height])
                    elevation_map, color_map, segment_map = photogrammetry(depth, elevation_map, pose, color, color_map, segment, segment_map, resolution, F_inv)
                    if i%20==0:
                        print(x,y)
                        print('Frame: %d/%d, Time: %f' % (i, len(positions), dt))
                    cv2.imshow('color', cv2.resize(color, (400,400), cv2.INTER_AREA))
                    cv2.imshow('color_map', cv2.resize(color_map, (800,800), cv2.INTER_AREA))
                    cv2.waitKey(2)
                    dt = time.time() - now
                    if i == 10:
                        TTC = dt*(len(positions) - 10)
                        print("estimated time to completion: {} minutes".format(TTC/60))
                except:
                    if camera is not None:
                        camera.remove()
                    cam_height *= 0.9
                    print("hold up")

        bng.resume()

        if not os.path.exists(map_dir_name + "/paths.png"):
            image_shape = color_map.shape
            path_map = np.ones_like(color_map)*255 if trail_data else np.zeros_like(color_map)
            if trail_data is not None:
                i = 0
                for line in trail_data:
                    line = json.loads(line.decode('utf-8'))
                    i += 1
                    try:
                        pos = np.array(line["nodes"])[:,:2]
                        img_pos = np.array(pos/resolution + (image_shape[0]//2), dtype= np.int32).reshape((-1, 1, 2))
                        path_map = cv2.polylines(path_map, [img_pos], False, (0,0,0), int(3/resolution) )
                    except:
                        print("no nodes on line: ", i)

            path_map = cv2.blur(path_map, (32, 32))

            cv2.imwrite(map_dir_name + '/paths.png', path_map)
        cv2.imwrite(map_dir_name + '/elevation_map.png', (elevation_map*65535/100).astype(np.uint16))
        cv2.imwrite(map_dir_name + '/color_map.png', color_map)
        cv2.imwrite(map_dir_name + '/segmt_map.png', segment_map)
        np.save(map_dir_name + '/elevation_map.npy', elevation_map)
        print("saved")

    bng.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name", type=str, default="Map_Gen_Config.yaml", help="name of the config file to use")
    parser.add_argument("--remote", type=bool, default=True, help="whether to connect to a remote beamng server")
    parser.add_argument("--host_IP", type=str, default="169.254.216.9", help="host ip address if using remote beamng")
    args = parser.parse_args()
    config_name = args.config_name
    config_path = str(ROOT_PATH.parent) + "/Configs/" + config_name
    main(config_path, args)