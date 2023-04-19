import time

from beamngpy import BeamNGpy, Scenario, Vehicle
from beamngpy import Camera
import cv2
import numpy as np


def main():
    bng = BeamNGpy('localhost', 64256, home='G:/BeamNG/BeamNG', user='G:/BeamNG/BeamNG/userfolder')
    # Launch BeamNG.tech
    bng.open()

    scenario = Scenario('small_island', name="test integration")

    vehicle = Vehicle('ego_vehicle', model='sunburst', partConfig='vehicles/sunburst/RACER.pc')
    
    NE = np.array([400,  400, 100])
    SW = np.array([-400, -400, 100])
    resolution = 50
    positions = np.mgrid[SW[0]:(NE[0]+resolution):resolution, SW[1]:(NE[1]+resolution):resolution].reshape(2,-1).T
    scenario.add_vehicle(vehicle, pos=(-60, 336, 35.5),
                         rot_quat=(0, 0, 0.3826834, 0.9238795))
    bng.set_tod(0.5)
    scenario.make(bng)
    bng.load_scenario(scenario)
    bng.start_scenario()
    time.sleep(5)
    color_list = []
    depth_list = []
    segment_list = []
    poses = []
    # bng.pause()
    time.sleep(2)
    for i in range(len(positions)):
        x = positions[i][0]
        y = positions[i][1]
        print(x,y)

        camera = Camera('camera', bng, vehicle, pos=(int(x), int(y), 100), dir=(0,0,-1), field_of_view_y=90, resolution=(100, 100), update_priority=1,
                        is_render_colours=True, is_render_depth=True, is_render_annotations=True,is_visualised=False,
                        requested_update_time=0.01, near_far_planes=(0.1, 2000.0), is_using_shared_memory=True, is_static=True) # get camera data
        time.sleep(1)
        camera_readings = camera.poll()
        color = camera_readings['colour']
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        depth = camera_readings['depth']
        segment = camera_readings['annotation']
        center_depth = depth[depth.shape[0]//2, depth.shape[1]//2]
        adjusted_height = 100 + (50 - center_depth)
        camera.remove()
        time.sleep(1)
        camera = Camera('camera', bng, vehicle, pos=(int(x), int(y), adjusted_height), dir=(0,0,-1), field_of_view_y=90, resolution=(1000, 1000), update_priority=1,
                        is_render_colours=True, is_render_depth=True, is_render_annotations=True,is_visualised=False,
                        requested_update_time=0.01, near_far_planes=(0.1, 2000.0), is_using_shared_memory=True, is_static=True) # get camera data
        time.sleep(1)
        camera_readings = camera.poll()
        color = camera_readings['colour']
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        depth = camera_readings['depth']
        segment = camera_readings['annotation']
        print(color.shape)
        cv2.imshow('window', cv2.resize(color, (400,400), cv2.INTER_AREA))
        cv2.waitKey(2)
        poses.append([x,y,adjusted_height])
        color_list.append(color)
        depth_list.append(depth)
        segment_list.append(segment)
        camera.remove()
        time.sleep(1)
    color_list = np.array(color_list)
    depth_list = np.array(depth_list)
    segment_list = np.array(segment_list)
    poses = np.array(poses)
    np.save("../../map_data_binary_50/color.npy", color_list)
    np.save("../../map_data_binary_50/depth.npy", depth_list)
    np.save("../../map_data_binary_50/segment.npy", segment_list)
    np.save("../../map_data_binary_50/positions.npy", poses)
    # bng.close()


if __name__ == '__main__':
    main()
'''
ENU
NE [400,  450   25.7465919 ]
SW: [-400, -450   32.47618165]
ZMAx: [ -49.16995027 -214.5803518    96.46927808]
'''