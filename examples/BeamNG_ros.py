import cv2
import numpy as np
from BeamNGRL.BeamNG.beamng_interface import *
import traceback
import yaml
import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Quaternion
from tf.transformations import quaternion_from_euler
from grid_map_msgs.msg import GridMap
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
from rosgraph_msgs.msg import Clock
import tf

class BeamNGROS:
    def __init__(self, map_name, start_pos, start_quat, config_path):
        with open(config_path + 'Map_config.yaml') as f:
            Map_config = yaml.safe_load(f)
        self.bng_interface = get_beamng_remote(
            car_model='RACER',
            start_pos=start_pos,
            start_quat=start_quat,
            map_name=map_name,
            car_make='sunburst',
            beamng_path=BNG_HOME,
            map_res=Map_config["map_res"],
            map_size=Map_config["map_size"],
            host_IP='10.18.171.121'
        )
        self.max_speed = 20
        self.max_steer = 0.5
        self.map_res = Map_config["map_res"]
        self.map_size = Map_config["map_size"]
        self.map_size_px = int(self.map_size / self.map_res)
        self.camera_pos = np.array([0.15, 0, 0.1])

        self.bng_interface.set_lockstep(True)
        # set up publisher for gridmap and vehicle state
        self.gridmap_pub = rospy.Publisher("/elevation_mapping/elevation_map_raw", GridMap, queue_size=1)
        # set up odom publisher
        self.odom_pub = rospy.Publisher('odom', Odometry, queue_size=1)
        # set up imu publisher
        self.imu_pub = rospy.Publisher('imu', Imu, queue_size=1)
        # set up clock publisher
        self.clock_pub = rospy.Publisher('clock', Clock, queue_size=10)
        # set up transform broadcaster:
        self.tf_broadcaster = tf.TransformBroadcaster()
        # set up control subscriber AckermannDriveStamped
        self.control_sub = rospy.Subscriber('hound/control', AckermannDriveStamped, self.control_callback)
    
        self.main_thread()
    
    def main_thread(self):
        while not rospy.is_shutdown():
            self.bng_interface.state_poll()
            state = self.bng_interface.state
            timestamp = rospy.Time.from_sec(self.bng_interface.timestamp)
            
            self.publish_state(state, timestamp)
            BEV_heght = self.bng_interface.BEV_heght + self.bng_interface.BEV_center[2]
            BEV_segmt = self.bng_interface.BEV_segmt
            BEV_color = self.bng_interface.BEV_color
            self.publish_gridmap(BEV_heght, BEV_segmt, BEV_color, timestamp, state)
            self.publish_clock(timestamp)

    def publish_state(self, state, timestamp):
        orientation_quat = Quaternion(*quaternion_from_euler(state[3], state[4], state[5]))
        # publish odometry message
        odom_msg = Odometry()
        odom_msg.header.frame_id = "map"
        odom_msg.child_frame_id = "base_link"
        odom_msg.pose.pose.position.x = state[0]
        odom_msg.pose.pose.position.y = state[1]
        odom_msg.pose.pose.position.z = state[2]
        odom_msg.pose.pose.orientation = orientation_quat
        odom_msg.twist.twist.linear.x = state[6]
        odom_msg.twist.twist.linear.y = state[7]
        odom_msg.twist.twist.linear.z = state[8]
        odom_msg.twist.twist.angular.x = state[12]
        odom_msg.twist.twist.angular.y = state[13]
        odom_msg.twist.twist.angular.z = state[14]
        self.odom_pub.publish(odom_msg)
        # publish imu message
        imu_msg = Imu()
        imu_msg.header.stamp = timestamp
        imu_msg.header.frame_id = "base_link"
        imu_msg.orientation = orientation_quat
        imu_msg.angular_velocity.x = state[12]
        imu_msg.angular_velocity.y = state[13]
        imu_msg.angular_velocity.z = state[14]
        imu_msg.linear_acceleration.x = state[9]
        imu_msg.linear_acceleration.y = state[10]
        imu_msg.linear_acceleration.z = state[11]
        self.imu_pub.publish(imu_msg)

        self.tf_broadcaster.sendTransform((state[0], state[1], state[2]),
                            (orientation_quat.x, orientation_quat.y, orientation_quat.z, orientation_quat.w),
                            timestamp,
                            "base_link",
                            "map")

        self.tf_broadcaster.sendTransform((self.camera_pos[0], self.camera_pos[1], self.camera_pos[2]),
                            (0, 0, 0, 1),
                            timestamp,
                            "camera_depth_frame",
                            "base_link")

        self.tf_broadcaster.sendTransform((0, 0, 0),
                            (-0.5, 0.5, -0.5, 0.5),
                            timestamp,
                            "camera_depth_optical_frame",
                            "camera_depth_frame")


    def publish_gridmap(self, BEV_heght, BEV_segmt, BEV_color, timestamp, state):
        # grid map callback is the inverse of this function:
        grid_map = GridMap()
        grid_map.info.header.stamp = timestamp
        grid_map.info.header.frame_id = "map"
        grid_map.info.resolution = self.map_res
        grid_map.info.length_x = self.map_size
        grid_map.info.length_y = self.map_size
        grid_map.info.pose.position.x = state[0]
        grid_map.info.pose.position.y = state[1]
        grid_map.info.pose.position.z = 0
        grid_map.layers=["elevation", "segmentation", "color"]
        grid_map.basic_layers=["elevation"]

        # add the elevation layer:
        matrix = BEV_heght
        data_array = Float32MultiArray()
        data_array.layout.dim.append(MultiArrayDimension("column_index", matrix.shape[1], matrix.shape[0]*matrix.shape[1]))
        data_array.layout.dim.append(MultiArrayDimension("row_index", matrix.shape[0], matrix.shape[0]))
        data_array.data = matrix.flatten(order='F')
        grid_map.data.append(data_array)

        # add the segmentation layer:
        matrix = BEV_segmt
        data_array = Float32MultiArray()
        data_array.layout.dim.append(MultiArrayDimension("column_index", matrix.shape[1], matrix.shape[0]*matrix.shape[1]))
        data_array.layout.dim.append(MultiArrayDimension("row_index", matrix.shape[0], matrix.shape[0]))
        data_array.data = matrix.flatten(order='F')
        grid_map.data.append(data_array)

        # add the color layer:
        matrix = BEV_color
        data_array = Float32MultiArray()
        data_array.layout.dim.append(MultiArrayDimension("column_index", matrix.shape[1], matrix.shape[0]*matrix.shape[1]))
        data_array.layout.dim.append(MultiArrayDimension("row_index", matrix.shape[0], matrix.shape[0]))
        data_array.data = matrix.flatten(order='F')
        grid_map.data.append(data_array)

        self.gridmap_pub.publish(grid_map)


    def publish_clock(self, timestamp):
        clock_msg = Clock()
        clock_msg.clock = timestamp
        self.clock_pub.publish(clock_msg)

    def control_callback(self, msg):
        ctrl = np.zeros(2)
        ctrl[0] = msg.drive.steering_angle/self.max_steer
        ctrl[1] = msg.drive.speed/self.max_speed
        self.bng_interface.send_ctrl(ctrl, speed_ctrl=True, speed_max = self.max_speed, Kp=1, Ki=0.05, Kd=0.0, FF_gain=0.0)

if __name__ == "__main__":
    # position of the vehicle for tripped_flat on grimap_v2
    start_point = np.array([-67, 336, 34.5])
    start_quat = np.array([0, 0, 0.3826834, 0.9238795])
    map_name = "small_island"
    #initialize the ros node:
    rospy.init_node('BeamNGROS', anonymous=True)
    #initialize the BeamNGROS class:
    config_path = str(Path(os.getcwd()).parent.absolute()) + "/BeamNGRL/control/UW_mppi/Configs/"
    bingchilling = BeamNGROS(map_name, start_point, start_quat, config_path)
    #start the beamng interface:
