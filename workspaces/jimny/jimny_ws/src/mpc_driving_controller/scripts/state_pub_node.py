#!/usr/bin/python3

"""
state_pub_node.py
Author: Axel Barbelanne, Matthijs Steyerberg
Date: 22-10-2024

Script for node publishing the vehicle state from GPS, IMU and steering angle data.
Modified from Vijay Govindarajan's code: https://github.com/MPC-Berkeley/genesis_path_follower?tab=MIT-1-ov-file

Subscribes to:
    - /gps/fix: GPS fix data
    - /gps/vel: GPS velocity data
    - /imu/data: IMU data
    - /vehicle/steering: Steering angle data
Publishes:
    - /state_est: Vehicle state data
"""

import rclpy
from rclpy.node import Node
import math as m
import numpy as np
from tf_transformations import euler_from_quaternion
from gps_msgs.msg import GPSFix
from sensor_msgs.msg import Imu
from mpc_driving_controller.msg import StateEst
from std_msgs.msg import Float64
from mpc_driving_controller_py.ref_gps_traj import latlon_to_XY

class StatePublisher(Node):
    """
    Node for publishing the vehicle state from GPS, IMU and steering angle data.
    """
    def __init__(self):
        super().__init__('state_publisher_node')

        attrs = ['tm_gps', 'lat', 'lon', 'x', 'y', 
                 'tm_vel', 'v', 'v_long', 'v_lat', 
                 'tm_imu', 'psi', 'long_accel', 'lat_accel', 'yaw_rate', 
                 'tm_df', 'df']
        for attr in attrs:
            setattr(self, attr, None)

        self.time_check_on = self.declare_parameter('time_check_on', True).value  # With descriptor

        self.initializing = True
        self.LAT0 = None
        self.LON0 = None
        self.gps_subscriber = self.create_subscription(GPSFix, 'gps_raw', self.gps_callback, 10)

        self.create_subscription(Imu, 'imu_attitude', self._parse_imu_data, 10)
        # self.create_subscription(Float64, '/vehicle/steering', self._parse_steering_angle, 10)
        self.create_subscription(GPSFix, 'gps_raw', self._parse_gps_fix, 10)

        self.state_pub = self.create_publisher(StateEst, 'state_est', 10)

        # Timer definition
        self.timer = self.create_timer(0.01, self.main_loop)

    def main_loop(self):
        if self.initializing:
            return

        # Check if all necessary timestamps are available
        if None in [self.tm_gps, self.tm_imu]:
            return

        curr_state = StateEst()
        curr_state.header.stamp = self.get_clock().now().to_msg()

        if self.time_check_on:
            time_valid = self._is_time_valid(self._extract_ros_time(curr_state),
                                       [self.tm_gps, self.tm_imu])
            if not time_valid:
                return  # Skip this cycle if the time is not valid

        # Update the current state values
        curr_state.lat = self.lat
        curr_state.lon = self.lon
        curr_state.x = self.x
        curr_state.y = self.y
        curr_state.psi = self.psi
        curr_state.v = self.v

        curr_state.yaw_rate = self.yaw_rate

        curr_state.a_long = self.long_accel
        curr_state.a_lat = self.lat_accel
        # curr_state.df = self.df

        # Publish the current state
        self.state_pub.publish(curr_state)
        self.get_logger().info("State est message published")

    def gps_callback(self, msg):
        # Initialize the reference trajectory generator, only runs once (for now)
        if self.initializing:
            self.initializing = False
            self.LAT0 = msg.latitude
            self.LON0 = msg.longitude

    def _extract_ros_time(self, msg):
        return msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec

    def _is_time_valid(self, tm_now, tm_arr, thresh=0.05):
        """ Check if the current time is close to the received times """
        for tm_rcvd in tm_arr:
            diff = m.fabs(tm_now - tm_rcvd)
            if diff > thresh:
                return False
        return True

    def _parse_gps_fix(self, msg):
        self.get_logger().info("GPS message received")
        self.v = msg.speed
        
        self.lat = msg.latitude
        self.lon = msg.longitude
        self.x, self.y = latlon_to_XY(self.LAT0, self.LON0, self.lat, self.lon)
        self.tm_gps = self._extract_ros_time(msg)

    def _parse_imu_data(self, msg):
        self.get_logger().info("IMU message received")
        ori = msg.orientation
        quat = (ori.x, ori.y, ori.z, ori.w)
        _, _, yaw = euler_from_quaternion(quat)

        # Offset and wrap yaw angle
        psi = yaw + 0.5 * m.pi
        self.psi = (psi + np.pi) % (2. * np.pi) - np.pi

        self.long_accel = msg.linear_acceleration.x
        self.lat_accel = -msg.linear_acceleration.y
        self.yaw_rate = msg.angular_velocity.z

        self.tm_imu = self._extract_ros_time(msg)

    # def _parse_steering_angle(self, msg):
    #     self.df = m.radians(msg.data) / 15.87  # Steering angle in radians
    #     self.tm_df = self._extract_ros_time(msg)


def main(args=None):
    rclpy.init(args=args)
    node = StatePublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
