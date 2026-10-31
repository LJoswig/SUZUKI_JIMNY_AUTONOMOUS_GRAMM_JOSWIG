# Capture and visualize the ZED X 3D point cloud with a green ROI.
# Output (ROS 2 topics):
# /zed/point_cloud  -> Full 3D point cloud (PointCloud2)
# /safety_stop/roi  -> Transparent green ROI (Marker)
# How to run on the Jetson:
#
# Terminal 1 – Start the Python script:
# source /opt/ros/humble/setup.bash
# python3 depth_sensing_rviz.py
#
# Terminal 2 – Open RViz2:
# source /opt/ros/humble/setup.bash
# rviz2
#
# RViz2 setup:
# 1. Set Fixed Frame to "zed_camera_frame".
# 2. Add PointCloud2 -> Topic: /zed/point_cloud
# 3. Add Marker -> Topic: /safety_stop/roi



import math
import numpy as np
import pyzed.sl as sl
import rclpy

from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
from visualization_msgs.msg import Marker


# -------------------------------------------------
# Vehicle and camera parameters (meters / degrees)
# -------------------------------------------------

SPEED_KMH = 5.0

# Initial estimates - calibrate on the actual Jimny
CAMERA_HEIGHT = 1.70
CAMERA_TO_BUMPER = 1.20
CAMERA_PITCH_DEG = 10.0  # Downward

# ROI relative to front bumper and road
ROI_WIDTH = 2.20
ROI_START = 0.50
ROI_END = 5.00
ROI_MIN_HEIGHT = 0.15
ROI_MAX_HEIGHT = 1.50

FRAME_ID = "zed_camera_frame"


class DepthSensingNode(Node):

    def __init__(self):
        super().__init__("depth_sensing_roi")

        # ROS 2 publishers
        self.cloud_pub = self.create_publisher(
            PointCloud2, "/zed/point_cloud", 10
        )

        self.roi_pub = self.create_publisher(
            Marker, "/safety_stop/roi", 10
        )

        # ZED camera settings
        init = sl.InitParameters(
            depth_mode=sl.DEPTH_MODE.NEURAL,
            coordinate_units=sl.UNIT.METER,
            coordinate_system=sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
        )

        self.zed = sl.Camera()
        status = self.zed.open(init)

        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"ZED camera error: {status}")

        self.point_cloud = sl.Mat()
        self.resolution = sl.Resolution(640, 360)

        # Capture at 10 Hz initially
        self.timer = self.create_timer(0.1, self.update)

        self.get_logger().info(
            f"ZED depth sensing started - {SPEED_KMH} km/h"
        )

    def publish_roi(self, stamp):
        marker = Marker()

        marker.header.frame_id = FRAME_ID
        marker.header.stamp = stamp
        marker.ns = "safety_roi"
        marker.id = 0

        marker.type = Marker.CUBE
        marker.action = Marker.ADD

        # ROI in vehicle coordinates:
        # X right, Y up, Z forward
        z_min = CAMERA_TO_BUMPER + ROI_START
        z_max = CAMERA_TO_BUMPER + ROI_END

        center_vehicle = np.array([
            0.0,
            (ROI_MIN_HEIGHT + ROI_MAX_HEIGHT) / 2,
            (z_min + z_max) / 2
        ])

        # Convert vehicle coordinates to camera coordinates
        theta = math.radians(CAMERA_PITCH_DEG)
        c = math.cos(theta)
        s = math.sin(theta)

        rotation = np.array([
            [1, 0, 0],
            [0, c, -s],
            [0, s, c]
        ])

        camera_position = np.array([
            0.0, CAMERA_HEIGHT, 0.0
        ])

        center_camera = rotation @ (
            center_vehicle - camera_position
        )

        marker.pose.position.x = float(center_camera[0])
        marker.pose.position.y = float(center_camera[1])
        marker.pose.position.z = float(center_camera[2])

        # Rotate box into camera frame
        marker.pose.orientation.x = math.sin(theta / 2)
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = math.cos(theta / 2)

        marker.scale.x = ROI_WIDTH
        marker.scale.y = ROI_MAX_HEIGHT - ROI_MIN_HEIGHT
        marker.scale.z = ROI_END - ROI_START

        # Green, transparent
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 0.20

        self.roi_pub.publish(marker)

    def update(self):
        if self.zed.grab() != sl.ERROR_CODE.SUCCESS:
            return

        # Retrieve XYZ point cloud in CPU memory
        self.zed.retrieve_measure(
            self.point_cloud,
            sl.MEASURE.XYZ,
            sl.MEM.CPU,
            self.resolution
        )

        data = self.point_cloud.get_data()
        xyz = np.asarray(data[:, :, :3])

        # Remove invalid depth measurements
        points = xyz.reshape(-1, 3)
        valid = np.isfinite(points).all(axis=1)
        points = points[valid].astype(np.float32)

        stamp = self.get_clock().now().to_msg()

        header = Header()
        header.stamp = stamp
        header.frame_id = FRAME_ID

        # Publish ROS 2 PointCloud2
        cloud_msg = point_cloud2.create_cloud_xyz32(
            header, points.tolist()
        )

        self.cloud_pub.publish(cloud_msg)

        # Publish green ROI
        self.publish_roi(stamp)

    def close(self):
        self.zed.close()


def main():
    rclpy.init()
    node = None

    try:
        node = DepthSensingNode()
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        if node is not None:
            node.close()
            node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
