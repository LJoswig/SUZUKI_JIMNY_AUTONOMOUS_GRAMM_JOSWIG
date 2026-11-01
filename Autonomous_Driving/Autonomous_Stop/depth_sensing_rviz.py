#!/usr/bin/env python3

# Capture and visualize the ZED X 3D point cloud with a green ROI.
#
# ROS 2 topics:
# /zed/point_cloud -> Full 3D point cloud (PointCloud2)
# /safety_stop/roi -> Transparent green ROI (Marker)
#
# Terminal 1 - Live camera:
# source /opt/ros/humble/setup.bash
# python3 depth_sensing_rviz.py
#
# Terminal 1 - SVO recording (no camera required):
# source /opt/ros/humble/setup.bash
# python3 depth_sensing_rviz.py --input_svo_file /path/recording.svo2 --loop
#
# Terminal 2 - RViz2:
# source /opt/ros/humble/setup.bash
# rviz2
#
# RViz2 setup:
# Fixed Frame: zed_camera_frame
# PointCloud2 Topic: /zed/point_cloud
# Marker Topic: /safety_stop/roi


import argparse
from pathlib import Path

import numpy as np
import pyzed.sl as sl
import rclpy

from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
from visualization_msgs.msg import Marker


# -------------------------------------------------
# ROI parameters (camera coordinates, meters)
# -------------------------------------------------

# ZED coordinate system: RIGHT_HANDED_Y_UP
# X = right
# Y = up
# Z = forward

# Temporary test ROI:
# These values are NOT calibrated to the Jimny.
ROI_CENTER_X = 0.0
ROI_CENTER_Y = -0.5
ROI_CENTER_Z = 3.0

ROI_WIDTH = 2.20
ROI_HEIGHT = 1.35
ROI_LENGTH = 4.50

FRAME_ID = "zed_camera_frame"


class DepthSensingNode(Node):

    def __init__(self, svo_path=None, loop=False):
        super().__init__("depth_sensing_roi")

        self.loop = loop
        self.is_svo = svo_path is not None
        self.finished = False

        # -------------------------------------------------
        # ROS 2 publishers
        # -------------------------------------------------

        self.cloud_pub = self.create_publisher(
            PointCloud2,
            "/zed/point_cloud",
            10
        )

        self.roi_pub = self.create_publisher(
            Marker,
            "/safety_stop/roi",
            10
        )

        # -------------------------------------------------
        # ZED camera settings
        # -------------------------------------------------

        init = sl.InitParameters(
            depth_mode=sl.DEPTH_MODE.NEURAL,
            coordinate_units=sl.UNIT.METER,
            coordinate_system=sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
        )

        # Select SVO recording or live camera
        if svo_path is not None:
            init.set_from_svo_file(str(svo_path))

            self.get_logger().info(
                f"Using SVO recording: {svo_path}"
            )

        else:
            self.get_logger().info(
                "Using live ZED camera"
            )

        # Open ZED camera or SVO recording
        self.zed = sl.Camera()
        status = self.zed.open(init)

        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(
                f"ZED initialization failed: {status}"
            )

        # Point cloud storage
        self.point_cloud = sl.Mat()

        # Point cloud output resolution
        self.resolution = sl.Resolution(640, 360)

        # Capture / playback at approximately 10 Hz
        self.timer = self.create_timer(
            0.1,
            self.update
        )

        self.get_logger().info(
            "ZED depth sensing started"
        )

    # -------------------------------------------------
    # Publish green ROI
    # -------------------------------------------------

    def publish_roi(self, stamp):

        marker = Marker()

        marker.header.frame_id = FRAME_ID
        marker.header.stamp = stamp

        marker.ns = "safety_roi_test"
        marker.id = 0

        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        
        
        # ROI position directly in camera coordinates
        marker.pose.position.x = ROI_CENTER_X
        marker.pose.position.y = ROI_CENTER_Y
        marker.pose.position.z = ROI_CENTER_Z

        # No rotation (temporary test)
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0

        # ROI dimensions
        marker.scale.x = ROI_WIDTH
        marker.scale.y = ROI_HEIGHT
        marker.scale.z = ROI_LENGTH

        # Green and transparent
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 0.20

        self.roi_pub.publish(marker)

    # -------------------------------------------------
    # Capture point cloud and publish ROS 2 topics
    # -------------------------------------------------

    def update(self):

        status = self.zed.grab()

        # Handle end of SVO recording
        if (
            self.is_svo
            and status == sl.ERROR_CODE.END_OF_SVOFILE_REACHED
        ):

            if self.loop:
                self.zed.set_svo_position(0)
                return

            self.get_logger().info(
                "End of SVO recording reached"
            )

            self.finished = True
            self.timer.cancel()
            return

        # Handle camera / playback errors
        if status != sl.ERROR_CODE.SUCCESS:

            self.get_logger().warning(
                f"ZED grab failed: {status}",
                throttle_duration_sec=5.0
            )

            return

        # Retrieve XYZ point cloud in CPU memory
        self.zed.retrieve_measure(
            self.point_cloud,
            sl.MEASURE.XYZ,
            sl.MEM.CPU,
            self.resolution
        )

        data = self.point_cloud.get_data()

        if data is None:
            return

        # Extract XYZ coordinates
        xyz = np.asarray(
            data[:, :, :3]
        )

        # Convert image-shaped array into list of 3D points
        points = xyz.reshape(-1, 3)

        # Remove invalid depth measurements
        valid = np.isfinite(points).all(axis=1)

        points = points[valid].astype(
            np.float32
        )

        # ROS 2 timestamp
        stamp = self.get_clock().now().to_msg()

        header = Header()
        header.stamp = stamp
        header.frame_id = FRAME_ID

        # Create ROS 2 PointCloud2 message
        cloud_msg = point_cloud2.create_cloud_xyz32(
            header,
            points.tolist()
        )

        # Publish full point cloud
        self.cloud_pub.publish(
            cloud_msg
        )

        # Publish green ROI
        self.publish_roi(
            stamp
        )

    # -------------------------------------------------
    # Close ZED camera / SVO recording
    # -------------------------------------------------

    def close(self):
        self.zed.close()


# -------------------------------------------------
# Main function
# -------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="ZED depth sensing with RViz2 ROI visualization"
    )

    parser.add_argument(
        "--input_svo_file",
        type=Path,
        default=None,
        help="Path to an SVO or SVO2 recording"
    )

    parser.add_argument(
        "--loop",
        action="store_true",
        help="Repeat the SVO recording continuously"
    )

    args = parser.parse_args()

    # Validate arguments
    if args.loop and args.input_svo_file is None:
        parser.error(
            "--loop requires --input_svo_file"
        )

    if args.input_svo_file is not None:

        if not args.input_svo_file.is_file():
            parser.error(
                f"SVO file not found: {args.input_svo_file}"
            )

        if args.input_svo_file.suffix.lower() not in (
            ".svo",
            ".svo2"
        ):
            parser.error(
                "Recording must have .svo or .svo2 extension"
            )

    # Initialize ROS 2
    rclpy.init()

    node = None

    try:

        node = DepthSensingNode(
            svo_path=args.input_svo_file,
            loop=args.loop
        )

        # Run until interrupted or recording finishes
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(
                node,
                timeout_sec=0.2
            )

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
