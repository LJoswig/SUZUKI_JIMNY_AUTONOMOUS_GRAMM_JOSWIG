
#!/usr/bin/env python3

# ZED X depth sensing and ROI visualization for Suzuki Jimny
#
# ROS 2 topics:
# /zed/point_cloud  -> 3D point cloud (PointCloud2)
# /safety_stop/roi  -> Transparent green ROI (Marker)
#
# Coordinate frame: zed_camera_level
# Origin: ZED X camera center
# X = forward (vehicle direction)
# Y = up
# Z = right
#
# The camera is mounted approximately 1.72 m above ground
# and tilted 10 degrees downward.
#
# The point cloud is transformed into a level coordinate
# frame so that the ROI can remain parallel to the ground.
#
# Terminal 1 - SVO recording:
# source /opt/ros/humble/setup.bash
# python3 depth_sensing_rviz.py --input_svo_file /path/recording.svo2 --loop
#
# Terminal 1 - Live camera:
# source /opt/ros/humble/setup.bash
# python3 depth_sensing_rviz.py
#
# Terminal 2 - RViz2:
# source /opt/ros/humble/setup.bash
# rviz2
#
# RViz2:
# Fixed Frame: zed_camera_level
# PointCloud2: /zed/point_cloud
# Marker: /safety_stop/roi

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
# Vehicle and camera geometry (meters)
# -------------------------------------------------

CAMERA_HEIGHT = 1.72
CAMERA_PITCH_DEG = 10.0

JIMNY_WIDTH = 1.645

# -------------------------------------------------
# ROI geometry (meters)
# -------------------------------------------------

# ROI begins 4 m ahead of the camera.
# It extends to 8.5 m ahead of the camera.

ROI_START = 4.0
ROI_LENGTH = 4.50

ROI_WIDTH = 2.20
ROI_HEIGHT = 1.50

# ROI center in level camera-centered coordinates
ROI_CENTER_X = ROI_START + ROI_LENGTH / 2.0
ROI_CENTER_Y = -CAMERA_HEIGHT + ROI_HEIGHT / 2.0
ROI_CENTER_Z = 0.0

FRAME_ID = "zed_camera_level"

# Point cloud output resolution
CLOUD_WIDTH = 640
CLOUD_HEIGHT = 360

# ROS 2 publication rate
PUBLISH_RATE_HZ = 10.0


# -------------------------------------------------
# Coordinate transformation
# -------------------------------------------------

def transform_to_level_frame(points):
    """
    Transform ZED SDK coordinates into a level,
    vehicle-aligned coordinate system.

    Input (RIGHT_HANDED_Y_UP):
        X = right
        Y = up
        Z = forward (camera optical direction)

    Output (zed_camera_level):
        X = forward (horizontal vehicle direction)
        Y = up
        Z = right

    Camera pitch is assumed to be 10 degrees downward.
    """

    theta = np.deg2rad(CAMERA_PITCH_DEG)

    x_cam = -points[:, 2]
    y_cam = points[:, 1]
    z_cam = points[:, 0]

    # Undo the downward camera pitch
    x_level = (
        np.cos(theta) * x_cam
        - np.sin(theta) * y_cam
    )

    y_level = (
        np.sin(theta) * x_cam
        + np.cos(theta) * y_cam
    )

    z_level = z_cam

    return np.column_stack((
        x_level,
        y_level,
        z_level
    )).astype(np.float32)


# -------------------------------------------------
# ROS 2 node
# -------------------------------------------------

class DepthSensingNode(Node):

    def __init__(self, svo_path=None, loop=False):

        super().__init__("depth_sensing_roi")

        self.loop = loop
        self.is_svo = svo_path is not None
        self.finished = False

        # ---------------------------------------------
        # ROS 2 publishers
        # ---------------------------------------------

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

        # ---------------------------------------------
        # ZED SDK initialization
        # ---------------------------------------------

        init = sl.InitParameters(
            depth_mode=sl.DEPTH_MODE.PERFORMANCE,
            coordinate_units=sl.UNIT.METER,
            coordinate_system=sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
        )

        if svo_path is not None:

            init.set_from_svo_file(str(svo_path))

            self.get_logger().info(
                f"Using SVO recording: {svo_path}"
            )

        else:

            self.get_logger().info(
                "Using live ZED camera"
            )

        self.zed = sl.Camera()

        status = self.zed.open(init)

        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(
                f"ZED initialization failed: {status}"
            )

        self.point_cloud = sl.Mat()

        self.resolution = sl.Resolution(
            CLOUD_WIDTH,
            CLOUD_HEIGHT
        )

        self.timer = self.create_timer(
            1.0 / PUBLISH_RATE_HZ,
            self.update
        )

        self.get_logger().info(
            "ZED depth sensing started"
        )

        self.get_logger().info(
            f"ROI: X={ROI_START:.2f} to "
            f"{ROI_START + ROI_LENGTH:.2f} m, "
            f"width={ROI_WIDTH:.2f} m, "
            f"height={ROI_HEIGHT:.2f} m"
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

        # ROI center
        marker.pose.position.x = ROI_CENTER_X
        marker.pose.position.y = ROI_CENTER_Y
        marker.pose.position.z = ROI_CENTER_Z

        # No rotation required in the level frame
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0

        # Dimensions:
        # X = length
        # Y = height
        # Z = width
        marker.scale.x = ROI_LENGTH
        marker.scale.y = ROI_HEIGHT
        marker.scale.z = ROI_WIDTH

        # Green, transparent
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 0.20

        self.roi_pub.publish(marker)

    # -------------------------------------------------
    # Capture and publish point cloud
    # -------------------------------------------------

    def update(self):

        status = self.zed.grab()

        # End of SVO recording
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

        if status != sl.ERROR_CODE.SUCCESS:

            self.get_logger().warning(
                f"ZED grab failed: {status}",
                throttle_duration_sec=5.0
            )

            return

        # ---------------------------------------------
        # Retrieve 3D point cloud
        # ---------------------------------------------

        self.zed.retrieve_measure(
            self.point_cloud,
            sl.MEASURE.XYZ,
            sl.MEM.CPU,
            self.resolution
        )

        data = self.point_cloud.get_data()

        if data is None:
            return

        xyz = np.asarray(data[:, :, :3])

        points = xyz.reshape(-1, 3)

        # Remove invalid measurements
        valid = np.isfinite(points).all(axis=1)

        points = points[valid].astype(np.float32)

        # ---------------------------------------------
        # Transform into level coordinate frame
        # ---------------------------------------------

        points = transform_to_level_frame(points)

        # ---------------------------------------------
        # ROS 2 timestamp and header
        # ---------------------------------------------

        stamp = self.get_clock().now().to_msg()

        header = Header()
        header.stamp = stamp
        header.frame_id = FRAME_ID

        # ---------------------------------------------
        # Publish PointCloud2
        # ---------------------------------------------

        cloud_msg = point_cloud2.create_cloud_xyz32(
            header,
            points.tolist()
        )

        self.cloud_pub.publish(cloud_msg)

        # ---------------------------------------------
        # Publish green ROI
        # ---------------------------------------------

        self.publish_roi(stamp)

    # -------------------------------------------------
    # Cleanup
    # -------------------------------------------------

    def close(self):
        self.zed.close()


# -------------------------------------------------
# Main
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

    rclpy.init()

    node = None

    try:

        node = DepthSensingNode(
            svo_path=args.input_svo_file,
            loop=args.loop
        )

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
