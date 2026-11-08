#ZED X depth sensing and Rviz visualization
#
# ROS 2 topics:
# /zed/point_cloud  -> 3D point cloud
# /safety_stop/roi  -> Green ROI marker
# /safety_stop/roi_points -> Filtered 3D point cloud
#
# Coordinate system: RIGHT_HANDED_Z_UP_X_FWD
# X = forward
# Y = left
# Z = up
#
# Origin: ZED X camera
# Camera height: 1.76m
# Camera mounted centrally on Jimny roof

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

# -----------------------------------------
# Camera and ROI parameters (meters)
# -----------------------------------------

FRAME_ID = "zed_camera"

CAMERA_HEIGHT = 1.76

ROI_START = 6.5 #before 4.0
ROI_LENGTH = 3.0
ROI_WIDTH = 2.2
ROI_HEIGHT = 1.5

## ROI center relative to camera
#ROI_CENTER_X = ROI_START + ROI_LENGTH / 2.0
#ROI_CENTER_Y = 0.0
#ROI_CENTER_Z = -CAMERA_HEIGHT + ROI_HEIGHT / 2.0
# Camera tilted 10 degrees downward
CAMERA_PITCH = np.deg2rad(10.0)
# ROI center in horizontal vehicle coordinates
GROUND_CLEARANCE = 0.20  # cm above ground
x = ROI_START + ROI_LENGTH / 2.0
z = -CAMERA_HEIGHT + GROUND_CLEARANCE + ROI_HEIGHT / 2.0
# Transform ROI center into camera coordinates
ROI_CENTER_X = np.cos(CAMERA_PITCH) * x - np.sin(CAMERA_PITCH) * z
ROI_CENTER_Y = 0.0
ROI_CENTER_Z = np.sin(CAMERA_PITCH) * x + np.cos(CAMERA_PITCH) * z

# Point cloud resolution
CLOUD_WIDTH = 640
CLOUD_HEIGHT = 360

PUBLISH_RATE_HZ = 10.0

# -----------------------------------------
# ROS 2 Node
# -----------------------------------------

class DepthSensingNode(Node):

    def __init__(self, svo_path=None, loop=False):

        super().__init__("depth_sensing_roi")

        self.is_svo = svo_path is not None
        self.loop = loop
        self.finished = False

        # ROS 2 publishers
        self.cloud_pub = self.create_publisher(
            PointCloud2,
            "/zed/point_cloud",
            10
        )

        # Publisher for points inside the ROI
        self.roi_points_pub = self.create_publisher(
            PointCloud2,
            "/safety_stop/roi_points",
            10
        )

        self.roi_pub = self.create_publisher(
            Marker,
            "/safety_stop/roi",
            10
        )

        # ZED initialization
        init = sl.InitParameters()

        init.coordinate_units = sl.UNIT.METER
        init.coordinate_system = (
            sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP_X_FWD
        )
        init.depth_mode = sl.DEPTH_MODE.NEURAL #PERFORMANCE

        if self.is_svo:
            init.set_from_svo_file(str(svo_path))

        self.zed = sl.Camera()
        status = self.zed.open(init)

        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(
                f"ZED camera open failed: {status}"
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

    # -----------------------------------------
    # Publish green ROI
    # -----------------------------------------

    def publish_roi(self, stamp):

        marker = Marker()

        marker.header.frame_id = FRAME_ID
        marker.header.stamp = stamp

        marker.ns = "safety_roi"
        marker.id = 0
        marker.type = Marker.CUBE
        marker.action = Marker.ADD

        # Position relative to ZED camera
        marker.pose.position.x = ROI_CENTER_X
        marker.pose.position.y = ROI_CENTER_Y
        marker.pose.position.z = ROI_CENTER_Z

        ## No rotation
        #marker.pose.orientation.w = 1.0
        # Rotate ROI by -10 degrees around Y axis
        marker.pose.orientation.y = -float(np.sin(CAMERA_PITCH / 2.0))
        marker.pose.orientation.w = float(np.cos(CAMERA_PITCH / 2.0))

        # ROI dimensions
        marker.scale.x = ROI_LENGTH
        marker.scale.y = ROI_WIDTH
        marker.scale.z = ROI_HEIGHT

        # Transparent green
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 0.4

        self.roi_pub.publish(marker)

    # -----------------------------------------
    # Filter points inside rotated ROI
    # -----------------------------------------

    def filter_roi(self, points):

        # Translate points relative to ROI center
        dx = points[:, 0] - ROI_CENTER_X
        dy = points[:, 1] - ROI_CENTER_Y
        dz = points[:, 2] - ROI_CENTER_Z

        # Rotate points into ROI coordinate system
        c = np.cos(CAMERA_PITCH)
        s = np.sin(CAMERA_PITCH)

        x_roi = c * dx - s * dz
        y_roi = dy
        z_roi = s * dx + c * dz

        # Check ROI boundaries
        inside = (
            (np.abs(x_roi) <= ROI_LENGTH / 2.0) &
            (np.abs(y_roi) <= ROI_WIDTH / 2.0) &
            (np.abs(z_roi) <= ROI_HEIGHT / 2.0)
        )
        
        if np.any(inside):
             print(
                "Filtered ROI Z range:",
                np.min(z_roi[inside]),
                np.max(z_roi[inside])
             )

        return points[inside]

    # -----------------------------------------
    # Capture and publish point cloud
    # -----------------------------------------

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

            self.finished = True
            self.timer.cancel()
            return

        if status != sl.ERROR_CODE.SUCCESS:
            return

        # Retrieve 3D point cloud
        self.zed.retrieve_measure(
            self.point_cloud,
            sl.MEASURE.XYZ,
            sl.MEM.CPU,
            self.resolution
        )

        data = self.point_cloud.get_data()

        if data is None:
            return

        points = np.asarray(
            data[:, :, :3]
        ).reshape(-1, 3)

        # Remove invalid points
        valid = np.isfinite(points).all(axis=1)
        points = points[valid].astype(np.float32)

        # ROS 2 header
        stamp = self.get_clock().now().to_msg()

        header = Header()
        header.stamp = stamp
        header.frame_id = FRAME_ID

        # Publish full point cloud
        cloud_msg = point_cloud2.create_cloud_xyz32(
            header,
            points.tolist()
        )
        self.cloud_pub.publish(cloud_msg)

        # Filter points inside ROI
        roi_points = self.filter_roi(points)

        self.get_logger().info(
            f"ROI points: {len(roi_points)}",
            throttle_duration_sec=1.0
        )

        # Publish filtered ROI point cloud
        roi_msg = point_cloud2.create_cloud_xyz32(
            header,
            roi_points.tolist()
        )
        self.roi_points_pub.publish(roi_msg)

        # Publish green ROI marker
        self.publish_roi(stamp)
        ## Publish point cloud
        #cloud_msg = point_cloud2.create_cloud_xyz32(
            #header,
            #points.tolist()
        #)

        #self.cloud_pub.publish(cloud_msg)

        # Publish ROI
        #self.publish_roi(stamp)

    # -----------------------------------------
    # Cleanup
    # -----------------------------------------

    def close(self):
        self.zed.close()

# -----------------------------------------
# Main
# -----------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_svo_file",
        type=Path,
        default=None
    )

    parser.add_argument(
        "--loop",
        action="store_true"
    )

    args = parser.parse_args()

    if args.loop and args.input_svo_file is None:
        parser.error("--loop requires --input_svo_file")

    if args.input_svo_file is not None:
        if not args.input_svo_file.is_file():
            parser.error("SVO file not found")

    rclpy.init()
    node = None

    try:
        node = DepthSensingNode(
            args.input_svo_file,
            args.loop
        )

        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node)

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

#Terminal 1:
#source /opt/ros/humble/setup.bash
#python3 depth_sensing_rviz.py --input_svo_file ~/zedx/svo_recording_data/testfahrt_20261101_165532.svo2 --loop

#Termianl 2:
#source /opt/ros/humble/setup.bash
#rviz2
