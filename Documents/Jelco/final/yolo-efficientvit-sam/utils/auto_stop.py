#ZED X depth sensing and Rviz visualization
#
# Moved here from Autonomous_Driving/Autonomous_Stop/depth_sensing_rviz.py so it can share
# the camera with the road following: main.py grabs the frame and calls update(zed).
# Tuning parameters are in utils/config.py under the same names.
#
# ROS 2 topics:
# /zed/point_cloud  -> 3D point cloud (only with AUTO_STOP_PUBLISH_FULL_CLOUD)
# /safety_stop/roi  -> Green ROI marker (box, or corridor outline when it follows the centerline)
# /safety_stop/roi_points -> obstacle points inside ROI (displayed as red points in RViz)
# /stop_request          -> Boolean obstacle detection signal
#
# Coordinate system: RIGHT_HANDED_Z_UP_X_FWD
# X = forward
# Y = left
# Z = up
#
# Origin: ZED X camera
# Camera height: 1.76m
# Camera mounted centrally on Jimny roof
# Camera is tilted downwards
#
# Frames used for the ROI:
# - camera: the point cloud frame, pitched down by CAMERA_PITCH
# - level:  camera position, z vertical, x = camera heading (the ROI is defined here)
#
# The ROI is a corridor of stations along a path in the level frame: ROI_START..ROI_START+ROI_LENGTH
# measured along the path from the camera, ROI_WIDTH wide, GROUND_CLEARANCE..+ROI_HEIGHT above the
# ground. The path is the centerline (set_path from main.py), or straight ahead without one.

import math
import time

import numpy as np
import pyzed.sl as sl
import rclpy

from geometry_msgs.msg import Point
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header, Bool
from visualization_msgs.msg import Marker

from .config import (
    AUTO_STOP_MIN_CONSECUTIVE_FRAMES,
    AUTO_STOP_PATH_MAX_AGE_SECONDS,
    AUTO_STOP_PUBLISH_FULL_CLOUD,
    AUTO_STOP_ROI_LEGACY_ROTATION,
    AUTO_STOP_ROI_STATION_STEP,
    CAMERA_HEIGHT,
    CAMERA_PITCH,
    CENTERLINE_CAMERA_X,
    CENTERLINE_CAMERA_Y,
    CENTERLINE_CAMERA_YAW_DEG,
    CLOUD_HEIGHT,
    CLOUD_WIDTH,
    FRAME_ID,
    GROUND_CLEARANCE,
    MIN_OBSTACLE_HEIGHT,
    MIN_OBSTACLE_POINTS,
    ROI_HEIGHT,
    ROI_LENGTH,
    ROI_START,
    ROI_WIDTH,
)
from .path_memory import inverse_transform_points

x = ROI_START + ROI_LENGTH / 2.0
z = -CAMERA_HEIGHT + GROUND_CLEARANCE + ROI_HEIGHT / 2.0
# Transform ROI center into camera coordinates
ROI_CENTER_X = np.cos(CAMERA_PITCH) * x - np.sin(CAMERA_PITCH) * z
ROI_CENTER_Y = 0.0
ROI_CENTER_Z = np.sin(CAMERA_PITCH) * x + np.cos(CAMERA_PITCH) * z

ROI_END = ROI_START + ROI_LENGTH
ROI_BOTTOM_Z = -CAMERA_HEIGHT + GROUND_CLEARANCE  # Level frame
ROI_TOP_Z = ROI_BOTTOM_Z + ROI_HEIGHT

MODE_CURVED = "curved"
MODE_STRAIGHT = "straight"
MODE_LEGACY = "straight (legacy)"


# -----------------------------------------
# Frame helpers
# -----------------------------------------

def camera_to_level(points):
    """(N, 3) camera points -> level frame (undo the downward pitch)."""
    c = np.cos(CAMERA_PITCH)
    s = np.sin(CAMERA_PITCH)
    x_l = c * points[:, 0] + s * points[:, 2]
    z_l = -s * points[:, 0] + c * points[:, 2]
    return np.column_stack((x_l, points[:, 1], z_l))


def level_to_camera(points):
    """(N, 3) level points -> camera frame (same rotation as ROI_CENTER_*)."""
    c = np.cos(CAMERA_PITCH)
    s = np.sin(CAMERA_PITCH)
    x_c = c * points[:, 0] - s * points[:, 2]
    z_c = s * points[:, 0] + c * points[:, 2]
    return np.column_stack((x_c, points[:, 1], z_c))


def base_link_to_level(ground_points):
    """(N, 2) base_link ground points -> (N, 2) level-frame xy (camera position as origin)."""
    return inverse_transform_points(
        np.asarray(ground_points, dtype=np.float64),
        (CENTERLINE_CAMERA_X, CENTERLINE_CAMERA_Y, math.radians(CENTERLINE_CAMERA_YAW_DEG)),
    )


# -----------------------------------------
# Corridor along a path
# -----------------------------------------

class Corridor:
    """ROI stations along a path in the level frame, from ROI_START to ROI_END (arc length)."""

    def __init__(self, centers, tangents, s):
        self.centers = centers      # (M, 2)
        self.tangents = tangents    # (M, 2) unit, along the path
        self.normals = np.column_stack((-tangents[:, 1], tangents[:, 0]))  # (M, 2) unit, left
        self.s = s                  # (M,) arc length from the camera
        half = ROI_WIDTH / 2.0 + AUTO_STOP_ROI_STATION_STEP
        self.xy_min = centers.min(axis=0) - half
        self.xy_max = centers.max(axis=0) + half

    def contains(self, xy):
        """Mask of level-frame (N, 2) points inside the corridor, and their arc length."""
        n = len(xy)
        if n == 0:
            return np.zeros(0, dtype=bool), np.zeros(0)
        diff = xy[:, None, :] - self.centers[None, :, :]
        nearest = np.argmin(np.einsum("nmk,nmk->nm", diff, diff), axis=1)
        offset = xy - self.centers[nearest]
        along = np.einsum("nk,nk->n", offset, self.tangents[nearest])
        lateral = np.einsum("nk,nk->n", offset, self.normals[nearest])
        s = self.s[nearest] + along
        inside = (s >= ROI_START) & (s <= ROI_END) & (np.abs(lateral) <= ROI_WIDTH / 2.0)
        return inside, s

    def outline(self):
        """Corridor edges as camera-frame polylines: 4 long edges and the near/far end frames."""
        left = self.centers + self.normals * (ROI_WIDTH / 2.0)
        right = self.centers - self.normals * (ROI_WIDTH / 2.0)

        def edge(xy, z):
            return level_to_camera(np.column_stack((xy, np.full(len(xy), z))))

        lines = [
            edge(left, ROI_BOTTOM_Z), edge(right, ROI_BOTTOM_Z),
            edge(left, ROI_TOP_Z), edge(right, ROI_TOP_Z),
        ]
        for i in (0, -1):
            frame = np.array([
                (*left[i], ROI_BOTTOM_Z), (*right[i], ROI_BOTTOM_Z),
                (*right[i], ROI_TOP_Z), (*left[i], ROI_TOP_Z), (*left[i], ROI_BOTTOM_Z),
            ])
            lines.append(level_to_camera(frame))
        return [line.astype(np.float32) for line in lines]


def build_corridor(level_path):
    """Corridor along a level-frame (N, 2) path sorted by x, or None if the path is unusable.

    Arc length starts where the path crosses x = 0 (the camera); the path is extended
    straight along its first/last segment to cover the camera and the far ROI end.
    """
    if level_path is None or len(level_path) < 2:
        return None
    path = np.asarray(level_path, dtype=np.float64)
    path = path[np.argsort(path[:, 0], kind="stable")]
    steps = np.hypot(*np.diff(path, axis=0).T)
    path = path[np.concatenate(([True], steps > 1e-6))]
    if len(path) < 2 or not np.all(np.isfinite(path)):
        return None

    # Back to the camera plane along the first segment
    if path[0, 0] > 0.0:
        first = path[1] - path[0]
        first /= np.hypot(*first)
        if first[0] < 1e-3:
            return None
        path = np.vstack((path[0] - first * (path[0, 0] / first[0]), path))

    s = np.concatenate(([0.0], np.cumsum(np.hypot(*np.diff(path, axis=0).T))))
    s0 = float(np.interp(0.0, path[:, 0], s))  # x is sorted, so this is the camera crossing
    s -= s0

    # Forward along the last segment to cover the far end of the ROI
    missing = ROI_END + AUTO_STOP_ROI_STATION_STEP - s[-1]
    if missing > 0.0:
        last = path[-1] - path[-2]
        last /= np.hypot(*last)
        path = np.vstack((path, path[-1] + last * missing))
        s = np.append(s, s[-1] + missing)

    count = int(math.ceil(ROI_LENGTH / AUTO_STOP_ROI_STATION_STEP)) + 1
    station_s = np.linspace(ROI_START, ROI_END, count)
    centers = np.column_stack((
        np.interp(station_s, s, path[:, 0]),
        np.interp(station_s, s, path[:, 1]),
    ))
    tangents = np.gradient(centers, axis=0)
    norms = np.hypot(tangents[:, 0], tangents[:, 1])
    if np.any(norms < 1e-9):
        return None
    tangents /= norms[:, None]
    return Corridor(centers, tangents, station_s)


STRAIGHT_CORRIDOR = build_corridor(np.array([[0.0, 0.0], [1.0, 0.0]]))

# -----------------------------------------
# ROS 2 Node
# -----------------------------------------

class DepthSensingNode(Node):

    def __init__(self):

        # Init ROS unless another part of main.py already did
        self._owns_rclpy_context = False
        if not rclpy.ok():
            rclpy.init(args=None)
            self._owns_rclpy_context = True

        super().__init__("depth_sensing_roi")

        self._consecutive_frames = 0
        # (Corridor, time.monotonic()) of the newest centerline; replaced as a whole, so the
        # camera thread can read it without a lock.
        self._path_corridor = None

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

        self.stop_pub = self.create_publisher(
        Bool, "/stop_request", 10
        )

        # The camera is opened (with depth) and grabbed by main.py
        self.point_cloud = sl.Mat()

        self.resolution = sl.Resolution(
            CLOUD_WIDTH,
            CLOUD_HEIGHT
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
    # Centerline path (from main.py, perception rate)
    # -----------------------------------------

    def set_path(self, ground_points):
        """Bend the ROI along this base_link centerline (N, 2); None clears it."""
        if ground_points is None or len(ground_points) < 2:
            self._path_corridor = None
            return
        corridor = build_corridor(base_link_to_level(ground_points))
        self._path_corridor = None if corridor is None else (corridor, time.monotonic())

    def clear_path(self):
        self._path_corridor = None

    def _current_corridor(self):
        """(corridor or None for the legacy box, mode)."""
        path_corridor = self._path_corridor
        if path_corridor is not None:
            corridor, stamp = path_corridor
            if time.monotonic() - stamp <= AUTO_STOP_PATH_MAX_AGE_SECONDS:
                return corridor, MODE_CURVED
        if AUTO_STOP_ROI_LEGACY_ROTATION:
            return None, MODE_LEGACY
        return STRAIGHT_CORRIDOR, MODE_STRAIGHT

    # -----------------------------------------
    # Publish green ROI
    # -----------------------------------------

    def publish_roi(self, stamp, corridor):

        marker = Marker()

        marker.header.frame_id = FRAME_ID
        marker.header.stamp = stamp

        marker.ns = "safety_roi"
        marker.id = 0
        marker.action = Marker.ADD

        if corridor is not None:
            # Corridor outline in camera coordinates
            marker.type = Marker.LINE_LIST
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.05
            for line in corridor.outline():
                for a, b in zip(line[:-1], line[1:]):
                    marker.points.append(Point(x=float(a[0]), y=float(a[1]), z=float(a[2])))
                    marker.points.append(Point(x=float(b[0]), y=float(b[1]), z=float(b[2])))
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 1.0
            self.roi_pub.publish(marker)
            return

        marker.type = Marker.CUBE

        # Position relative to ZED camera
        marker.pose.position.x = ROI_CENTER_X
        marker.pose.position.y = ROI_CENTER_Y
        marker.pose.position.z = ROI_CENTER_Z

        ## No rotation
        #marker.pose.orientation.w = 1.0
        # Rotate ROI by CAMERA_PITCH degrees
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
    # Filter points inside the ROI
    # -----------------------------------------

    def _log_near_far(self, heights, near, far):
        if np.any(near) and np.any(far):
            self.get_logger().info(
                f"Near ROI median height: "
                f"{np.median(heights[near]):.2f} m | "
                f"Far ROI median height: "
                f"{np.median(heights[far]):.2f} m",
                throttle_duration_sec=1.0
            )

    def filter_roi_legacy(self, points):
        """Old straight box (rotates by +CAMERA_PITCH, see AUTO_STOP_ROI_LEGACY_ROTATION)."""

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

        # Calculate height above assumed ground
        height_above_ground = (
            z_roi + ROI_HEIGHT / 2.0 + GROUND_CLEARANCE
        )

        # Select obstacle candidates inside ROI
        obstacle_mask = (
            inside &
            (height_above_ground >= MIN_OBSTACLE_HEIGHT)
        )

        # Compare heights at near and far end of ROI
        self._log_near_far(height_above_ground, inside & (x_roi < 0), inside & (x_roi >= 0))

        return points[obstacle_mask]

    def filter_roi(self, points, corridor):
        """Obstacle points (camera frame) inside the corridor."""
        # Level the points step by step (same rotation as camera_to_level), so each
        # prefilter only touches the points that survived the previous one.
        c = np.float32(np.cos(CAMERA_PITCH))
        s_p = np.float32(np.sin(CAMERA_PITCH))
        y = points[:, 1]
        index = np.flatnonzero((y >= corridor.xy_min[1]) & (y <= corridor.xy_max[1]))
        x_c = points[index, 0]
        z_c = points[index, 2]
        height = c * z_c - s_p * x_c + np.float32(CAMERA_HEIGHT)
        keep = (height >= GROUND_CLEARANCE) & (height <= GROUND_CLEARANCE + ROI_HEIGHT)
        index, x_c, z_c, height = index[keep], x_c[keep], z_c[keep], height[keep]
        x_l = c * x_c + s_p * z_c
        keep = (x_l >= corridor.xy_min[0]) & (x_l <= corridor.xy_max[0])
        index, height, x_l = index[keep], height[keep], x_l[keep]

        inside, s = corridor.contains(
            np.column_stack((x_l, points[index, 1])).astype(np.float64)
        )
        index = index[inside]
        s = s[inside]
        heights = height[inside]

        mid = ROI_START + ROI_LENGTH / 2.0
        self._log_near_far(heights, s < mid, s >= mid)

        return points[index[heights >= MIN_OBSTACLE_HEIGHT]]

    # -----------------------------------------
    # Process the grabbed frame and publish
    # -----------------------------------------

    def update(self, zed):
        """Run on the frame main.py just grabbed; returns (roi_points, obstacle_detected, roi_info).

        roi_info = {"mode": curved / straight / straight (legacy), "lines": camera-frame polylines}.
        """

        corridor, mode = self._current_corridor()
        roi_info = {
            "mode": mode,
            "lines": legacy_box_lines() if corridor is None else corridor.outline(),
        }

        # Retrieve 3D point cloud
        zed.retrieve_measure(
            self.point_cloud,
            sl.MEASURE.XYZ,
            sl.MEM.CPU,
            self.resolution
        )

        data = self.point_cloud.get_data()

        if data is None:
            return None, False, roi_info

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

        # Publish full point cloud (slow, only for tuning in RViz)
        if AUTO_STOP_PUBLISH_FULL_CLOUD:
            cloud_msg = point_cloud2.create_cloud_xyz32(
                header,
                points.tolist()
            )
            self.cloud_pub.publish(cloud_msg)

        # Filter points inside ROI
        if corridor is None:
            roi_points = self.filter_roi_legacy(points)
        else:
            roi_points = self.filter_roi(points, corridor)

        # Check whether enough obstacle points are detected, in enough frames in a row
        if len(roi_points) >= MIN_OBSTACLE_POINTS:
            self._consecutive_frames += 1
        else:
            self._consecutive_frames = 0
        obstacle_detected = self._consecutive_frames >= AUTO_STOP_MIN_CONSECUTIVE_FRAMES

        # Publish ROS 2 stop request
        stop_msg = Bool()
        stop_msg.data = obstacle_detected

        self.stop_pub.publish(stop_msg)

        # Debug output
        self.get_logger().info(
            f"ROI: {mode} | "
            f"Obstacle points: {len(roi_points)} | "
            f"Frames in a row: {self._consecutive_frames} | "
            f"Stop request: {obstacle_detected}",
            throttle_duration_sec=1.0
        )

        # Publish filtered ROI point cloud
        roi_msg = point_cloud2.create_cloud_xyz32(
            header,
            roi_points.tolist()
        )
        self.roi_points_pub.publish(roi_msg)

        # Publish green ROI marker
        self.publish_roi(stamp, corridor)

        return roi_points, obstacle_detected, roi_info

    # -----------------------------------------
    # Cleanup (the camera is closed by main.py)
    # -----------------------------------------

    def close(self):
        self.destroy_node()
        if self._owns_rclpy_context and rclpy.ok():
            rclpy.shutdown()


def roi_corners():
    """The 8 legacy ROI box corners in camera coordinates (inverse of the rotation in filter_roi_legacy)."""
    c = np.cos(CAMERA_PITCH)
    s = np.sin(CAMERA_PITCH)
    corners = []
    for x_roi in (-ROI_LENGTH / 2.0, ROI_LENGTH / 2.0):
        for y_roi in (-ROI_WIDTH / 2.0, ROI_WIDTH / 2.0):
            for z_roi in (-ROI_HEIGHT / 2.0, ROI_HEIGHT / 2.0):
                corners.append((
                    ROI_CENTER_X + c * x_roi + s * z_roi,
                    ROI_CENTER_Y + y_roi,
                    ROI_CENTER_Z - s * x_roi + c * z_roi,
                ))
    return np.array(corners, dtype=np.float32)


def legacy_box_lines():
    """The 12 legacy box edges as 2-point camera-frame polylines."""
    corners = roi_corners()
    # Corner index bits: 4 = far end, 2 = left side, 1 = top
    return [
        corners[[a, a | bit]]
        for a in range(8)
        for bit in (1, 2, 4)
        if a | bit != a
    ]
