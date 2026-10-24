#!/usr/bin/env python3
"""Publish the centerline path and its status for the road follower."""

from __future__ import annotations

import math
from typing import Optional

import numpy as np


class CenterlineRosPublisher:
    """Publish the ground centerline as nav_msgs/Path plus a status string.

    The path is in base_link (rear axle centre on the ground, x forward, y left) at the
    moment the image was taken; header.stamp carries that image time so subscribers can
    compensate the perception delay with odometry. A lost centerline is published as an
    empty path.
    """

    def __init__(self, path_topic: str, status_topic: str, frame_id: str = "base_link"):
        try:
            import rclpy
            from geometry_msgs.msg import PoseStamped
            from nav_msgs.msg import Path
            from std_msgs.msg import String
        except ImportError as exc:
            raise RuntimeError(
                "CenterlineRosPublisher requires ROS2 Python packages "
                "(rclpy, nav_msgs, geometry_msgs, std_msgs); source ROS first."
            ) from exc

        self._rclpy = rclpy
        self._path_class = Path
        self._pose_class = PoseStamped
        self._string_class = String
        self.frame_id = frame_id

        self._owns_rclpy_context = False
        if not rclpy.ok():
            rclpy.init(args=None)
            self._owns_rclpy_context = True
        self._node = rclpy.create_node("centerline_publisher")
        self._path_publisher = self._node.create_publisher(Path, path_topic, 10)
        self._status_publisher = self._node.create_publisher(String, status_topic, 10)

    def publish(self, ground_points: Optional[np.ndarray], status: str, timestamp: float) -> None:
        seconds = int(math.floor(timestamp))
        nanoseconds = min(int(round((timestamp - seconds) * 1e9)), 999_999_999)

        path = self._path_class()
        path.header.frame_id = self.frame_id
        path.header.stamp.sec = seconds
        path.header.stamp.nanosec = nanoseconds
        if ground_points is not None and len(ground_points) > 1:
            points = np.asarray(ground_points, dtype=np.float64)
            # Orientation along the path, so it is a usable plan for Nav2 later.
            headings = np.arctan2(np.gradient(points[:, 1]), np.gradient(points[:, 0]))
            for (x, y), heading in zip(points, headings):
                pose = self._pose_class()
                pose.header = path.header
                pose.pose.position.x = float(x)
                pose.pose.position.y = float(y)
                pose.pose.orientation.z = math.sin(heading / 2.0)
                pose.pose.orientation.w = math.cos(heading / 2.0)
                path.poses.append(pose)
        self._path_publisher.publish(path)

        status_message = self._string_class()
        status_message.data = status
        self._status_publisher.publish(status_message)

    def close(self) -> None:
        self._node.destroy_node()
        if self._owns_rclpy_context and self._rclpy.ok():
            self._rclpy.shutdown()
