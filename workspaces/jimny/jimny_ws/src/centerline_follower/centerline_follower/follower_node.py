#!/usr/bin/env python3
"""
Pure pursuit road follower on the camera centerline, in shadow mode.

Subscribes:
  /centerline/path    nav_msgs/Path, base_link at image time (header.stamp = image time)
  /centerline/status  std_msgs/String: observed, coasting, remembered or lost
  /imu_attitude       sensor_msgs/Imu, angular_velocity.z (cube: positive = right turn)
  /feedback_data      std_msgs/Float32MultiArray, data[1] = measured speed in m/s

Publishes:
  /centerline_follower/cmd    ackermann_msgs/AckermannDrive (steering-wheel degrees, positive =
                              right, like /ackermann_cmd); speed 0 means not tracking. mode_switch
                              uses it in ACRO; in other modes nothing acts on it (shadow mode)
  /centerline_follower/debug       std_msgs/String with a JSON object per control cycle

The path is kept in a dead-reckoned odom frame at the pose where its image was taken and
evaluated at the current pose, which compensates the perception delay. A path that jumps
sideways by more than max_path_jump, or lies more than max_lateral_offset to the side at the
lookahead, is rejected; the previous path then goes stale and the follower stops. All times
come from the node clock, so the node also works on bag time with use_sim_time:=true.
"""

import json
import math
from collections import deque

import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDrive
from nav_msgs.msg import Path
from rclpy.node import Node
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32MultiArray, String

from .odometry import PlanarOdometry
from .pure_pursuit import PurePursuitFollower, inverse_transform_points


class CenterlineFollowerNode(Node):

    def __init__(self):
        super().__init__('centerline_follower')
        parameters = {
            'wheelbase': 2.27,
            'steering_ratio': 20.0,
            'steering_sign': -1.0,
            'lookahead_distance': 6.0,
            'max_steering_wheel_deg': 620.0,
            'max_steering_rate_deg_s': 120.0,
            'target_speed': 1.39,
            'max_path_age_seconds': 1.0,
            'max_path_jump': 1.0,
            'max_lateral_offset': 1.5,
            'rate_hz': 20.0,
            'yaw_rate_sign': -1.0,
            'max_speed_age_seconds': 0.5,
            'path_topic': '/centerline/path',
            'status_topic': '/centerline/status',
            'imu_topic': '/imu_attitude',
            'feedback_topic': '/feedback_data',
            'command_topic': '/centerline_follower/cmd',
            'debug_topic': '/centerline_follower/debug',
        }
        for name, default in parameters.items():
            self.declare_parameter(name, default)
        param = lambda name: self.get_parameter(name).value  # noqa: E731

        self.follower = PurePursuitFollower(
            wheelbase=param('wheelbase'),
            steering_ratio=param('steering_ratio'),
            steering_sign=param('steering_sign'),
            lookahead_distance=param('lookahead_distance'),
            max_steering_wheel_deg=param('max_steering_wheel_deg'),
            max_steering_rate_deg_s=param('max_steering_rate_deg_s'),
            target_speed=param('target_speed'),
            max_path_age_seconds=param('max_path_age_seconds'),
        )
        self.odometry = PlanarOdometry(
            yaw_rate_sign=param('yaw_rate_sign'),
            max_speed_age_seconds=param('max_speed_age_seconds'),
        )
        # Recent poses, to look up where the vehicle was when a path's image was taken.
        self.pose_history = deque(maxlen=400)
        self.path_status = None
        self.max_path_jump = param('max_path_jump')
        self.max_lateral_offset = param('max_lateral_offset')
        self.rejected_paths = 0
        self.previous_steering = None
        self.previous_time = None

        self.create_subscription(Path, param('path_topic'), self._path_callback, 10)
        self.create_subscription(String, param('status_topic'), self._status_callback, 10)
        self.create_subscription(Imu, param('imu_topic'), self._imu_callback, 50)
        self.create_subscription(
            Float32MultiArray, param('feedback_topic'), self._feedback_callback, 10)
        self.command_publisher = self.create_publisher(
            AckermannDrive, param('command_topic'), 10)
        self.debug_publisher = self.create_publisher(String, param('debug_topic'), 10)
        self.create_timer(1.0 / param('rate_hz'), self._control_callback)
        self.get_logger().info(
            f"Follower publishing on {param('command_topic')} "
            f"(lookahead {param('lookahead_distance')} m, "
            f"steering limit {param('max_steering_wheel_deg')} deg)")

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _feedback_callback(self, msg):
        # [velocity_ctrl, velocity_fb, steering_rate_ctrl, steering_angle_fb]
        if len(msg.data) > 1:
            self.odometry.update_speed(msg.data[1], self._now())

    def _imu_callback(self, msg):
        now = self._now()
        self.odometry.update_yaw_rate(msg.angular_velocity.z, now)
        pose, epoch = self.odometry.pose(now)
        if pose is not None:
            self.pose_history.append((now, pose, epoch))

    def _pose_at(self, timestamp):
        """Odom pose at `timestamp` from the recent history, or (None, None)."""
        for recorded, pose, epoch in reversed(self.pose_history):
            if recorded <= timestamp:
                if timestamp - recorded <= self.odometry.max_step_seconds:
                    return pose, epoch
                return None, None
        return None, None

    def _status_callback(self, msg):
        self.path_status = msg.data

    def _path_callback(self, msg):
        if len(msg.poses) < 2:
            self.follower.clear_path()
            return
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        points = np.array([[p.pose.position.x, p.pose.position.y] for p in msg.poses])
        pose, epoch = self._pose_at(stamp)
        # Implausible paths (e.g. a false detection beside the road) are not used: the previous
        # path then goes stale if the problem persists, so the follower stops instead of following.
        offset = self._lateral_at_lookahead(points)
        jump = self._path_jump(points, pose, epoch, stamp)
        if offset is not None and abs(offset) > self.max_lateral_offset:
            self._reject(f'lateral offset {offset:+.2f} m at the lookahead')
            return
        if jump is not None and jump > self.max_path_jump:
            self._reject(f'lateral jump {jump:.2f} m at the lookahead')
            return
        self.follower.update_path(points, pose, epoch, stamp, self.path_status)

    def _reject(self, reason):
        self.rejected_paths += 1
        self.get_logger().warn(f'Rejected path: {reason}', throttle_duration_sec=1.0)

    def _lateral_at_lookahead(self, points):
        points = points[np.argsort(points[:, 0])]
        lookahead = self.follower.lookahead_distance
        if points[0, 0] > lookahead or points[-1, 0] < lookahead:
            return None
        return float(np.interp(lookahead, points[:, 0], points[:, 1]))

    def _path_jump(self, points, pose, epoch, stamp):
        """Lateral jump in metres between a new path and the current one, or None."""
        follower = self.follower
        if follower.path_points is None:
            return None
        if stamp - follower.path_stamp > follower.max_path_age_seconds:
            return None
        if follower.path_pose is None:
            previous = follower.path_points
        elif pose is None or epoch != follower.path_epoch:
            return None
        else:
            previous = inverse_transform_points(follower.path_points, pose)
        before = self._lateral_at_lookahead(previous)
        now = self._lateral_at_lookahead(points)
        if before is None or now is None:
            return None
        return abs(now - before)

    def _control_callback(self):
        now = self._now()
        pose, epoch = self.odometry.pose(now)
        command = self.follower.compute(pose, epoch, now)

        steering_rate = 0.0
        if self.previous_time is not None and now > self.previous_time:
            steering_rate = (command.steering_wheel_deg - self.previous_steering) / (
                now - self.previous_time)
        self.previous_steering = command.steering_wheel_deg
        self.previous_time = now

        drive = AckermannDrive()
        drive.steering_angle = float(command.steering_wheel_deg)
        drive.steering_angle_velocity = float(steering_rate)
        drive.speed = float(command.speed)
        self.command_publisher.publish(drive)

        lookahead = command.lookahead_point or (math.nan, math.nan)
        debug = String()
        debug.data = json.dumps({
            'time': now,
            'tracking': command.tracking,
            'reason': command.reason,
            'path_status': command.path_status,
            'steering_wheel_deg': round(command.steering_wheel_deg, 2),
            'speed': command.speed,
            'lookahead_x': None if math.isnan(lookahead[0]) else round(lookahead[0], 3),
            'lookahead_y': None if math.isnan(lookahead[1]) else round(lookahead[1], 3),
            'curvature': round(command.curvature, 5),
            'path_age': None if command.path_age is None else round(command.path_age, 3),
            'odometry_ok': pose is not None,
            'rejected_paths': self.rejected_paths,
        })
        self.debug_publisher.publish(debug)


def main(args=None):
    rclpy.init(args=args)
    node = CenterlineFollowerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
