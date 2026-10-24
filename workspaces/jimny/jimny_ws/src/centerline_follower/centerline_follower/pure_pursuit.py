"""
Pure pursuit road follower on the ground-projected centerline.

Copy of yolo-efficientvit-sam/utils/follower.py (plus the two frame transforms from
utils/path_memory.py) so this package is self-contained on the car. Keep both in sync;
the desk test (offline shadow tool vs. this node on the same bag) shows any drift.

The latest path is stored in the odom frame at the pose where its image was taken, and
evaluated at the vehicle's pose at command time. That cancels the perception delay and
lets the follower run faster than perception, between path updates.

Steering output follows the vehicle convention of /ackermann_cmd: steering-wheel degrees,
positive = right turn. The steering centre offset is not handled here (encoder issue).
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


def _rotation_2d(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def transform_points(points, pose):
    """Map (N, 2) points from a frame located at pose=(x, y, yaw) into its parent frame."""
    x, y, yaw = pose
    return points @ _rotation_2d(yaw).T + np.array([x, y])


def inverse_transform_points(points, pose):
    x, y, yaw = pose
    return (points - np.array([x, y])) @ _rotation_2d(yaw)


@dataclass
class FollowerCommand:
    steering_wheel_deg: float
    speed: float
    tracking: bool
    reason: str
    path_status: Optional[str] = None
    lookahead_point: Optional[Tuple[float, float]] = None
    curvature: float = 0.0
    path_age: Optional[float] = None


class PurePursuitFollower:
    TRACKING = "tracking"
    NO_PATH = "no_path"
    STALE_PATH = "stale_path"
    NO_ODOMETRY = "no_odometry"
    PATH_TOO_SHORT = "path_too_short"

    def __init__(
        self,
        wheelbase=2.3,
        steering_ratio=20.0,
        steering_sign=-1.0,
        lookahead_distance=6.0,
        max_steering_wheel_deg=620.0,
        max_steering_rate_deg_s=120.0,
        target_speed=1.39,
        max_path_age_seconds=1.0,
    ):
        if wheelbase <= 0.0:
            raise ValueError("Wheelbase must be positive")
        if steering_ratio <= 0.0:
            raise ValueError("Steering ratio must be positive")
        if steering_sign not in (-1.0, 1.0):
            raise ValueError("Steering sign must be -1 or 1")
        if lookahead_distance <= 0.0:
            raise ValueError("Lookahead distance must be positive")
        if max_steering_wheel_deg <= 0.0 or max_steering_rate_deg_s <= 0.0:
            raise ValueError("Steering limits must be positive")
        if target_speed < 0.0:
            raise ValueError("Target speed must be non-negative")
        if max_path_age_seconds <= 0.0:
            raise ValueError("Maximum path age must be positive")
        self.wheelbase = float(wheelbase)
        self.steering_ratio = float(steering_ratio)
        self.steering_sign = float(steering_sign)
        self.lookahead_distance = float(lookahead_distance)
        self.max_steering_wheel_deg = float(max_steering_wheel_deg)
        self.max_steering_rate_deg_s = float(max_steering_rate_deg_s)
        self.target_speed = float(target_speed)
        self.max_path_age_seconds = float(max_path_age_seconds)
        self.reset()

    def reset(self):
        self.path_points = None
        self.path_pose = None
        self.path_epoch = None
        self.path_stamp = None
        self.path_status = None
        self.steering_wheel_deg = 0.0
        self.last_command_time = None

    def update_path(self, ground_points, pose, epoch, stamp, path_status=None):
        """
        Store a base_link path observed at `stamp`, when the vehicle was at `pose`.

        Without odometry (pose None) the path is used as-is, without delay compensation.
        """
        if ground_points is None or len(ground_points) < 2:
            return
        points = np.asarray(ground_points, dtype=np.float64)
        self.path_points = transform_points(points, pose) if pose is not None else points
        self.path_pose = pose
        self.path_epoch = epoch
        self.path_stamp = float(stamp)
        self.path_status = path_status

    def clear_path(self):
        """Forget the path, e.g. when perception reports the road as lost."""
        self.path_points = None
        self.path_pose = None
        self.path_epoch = None
        self.path_stamp = None

    def _path_in_base_link(self, pose, epoch):
        if self.path_pose is None:
            return self.path_points
        if pose is None or epoch != self.path_epoch:
            return None
        return inverse_transform_points(self.path_points, pose)

    def _lookahead_point(self, ground):
        """First point along the path at the lookahead distance from the rear axle."""
        ahead = ground[ground[:, 0] > 0.0]
        if len(ahead) < 2:
            return None
        distances = np.hypot(ahead[:, 0], ahead[:, 1])
        beyond = np.flatnonzero(distances >= self.lookahead_distance)
        if len(beyond) == 0:
            return None
        index = beyond[0]
        if index == 0:
            # The visible path starts beyond the lookahead (camera blind zone): aim at its start.
            return ahead[0]
        fraction = (self.lookahead_distance - distances[index - 1]) / (
            distances[index] - distances[index - 1]
        )
        return ahead[index - 1] + fraction * (ahead[index] - ahead[index - 1])

    def _limit(self, target_deg, now):
        limit = self.max_steering_wheel_deg
        target_deg = float(np.clip(target_deg, -limit, limit))
        if self.last_command_time is None:
            self.steering_wheel_deg = target_deg
        else:
            delta_time = min(max(now - self.last_command_time, 0.0), 0.5)
            max_step = self.max_steering_rate_deg_s * delta_time
            step = np.clip(target_deg - self.steering_wheel_deg, -max_step, max_step)
            self.steering_wheel_deg += float(step)
        self.last_command_time = now
        return self.steering_wheel_deg

    def _hold(self, reason, now, path_age=None):
        # Keep the wheel where it is and stop; the emergency decision belongs to mode_switch.
        steering = self._limit(self.steering_wheel_deg, now)
        return FollowerCommand(steering, 0.0, False, reason, self.path_status, path_age=path_age)

    def compute(self, pose, epoch, now):
        """Steering and speed command for the vehicle at `pose` (odom) at time `now`."""
        if self.path_points is None:
            return self._hold(self.NO_PATH, now)
        path_age = now - self.path_stamp
        if path_age > self.max_path_age_seconds:
            return self._hold(self.STALE_PATH, now, path_age)
        ground = self._path_in_base_link(pose, epoch)
        if ground is None:
            return self._hold(self.NO_ODOMETRY, now, path_age)
        target = self._lookahead_point(ground)
        if target is None:
            return self._hold(self.PATH_TOO_SHORT, now, path_age)

        distance_squared = float(target[0] ** 2 + target[1] ** 2)
        curvature = 2.0 * float(target[1]) / distance_squared
        road_wheel_deg = math.degrees(math.atan(self.wheelbase * curvature))
        steering = self._limit(self.steering_sign * self.steering_ratio * road_wheel_deg, now)
        return FollowerCommand(
            steering_wheel_deg=steering,
            speed=self.target_speed,
            tracking=True,
            reason=self.TRACKING,
            path_status=self.path_status,
            lookahead_point=(float(target[0]), float(target[1])),
            curvature=curvature,
            path_age=path_age,
        )
