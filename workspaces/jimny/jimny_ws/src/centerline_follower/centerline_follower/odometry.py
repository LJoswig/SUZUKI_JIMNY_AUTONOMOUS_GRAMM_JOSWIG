"""
Dead-reckoned rear-axle pose from measured speed and IMU yaw rate.

Copy of PlanarOdometry from yolo-efficientvit-sam/utils/motion_inputs.py; keep both in sync.
"""

import math
from typing import Optional, Tuple


class PlanarOdometry:
    """Dead-reckon the rear-axle pose from measured speed and IMU yaw rate."""

    def __init__(
        self,
        yaw_rate_sign: float = -1.0,
        max_speed_age_seconds: float = 0.5,
        max_step_seconds: float = 0.25,
        min_moving_speed: float = 0.05,
    ):
        self.yaw_rate_sign = float(yaw_rate_sign)
        self.max_speed_age_seconds = float(max_speed_age_seconds)
        self.max_step_seconds = float(max_step_seconds)
        self.min_moving_speed = float(min_moving_speed)
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.epoch = 0
        self.valid = False
        self._speed = None
        self._speed_timestamp = None
        self._last_yaw_timestamp = None
        self._last_pose_timestamp = None

    def _restart(self) -> None:
        self.epoch += 1
        self.valid = False

    def update_speed(self, speed: float, timestamp: float) -> None:
        self._speed = float(speed)
        self._speed_timestamp = float(timestamp)

    def speed(self, timestamp: float) -> Optional[float]:
        if self._speed_timestamp is None:
            return None
        if timestamp - self._speed_timestamp > self.max_speed_age_seconds:
            return None
        return self._speed

    def update_yaw_rate(self, yaw_rate: float, timestamp: float) -> None:
        timestamp = float(timestamp)
        previous = self._last_yaw_timestamp
        self._last_yaw_timestamp = timestamp
        speed = self.speed(timestamp)
        if previous is None or speed is None:
            self._restart()
            return
        delta_time = timestamp - previous
        if delta_time <= 0.0 or delta_time > self.max_step_seconds:
            self._restart()
            return

        if not self.valid:
            self.valid = True
        # An Ackermann vehicle cannot turn in place; skipping yaw at standstill
        # also stops the gyro bias from accumulating while parked.
        if abs(speed) >= self.min_moving_speed:
            yaw_rate_ccw = self.yaw_rate_sign * float(yaw_rate)
            mid_yaw = self.yaw + 0.5 * yaw_rate_ccw * delta_time
            self.x += speed * math.cos(mid_yaw) * delta_time
            self.y += speed * math.sin(mid_yaw) * delta_time
            self.yaw += yaw_rate_ccw * delta_time
        self._last_pose_timestamp = timestamp

    def pose(self, timestamp: float) -> Tuple[Optional[Tuple[float, float, float]], int]:
        if not self.valid or self._last_pose_timestamp is None:
            return None, self.epoch
        if abs(timestamp - self._last_pose_timestamp) > self.max_step_seconds:
            return None, self.epoch
        return (self.x, self.y, self.yaw), self.epoch
