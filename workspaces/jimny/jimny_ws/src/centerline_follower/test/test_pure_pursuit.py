import math

import numpy as np

from centerline_follower.odometry import PlanarOdometry
from centerline_follower.pure_pursuit import PurePursuitFollower


def straight(y=0.0):
    x = np.linspace(5.0, 15.0, 30)
    return np.column_stack((x, np.full_like(x, y)))


def test_straight_road_needs_no_steering():
    follower = PurePursuitFollower()
    follower.update_path(straight(), (0.0, 0.0, 0.0), 1, 0.0)
    command = follower.compute((0.0, 0.0, 0.0), 1, 0.05)
    assert command.tracking
    assert abs(command.steering_wheel_deg) < 1e-6


def test_left_curve_gives_negative_steering():
    radius = 20.0
    angles = np.linspace(0.2, 0.8, 40)
    path = np.column_stack((radius * np.sin(angles), radius - radius * np.cos(angles)))
    follower = PurePursuitFollower(max_steering_rate_deg_s=1e6)
    follower.update_path(path, (0.0, 0.0, 0.0), 1, 0.0)
    command = follower.compute((0.0, 0.0, 0.0), 1, 0.05)
    expected = -20.0 * math.degrees(math.atan(2.27 / radius))
    assert abs(command.steering_wheel_deg - expected) < 3.0


def test_delay_compensation_uses_current_pose():
    follower = PurePursuitFollower(max_steering_rate_deg_s=1e6)
    follower.update_path(straight(1.0), (0.0, 0.0, 0.0), 1, 0.0)
    assert follower.compute((0.0, 1.0, 0.0), 1, 0.1).steering_wheel_deg == 0.0


def test_stops_on_stale_cleared_or_foreign_epoch_path():
    follower = PurePursuitFollower()
    follower.update_path(straight(), (0.0, 0.0, 0.0), 1, 0.0)
    assert follower.compute((0.0, 0.0, 0.0), 1, 2.0).reason == PurePursuitFollower.STALE_PATH
    assert follower.compute((0.0, 0.0, 0.0), 2, 0.5).reason == PurePursuitFollower.NO_ODOMETRY
    follower.clear_path()
    command = follower.compute((0.0, 0.0, 0.0), 1, 0.6)
    assert command.reason == PurePursuitFollower.NO_PATH and command.speed == 0.0


def test_steering_rate_and_angle_limits():
    follower = PurePursuitFollower(max_steering_rate_deg_s=120.0)
    sharp = np.column_stack((np.linspace(5.0, 7.0, 10), np.linspace(0.0, -6.0, 10)))
    follower.update_path(sharp, (0.0, 0.0, 0.0), 1, 0.0)
    follower.compute((0.0, 0.0, 0.0), 1, 0.0)
    follower.steering_wheel_deg = 0.0
    assert abs(follower.compute((0.0, 0.0, 0.0), 1, 0.05).steering_wheel_deg) <= 6.0 + 1e-9
    for step in range(200):
        command = follower.compute((0.0, 0.0, 0.0), 1, 0.05 + 0.05 * step)
    assert command.steering_wheel_deg <= 620.0


def test_odometry_drives_a_left_half_circle():
    odometry = PlanarOdometry(yaw_rate_sign=-1.0)
    t = 0.0
    for _ in range(int(round(math.pi / 0.2 / 0.05))):
        t += 0.05
        odometry.update_speed(2.0, t)
        odometry.update_yaw_rate(-0.2, t)
    (x, y, yaw), _ = odometry.pose(t)
    assert abs(y - 20.0) < 0.3 and abs(yaw - math.pi) < 0.02
