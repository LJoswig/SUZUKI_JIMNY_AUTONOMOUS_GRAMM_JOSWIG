#!/usr/bin/env python3
"""Record a rosbag of the vehicle and centerline topics next to a camera recording."""

import shutil
import signal
import subprocess
import time
from pathlib import Path

# Recorded next to the SVO so every camera recording has a matching bag.
ROSBAG_TOPICS = [
    # Motion and odometry
    "/imu_attitude",
    "/feedback_data",
    "/steering_feedback",
    "/gps",
    "/gps_raw",
    # Commands, modes and safety state
    "/ackermann_cmd",
    "/rc_inputs",
    "/rc_steer_debug",
    "/mode",
    "/heartbeat",
    "/signal_error",
    # Centerline perception and road follower
    "/centerline/path",
    "/centerline/status",
    "/centerline_follower/cmd",
    "/centerline_follower/debug",
]


def start_rosbag(bag_dir: Path, topics, log_path: Path):
    """Start `ros2 bag record` in its own process group; returns the process or None."""
    if shutil.which("ros2") is None:
        print("[WARNING] ros2 not found (ROS not sourced?); recording the SVO without a rosbag.")
        return None
    log_file = open(log_path, "w")
    process = subprocess.Popen(
        ["ros2", "bag", "record", "-o", str(bag_dir), *topics],
        # No terminal input: the recorder's keyboard controls (space = pause) would otherwise
        # react to keys typed in the terminal running main.py.
        stdin=subprocess.DEVNULL,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,  # CTRL+C stops the SVO first; the bag is stopped explicitly after it
    )
    time.sleep(2.0)
    if process.poll() is not None:
        print(f"[WARNING] rosbag recorder exited immediately; see {log_path}")
        return None
    print(f"[INFO] Rosbag recording started: {bag_dir}")
    return process


def stop_rosbag(process) -> bool:
    if process is None:
        return False
    process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        print("[WARNING] rosbag recorder did not stop in time; terminating it.")
        process.terminate()
        process.wait(timeout=5)
    return process.returncode == 0

