#!/usr/bin/env python3

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass
class MotionSample:
    timestamp: float
    steering_angle_meas: Optional[float] = None
    steering_angle_cmd: Optional[float] = None
    speed_cmd: Optional[float] = None
    # Raw IMU angular_velocity.z as published by the cube (positive = right turn).
    yaw_rate: Optional[float] = None
    # Measured vehicle speed from the sterfBoard feedback, m/s (forward only).
    speed_meas: Optional[float] = None
    # Wheel/IMU odometry of the rear axle in a local odom frame (x forward at
    # start, y left, yaw counter-clockwise). odom_epoch changes whenever the
    # integration had to restart, so poses from different epochs are unrelated.
    odom_pose: Optional[Tuple[float, float, float]] = None
    odom_epoch: Optional[int] = None

    def has_any_signal(self) -> bool:
        return any(
            value is not None
            for value in (
                self.steering_angle_meas,
                self.steering_angle_cmd,
                self.speed_cmd,
                self.yaw_rate,
                self.speed_meas,
                self.odom_pose,
            )
        )


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


class BaseMotionProvider:
    def get_sample(self, frame_timestamp: float) -> Optional[MotionSample]:
        raise NotImplementedError

    def close(self) -> None:
        return None


class NullMotionProvider(BaseMotionProvider):
    def get_sample(self, frame_timestamp: float) -> Optional[MotionSample]:
        return None


class RosbagMotionProvider(BaseMotionProvider):
    """Replay motion topics from a ROS2 bag and align to frame timestamps."""

    def __init__(
        self,
        bag_uri: str,
        topic_names: Dict[str, str],
        max_signal_age_seconds: float = 0.15,
        lookahead_seconds: float = 0.02,
        odometry: Optional[PlanarOdometry] = None,
    ):
        self.max_signal_age_seconds = float(max_signal_age_seconds)
        self.lookahead_seconds = float(lookahead_seconds)
        self.odometry = odometry if odometry is not None else PlanarOdometry()

        self._latest_values: Dict[str, Optional[float]] = {
            "steering_angle_meas": None,
            "steering_angle_cmd": None,
            "speed_cmd": None,
            "yaw_rate": None,
        }
        self._latest_timestamps: Dict[str, Optional[float]] = {
            key: None for key in self._latest_values
        }

        self._topic_to_signal: Dict[str, str] = {}
        self._topic_types: Dict[str, str] = {}
        self._topic_msg_classes: Dict[str, object] = {}

        try:
            import rosbag2_py
            from rclpy.serialization import deserialize_message
            from rosidl_runtime_py.utilities import get_message
        except ImportError as exc:
            raise RuntimeError(
                "RosbagMotionProvider requires ROS2 Python packages "
                "(rosbag2_py, rclpy, rosidl_runtime_py)."
            ) from exc

        self._deserialize_message = deserialize_message

        storage_options = rosbag2_py.StorageOptions(uri=bag_uri, storage_id="sqlite3")
        converter_options = rosbag2_py.ConverterOptions(
            input_serialization_format="",
            output_serialization_format="",
        )
        reader = rosbag2_py.SequentialReader()
        reader.open(storage_options, converter_options)
        self._reader = reader

        for topic_info in self._reader.get_all_topics_and_types():
            self._topic_types[topic_info.name] = topic_info.type

        mapping = {
            "ackermann_cmd": "ackermann",
            "steering_feedback": "steering_feedback",
            "imu_attitude": "imu",
            "feedback_data": "feedback",
        }
        for logical_name, topic_name in topic_names.items():
            if logical_name not in mapping:
                continue
            if topic_name not in self._topic_types:
                continue
            self._topic_to_signal[topic_name] = mapping[logical_name]
            self._topic_msg_classes[topic_name] = get_message(self._topic_types[topic_name])

        if not self._topic_to_signal:
            raise RuntimeError(
                "No configured motion topics were found in rosbag. "
                "Check topic names in config."
            )

        self._next_record = None
        self._advance_reader()

    def _advance_reader(self) -> None:
        if self._reader.has_next():
            self._next_record = self._reader.read_next()
        else:
            self._next_record = None

    @staticmethod
    def _to_seconds(nanoseconds: int) -> float:
        return float(nanoseconds) * 1e-9

    def _update_signal(self, signal_key: str, value: Optional[float], timestamp: float) -> None:
        if value is None:
            return
        self._latest_values[signal_key] = float(value)
        self._latest_timestamps[signal_key] = timestamp

    def _consume_record(self, topic: str, payload, timestamp: float) -> None:
        signal_group = self._topic_to_signal.get(topic)
        if signal_group is None:
            return

        msg = self._deserialize_message(payload, self._topic_msg_classes[topic])

        if signal_group == "ackermann":
            self._update_signal("steering_angle_cmd", getattr(msg, "steering_angle", None), timestamp)
            self._update_signal("speed_cmd", getattr(msg, "speed", None), timestamp)
            return

        if signal_group == "steering_feedback":
            data = getattr(msg, "data", None)
            if data is not None and len(data) > 0:
                self._update_signal("steering_angle_meas", data[0], timestamp)
            return

        if signal_group == "feedback":
            # [velocity_ctrl, velocity_fb, steering_rate_ctrl, steering_angle_fb]
            data = getattr(msg, "data", None)
            if data is not None and len(data) > 1:
                self.odometry.update_speed(data[1], timestamp)
            return

        if signal_group == "imu":
            angular_velocity = getattr(msg, "angular_velocity", None)
            if angular_velocity is not None:
                yaw_rate = getattr(angular_velocity, "z", None)
                self._update_signal("yaw_rate", yaw_rate, timestamp)
                if yaw_rate is not None:
                    self.odometry.update_yaw_rate(yaw_rate, timestamp)

    def _pump_until(self, frame_timestamp: float) -> None:
        limit = frame_timestamp + self.lookahead_seconds
        while self._next_record is not None:
            topic, payload, ros_time_ns = self._next_record
            timestamp = self._to_seconds(ros_time_ns)
            if timestamp > limit:
                break
            self._consume_record(topic, payload, timestamp)
            self._advance_reader()

    def _value_if_fresh(self, signal_key: str, frame_timestamp: float) -> Optional[float]:
        value = self._latest_values[signal_key]
        signal_ts = self._latest_timestamps[signal_key]
        if value is None or signal_ts is None:
            return None
        if (frame_timestamp - signal_ts) > self.max_signal_age_seconds:
            return None
        return value

    def get_sample(self, frame_timestamp: float) -> Optional[MotionSample]:
        self._pump_until(frame_timestamp)
        odom_pose, odom_epoch = self.odometry.pose(frame_timestamp)
        sample = MotionSample(
            timestamp=float(frame_timestamp),
            steering_angle_meas=self._value_if_fresh("steering_angle_meas", frame_timestamp),
            steering_angle_cmd=self._value_if_fresh("steering_angle_cmd", frame_timestamp),
            speed_cmd=self._value_if_fresh("speed_cmd", frame_timestamp),
            yaw_rate=self._value_if_fresh("yaw_rate", frame_timestamp),
            speed_meas=self.odometry.speed(frame_timestamp),
            odom_pose=odom_pose,
            odom_epoch=odom_epoch,
        )
        return sample if sample.has_any_signal() else None


class LiveRosMotionProvider(BaseMotionProvider):
    """Subscribe to live ROS2 topics and expose frame-aligned motion samples."""

    def __init__(
        self,
        topic_names: Dict[str, str],
        max_signal_age_seconds: float = 0.15,
        odometry: Optional[PlanarOdometry] = None,
    ):
        self.max_signal_age_seconds = float(max_signal_age_seconds)
        self.odometry = odometry if odometry is not None else PlanarOdometry()
        self._lock = threading.Lock()
        self._latest_values: Dict[str, Optional[float]] = {
            "steering_angle_meas": None,
            "steering_angle_cmd": None,
            "speed_cmd": None,
            "yaw_rate": None,
        }
        self._latest_timestamps: Dict[str, Optional[float]] = {
            key: None for key in self._latest_values
        }

        try:
            import rclpy
            from rclpy.executors import SingleThreadedExecutor
            from rclpy.node import Node
            from ackermann_msgs.msg import AckermannDrive
            from sensor_msgs.msg import Imu
            from std_msgs.msg import Float32MultiArray
        except ImportError as exc:
            raise RuntimeError(
                "LiveRosMotionProvider requires ROS2 Python packages "
                "(rclpy, ackermann_msgs, sensor_msgs, std_msgs)."
            ) from exc

        self._rclpy = rclpy
        self._owns_rclpy_context = False
        if not self._rclpy.ok():
            self._rclpy.init(args=None)
            self._owns_rclpy_context = True

        # The executor binds to the default context, so it must exist after init.
        self._executor = SingleThreadedExecutor()

        provider = self

        class _MotionNode(Node):
            def __init__(self):
                super().__init__("centerline_motion_provider")

        self._node = _MotionNode()
        self._executor.add_node(self._node)

        def now_seconds() -> float:
            return float(self._node.get_clock().now().nanoseconds) * 1e-9

        def update(signal_key: str, value: Optional[float]) -> None:
            if value is None:
                return
            with provider._lock:
                provider._latest_values[signal_key] = float(value)
                provider._latest_timestamps[signal_key] = now_seconds()

        def ackermann_callback(msg: AckermannDrive) -> None:
            update("steering_angle_cmd", msg.steering_angle)
            update("speed_cmd", msg.speed)

        def steering_feedback_callback(msg: Float32MultiArray) -> None:
            if len(msg.data) > 0:
                update("steering_angle_meas", msg.data[0])

        def feedback_callback(msg: Float32MultiArray) -> None:
            if len(msg.data) > 1:
                with provider._lock:
                    provider.odometry.update_speed(msg.data[1], now_seconds())

        def imu_callback(msg: Imu) -> None:
            update("yaw_rate", msg.angular_velocity.z)
            with provider._lock:
                provider.odometry.update_yaw_rate(msg.angular_velocity.z, now_seconds())

        self._node.create_subscription(
            AckermannDrive,
            topic_names["ackermann_cmd"],
            ackermann_callback,
            10,
        )
        self._node.create_subscription(
            Float32MultiArray,
            topic_names["steering_feedback"],
            steering_feedback_callback,
            10,
        )
        self._node.create_subscription(
            Imu,
            topic_names["imu_attitude"],
            imu_callback,
            10,
        )
        if "feedback_data" in topic_names:
            self._node.create_subscription(
                Float32MultiArray,
                topic_names["feedback_data"],
                feedback_callback,
                10,
            )

        self._running = True

        def spin_loop() -> None:
            while self._running:
                self._executor.spin_once(timeout_sec=0.01)

        self._spin_thread = threading.Thread(target=spin_loop, daemon=True)
        self._spin_thread.start()

    def _value_if_fresh(self, signal_key: str, frame_timestamp: float) -> Optional[float]:
        value = self._latest_values[signal_key]
        signal_ts = self._latest_timestamps[signal_key]
        if value is None or signal_ts is None:
            return None
        if (frame_timestamp - signal_ts) > self.max_signal_age_seconds:
            return None
        return value

    def get_sample(self, frame_timestamp: float) -> Optional[MotionSample]:
        with self._lock:
            odom_pose, odom_epoch = self.odometry.pose(frame_timestamp)
            sample = MotionSample(
                timestamp=float(frame_timestamp),
                steering_angle_meas=self._value_if_fresh("steering_angle_meas", frame_timestamp),
                steering_angle_cmd=self._value_if_fresh("steering_angle_cmd", frame_timestamp),
                speed_cmd=self._value_if_fresh("speed_cmd", frame_timestamp),
                yaw_rate=self._value_if_fresh("yaw_rate", frame_timestamp),
                speed_meas=self.odometry.speed(frame_timestamp),
                odom_pose=odom_pose,
                odom_epoch=odom_epoch,
            )
        return sample if sample.has_any_signal() else None

    def close(self) -> None:
        self._running = False
        if hasattr(self, "_spin_thread"):
            self._spin_thread.join(timeout=1.0)
        if hasattr(self, "_executor") and hasattr(self, "_node"):
            self._executor.remove_node(self._node)
            self._node.destroy_node()
        if getattr(self, "_owns_rclpy_context", False) and self._rclpy.ok():
            self._rclpy.shutdown()
