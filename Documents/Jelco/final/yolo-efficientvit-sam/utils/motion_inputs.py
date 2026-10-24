#!/usr/bin/env python3

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class MotionSample:
    timestamp: float
    steering_angle_meas: Optional[float] = None
    steering_angle_cmd: Optional[float] = None
    speed_cmd: Optional[float] = None
    yaw_rate: Optional[float] = None

    def has_any_signal(self) -> bool:
        return any(
            value is not None
            for value in (
                self.steering_angle_meas,
                self.steering_angle_cmd,
                self.speed_cmd,
                self.yaw_rate,
            )
        )


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
    ):
        self.max_signal_age_seconds = float(max_signal_age_seconds)
        self.lookahead_seconds = float(lookahead_seconds)

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

        if signal_group == "imu":
            angular_velocity = getattr(msg, "angular_velocity", None)
            if angular_velocity is not None:
                self._update_signal("yaw_rate", getattr(angular_velocity, "z", None), timestamp)

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
        sample = MotionSample(
            timestamp=float(frame_timestamp),
            steering_angle_meas=self._value_if_fresh("steering_angle_meas", frame_timestamp),
            steering_angle_cmd=self._value_if_fresh("steering_angle_cmd", frame_timestamp),
            speed_cmd=self._value_if_fresh("speed_cmd", frame_timestamp),
            yaw_rate=self._value_if_fresh("yaw_rate", frame_timestamp),
        )
        return sample if sample.has_any_signal() else None


class LiveRosMotionProvider(BaseMotionProvider):
    """Subscribe to live ROS2 topics and expose frame-aligned motion samples."""

    def __init__(
        self,
        topic_names: Dict[str, str],
        max_signal_age_seconds: float = 0.15,
    ):
        self.max_signal_age_seconds = float(max_signal_age_seconds)
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
        self._executor = SingleThreadedExecutor()

        self._owns_rclpy_context = False
        if not self._rclpy.ok():
            self._rclpy.init(args=None)
            self._owns_rclpy_context = True

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

        def imu_callback(msg: Imu) -> None:
            update("yaw_rate", msg.angular_velocity.z)

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
            sample = MotionSample(
                timestamp=float(frame_timestamp),
                steering_angle_meas=self._value_if_fresh("steering_angle_meas", frame_timestamp),
                steering_angle_cmd=self._value_if_fresh("steering_angle_cmd", frame_timestamp),
                speed_cmd=self._value_if_fresh("speed_cmd", frame_timestamp),
                yaw_rate=self._value_if_fresh("yaw_rate", frame_timestamp),
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
