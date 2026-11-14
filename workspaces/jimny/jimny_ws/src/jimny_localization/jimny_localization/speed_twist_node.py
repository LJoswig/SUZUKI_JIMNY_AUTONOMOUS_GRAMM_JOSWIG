"""
GPS ground speed -> TwistWithCovarianceStamped for robot_localization.

Subscribes:
  /gps_raw         gps_msgs/GPSFix from cuberos (speed = Doppler ground speed in m/s, ~10 Hz)
Publishes:
  /speed/twist     geometry_msgs/TwistWithCovarianceStamped in base_link (rear axle)

Why GPS speed: the car has no wheel speed sensor in ROS. /feedback_data[1] is this same GPS speed
after a round trip through sterfBoard -> CAN -> axelBrake -> CAN -> sterfBoard -> serial, so
reading /gps_raw directly gives the same value with less delay and without depending on
/ackermann_cmd being published. Doppler speed (~0.05-0.1 m/s) is much better than GPS position.

Limitations:
  - GPS speed has no sign: reversing is integrated as driving forward.
  - Below min_speed the value is mostly noise and is published as 0, so the EKF does not creep
    while parked (same idea as PlanarOdometry.min_moving_speed).
  - Without a fix nothing is published; the EKF then warns about the sensor timeout.

vy is published as 0 with a small variance: an Ackermann vehicle does not slide sideways at the
rear axle, and fusing that constraint keeps the EKF from drifting laterally.
"""

import rclpy
from geometry_msgs.msg import TwistWithCovarianceStamped
from gps_msgs.msg import GPSFix
from rclpy.node import Node

# Variance for twist components the EKF is not configured to fuse.
UNUSED_VARIANCE = 1e3


class SpeedTwistNode(Node):

    def __init__(self):
        super().__init__('speed_twist')
        self.declare_parameter('input_topic', '/gps_raw')
        self.declare_parameter('output_topic', '/speed/twist')
        self.declare_parameter('frame_id', 'base_link')
        self.declare_parameter('min_speed', 0.1)            # m/s, below this the car is standing
        self.declare_parameter('min_speed_variance', 0.01)  # (0.1 m/s)^2, floor for err_speed^2
        self.declare_parameter('lateral_variance', 0.001)   # nonholonomic constraint

        self.frame_id = self.get_parameter('frame_id').value
        self.min_speed = float(self.get_parameter('min_speed').value)
        self.min_speed_variance = float(self.get_parameter('min_speed_variance').value)
        self.lateral_variance = float(self.get_parameter('lateral_variance').value)
        self._last_sample = None

        self.publisher = self.create_publisher(
            TwistWithCovarianceStamped, self.get_parameter('output_topic').value, 10)
        self.create_subscription(
            GPSFix, self.get_parameter('input_topic').value, self._callback, 10)

    def _callback(self, msg):
        # cuberos publishes on a timer and can repeat a sample; the EKF must see each one once.
        sample = (msg.latitude, msg.longitude, msg.speed, msg.track)
        if sample == self._last_sample:
            return
        self._last_sample = sample

        # cuberos maps MAVLink fix_type 0/1 to status -1 (no fix)
        if msg.status.status < 0:
            self.get_logger().warn('no GPS fix, not publishing speed', throttle_duration_sec=5.0)
            return

        speed = float(msg.speed)
        if speed < self.min_speed:
            speed = 0.0

        out = TwistWithCovarianceStamped()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = self.frame_id
        out.twist.twist.linear.x = speed
        out.twist.twist.linear.y = 0.0

        covariance = [0.0] * 36
        for i in range(6):
            covariance[i * 7] = UNUSED_VARIANCE
        covariance[0] = max(float(msg.err_speed) ** 2, self.min_speed_variance)
        covariance[7] = self.lateral_variance
        out.twist.covariance = covariance
        self.publisher.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = SpeedTwistNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
