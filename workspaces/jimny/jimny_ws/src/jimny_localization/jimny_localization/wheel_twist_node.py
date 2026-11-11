"""
Measured vehicle speed -> TwistWithCovarianceStamped for robot_localization.

Subscribes:
  /feedback_data   std_msgs/Float32MultiArray, data[speed_index] = measured speed in m/s (~5 Hz)
Publishes:
  /wheel/twist     geometry_msgs/TwistWithCovarianceStamped in base_link (rear axle)

vx is the measured speed. vy is published as 0 with a small variance: an Ackermann vehicle does
not slide sideways at the rear axle, and fusing that constraint keeps the EKF from drifting
laterally. Float32MultiArray has no header, so messages are stamped on receive.
"""

import rclpy
from geometry_msgs.msg import TwistWithCovarianceStamped
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

# Variance for twist components the EKF is not configured to fuse.
UNUSED_VARIANCE = 1e3


class WheelTwistNode(Node):

    def __init__(self):
        super().__init__('wheel_twist')
        self.declare_parameter('input_topic', '/feedback_data')
        self.declare_parameter('output_topic', '/wheel/twist')
        self.declare_parameter('speed_index', 1)
        self.declare_parameter('frame_id', 'base_link')
        self.declare_parameter('speed_variance', 0.01)       # (0.1 m/s)^2
        self.declare_parameter('lateral_variance', 0.001)    # nonholonomic constraint

        self.speed_index = self.get_parameter('speed_index').value
        self.frame_id = self.get_parameter('frame_id').value
        self.speed_variance = float(self.get_parameter('speed_variance').value)
        self.lateral_variance = float(self.get_parameter('lateral_variance').value)

        self.publisher = self.create_publisher(
            TwistWithCovarianceStamped, self.get_parameter('output_topic').value, 10)
        self.create_subscription(
            Float32MultiArray, self.get_parameter('input_topic').value, self._callback, 10)

    def _callback(self, msg):
        if len(msg.data) <= self.speed_index:
            self.get_logger().warn(
                f'feedback message has {len(msg.data)} values, expected index {self.speed_index}',
                throttle_duration_sec=5.0)
            return

        out = TwistWithCovarianceStamped()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = self.frame_id
        out.twist.twist.linear.x = float(msg.data[self.speed_index])
        out.twist.twist.linear.y = 0.0

        covariance = [0.0] * 36
        for i in range(6):
            covariance[i * 7] = UNUSED_VARIANCE
        covariance[0] = self.speed_variance
        covariance[7] = self.lateral_variance
        out.twist.covariance = covariance
        self.publisher.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = WheelTwistNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
