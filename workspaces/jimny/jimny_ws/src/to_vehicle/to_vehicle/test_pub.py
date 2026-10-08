"""
Node to publish simulated Ackermann and GPS messages for testing purposes.
"""

import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDrive
from gps_msgs.msg import GPSFix
import random
import time


class DataPublisherNode(Node):
    def __init__(self):
        super().__init__('data_publisher_node')

        # Publishers for AckermannDrive and GPSFix messages
        self.ackermann_publisher = self.create_publisher(AckermannDrive, 'ackermann_cmd', 10)
        self.gps_publisher = self.create_publisher(GPSFix, 'gps_raw', 10)

        # Timer to publish messages at regular intervals
        self.create_timer(0.5, self.publish_constant_ackermann)
        self.create_timer(0.5, self.publish_gps)

    def publish_random_ackermann(self):
        # Create and publish an AckermannDrive message
        ackermann_msg = AckermannDrive()
        ackermann_msg.steering_angle = random.uniform(-0.5, 0.5)  # Simulate steering angle between -0.5 and 0.5 radians
        ackermann_msg.steering_angle_velocity = random.uniform(-1.0, 1.0)  # Simulate steering velocity
        ackermann_msg.speed = random.uniform(0.0, 10.0)  # Simulate vehicle speed (m/s)
        ackermann_msg.acceleration = random.uniform(-1.0, 1.0)  # Simulate acceleration
        ackermann_msg.jerk = random.uniform(-0.5, 0.5)  # Simulate jerk

        self.ackermann_publisher.publish(ackermann_msg)
        self.get_logger().info(f'Published AckermannDrive message: {ackermann_msg}')

    def publish_constant_ackermann(self):
        # Create and publish an AckermannDrive message
        ackermann_msg = AckermannDrive()
        ackermann_msg.steering_angle = 2  # Simulate steering angle between -0.5 and 0.5 radians
        ackermann_msg.steering_angle_velocity = 0 # Simulate steering velocity
        ackermann_msg.speed = 1  # Simulate vehicle speed (m/s)
        ackermann_msg.acceleration = 0  # Simulate acceleration
        ackermann_msg.jerk = 0  # Simulate jerk

        self.ackermann_publisher.publish(ackermann_msg)
        self.get_logger().info(f'Published AckermannDrive message: {ackermann_msg}')

    def publish_gps(self):
        # Create and publish a GPSFix message
        gps_msg = GPSFix()
        gps_msg.latitude = random.uniform(-90.0, 90.0)  # Simulate latitude
        gps_msg.longitude = random.uniform(-180.0, 180.0)  # Simulate longitude
        gps_msg.altitude = random.uniform(0, 1000)  # Simulate altitude
        gps_msg.speed = random.uniform(0.0, 10.0)  # Simulate GPS speed (m/s)

        self.gps_publisher.publish(gps_msg)
        self.get_logger().info(f'Published GPSFix message: {gps_msg}')


def main(args=None):
    rclpy.init(args=args)
    node = DataPublisherNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Node interrupted. Shutting down.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()