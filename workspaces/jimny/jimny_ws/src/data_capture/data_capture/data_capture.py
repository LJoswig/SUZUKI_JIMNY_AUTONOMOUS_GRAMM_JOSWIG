"""
data_capture.py
Author: Matthijs Steyerberg, Axel Barbelanne
Date: 22-10-2024

Data recording node.
During driving this node will save datapoints that are `x` meters apart.
Datapoints include:
- Left and Right ZED camera images
- Vehicle states
  - velocity
  - yaw rate
  - steering angle
  - pose
- Vehicle GPS position
The images are saved in an `image` folder while the other datapoints are entered into an CSV file.
The CSV file also contains additional vehicle state entries between different images taken.
This is indexed using the sub_count, which is reset everytime a new image is recorded.
The datapoint index starts at 0 and increases by one for every image recorded.
All data is stored in the recordings folder, where a new folder is created using the current timestamp dd_mm_YYYY_HH_MM
Within this folder the CSV file is directly stored and the images are in a subfolder called "images"
"""

import rclpy
from rclpy.node import Node
from gps_msgs.msg import GPSFix
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32
from geopy.distance import geodesic
from zed_capture import ZEDCAMERA
import csv
import os
from datetime import datetime


class DataRecorderNode(Node):
    def __init__(self):
        super().__init__('data_recorder')

        # Parameters
        self.declare_parameter('distance_threshold', 1.0)  # meters
        self.distance_threshold = self.get_parameter('distance_threshold').get_parameter_value().double_value

        # Initialize subscribers
        self.gps_raw_subscription = self.create_subscription(GPSFix, 'gps', self.gps_callback, 10)
        self.gps_subscription = self.create_subscription(GPSFix, 'gps_raw', self.gps_raw_callback, 10)
        self.imu_subscription = self.create_subscription(Imu, 'imu_attitude', self.imu_callback, 10)
        self.steer_subscription = self.create_subscription(Float32, 'steering_angle', self.steer_callback, 10)

        # Variables to store GPS and vehicle status
        self.last_lat = None
        self.last_lon = None
        self.datapoint = 0      # index of saved datapoints
        self.sub_count = 0      # Index between datapoints to save extra CSV entries
        self.speed = None       # GPS velocity
        self.track = None       # Course over ground, not compass heading
        self.heading = None     # Compass heading
        self.orientation = None  # Using Quaternion
        self.roll = None
        self.pitch = None
        self.yaw = None
        self.steering_angle = None

        # Initialize ZED camera
        self.forward_zed = ZEDCAMERA()

        # Get the current date and time
        timestamp = datetime.now().strftime('%d_%m_%Y_%H_%M')

        # Create the directory path
        self.folder_path = os.path.join('..', 'recordings', timestamp)
        self.images_path = os.path.join(self.folder_path, 'images')
        os.makedirs(self.images_path, exist_ok=True)

        # Create and set up the CSV file in the new folder
        self.csv_filename = os.path.join(self.folder_path, 'recorded_data.csv')
        self.setup_csv()

    def setup_csv(self):
        """Set up the CSV file and write the header row."""
        # If the file does not exist, create it and write the header
        file_exists = os.path.isfile(self.csv_filename)
        with open(self.csv_filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                # Write header only if the file doesn't exist yet
                writer.writerow(['DataPoint', 'SubCount', 'Latitude', 'Longitude', 'Speed', 'Track', 'Heading',
                                 'Orientation_Quaternion'])

    def gps_callback(self, msg):
        self.heading = msg.track

    def gps_raw_callback(self, gps_msg):
        current_lat = gps_msg.latitude
        current_lon = gps_msg.longitude
        self.speed = gps_msg.speed
        self.track = gps_msg.track

        if self.last_lat is None or self.last_lon is None:
            # First data point, so just store the position
            self.last_lat = current_lat
            self.last_lon = current_lon
            self.record_data(img=True)
        else:
            # Check if the distance threshold has been exceeded
            distance = self.calculate_distance(current_lat, current_lon)
            if distance >= self.distance_threshold:
                self.last_lat = current_lat
                self.last_lon = current_lon
                self.sub_count = 0
                self.datapoint += 1
                self.record_data(img=True)
            else:
                # Do not record an image but save CSV data
                self.sub_count += 1
                self.record_data(img=False)

    def imu_callback(self, imu_msg):
        self.orientation = imu_msg.orientation

    def steer_callback(self, steering_angle):
        self.steering_angle = steering_angle.data

    def calculate_distance(self, lat2, lon2):
        """Uses geopy to calculate geodesic distance between two points."""
        point1 = (self.last_lat, self.last_lon)
        point2 = (lat2, lon2)
        return geodesic(point1, point2).meters  # Returns the distance in meters

    def record_data(self, img):
        """Records data to the CSV file and logs the message."""
        self.get_logger().info(f'Recording data at lat: {self.last_lat}, lon: {self.last_lon}, speed: {self.speed}')
        if img:
            self.forward_zed.capture_img(self.datapoint)

        # Prepare row data
        row = [
            self.datapoint,     # DataPoint index
            self.sub_count,     # Sub index between images taken, should be 0 when image recorded
            self.last_lon,      # Latitude
            self.last_lon,      # Longitude
            self.speed,         # Speed
            self.track,         # Track, heading of the velocity
            self.heading,       # Heading
            self.orientation    # Orientation Quaternion
        ]

        # Write data to CSV file
        with open(self.csv_filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(row)


def main(args=None):
    rclpy.init(args=args)
    data_recorder_node = DataRecorderNode()
    rclpy.spin(data_recorder_node)
    data_recorder_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
