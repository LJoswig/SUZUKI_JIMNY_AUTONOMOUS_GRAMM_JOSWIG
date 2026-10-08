#!/usr/bin/python3
"""
waypoint_generator_node.py
Author: Axel Barbelanne, Matthijs Steyerberg
Date: 22-10-2024

Script for node providing a service to generate waypoints for the vehicle to follow, from a GPS trajectory. 
The vehicle already takes into account the vehicle's target velocity, path cruvature and MPC horizon.
Modified from Vijay Govindarajan's code: https://github.com/MPC-Berkeley/genesis_path_follower?tab=MIT-1-ov-file

Publishes:
    - /traj_ref: Reference trajectory data
Subscribes
    - /gps_raw: Raw GPS data for initial GPS
Services provided:
    - /get_waypoints: Service to generate waypoints for the vehicle to follow
"""
import rclpy
from rclpy.node import Node
from mpc_driving_controller_py.ref_gps_traj import GPSRefTrajectory
from mpc_driving_controller.srv import GenerateWaypoints
from mpc_driving_controller.msg import RefTraj
from gps_msgs.msg import GPSFix
import time


class WaypointGeneratorNode(Node):
    def __init__(self):
        super().__init__('waypoint_generator_node')
        # Define parameters with descriptions and initial values
        params = [
            ('wp_csv_path', ''),             # Path for waypoints CSV file
            ('do_track_velocity', False),    # Flag for tracking velocity
            ('v_track_scale', 1.0),          # Scale factor for velocity tracking
            ('is_simulation', False),   # Flag for simulation
            ('lat0', 0.0),              # Initial latitude
            ('lon0', 0.0),              # Initial longitude
            # MPC parameters
            ('f', 2.0),                      # Frequency
            ('N', 10),                       # Horizon length
            ('DT', 0.2),                     # Time step
            ('v_ref', 5.0),                  # Reference velocity
            ('Q', [0.0, 100.0, 500.0, 1.0]), # State cost weights
            ('R', [0.01, 0.001])             # Control cost weights
        ]

        # Declare parameters
        self.declare_parameters(
            namespace='',
            parameters=params
        )

        # Extract parameter keys
        parameter_keys = [param[0] for param in params]

        # Fetch and dynamically assign parameters
        params = {key: self.get_parameter(key).value for key in parameter_keys}

        for key, value in params.items():
            setattr(self, key, value)

        self.service_timeout = 1.0

        if self.is_simulation:
            self.LAT0 = self.lat0
            self.LON0 = self.lon0
            self.initializing = False
            # Initialize trajectory generator
            self.ref_trajectory = GPSRefTrajectory(
                csv_filename=self.wp_csv_path,
                LAT0=self.LAT0,
                LON0=self.LON0,
                traj_horizon=self.N,
                traj_dt=self.DT
            )
        else:
            self.LAT0 = None
            self.LON0 = None
            self.gps_subscriber = self.create_subscription(GPSFix, 'gps_raw', self.gps_callback, 10)
            self.initializing = True

        # Reference trajectory for plotting
        self.traj_pub = self.create_publisher(RefTraj, 'traj_ref', 10)

        # Set up the service to handle waypoint requests
        self.srv = self.create_service(GenerateWaypoints, 'get_waypoints', self.handle_get_waypoints)

    def gps_callback(self, msg):
        # Initialize the reference trajectory generator, only runs once (for now)
        if self.initializing:
            self.initializing = False
            self.LAT0 = msg.latitude
            self.LON0 = msg.longitude
            # Initialize trajectory generator
            self.ref_trajectory = GPSRefTrajectory(
                csv_filename=self.wp_csv_path,
                LAT0=self.LAT0,
                LON0=self.LON0,
                traj_horizon=self.N,
                traj_dt=self.DT
            )

    def handle_get_waypoints(self, request, response):
        start_time = time.time()

        while self.initializing:
            if time.time() - start_time > self.service_timeout:
                raise RuntimeError("GPS data timeout")
            time.sleep(0.1)  # sleep briefly to avoid busy waiting
        
        # Extract data from the service request (current pose and optionally target velocity)
        X_init = request.x0
        Y_init = request.y0
        psi_init = request.psi0

        if self.do_track_velocity:
            waypoint_dict = self.ref_trajectory.get_waypoints(X_init, Y_init, psi_init, v_track_scale=self.v_track_scale)
        else:
            waypoint_dict = self.ref_trajectory.get_waypoints(X_init, Y_init, psi_init, v_target=self.v_ref)

        # Fill the response with the generated waypoint data
        response.s0 = waypoint_dict['s0']
        response.e_y0 = waypoint_dict['e_y0']
        response.e_psi0 = waypoint_dict['e_psi0']
        response.x_ref = list(waypoint_dict['x_ref'])
        response.y_ref = list(waypoint_dict['y_ref'])
        response.psi_ref = list(waypoint_dict['psi_ref'])
        response.cdist_ref = list(waypoint_dict['cdist_ref'])
        response.curv_ref = list(waypoint_dict['curv_ref'])
        response.v_ref = list(waypoint_dict['v_ref'])
        response.stop = waypoint_dict['stop']
        
        ref_traj = RefTraj()
        ref_traj.x_traj = response.x_ref
        ref_traj.y_traj = response.y_ref
        self.traj_pub.publish(ref_traj)

        return response


if __name__ == '__main__':
    rclpy.init()

    node = WaypointGeneratorNode()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()
