#!/usr/bin/python3

"""
traj_plotter_node.py
Author: Axel Barbelanne, Matthijs Steyerberg
Date: 22-10-2024

Script for node plotting the global GPS trajectory and the vehicle's path tracking behavior.
Modified from Vijay Govindarajan's code: https://github.com/MPC-Berkeley/genesis_path_follower?tab=MIT-1-ov-file

Subscribes to:
    - /state_est: Vehicle state data
    - /mpc_path: MPC planned path
    - /traj_ref: Reference trajectory data
"""

import rclpy
from rclpy.node import Node
import numpy as np
import matplotlib
import os
from mpc_driving_controller_py import ref_gps_traj as r
from mpc_driving_controller.msg import StateEst, MpcPath, RefTraj
from gps_msgs.msg import GPSFix

matplotlib.use('Agg' if os.environ.get('DISPLAY', '') == '' else 'GTK3Agg')

import matplotlib.pyplot as plt



class PlotGPSTrajectory(Node):
    '''
    A class to plot the global GPS trajectory and the vehicle's path tracking behavior
    '''

    def __init__(self):
        super().__init__('vehicle_plotter')

        # Define parameters with descriptions and initial values
        params = [
            ('wp_csv_path', ''),        # CSV path for waypoints
            ('is_simulation', False),   # Flag for simulation
            ('lat0', 0.0),              # Initial latitude
            ('lon0', 0.0),              # Initial longitude
            ('N', 10),                  # Horizon length
            ('DT', 0.2)                 # Time step
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


        if not self.wp_csv_path or self.lat0 is None or self.lon0 is None:
            raise ValueError("Missing required parameters: wp_csv_path, lat0, lon0")

        self.service_timeout = 1.0

        if self.is_simulation:
            self.initializing = False
            self.LAT0 = self.lat0
            self.LON0 = self.lon0
            # Initialize trajectory generator
            self.ref_trajectory = r.GPSRefTrajectory(
                csv_filename=self.wp_csv_path,
                LAT0=self.LAT0,
                LON0=self.LON0,
                traj_horizon=self.N,
                traj_dt=self.DT
            )
            self.setup_graph()
        else:
            self.initializing = True
            self.LAT0 = None
            self.LON0 = None
            self.gps_subscriber = self.create_subscription(GPSFix, 'gps_raw', self.gps_callback, 10)

        self.create_subscription(StateEst, 'state_est', self._update_state, 1)
        self.create_subscription(MpcPath, 'mpc_path', self._update_mpc_trajectory, 1)
        self.create_subscription(RefTraj, 'traj_ref', self._update_ref_traj, 1)

    def gps_callback(self, msg):
        # Initialize the reference trajectory generator, only runs once (for now)
        if self.initializing:
            self.LAT0 = msg.latitude
            self.LON0 = msg.longitude
            # Initialize trajectory generator
            self.ref_trajectory = r.GPSRefTrajectory(
                csv_filename=self.wp_csv_path,
                LAT0=self.LAT0,
                LON0=self.LON0,
                traj_horizon=self.N,
                traj_dt=self.DT
            )
            self.setup_graph()
            self.initializing = False

    def setup_graph(self):
        # Set up Data
        self.x_global_traj = self.ref_trajectory.trajectory[:, self.ref_trajectory.access_map['x']]
        self.y_global_traj = self.ref_trajectory.trajectory[:, self.ref_trajectory.access_map['y']]
        self.x_ref_traj = self.x_global_traj[0]
        self.y_ref_traj = self.y_global_traj[0]
        self.x_mpc_traj = self.x_global_traj[0]
        self.y_mpc_traj = self.y_global_traj[0]
        self.x_vehicle = self.x_global_traj[0]
        self.y_vehicle = self.y_global_traj[0]
        self.psi_vehicle = 0.0
        self.df_vehicle = 0.0

        # Set up Plot
        self.f = plt.figure()
        self.ax = plt.gca()
        plt.ion()

        # Set plot limits
        x_min, x_max = np.min(self.x_global_traj), np.max(self.x_global_traj)
        y_min, y_max = np.min(self.y_global_traj), np.max(self.y_global_traj)

        axis_margin = 100
        self.ax.set_xlim([x_min - axis_margin, x_max + axis_margin])
        self.ax.set_ylim([y_min - axis_margin, y_max + axis_margin])
        

        self.l1, = self.ax.plot(self.x_global_traj, self.y_global_traj, 'k')    # Global Trajectory
        self.l2, = self.ax.plot(self.x_ref_traj, self.y_ref_traj, 'b')          # Reference Trajectory
        self.l3, = self.ax.plot(self.x_mpc_traj, self.y_mpc_traj, 'g*')         # MPC Trajectory
        self.l4, = self.ax.plot(self.x_vehicle, self.y_vehicle, 'rx')           # Vehicle Trajectory

    def _update_ref_traj(self, msg):
        if self.initializing:
            return
        self.l2.set_xdata(msg.x_traj)
        self.l2.set_ydata(msg.y_traj)

        # Update the plot
        plt.pause(0.001)

    def _update_state(self, msg):
        if self.initializing:
            return
        # Append the new vehicle position to lists for x and y
        if not hasattr(self, 'x_vehicle_list'):
            self.x_vehicle_list = []
        if not hasattr(self, 'y_vehicle_list'):
            self.y_vehicle_list = []

        self.x_vehicle_list.append(msg.x)
        self.y_vehicle_list.append(msg.y)

        self.l4.set_xdata(self.x_vehicle_list)
        self.l4.set_ydata(self.y_vehicle_list)

        # Update the plot
        plt.pause(0.001)


    def _update_mpc_trajectory(self, msg):
        if self.initializing:
            return
        # Update the MPC planned trajectory
        self.x_mpc_traj = msg.xs
        self.y_mpc_traj = msg.ys

        self.l3.set_xdata(self.x_mpc_traj)
        self.l3.set_ydata(self.y_mpc_traj)

        # Update the plot
        plt.pause(0.001)

def main(args=None):
    rclpy.init(args=args)
    node = PlotGPSTrajectory()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
