#!/usr/bin/python3

"""
vehicle_simulator_node.py
Author: Axel Barbelanne, Matthijs Steyerberg
Date: 22-10-2024

Script for node simulating the vehicle dynamics. The simulator uses a linear tire model, more precise than the MPC's. 
Modified from Ugo Rosolia's Code: https://github.com/urosolia/RacingLMPC/blob/master/src/fnc/SysModel.py

Subscribes to:
    - /ackermann_auto: Ackermann drive commands
Publishes:
    - /state_est: Vehicle state data
"""

import rclpy
import numpy as np
import math
from rclpy.node import Node
from mpc_driving_controller.msg import StateEst
from ackermann_msgs.msg import AckermannDrive

class VehicleSimulator(Node):
    """ Node for simulating the vehicle dynamics """
    def __init__(self):
        super().__init__('vehicle_simulator_node')
        
        # Define parameters with default values
        params = [
            ('X0', 0.0),            # X position (m)
            ('Y0', 0.0),            # Y position (m)
            ('Psi0', np.pi / 2),    # Yaw angle (rad)
            ('V0', 0.0),            # Longitudinal velocity (m/s)
            ('L_R', 1.5213),        # Rear axle distance (m)
            ('L_F', 1.4987),        # Front axle distance (m)
            ('STEER_RATIO', 14.)     # Steering ratio between steering wheel and front wheels
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

        self.vy = 0.0  # lateral velocity (m/s)
        self.wz = 0.0  # yaw rate (rad/s)
        self.a_y = 0.0  # lateral acceleration (m/s^2)

        self.acc_time_constant = 0.1  # s
        self.df_time_constant = 0.05  # s

        # Commands and vehicle state
        self.tcmd_a = None  # time of received acc command
        self.tcmd_d = None  # time of received df command
        self.acc = 0.0  # actual longitudinal acceleration (m/s^2)
        self.df = 0.0  # actual steering angle (rad)
        self.acc_des = 0.0  # desired longitudinal acceleration (m/s^2)
        self.df_des = 0.0  # desired steering_angle (rad)

        self.dt_model = 0.2  # vehicle model update period (s)
        self.hz = int(1.0 / self.dt_model)

        # ROS 2 Publishers and Subscribers
        self.state_pub = self.create_publisher(StateEst, 'state_est', 10)
        self.create_subscription(AckermannDrive, 'mpc_commands', self._command_callback, 10)

        # Timer to simulate vehicle dynamics at fixed rate (equivalent to rospy.Rate)
        self.timer = self.create_timer(self.dt_model, self.pub_loop)

    def pub_loop(self):
        # Update vehicle model
        self._update_vehicle_model()

        # Create and publish the state message
        curr_state = StateEst()
        curr_state.header.stamp = self.get_clock().now().to_msg()
        curr_state.x = self.X0
        curr_state.y = self.Y0
        curr_state.psi = self.Psi0
        curr_state.v = (self.V0**2 + self.vy**2)**0.5

        curr_state.v_long = self.V0
        curr_state.v_lat = self.vy
        curr_state.yaw_rate = self.wz

        curr_state.a_long = self.acc
        curr_state.a_lat = self.a_y
        curr_state.df = self.df

        self.state_pub.publish(curr_state)

    def _command_callback(self, msg):
        self.tcmd_a = self.get_clock().now()
        self.acc_des = msg.acceleration
        self.tcmd_d = self.get_clock().now()
        self.df_des = - msg.steering_angle * np.pi / 180 / self.STEER_RATIO

    def _update_vehicle_model(self, disc_steps=10):
        # Genesis Parameters from HCE:
        m = 2303.1  # kg (vehicle mass)
        Iz = 5520.1  # kg*m2 (vehicle inertia)
        C_alpha_f = 7.6419e4 * 2  # N/rad (front axle cornering stiffness)
        C_alpha_r = 13.4851e4 * 2  # N/rad (rear axle cornering stiffness)

        deltaT = self.dt_model / disc_steps
        self._update_low_level_control(self.dt_model)
        for i in range(disc_steps):
            # Compute tire slip angle
            alpha_f = 0.0
            alpha_r = 0.0
            if math.fabs(self.V0) > 1.0:
                alpha_f = self.df - np.arctan2(self.vy + self.L_F * self.wz, self.V0)
                alpha_r = -np.arctan2(self.vy - self.L_R * self.wz, self.V0)

            # Approximate tire force saturation.
            alpha_f = np.clip(alpha_f, -0.1, 0.1)
            alpha_r = np.clip(alpha_r, -0.1, 0.1)

            # Compute lateral force at front and rear tire (linear model)
            Fyf = C_alpha_f * alpha_f
            Fyr = C_alpha_r * alpha_r

            # Propagate the vehicle dynamics deltaT seconds ahead.
            vx_n = max(0.0, self.V0 + deltaT * (self.acc - 1 / m * Fyf * np.sin(self.df) + self.wz * self.vy))

            if vx_n > 1e-6:
                vy_n = self.vy + deltaT * (1.0 / m * (Fyf * np.cos(self.df) + Fyr) - self.wz * self.V0)
                wz_n = self.wz + deltaT * (1.0 / Iz * (self.L_F * Fyf * np.cos(self.df) - self.L_R * Fyr))
                a_y = 1.0 / m * (Fyf * np.cos(self.df) + Fyr)
            else:
                vy_n = 0.0
                wz_n = 0.0
                a_y = 0.0

            psi_n = self.Psi0 + deltaT * self.wz
            X_n = self.X0 + deltaT * (self.V0 * np.cos(self.Psi0) - self.vy * np.sin(self.Psi0))
            Y_n = self.Y0 + deltaT * (self.V0 * np.sin(self.Psi0) + self.vy * np.cos(self.Psi0))

            self.X0 = X_n
            self.Y0 = Y_n
            self.Psi0 = (psi_n + np.pi) % (2.0 * np.pi) - np.pi
            self.V0 = vx_n
            self.vy = vy_n
            self.wz = wz_n
            self.a_y = a_y

    def _update_low_level_control(self, dt_control):
        # Simulate first order control delay in acceleration/steering.
        self.acc = dt_control / (dt_control + self.acc_time_constant) * (self.acc_des - self.acc) + self.acc
        self.df = dt_control / (dt_control + self.df_time_constant) * (self.df_des - self.df) + self.df

def main(args=None):
    rclpy.init(args=args)
    simulator = VehicleSimulator()

    try:
        rclpy.spin(simulator)
    except KeyboardInterrupt:
        pass
    finally:
        simulator.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
