#!/usr/bin/python3

"""
controller_node.py
Author: Axel Barbelanne, Matthijs Steyerberg
Date: 22-10-2024

Script  for the MPC controller node that generates control commands.
Implements an asynchronous service client to get waypoints for the vehicle to follow.
Modified from Vijay Govindarajan's code: https://github.com/MPC-Berkeley/genesis_path_follower?tab=MIT-1-ov-file

Subscribes to:
    - /state_est: Current vehicle state
    - /mode: Vehicle mode (manual/autonomous)
Publishes:
    - /mpc_commands: Control commands for the vehicle
    - /error_state_pub: Error state of the MPC controller
    - /mpc_path: MPC path for plotting
Service clients:
    - /get_waypoints: Waypoint generation service
"""

import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDrive
from std_msgs.msg import String, Bool
from mpc_driving_controller.srv import GenerateWaypoints
from mpc_driving_controller.msg import StateEst, MpcPath
from std_msgs.msg import Float32MultiArray
import numpy as np

from mpc_driving_controller_py.kinematic_frenet_mpc import KinFrenetMPCPathFollower as F_MPC
from mpc_driving_controller_py.kinematic_cartesian_mpc import KinCartMPCPathFollower as C_MPC


class MPCControllerNode(Node):
    """
    Node for the MPC controller that generates control commands.
    """

    def __init__(self):
        super().__init__('mpc_controller_node')

        params = [
            ('use_frenet_mpc', True),
            ('is_simulation', False),
            # MPC parameters
            ('f', 5.0),
            ('N', 10),
            ('DT', 0.2),
            ('Q', [0.0, 100.0, 500.0, 1.0]),
            ('R', [0.01, 0.01, 0.001]),
            ('s', 100.0),
            # Vehicle parameters
            ('L_F', 1.5213),
            ('L_R', 1.4987),
            ('DF_MAX', 30 * np.pi / 180),
            ('DF_MIN', -30 * np.pi / 180),
            ('DF_DOT_MAX', 30 * np.pi / 180),
            ('DF_DOT_MIN', -30 * np.pi / 180),
            ('STEER_RATIO', 14.),
            # Frenet specific parameters
            ('AX_MAX', 5.0),
            ('AX_MIN', -10.0),
            ('AY_MAX', 3.0),
            ('AY_MIN', -3.0),
            ('AX_DOT_MAX', 1.5),
            ('AX_DOT_MIN', -1.5),
            ('AY_DOT_MAX', 5.0),
            ('AY_DOT_MIN', -5.0),
            ('EY_MAX', 0.8),
            ('EY_MIN', -0.8),
            ('EPSI_MAX', 10 * np.pi / 180),
            ('EPSI_MIN', -10 * np.pi / 180),
            # Cartesian specific parameters
            ('V_MAX', 20.),
            ('V_MIN', 0.),
            ('A_MAX', 2.),
            ('A_MIN', 0.),
            ('A_DOT_MAX', 1.5),
            ('A_DOT_MIN', -1.5)
        ]
        # See launch file for parameter descriptions
        self.declare_parameters(
            namespace='',
            parameters=params
        )

        parameter_keys = [param[0] for param in params]

        # Fetch and dynamically assign parameters
        params = {key: self.get_parameter(key).value for key in parameter_keys}

        for key, value in params.items():
            setattr(self, key, value)

        # Initialize MPC with parameters
        if self.use_frenet_mpc:
            # Frenet MPC
            self.mpc = F_MPC(N=self.N, DT=self.DT, L_F=self.L_F, L_R=self.L_R, Q=self.Q, R=self.R, s=self.s,
                            V_MIN=self.V_MIN, V_MAX=self.V_MAX, A_MIN=self.A_MIN, A_MAX=self.A_MAX, A_DOT_MIN=self.A_DOT_MIN,
                            A_DOT_MAX=self.A_DOT_MAX, DF_MIN=self.DF_MIN, DF_MAX=self.DF_MAX, DF_DOT_MIN=self.DF_DOT_MIN,
                            DF_DOT_MAX=self.DF_DOT_MAX, STEER_RATIO = self.STEER_RATIO)
        else:
            # Cartesian MPC
            self.mpc = C_MPC(N=self.N, DT=self.DT, L_F=self.L_F, L_R=self.L_R, Q=self.Q, R=self.R, s=self.s,
                            V_MIN=self.V_MIN, V_MAX=self.V_MAX, A_MIN=self.A_MIN, A_MAX=self.A_MAX, A_DOT_MIN=self.A_DOT_MIN,
                            A_DOT_MAX=self.A_DOT_MAX, DF_MIN=self.DF_MIN, DF_MAX=self.DF_MAX, DF_DOT_MIN=self.DF_DOT_MIN,
                            DF_DOT_MAX=self.DF_DOT_MAX, STEER_RATIO = self.STEER_RATIO)


        self.finishing = False  # Final state in horizon
        self.final_it = 0       # Remaining iterations when end is reached

        # Commands as Ackermann message (steering angle + acceleration value)
        self.mpc_cmd_pub = self.create_publisher(AckermannDrive, 'mpc_commands', 10)
        # Path pub for plotting:
        self.mpc_path_pub = self.create_publisher(MpcPath, 'mpc_path', 10)

        self.error_state_pub = self.create_publisher(Bool, 'mpc_error_state', 10)
        
        self.state_sub = self.create_subscription(StateEst, 'state_est', self._state_callback, 10)

        self.feedback_sub = self.create_subscription(Float32MultiArray, 'feedback_data', self._feedback_callback, 10)

        if self.is_simulation:
            # Always active in simulation
            self.mpc_active = True
        else:
            # Depends on the mode
            self.mpc_active = False
            self.mode_sub = self.create_subscription(String, 'mode', self._mode_callback, 10)

        # Client for the waypoint generation service
        self.client = self.create_client(GenerateWaypoints, 'get_waypoints')

        # Control loop rate
        self.timer = self.create_timer(1 / self.f, self._control_loop)



        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for service to become available...')

        self.current_state = {'t': -1., 'x0': 0., 'y0': 0., 'psi0': 0., 'v0': 0}
        self.acc_prev = 0
        self.steer_prev = 0

        # Store waypoints after receiving from the service
        self.waypoint_dict = None

        self.df_feedback = None

    def _state_callback(self, msg):
        """Callback function to update the current vehicle state"""
        self.current_state['t']    = msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec
        self.current_state['x0'] = msg.x
        self.current_state['y0'] = msg.y
        self.current_state['psi0'] = msg.psi
        self.current_state['v0'] = msg.v

    def _feedback_callback(self, msg):
        """Callback function to update the current vehicle state from feedback"""
        self.df_feedback = msg.data[3]  # Steering angle feedback value from vehicle

    def _control_loop(self):
        """Main control loop that updates MPC and publishes control commands"""

        if not self.mpc_active:
            return

        # Request waypoints
        self._get_waypoints()

        if self.waypoint_dict is None:
            self.get_logger().info("Waiting for waypoints...")
            return 

        update_dict = {}
        update_dict.update(self.current_state)
        update_dict.update(self.waypoint_dict)
        if self.acc_prev != None:
            update_dict['acc_prev'] = self.acc_prev
            update_dict['df_prev'] = self.steer_prev
        
        update_dict['ending'] = False

        if self.finishing:
            if self.final_it == self.N - 1:
                self.get_logger().info("Finishing trajectory. Shutting down.")
                rclpy.shutdown()
            update_dict['ending'] = True
            self.final_it += 1

        self.mpc.update(update_dict)
        sol_dict = self.mpc.solve()

        # Only publish control commands if the solution is optimal
        
        error_msg = Bool()
        error_msg.data = not sol_dict['optimal']
        self.error_state_pub.publish(error_msg)

        if sol_dict['optimal']:
            acc_sol = sol_dict['u_control'][0]
            steer_sol = sol_dict['u_control'][1]
            acc_rate = (acc_sol - self.acc_prev) / self.DT
            steer_rate = (steer_sol - self.steer_prev) / self.DT
            self.acc_prev = acc_sol
            self.steer_prev = steer_sol

            ackermann_msg = AckermannDrive()
            # ackermann_msg.header.stamp = self.get_clock().now().to_msg()
            
            ackermann_msg.steering_angle = - steer_sol * 180 / np.pi    # Steering angle inverted
            ackermann_msg.acceleration = acc_sol
            ackermann_msg.speed = sol_dict['z_mpc'][1, 3]
            ackermann_msg.jerk = acc_rate
            ackermann_msg.steering_angle_velocity = steer_rate * 180 / np.pi

            self.mpc_cmd_pub.publish(ackermann_msg)

            if self.waypoint_dict['stop'] and not self.finishing:
                self.get_logger().info("End in sight. Stopping MPC.")
                self.finishing = True
                update_dict['ending'] = True

        else:
            self.get_logger().warn("MPC solution not optimal!")

        
        
        # Warm start information for increased performance
        update_dict['warm_start'] = {
            'z_ws': sol_dict['z_mpc'],
            'u_ws': sol_dict['u_mpc'],
            'sl_ws': sol_dict['sl_mpc']
        }
        self._publish_mpc_path_message(sol_dict)

    def _get_waypoints(self):
        request = GenerateWaypoints.Request()
        request.x0 = self.current_state['x0']
        request.y0 = self.current_state['y0']
        request.psi0 = self.current_state['psi0']

        # Asynchronous call for the service request
        future = self.client.call_async(request)
        future.add_done_callback(self._waypoint_response_callback)

    def _waypoint_response_callback(self, future):
        """Callback executed when the service returns a result."""
        try:
            result = future.result()
            if result is not None:
                # Update the waypoint dictionary after successful service call
                self.waypoint_dict = {
                    's0': result.s0,
                    'e_y0': result.e_y0,
                    'e_psi0': result.e_psi0,
                    'x_ref': np.array(result.x_ref),
                    'y_ref': np.array(result.y_ref),
                    'psi_ref': np.array(result.psi_ref),
                    'cdist_ref': np.array(result.cdist_ref),
                    'curv_ref': np.array(result.curv_ref),
                    'v_ref': np.array(result.v_ref),
                    'stop': result.stop
                }
            else:
                self.get_logger().error('Service call returned None!')
        except Exception as e:
            self.get_logger().error(f'Service call failed: {e}')

    def _mode_callback(self, msg):
        """Callback function to update the vehicle mode"""
        # for initial switch to MPC
        if msg.data == 'ACRO' and self.df_feedback is not None:
            if not self.mpc_active:
                self.get_logger().info("MPC Controller activated.")
                self.mpc_active = True
                # Set previous solutions as feedback values
                self.acc_prev = 0
                self.steer_prev = self.df_feedback
        else:
            self.mpc_active = False

    def _publish_mpc_path_message(self, sol_dict):
        mpc_path_msg = MpcPath()

        mpc_path_msg.header.stamp = self.get_clock().now().to_msg()

        mpc_path_msg.solve_status = 'optimal' if sol_dict['optimal'] else 'suboptimal'
        mpc_path_msg.solve_time = sol_dict['solve_time']

        # MPC path states
        mpc_path_msg.xs = list(sol_dict['z_mpc'][:, 0])  # x_mpc
        mpc_path_msg.ys = list(sol_dict['z_mpc'][:, 1])  # y_mpc
        mpc_path_msg.psis = list(sol_dict['z_mpc'][:, 2])  # psi_mpc
        mpc_path_msg.vs = list(sol_dict['z_mpc'][:, 3])  # v_mpc

        # Frenet coordinates, if available
        if 'z_mpc_frenet' in sol_dict:
            mpc_path_msg.ss = list(sol_dict['z_mpc_frenet'][:, 0])  # s_frenet
            mpc_path_msg.eys = list(sol_dict['z_mpc_frenet'][:, 1])  # e_y_frenet
            mpc_path_msg.epsis = list(sol_dict['z_mpc_frenet'][:, 2])  # e_psi_frenet

            mpc_path_msg.vrf = sol_dict['v_ref_frenet']  # v_ref_frenet
            mpc_path_msg.crf = list(sol_dict['curv_ref_frenet'])  # curvature_ref_frenet

        # Reference trajectory
        mpc_path_msg.xr = list(sol_dict['z_ref'][:, 0])  # x_ref
        mpc_path_msg.yr = list(sol_dict['z_ref'][:, 1])  # y_ref
        mpc_path_msg.psir = list(sol_dict['z_ref'][:, 2])  # psi_ref
        mpc_path_msg.vr = list(sol_dict['z_ref'][:, 3])  # v_ref

        # Control inputs (acceleration and steering angle)
        mpc_path_msg.acc = list(sol_dict['u_mpc'][:, 0])  # acc_mpc
        mpc_path_msg.df = list(sol_dict['u_mpc'][:, 1])  # df_mpc

        # Publish the message
        self.mpc_path_pub.publish(mpc_path_msg)


if __name__ == '__main__':
    rclpy.init()
    node = MPCControllerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("MPC Controller Node has been stopped.")
    finally:
        node.destroy_node()
        rclpy.shutdown()
