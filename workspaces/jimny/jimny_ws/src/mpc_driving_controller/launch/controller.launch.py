"""
controller.launch.py
Author: Axel Barbelanne, Matthijs Steyerberg
Date: 22-10-2024

Launch file for running the MPC controller on a vehicle.
The following nodes are launched:
1. state_pub_node: Publishes the vehicle state from GPS, IMU and steering angle 
2. waypoint_generator_node: Generates waypoints for the vehicle to follow
3. controller_node: MPC controller node
"""

from launch import LaunchDescription
from launch_ros.actions import Node
import numpy as np
from ament_index_python.packages import get_package_prefix
import os

def generate_launch_description():
    waypoint_file_name = "y_line_up.csv"  # Set CSV file name

    workspace_path = os.environ.get('ROS_WORKSPACE_PATH', None)
    assert workspace_path is not None, "No workspace directory found. Run start.sh script or export ROS_WORKSPACE_PATH=..."
    waypoint_csv_path = os.path.join(workspace_path, "src/mpc_driving_controller/data/waypoints", waypoint_file_name)

    # Flag for simulation
    IS_SIMULATION = False

    # Flag to use Frenet MPC (True) or Cartesian MPC (False)
    USE_FRENET_MPC = False

    # Flag for tracking recorded velocity
    TRACK_VELOCITY = False
    TRACK_SCALE = 0.2

    mpc_params = {
        'f': 5.0,       # Frequency of the controller
        'N': 20,        # Prediction horizon
        'DT': 0.25,      # Time step
        'v_ref': 2.0,   # Reference velocity
    }
    if USE_FRENET_MPC:
        mpc_params['Q']= [1.0, 1.0, 100.0, 0]    # State cost - [long_err, lat_err, delta_psi, delta_v]
        mpc_params['R']= [10., 100., 100.]        # Control cost - [acc, acc rate, steering rate]
        mpc_params['s']= 1000.                  # Cost on final velocity 
    else:
        mpc_params['Q']= [1.0, 1.0, 10.0, 0]    # State cost - [long_err, lat_err, delta_psi, delta_v]
        mpc_params['R']= [10., 100., 100.]        # Control cost - [acc, acc rate, steering rate]
        mpc_params['s']= 1000.                  # Cost on final velocity 


    vehicle_params = {
        'L_F': 1.035,                  # Distance from the center of mass to the front axle
        'L_R': 1.265,                  # Distance from the center of mass to the rear axle
        'STEER_RATIO': 14.,                 # Steering ratio steering wheel/front wheel
        'DF_MAX': 30*np.pi/180,
        'DF_MIN': -30*np.pi/180,        # min/max front steer angle constraint (rad)
        'DF_DOT_MAX': 10*np.pi/180,
        'DF_DOT_MIN': -10*np.pi/180,    # min/max front steer angle rate constraint (rad/s)
    }


    vehicle_params['V_MAX'] = 15.
    vehicle_params['V_MIN'] = 0.        # min/max velocity constraint (m/s)
    vehicle_params['A_MAX'] = 2.
    vehicle_params['A_MIN'] = -3.       # min/max acceleration constraint (m/s^2)
    vehicle_params['A_DOT_MAX'] = 1.5
    vehicle_params['A_DOT_MIN'] = -1.5  # min/max jerk constraint (m/s^3)
        
        

    return LaunchDescription([
        Node(
            package='mpc_driving_controller',
            executable='state_pub_node.py',
            name='state_pub_node',
            output='screen',
        ),
        Node(
            package='mpc_driving_controller',
            executable='waypoint_generator_node.py',
            name='waypoint_generator_node',
            output='screen',
            parameters=[{'wp_csv_path': waypoint_csv_path,
                         'do_track_velocity': TRACK_VELOCITY,
                         'v_track_scale': TRACK_SCALE,
                         'is_simulation': IS_SIMULATION},
                         mpc_params]
        ),
        Node(
            package='mpc_driving_controller',
            executable='controller_node.py',
            name='controller_node',
            output='screen',
            parameters=[{'use_frenet_mpc': USE_FRENET_MPC,
                         'is_simulation': IS_SIMULATION},
                         mpc_params, vehicle_params]
        ),
        Node(
            package='mpc_driving_controller',
            executable='traj_plotter_node.py',
            name='traj_plotter_node',
            output='screen',
            parameters=[{'wp_csv_path': waypoint_csv_path,
                        'is_simulation': IS_SIMULATION},
                         mpc_params]
        )
    ])
