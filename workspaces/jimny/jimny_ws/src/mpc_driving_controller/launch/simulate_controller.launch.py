"""
simulate_controller.launch.py
Author: Axel Barbelanne, Matthijs Steyerberg
Date: 22-10-2024

Launch file for running the MPC controller with the vehicle simulator, waypoint generator and trajectory plotter.
The following nodes are launched:
1. vehicle_simulator_node: Simulates the vehicle dynamics
2. traj_plotter_node: Plots the vehicle trajectory
3. waypoint_generator_node: Generates waypoints for the vehicle to follow
4. controller_node: MPC controller node
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import numpy as np
import os

lat0_arg = DeclareLaunchArgument('lat0', default_value='34.056135', description='Initial latitude')
lon0_arg = DeclareLaunchArgument('lon0', default_value='-118.243683', description='Initial longitude')
waypoint_file_arg = DeclareLaunchArgument('waypoint_file_name', default_value='dk-2024.10.26-sample.csv', description='Waypoint file name')


def generate_launch_description():

    # CSV waypoint file
    waypoint_file_name = "dk-2024.10.26-sample.csv"  # Set CSV file name

    workspace_path = os.environ.get('ROS_WORKSPACE_PATH', None)
    assert workspace_path is not None, "No workspace directory found. Run start.sh script or export ROS_WORKSPACE_PATH=..."
    waypoint_csv_path = os.path.join(workspace_path, "src/mpc_driving_controller/data/waypoints", waypoint_file_name)

    # Flag for simulation
    IS_SIMULATION = True 

    # Flag to use Frenet MPC (True) or Cartesian MPC (False)
    USE_FRENET_MPC = False

    # Flag for tracking recorded velocity
    TRACK_VELOCITY = False
    TRACK_SCALE = 0.2

    # Initial conditions for simulation
    init_sim = {
        'X0': 0.0,
        'Y0': 0.0, 
        'Psi0': -np.pi/8,
        'V0': 0.0
    }

    init_params = {
        'lat0': -25.4074883217,   # Starting latitude
        'lon0': 28.3384085838  # Starting longitude
    }

    mpc_params = {
        'f': 2.0,       # Frequency of the controller
        'N': 15,        # Prediction horizon
        'DT': 0.5,      # Time step
        'v_ref': 5.0,   # Reference velocity
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
        'DF_MAX': 60*np.pi/180,
        'DF_MIN': -60*np.pi/180,        # min/max front steer angle constraint (rad)
        'DF_DOT_MAX': 60*np.pi/180,
        'DF_DOT_MIN': -60*np.pi/180,    # min/max front steer angle rate constraint (rad/s)
    }


    # Can be reused to improve Frenet MPC
    # if USE_FRENET_MPC:
    #     vehicle_params['AX_MAX'] = 2. 
    #     vehicle_params['AX_MIN'] = -3.              # min/max longitudinal acceleration constraint (m/s^2)
    #     vehicle_params['AY_MAX'] = 3.
    #     vehicle_params['AY_MIN'] = -3.              # min/max lateral acceleration constraint (m/s^2)
    #     vehicle_params['AX_DOT_MAX'] = 1.5      
    #     vehicle_params['AX_DOT_MIN'] = -1.5         # min/max longitudinal jerk constraint (m/s^3)
    #     vehicle_params['AY_DOT_MAX'] = 5.
    #     vehicle_params['AY_DOT_MIN'] = -5.          # min/max lateral jerk constraint (m/s^3)
    #     vehicle_params['EY_MAX'] = 0.8
    #     vehicle_params['EY_MIN'] = -0.8             # min/max lateral error constraint (m)
    #     vehicle_params['EPSI_MAX'] = 30*np.pi/180
    #     vehicle_params['EPSI_MIN'] = -30*np.pi/180  # min/max heading error constraint (rad)
        
    # else:

    vehicle_params['V_MAX'] = 20.
    vehicle_params['V_MIN'] = 0.        # min/max velocity constraint (m/s)
    vehicle_params['A_MAX'] = 2.
    vehicle_params['A_MIN'] = -3.       # min/max acceleration constraint (m/s^2)
    vehicle_params['A_DOT_MAX'] = 1.5
    vehicle_params['A_DOT_MIN'] = -1.5  # min/max jerk constraint (m/s^3)
        
        

    return LaunchDescription([
        Node(
            package='mpc_driving_controller',
            executable='vehicle_simulator_node.py',
            name='vehicle_simulator_node',
            output='screen',
            parameters=[init_sim, vehicle_params]
        ),
        Node(
            package='mpc_driving_controller',
            executable='traj_plotter_node.py',
            name='traj_plotter_node',
            output='screen',
            parameters=[{'wp_csv_path': waypoint_csv_path,
                        'is_simulation': IS_SIMULATION},
                         init_params, mpc_params]
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
                         init_params,
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
    ])
