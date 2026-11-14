"""
REP-105 localization: map -> odom -> base_link (rear axle) -> {imu_link, gps_link}.

  speed_twist        /gps_raw (Doppler speed) -> /speed/twist
  ekf_local          /speed/twist + /imu/data          -> /odometry/local,  TF odom -> base_link
  ekf_global         + /odometry/gps                   -> /odometry/global, TF map -> odom
  navsat_transform   /gps/fix + /imu/data + /odometry/global -> /odometry/gps, /gps/filtered

Sensor mounting offsets are given in base_link (REP-103), in metres:
  origin = centre of the rear axle (midway between the rear wheels, at axle height)
  x = forward along the car's centreline, y = left (right of centre is negative), z = up
  e.g. gps_z = antenna height above ground - rear axle height above ground.
GPS antenna measured 2026-11-14: 0.93 m ahead of the rear axle, centred, 1.35 m above the axle.
The GPS lever arm matters: an antenna 2 m ahead of the axle appears 2 m off in the map if it
is not set.
imu_yaw stays 0 if ArduPilot's AHRS_ORIENTATION already accounts for how the Cube is mounted.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def static_tf(child, prefix):
    return Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name=f'base_link_to_{child}',
        arguments=[
            '--x', LaunchConfiguration(f'{prefix}_x'),
            '--y', LaunchConfiguration(f'{prefix}_y'),
            '--z', LaunchConfiguration(f'{prefix}_z'),
            '--yaw', LaunchConfiguration(f'{prefix}_yaw'),
            '--frame-id', 'base_link',
            '--child-frame-id', child,
        ],
    )


def generate_launch_description():
    config_dir = os.path.join(get_package_share_directory('jimny_localization'), 'config')
    ekf_config = os.path.join(config_dir, 'ekf.yaml')
    navsat_config = os.path.join(config_dir, 'navsat.yaml')

    mount_args = []
    for prefix, defaults in (('gps', ('0.93', '0.0', '1.35', '0.0')),
                             ('imu', ('0.0', '0.0', '0.0', '0.0'))):
        for axis, default in zip(('x', 'y', 'z', 'yaw'), defaults):
            mount_args.append(DeclareLaunchArgument(
                f'{prefix}_{axis}', default_value=default,
                description=f'{prefix} mount {axis} relative to the rear axle (m / rad)'))

    # true when replaying a bag with `ros2 bag play --clock`
    sim_time = {'use_sim_time': LaunchConfiguration('use_sim_time')}

    return LaunchDescription(mount_args + [
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        static_tf('gps_link', 'gps'),
        static_tf('imu_link', 'imu'),

        Node(
            package='jimny_localization',
            executable='speed_twist',
            name='speed_twist',
            output='screen',
            parameters=[sim_time],
        ),
        Node(
            package='robot_localization',
            executable='ekf_node',
            name='ekf_local',
            output='screen',
            parameters=[ekf_config, sim_time],
            remappings=[('odometry/filtered', '/odometry/local')],
        ),
        Node(
            package='robot_localization',
            executable='ekf_node',
            name='ekf_global',
            output='screen',
            parameters=[ekf_config, sim_time],
            remappings=[('odometry/filtered', '/odometry/global')],
        ),
        Node(
            package='robot_localization',
            executable='navsat_transform_node',
            name='navsat_transform',
            output='screen',
            parameters=[navsat_config, sim_time],
            remappings=[
                ('imu', '/imu/data'),
                ('gps/fix', '/gps/fix'),
                ('odometry/filtered', '/odometry/global'),
                ('odometry/gps', '/odometry/gps'),
                ('gps/filtered', '/gps/filtered'),
            ],
        ),
    ])
