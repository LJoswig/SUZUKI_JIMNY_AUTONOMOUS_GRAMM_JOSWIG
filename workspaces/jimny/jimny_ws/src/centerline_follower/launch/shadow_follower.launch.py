"""Launch the road follower in shadow mode (publishes only /centerline_follower/*)."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    parameters = os.path.join(
        get_package_share_directory('centerline_follower'), 'config', 'shadow_follower.yaml')
    use_sim_time = LaunchConfiguration('use_sim_time')
    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time', default_value='false',
            description='true when replaying a bag with ros2 bag play --clock'),
        Node(
            package='centerline_follower',
            executable='follower',
            name='centerline_follower',
            output='screen',
            parameters=[parameters, {'use_sim_time': use_sim_time}],
        ),
    ])
