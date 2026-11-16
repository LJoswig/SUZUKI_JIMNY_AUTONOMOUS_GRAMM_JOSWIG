"""
Road graph + trails in RViz.

  ros2 launch jimny_map map_view.launch.py graph_file:=test_area.geojson

graph_file is looked up in jimny_map/maps (installed share dir) unless it is an absolute path.
Needs jimny_localization running (navsat_transform provides the lat/lon -> map conversion).
Set rviz:=false on the Jetson when RViz runs on a laptop in the same ROS_DOMAIN_ID.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    rviz_config = os.path.join(get_package_share_directory('jimny_map'), 'rviz', 'map.rviz')
    use_sim_time = LaunchConfiguration('use_sim_time')

    return LaunchDescription([
        DeclareLaunchArgument('graph_file', default_value='test_area.geojson'),
        DeclareLaunchArgument('rviz', default_value='true'),
        DeclareLaunchArgument('use_sim_time', default_value='false',
                              description='true when replaying a bag with --clock'),

        Node(
            package='jimny_map',
            executable='map_visualizer',
            name='map_visualizer',
            output='screen',
            parameters=[{
                'graph_file': LaunchConfiguration('graph_file'),
                'use_sim_time': use_sim_time,
            }],
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config],
            parameters=[{'use_sim_time': use_sim_time}],
            condition=IfCondition(LaunchConfiguration('rviz')),
        ),
    ])
