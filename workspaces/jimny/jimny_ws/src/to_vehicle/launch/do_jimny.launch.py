from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    # Path to controller launch file
    controller_launch_file = os.path.join(
        get_package_share_directory('mpc_driving_controller'), 'launch', 'controller.launch.py')

    return LaunchDescription([
        # The GPS waypoint MPC is not used; start it only with use_mpc:=true.
        # While it runs it publishes /mpc_commands, which mode_switch forwards in ACRO.
        DeclareLaunchArgument(
            'use_mpc',
            default_value='false',
            description='Start the GPS waypoint MPC controller (mpc_driving_controller)'
        ),

        # Launch cuberos_node.py
        Node(
            package='cuberos',
            executable='cuberos_node.py',
            name='cuberos_node',
            output='screen'
        ),

        # Launch mode_switch switch
        Node(
            package='mode_switching',
            executable='switch',
            name='mode_switch',
            output='screen'
        ),

        # Launch to_vehicle node
        Node(
            package='to_vehicle',
            executable='to_vehicle',
            name='to_vehicle',
            output='screen'
        ),

        # Include controller.launch.py from mpc_driving_controller
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(controller_launch_file),
            condition=IfCondition(LaunchConfiguration('use_mpc'))
        ),
    ])
