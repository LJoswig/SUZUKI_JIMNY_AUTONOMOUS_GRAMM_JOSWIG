from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    # Path to controller launch file
    controller_launch_file = os.path.join(
        get_package_share_directory('mpc_driving_controller'), 'launch', 'controller.launch.py')

    return LaunchDescription([
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
            PythonLaunchDescriptionSource(controller_launch_file)
        ),
    ])
