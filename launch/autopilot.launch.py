import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    """
    Launches the autopilot node with parameters loaded from a YAML file.
    """

    package_name = 'autopilot_neural_network'
    package_share = get_package_share_directory(package_name)

    # Path to the YAML configuration file
    params_file = os.path.join(package_share, 'config', 'parameters.yaml')

    # Define the node to launch
    autopilot_node = Node(package=package_name,
                          executable='autopilot',
                          name='autopilot',
                          output='screen',
                          parameters=[params_file])

    return LaunchDescription([autopilot_node])
