from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'autopilot_neural_network'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(),  # includes scripts/ automatically
    data_files=[
        # Package index
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),

        # package.xml
        ('share/' + package_name, ['package.xml']),

        # Launch files
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),

        # Config files
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Lucas Mazzetto',
    maintainer_email='workabotic@gmail.com',
    description='An end-to-end deep learning pipeline for autonomous driving in ROS 2.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'autopilot = autopilot_neural_network.autopilot:main',
            'data_collector = autopilot_neural_network.data_collector:main',
        ],
    },
)
