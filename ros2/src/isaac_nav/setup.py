import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'isaac_nav'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='peterson',
    maintainer_email='livealive7@gmail.com',
    description='Isaac Sim (Pegasus) to NavRL ROS2 glue',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'odom_bridge_node = isaac_nav.odom_bridge_node:main',
            'depth_restamp_node = isaac_nav.depth_restamp_node:main',
            'cmd_vel_ardupilot_bridge = isaac_nav.cmd_vel_ardupilot_bridge:main',
            'cmd_vel_px4_bridge = isaac_nav.cmd_vel_px4_bridge:main',
        ],
    },
)
