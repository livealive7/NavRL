"""Random goal generator for NavRL (see isaac_nav/goal_manager_node.py).

Start it after isaac_navrl.launch.py. Every argument below is passed straight
to the node as a parameter of the same name.

    ros2 launch isaac_nav goal_manager.launch.py num_goals:=8 seed:=3 \
        map_file:=/path/to/warehouse_grid.npz use_goal_height:=true
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

# The grid goal_display.py exports by itself when a scene is loaded (host path of
# the container's /Volume/NavRL/maps/scene_grid.npz). Used if it exists.
_grid = os.path.expanduser('~/Volume/NavRL/maps/scene_grid.npz')
DEFAULT_MAP = _grid if os.path.isfile(_grid) else ''

# name -> (default, python type)
ARGS = {
    'num_goals': ('5', int),
    'seed': ('0', int),
    'x_min': ('-20.0', float),
    'x_max': ('20.0', float),
    'y_min': ('-20.0', float),
    'y_max': ('20.0', float),
    'z_min': ('1.0', float),
    'z_max': ('3.0', float),
    'min_goal_spacing': ('3.0', float),
    'min_start_distance': ('3.0', float),
    'reach_radius': ('0.5', float),
    'marker_radius': ('0.1', float),
    'use_planner': ('true', bool),
    'planner_source': ('static', str),
    'live_map_topic': ('/occupancy_map/voxel_map', str),
    'live_resolution': ('0.5', float),
    'live_inflate': ('0.6', float),
    'live_replan_s': ('1.0', float),
    'goal_ignore_radius': ('0.0', float),
    'live_dump_dir': ('', str),
    'use_detour': ('false', bool),
    'stuck_window_s': ('8.0', float),
    'stuck_progress': ('0.5', float),
    'detour_min_altitude': ('1.0', float),
    'detour_offset': ('2.0', float),
    'detour_growth': ('2.0', float),
    'detour_max': ('16.0', float),
    'detour_reach': ('0.8', float),
    'detour_timeout_s': ('20.0', float),
    'lookahead': ('3.0', float),
    'replan_distance': ('1.5', float),
    'dwell_s': ('0.5', float),
    'goal_timeout_s': ('120.0', float),
    'use_goal_height': ('false', bool),
    'map_file': (DEFAULT_MAP, str),
    'odom_topic': ('/drone0/odom', str),
    'udp_host': ('127.0.0.1', str),
    'udp_port': ('9870', int),
    'use_sim_time': ('true', bool),
}


def generate_launch_description():
    declarations = [DeclareLaunchArgument(k, default_value=v[0])
                    for k, v in ARGS.items()]
    params = {k: ParameterValue(LaunchConfiguration(k), value_type=v[1])
              for k, v in ARGS.items()}
    node = Node(package='isaac_nav', executable='goal_manager_node',
                name='goal_manager', parameters=[params], output='screen')
    return LaunchDescription(declarations + [node])
