"""One command for the whole NavRL-on-Isaac-Sim stack, except Isaac Sim itself.

Starts, in this order:
  1. PX4 SITL      ``run_px4.sh`` (make px4_sitl none_iris + the extra mavlink stream).
  2. NavRL         isaac_navrl.launch.py  (bridges, map_manager, safe_action, navigation_node).
  3. goal_manager  goal_manager.launch.py, ``goal_manager_delay`` seconds later (it waits for
                   odometry by itself, the delay only keeps the first log lines readable).

Isaac Sim is started separately: the Pegasus extension window, or
``examples/12_multi_backend_standalone.py`` inside the Isaac Sim container.

Every ``name:=value`` given on the command line is passed to BOTH included launch files; each
one uses the arguments it declares and ignores the rest. The defaults below reproduce the
usual two commands and can be overridden the same way:

    ros2 launch isaac_nav navrl_stack.launch.py
    ros2 launch isaac_nav navrl_stack.launch.py num_goals:=5 seed:=3 vel_limit:=1.5
    # Only PX4 (e.g. after reloading the scene in Isaac Sim; PX4 must be restarted then):
    ros2 launch isaac_nav navrl_stack.launch.py navrl:=false goal_manager:=false
    # Only the ROS side (PX4 started by hand):
    ros2 launch isaac_nav navrl_stack.launch.py run_px4:=false

Control arguments of this file (not forwarded): run_px4, px4_script, navrl, goal_manager,
goal_manager_delay.

Stopping (Ctrl-C) also kills the PX4 process started here (``run_px4.sh`` runs it as a
background job, which ignores SIGINT).
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, ExecuteProcess, GroupAction,
                            IncludeLaunchDescription, OpaqueFunction, TimerAction)
from launch.launch_description_sources import PythonLaunchDescriptionSource

CONTROL_ARGS = ('run_px4', 'px4_script', 'px4_kill_pattern', 'navrl', 'goal_manager',
                'goal_manager_delay')

HOME = os.path.expanduser('~')
SITE_PACKAGES = os.path.join(HOME, 'Volume/IsaacLab/.venv/lib/python3.12/site-packages')
SMALL_MAP = os.path.join(HOME, 'Volume/NavRL/maps/scene_grid_small.npz')

# Used unless given on the command line. Only entries that make sense are added (paths that
# do not exist are left to the launch files' own defaults).
STACK_DEFAULTS = {
    'use_goal_height': 'true',
    'vel_limit': '1.0',
    'goal_ignore_radius': '0.3',
    'goal_clip_behind': '0.3',
    'num_goals': '10',
    'seed': '0',
    'marker_radius': '0.01',
    'use_planner': 'true',
    'use_detour': 'false',
    'enable_foxglove': 'true',   # foxglove_bridge on port 8765 (foxglove_port)
}
if os.path.isdir(SITE_PACKAGES):
    STACK_DEFAULTS['navrl_site_packages'] = SITE_PACKAGES
if os.path.isfile(SMALL_MAP):
    STACK_DEFAULTS['map_file'] = SMALL_MAP


def _truthy(context, name):
    return context.launch_configurations[name].lower() == 'true'


def _build(context):
    share = get_package_share_directory('isaac_nav')
    forwarded = dict(STACK_DEFAULTS)
    forwarded.update({k: v for k, v in context.launch_configurations.items()
                      if k not in CONTROL_ARGS})

    def include(name):
        # Scoped: the arguments an included file declares (e.g. isaac_navrl's odom_topic with
        # an empty default) must not leak into the next one and override its own defaults.
        return GroupAction(scoped=True, actions=[IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(share, 'launch', name)),
            launch_arguments=forwarded.items())])

    actions = []
    if _truthy(context, 'run_px4'):
        script = os.path.expanduser(context.launch_configurations['px4_script'])
        # run_px4.sh starts the px4 binary as a background job, which ignores SIGINT, so
        # stopping the launch would leave PX4 running. The wrapper kills it on the way out
        # (the pattern goes through the environment so pkill cannot match the wrapper itself).
        actions.append(ExecuteProcess(
            cmd=['bash', '-c',
                 'trap \'pkill -f "$PX4_KILL_PATTERN"\' EXIT; bash "$0"', script],
            additional_env={'PX4_KILL_PATTERN': context.launch_configurations['px4_kill_pattern']},
            name='run_px4', output='screen', sigterm_timeout='2'))
    if _truthy(context, 'navrl'):
        actions.append(include('isaac_navrl.launch.py'))
    if _truthy(context, 'goal_manager'):
        delay = float(context.launch_configurations['goal_manager_delay'])
        actions.append(TimerAction(period=delay, actions=[include('goal_manager.launch.py')]))
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('run_px4', default_value='true',
                              description='Start PX4 SITL with run_px4.sh'),
        DeclareLaunchArgument('px4_script', default_value=os.path.join(HOME, 'run_px4.sh'),
                              description='Path of run_px4.sh'),
        DeclareLaunchArgument('px4_kill_pattern', default_value='px4_sitl_default/bin/px4',
                              description='pkill -f pattern of the PX4 process to stop on exit'),
        DeclareLaunchArgument('navrl', default_value='true',
                              description='Start isaac_navrl.launch.py'),
        DeclareLaunchArgument('goal_manager', default_value='true',
                              description='Start goal_manager.launch.py'),
        DeclareLaunchArgument('goal_manager_delay', default_value='5.0',
                              description='Seconds to wait before starting goal_manager'),
        OpaqueFunction(function=_build),
    ])
