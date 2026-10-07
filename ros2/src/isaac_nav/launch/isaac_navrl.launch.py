"""Launch the NavRL ROS2 stack on top of an Isaac Sim (Pegasus) drone.

Starts, for one drone:
  1. odom_bridge_node        Pegasus state/pose + state/twist -> nav_msgs/Odometry.
  2. depth_restamp_node      Re-stamps Isaac's depth image into the odom clock domain.
  3. cmd_vel bridge          NavRL Twist -> ArduPilot or PX4 SITL over MAVLink
                             (backend:=px4|ardupilot, or none to skip).
  4. map_manager             Occupancy map + the /occupancy_map/raycast service.
  5. onboard_detector        Dynamic obstacle detector + /onboard_detector/get_dynamic_obstacles.
     (+ optional YOLO)
  6. safe_action_node        /safe_action/get_safe_action service.
  7. navigation_node         The NavRL policy: /goal_pose + odom in, cmd_vel out.
  8. foxglove_bridge         Optional visualization.

NavRL's own topic and service interface is left untouched. The NavRL nodes are
deliberately started WITHOUT a namespace: their service names
(/occupancy_map/raycast, /onboard_detector/get_dynamic_obstacles,
/safe_action/get_safe_action) and /goal_pose, /navigation_emergency_stop are
relative to the node namespace in the NavRL sources, so a namespace would
silently rename them. Only the Isaac-facing bridges live under /drone<id>.

Typical use:
    ros2 launch isaac_nav isaac_navrl.launch.py backend:=px4

Prerequisites:
  * Pegasus' ROS2Backend publishes /clock (pub_clock: True); every node here
    runs with use_sim_time:=true by default. If /clock is missing, ROS timers
    never fire, so pass use_sim_time:=false in that case.
  * navigation_node.py imports torch, torchrl, tensordict and hydra. They are
    usually not in the system Python, so point navrl_site_packages at a
    site-packages directory that has them (see README.md).
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription,
                            OpaqueFunction)
from launch.conditions import IfCondition
from launch.launch_description_sources import AnyLaunchDescriptionSource
from launch.substitutions import EnvironmentVariable, LaunchConfiguration
from launch.utilities import perform_substitutions
from launch_ros.actions import Node

DEFAULT_PYMAVLINK_SITE_PACKAGES = os.path.expanduser(
    '~/venv-ardupilot/lib/python3.12/site-packages')


def optical_to_body_matrix(x, y, z):
    """Row-major 4x4 pose of a camera optical frame in the FLU body frame.

    The optical frame is x right, y down, z forward (REP 103/104); the body
    frame is x forward, y left, z up. (x, y, z) is the camera position in the
    body frame. NavRL's body_to_*_sensor parameters use this convention.
    """
    return [0.0, 0.0, 1.0, x,
            -1.0, 0.0, 0.0, y,
            0.0, -1.0, 0.0, z,
            0.0, 0.0, 0.0, 1.0]


def launch_setup(context, *args, **kwargs):
    def arg(name):
        return perform_substitutions(context, [LaunchConfiguration(name)])

    drone_id = int(arg('drone_id'))
    ns = f'drone{drone_id}'
    backend = arg('backend').strip().lower()
    if backend not in ('ardupilot', 'px4', 'none'):
        raise ValueError(
            f"backend must be 'ardupilot', 'px4' or 'none', got {backend!r}")

    use_sim_time = LaunchConfiguration('use_sim_time')
    cfg_dir = os.path.join(get_package_share_directory('isaac_nav'), 'config')

    # ---- topic names ---------------------------------------------------
    odom_topic = arg('odom_topic') or f'/{ns}/odom'
    cmd_vel_topic = arg('cmd_vel_topic') or f'/{ns}/cmd_vel'
    depth_topic = f'/{ns}/camera/depth'
    depth_restamped_topic = f'/{ns}/camera/depth_restamped'
    color_topic = f'/{ns}/camera/color/image_raw'

    # ---- MAVLink endpoint ----------------------------------------------
    mavlink_connection = arg('mavlink_connection')
    if not mavlink_connection:
        if backend == 'ardupilot':
            # Not 14550: Pegasus' own ArduPilot backend uses that port for its
            # HIL channel. 14551 is the dedicated companion-computer link.
            mavlink_connection = f'udpout:127.0.0.1:{14551 + 10 * drone_id}'
        else:
            # PX4 SITL streams offboard MAVLink to 14540 + 10 * instance.
            mavlink_connection = f'udpin:0.0.0.0:{14540 + 10 * drone_id}'

    # ---- PYTHONPATH for nodes that need venv packages -------------------
    current_pythonpath = perform_substitutions(
        context, [EnvironmentVariable('PYTHONPATH', default_value='')])

    def merged_pythonpath(extra):
        return os.pathsep.join(p for p in [current_pythonpath, extra] if p)

    pymavlink_path = arg('pymavlink_site_packages')
    navrl_path = arg('navrl_site_packages')

    # ---- shared perception overrides -------------------------------------
    cam_x, cam_y, cam_z = (float(arg('camera_x')), float(arg('camera_y')),
                           float(arg('camera_z')))
    intrinsics = [float(arg('fx')), float(arg('fy')),
                  float(arg('cx')), float(arg('cy'))]
    body_to_camera = optical_to_body_matrix(cam_x, cam_y, cam_z)
    image_cols = int(arg('image_cols'))
    image_rows = int(arg('image_rows'))
    ground_height = float(arg('ground_height'))

    map_overrides = {
        'depth_image_topic': depth_restamped_topic,
        'odom_topic': odom_topic,
        'pose_topic': f'/{ns}/state/pose',
        'depth_intrinsics': intrinsics,
        'image_cols': image_cols,
        'image_rows': image_rows,
        'body_to_depth_sensor': body_to_camera,
        'ground_height': ground_height,
        'map_size': [float(arg('map_size_x')), float(arg('map_size_y')),
                     float(arg('map_size_z'))],
        'use_sim_time': use_sim_time,
    }
    detector_overrides = {
        'depth_image_topic': depth_restamped_topic,
        'color_image_topic': color_topic,
        'odom_topic': odom_topic,
        'pose_topic': f'/{ns}/state/pose',
        'depth_intrinsics': intrinsics,
        'color_intrinsics': intrinsics,
        'image_cols': image_cols,
        'image_rows': image_rows,
        'body_to_camera_depth': body_to_camera,
        'body_to_camera_color': body_to_camera,
        'ground_height': ground_height,
        'use_sim_time': use_sim_time,
    }

    # ---- Isaac-facing bridges (namespaced under /drone<id>) ---------------
    odom_node = Node(
        package='isaac_nav',
        executable='odom_bridge_node',
        name='isaac_odom_bridge',
        namespace=ns,
        parameters=[{
            'pose_topic': 'state/pose',
            'twist_topic': arg('twist_topic'),
            'output_topic': odom_topic,
            'world_frame_id': 'map',
            'use_sim_time': use_sim_time,
        }],
        output='screen',
    )

    depth_restamp_node = Node(
        package='isaac_nav',
        executable='depth_restamp_node',
        name='depth_restamp_node',
        namespace=ns,
        parameters=[{
            'input_topic': depth_topic,
            'output_topic': depth_restamped_topic,
            'use_sim_time': use_sim_time,
        }],
        output='screen',
    )

    actions = [odom_node, depth_restamp_node]

    if backend != 'none':
        executable = ('cmd_vel_ardupilot_bridge' if backend == 'ardupilot'
                      else 'cmd_vel_px4_bridge')
        bridge_params = {
            'connection_string': mavlink_connection,
            'cmd_vel_topic': cmd_vel_topic,
            'use_sim_time': use_sim_time,
        }
        if backend == 'px4':
            bridge_params['odom_topic'] = odom_topic
            bridge_params['takeoff_height'] = float(arg('takeoff_height'))
        actions.append(Node(
            package='isaac_nav',
            executable=executable,
            name=executable,
            namespace=ns,
            parameters=[bridge_params],
            # pymavlink usually lives in a venv the colcon-generated
            # (system python) entry point cannot see.
            additional_env={
                'PYTHONPATH': merged_pythonpath(pymavlink_path)},
            output='screen',
        ))

    # ---- NavRL stack (root namespace on purpose, see module docstring) ----
    actions.append(Node(
        package='map_manager',
        executable='occupancy_map_node',
        name='map_manager_node',
        parameters=[os.path.join(cfg_dir, 'map_param.yaml'), map_overrides],
        output='screen',
    ))

    actions.append(Node(
        package='onboard_detector',
        executable='dynamic_detector_node',
        name='dynamic_detector_node',
        parameters=[os.path.join(cfg_dir, 'dynamic_detector_param.yaml'),
                    detector_overrides],
        output='screen',
    ))

    actions.append(Node(
        package='onboard_detector',
        executable='yolo_detector_node.py',
        name='yolo_detector_node',
        parameters=[os.path.join(cfg_dir, 'yolo_detector_param.yaml'),
                    {'color_image_topic': color_topic,
                     'use_sim_time': use_sim_time}],
        condition=IfCondition(LaunchConfiguration('use_yolo')),
        output='screen',
    ))

    actions.append(Node(
        package='navigation_runner',
        executable='safe_action_node',
        name='safe_action_node',
        parameters=[os.path.join(cfg_dir, 'safe_action_param.yaml'),
                    {'min_height': float(arg('min_height')),
                     'max_height': float(arg('max_height')),
                     'use_sim_time': use_sim_time}],
        output='screen',
    ))

    actions.append(Node(
        package='navigation_runner',
        executable='navigation_node.py',
        name='navigation_node',
        parameters=[os.path.join(cfg_dir, 'navigation_param.yaml'),
                    {'odom_topic': odom_topic,
                     'cmd_topic': cmd_vel_topic,
                     'vel_limit': float(arg('vel_limit')),
                     'use_goal_height': arg('use_goal_height').lower() == 'true',
                     'goal_min_height': float(arg('min_height')),
                     'goal_max_height': float(arg('max_height')),
                     'checkpoint_file': arg('checkpoint_file'),
                     'use_sim_time': use_sim_time}],
        additional_env={'PYTHONPATH': merged_pythonpath(navrl_path)},
        output='screen',
    ))

    actions.append(IncludeLaunchDescription(
        AnyLaunchDescriptionSource(os.path.join(
            get_package_share_directory('foxglove_bridge'),
            'launch', 'foxglove_bridge_launch.xml')),
        launch_arguments={'port': LaunchConfiguration('foxglove_port')}.items(),
        condition=IfCondition(LaunchConfiguration('enable_foxglove')),
    ))

    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('drone_id', default_value='0'),
        DeclareLaunchArgument(
            'backend', default_value='px4',
            description="Flight controller behind the cmd_vel bridge: "
                        "'px4' (default), 'ardupilot', or 'none' to start no bridge "
                        "(e.g. when cmd_vel is consumed by something else)."),
        DeclareLaunchArgument(
            'use_sim_time', default_value='true',
            description='Use /clock published by Pegasus (pub_clock: True).'),
        DeclareLaunchArgument(
            'use_yolo', default_value='false',
            description='Also start the YOLO detector. The dynamic detector '
                        'works from depth alone without it.'),

        # Topics (empty = derive from drone_id).
        DeclareLaunchArgument(
            'use_goal_height', default_value='false',
            description='Honour the z of /goal_pose (clamped to min/max_height). '
                        'Default keeps upstream NavRL: fly at the current height.'),
        DeclareLaunchArgument(
            'takeoff_height', default_value='2.0',
            description='px4 only: climb this many metres after arming before '
                        'cmd_vel is forwarded (0 disables)'),
        DeclareLaunchArgument(
            'odom_topic', default_value='',
            description='Odometry published for NavRL. Default /drone<id>/odom.'),
        DeclareLaunchArgument(
            'cmd_vel_topic', default_value='',
            description='Twist published by NavRL. Default /drone<id>/cmd_vel.'),
        DeclareLaunchArgument(
            'twist_topic', default_value='state/twist',
            description="Twist paired with state/pose in the odometry. NavRL "
                        "needs the BODY-frame one ('state/twist'); "
                        "'state/twist_inertial' is world-frame."),

        # MAVLink / Python environments.
        DeclareLaunchArgument(
            'mavlink_connection', default_value='',
            description='pymavlink connection string. Empty = derive from '
                        'backend and drone_id.'),
        DeclareLaunchArgument(
            'pymavlink_site_packages',
            default_value=(DEFAULT_PYMAVLINK_SITE_PACKAGES
                           if os.path.isdir(DEFAULT_PYMAVLINK_SITE_PACKAGES)
                           else ''),
            description='site-packages directory that provides pymavlink.'),
        DeclareLaunchArgument(
            'navrl_site_packages', default_value='',
            description='site-packages directory that provides torch, '
                        'torchrl, tensordict, hydra-core and einops for '
                        'navigation_node.py.'),

        # NavRL policy.
        DeclareLaunchArgument('vel_limit', default_value='1.0'),
        DeclareLaunchArgument('checkpoint_file',
                              default_value='navrl_checkpoint.pt'),

        # Camera: Pegasus MonocularCamera at 320x240 and 70 deg HFOV.
        DeclareLaunchArgument('image_cols', default_value='320'),
        DeclareLaunchArgument('image_rows', default_value='240'),
        DeclareLaunchArgument('fx', default_value='228.50367736816406'),
        DeclareLaunchArgument('fy', default_value='228.50367736816406'),
        DeclareLaunchArgument('cx', default_value='160.0'),
        DeclareLaunchArgument('cy', default_value='120.0'),
        DeclareLaunchArgument(
            'camera_x', default_value='0.30',
            description='Camera position in the body frame (FLU), meters. '
                        "Pegasus MonocularCamera default is [0.30, 0, 0]."),
        DeclareLaunchArgument('camera_y', default_value='0.0'),
        DeclareLaunchArgument('camera_z', default_value='0.0'),

        # Map and safety layer.
        DeclareLaunchArgument('map_size_x', default_value='60.0'),
        DeclareLaunchArgument('map_size_y', default_value='60.0'),
        DeclareLaunchArgument(
            'map_size_z', default_value='6.0',
            description='Vertical extent above ground_height; must cover '
                        'the cruise altitude.'),
        DeclareLaunchArgument(
            'ground_height', default_value='0.15',
            description='Map floor / ground-point cutoff in meters.'),
        DeclareLaunchArgument('min_height', default_value='0.5'),
        DeclareLaunchArgument('max_height', default_value='5.0'),

        # Visualization.
        DeclareLaunchArgument('enable_foxglove', default_value='false'),
        DeclareLaunchArgument('foxglove_port', default_value='8765'),

        OpaqueFunction(function=launch_setup),
    ])
