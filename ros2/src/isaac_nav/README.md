# isaac_nav

Glue between an Isaac Sim drone (Pegasus ROS2 backend) and the NavRL ROS2 stack
(`map_manager`, `onboard_detector`, `navigation_runner`). ROS2 Jazzy, Isaac Sim 6.0.1.

NavRL's own topic/service interface is unchanged. This package only adds what
NavRL's Unitree-Go2 setup got from the Go2 driver: an `Odometry` topic, a depth
image in the same clock domain, a path from `cmd_vel` to the flight controller,
Isaac-specific parameters and one launch file.

```
isaac_nav/
├── package.xml  setup.py  setup.cfg  resource/isaac_nav     ament_python
├── isaac_nav/
│   ├── odom_bridge_node.py          state/pose + state/twist -> nav_msgs/Odometry
│   ├── depth_restamp_node.py        depth Image, header.stamp := node clock
│   ├── cmd_vel_ardupilot_bridge.py  Twist -> SET_POSITION_TARGET_LOCAL_NED (ArduPilot SITL)
│   └── cmd_vel_px4_bridge.py        Twist -> offboard setpoints, arm + OFFBOARD (PX4 SITL)
├── config/                          NavRL parameters for the Pegasus Iris
│   ├── map_param.yaml  dynamic_detector_param.yaml  yolo_detector_param.yaml
│   └── safe_action_param.yaml  navigation_param.yaml
└── launch/isaac_navrl.launch.py     starts everything
```

## Build

```bash
source /opt/ros/jazzy/setup.bash
sudo apt install ros-jazzy-vision-msgs        # needed by onboard_detector
cd <ros2_ws>/src && cp -r /path/to/NavRL/ros2/{map_manager,onboard_detector,navigation_runner,isaac_nav} .
cd .. && colcon build --symlink-install       # symlink-install is required, see below
source install/setup.bash
```

`--symlink-install` matters: `navigation_runner` only installs `navigation_node.py`
and `cfg/`; `navigation_node.py` imports `navigation`, `ppo`, `utils` and loads
`ckpts/` from its own directory, which only resolves through the symlink.

## Run

```bash
# 1. Isaac Sim with Pegasus (ROS2Backend pub_clock=True, 320x240 camera with depth) and the SITL
# 2. NavRL
ros2 launch isaac_nav isaac_navrl.launch.py \
    backend:=px4 \
    navrl_site_packages:=/path/to/venv/lib/python3.12/site-packages
# 3. Fly (take off first), then send a goal
ros2 topic pub --once /goal_pose geometry_msgs/PoseStamped \
    "{header: {frame_id: map}, pose: {position: {x: 5.0, y: 2.0, z: 0.0}}}"
```

Useful arguments: `drone_id`, `backend:=px4|ardupilot|none` (default `px4`), `use_sim_time`,
`use_yolo`, `odom_topic`, `cmd_vel_topic`, `twist_topic`, `mavlink_connection`,
`pymavlink_site_packages`, `navrl_site_packages`, `vel_limit`, camera
(`fx fy cx cy image_cols image_rows camera_x camera_y camera_z`), map
(`map_size_x/y/z ground_height`), safety layer (`min_height max_height`),
`enable_foxglove`.

## Topic map (drone_id 0)

| Isaac Sim (Pegasus) | isaac_nav | NavRL |
|---|---|---|
| `/drone0/state/pose` PoseStamped, BEST_EFFORT, ENU | `odom_bridge_node` input | |
| `/drone0/state/twist` TwistStamped, BEST_EFFORT, FLU body | `odom_bridge_node` input | |
| | `/drone0/odom` Odometry, frame `map`, body-frame twist | `odom_topic` of `navigation_node`, `map_manager_node`, `dynamic_detector_node` |
| `/drone0/camera/depth` Image 32FC1 (m) | `depth_restamp_node` input | |
| | `/drone0/camera/depth_restamped` | `depth_image_topic` of `map_manager_node`, `dynamic_detector_node` |
| `/drone0/camera/color/image_raw` | | `color_image_topic` of `dynamic_detector_node`, `yolo_detector_node` |
| `/drone0/camera/color/camera_info` | source of the default intrinsics (not subscribed) | |
| `/clock` | `use_sim_time` | all nodes |
| | `/drone0/cmd_vel` Twist (FLU body) | published by `navigation_node` (`cmd_topic`), consumed by the cmd_vel bridge |
| MAVLink to SITL | cmd_vel bridge (ArduPilot `udpout:127.0.0.1:14551+10*id`, PX4 `udpin:0.0.0.0:14540+10*id`) | |
| | | `/goal_pose` PoseStamped in, `/navigation_emergency_stop` Bool in (unchanged) |
| | | services `/occupancy_map/raycast`, `/onboard_detector/get_dynamic_obstacles`, `/safe_action/get_safe_action` (unchanged) |

The NavRL nodes are started without a namespace on purpose: their service and
goal topic names are relative in the NavRL sources, so a namespace would rename them.

## Verified against a live Pegasus + PX4 SITL

* **Camera mount.** The Pegasus camera looks straight ahead, upright, not mirrored: NavRL's own
  `/occupancy_map/raycast` against a flat wall 2.47 m ahead returned 2.5 m at 0 deg and symmetric
  distances at +-10/20/30 deg (2.5, 2.7, 2.9), matching 2.47/cos(theta). So `camera_x = 0.30` and the
  optical-to-FLU rotation in `body_to_depth_sensor` are right, and so are the intrinsics from `camera_info`.
* **Depth.** 320x240 `32FC1` in meters at ~26 Hz; the floor does not appear in the map at 2.2-2.6 m altitude
  with `ground_height: 0.15`.
* **Closed loop.** `backend:=px4`, vehicle hovering at ~2.2-2.5 m: the bridge switched to OFFBOARD by itself
  (vehicle was already armed). A goal 4 m to the side: NavRL yawed to the goal, flew at 1.0 m/s and stopped
  1.0 m short (its design), altitude deviation 0.04 m. A goal behind a wall: no collision (closest approach
  4.0 m in x, wall surface at ~5.5 m), altitude deviation 0.08 m; it dithers in front of the wall because the
  policy is purely local.
* **Clocks.** Pegasus stamps `state/*` with wall time (~1.79e9 s) but `/clock` is sim time (~3000 s);
  the bridges re-stamp with the node clock, which under `use_sim_time:=true` is the sim clock. The sim ran at
  roughly 0.4x real time, so NavRL's 20 Hz timers show up at ~9 Hz wall rate.

## Unverified assumptions

1. **Body-frame twist.** `state/twist` is documented as FLU body velocity; it was not compared against
   `state/twist_inertial` with a yawed vehicle (the flights above only indirectly support it).
2. **Safety layer heights.** `min_height 0.5` / `max_height 5.0` merely bracket the cruise altitude; they were
   not exercised near their limits.
3. **ArduPilot backend.** Only the PX4 path was flown. The ArduPilot bridge and its port 14551 (companion link,
   assumed from the reference launch file's comments) were not run.
4. **Moving obstacles.** `dynamic_detector` reported 0 obstacles in a static scene; detection of moving
   obstacles was not tested, nor was `use_yolo`.
5. **Image-size dependent filters.** `depth_filter_margin` was halved (5) for 320x240.
6. **PX4 bridge arms and enters OFFBOARD by itself** (`auto_arm_offboard` defaults to true in the reference
   bridge, kept unchanged), also before take-off.
7. **Yaw alignment.** The reference angle controller feeds the raw (unwrapped) angle difference to its PID, so
   it may turn the long way round (observed once: ~180 deg turn before flying off).
8. **Python environment.** `navigation_node.py` needs only `torch`, `hydra-core`/`omegaconf` and `numpy`
   (torchrl/tensordict/einops are no longer used). Point `navrl_site_packages` at a site-packages directory
   that has them; the Isaac Lab venv worked.

## Changes made to the NavRL sources (needed on Jazzy / without old torchrl)

* `map_manager/.../occupancyMap.h`, `onboard_detector/.../dynamicDetector.h`: `cv_bridge/cv_bridge.h`
  -> `cv_bridge/cv_bridge.hpp` (Jazzy only ships the `.hpp`).
* `onboard_detector/.../dynamicDetector.cpp`: `visCB()` returns until the first `detectionCB()` has
  created `uvDetector_`; without it the node segfaults at start-up when no depth image has arrived yet.
* `navigation_runner/scripts/navigation.py`: also waits for `/onboard_detector/get_dynamic_obstacles` at
  start-up. Only the raycast service was awaited; a synchronous `call()` made before the other server is
  discovered is lost and never returns, which froze the control loop (no `cmd_vel` at all) whenever the nodes
  were started together.
* `navigation_runner/scripts/ppo.py`: plain-PyTorch inference policy replacing the torchrl/tensordict
  one. State-dict keys are unchanged, so `ckpts/navrl_checkpoint.pt` loads with `strict=True`.
* `navigation_runner/scripts/navigation.py`, `utils.py`: no torchrl/tensordict imports (specs and
  `TensorDict` observations became a plain dict; training-only `make_batch` removed).

## Other tests

* New policy vs the original torchrl 0.6 code on the official checkpoint, 40 random observations:
  bit-identical outputs under torch 2.5.1; max difference 1.5e-6 under torch 2.11 without torchrl.
* `odom_bridge_node` / `depth_restamp_node` with synthetic publishers; `colcon build` of all four packages
  on Jazzy; launch argument handling for `backend:=px4|ardupilot|none` (default `px4`).

## Gotchas

* Run with `use_sim_time:=false` if Pegasus does not publish `/clock`, otherwise no timer ever fires.
* Do not run MAVROS and `cmd_vel_px4_bridge` together: both bind UDP 14540.
* After stopping the launch, check for a leftover `navigation_node.py` (it ignores SIGTERM while blocked in a
  service call).

## Random goals (goal_manager_node + Isaac Sim display)

`goal_manager_node` (ROS2) samples N random 3D goals, publishes them one at a time
on `/goal_pose` and moves to the next once the drone stays within `reach_radius`
for `dwell_s` (or after `goal_timeout_s`, marked failed). Isaac Sim only draws
them: `isaac_scripts/goal_display.py` is loaded automatically by the Pegasus
`multi_backend_example` extension (container path `/Volume/NavRL/...`, override
with env `NAVRL_GOAL_DISPLAY`; restart Isaac Sim after changing the extension).
It listens on UDP 9870 and draws each goal as a coloured USD sphere without collision
(grey pending, cyan next, yellow current = 1.6x larger, green reached, red failed). The
radius is `marker_radius` of the goal manager (default 0.25 m). The spheres are real
meshes, so the drone's depth camera sees them and NavRL treats them as obstacles: it
steers around the sphere instead of flying into it. Keep the radius small (the goal
then counts as reached once within `reach_radius`, see below). The same states go out
as `/goal_manager/markers` (MarkerArray) for Foxglove / RViz.

- Obstacle / reachability filtering needs a static grid. `goal_display.py` exports it
  by itself once per loaded scene (drone spawned, simulation playing), with x/y
  bounds from the scene's colliders (drone, goal markers and ground planes left out,
  cut to +-40 m around the drone) and z 0..5 m, to `~/Volume/NavRL/maps/scene_grid.npz`.
  `goal_manager.launch.py` uses that file as `map_file` when it exists. Goals are then
  free cells connected to the drone's start cell. Without a map only NavRL's live
  map (what the camera has seen) is used. To redo it by hand or with other bounds:
  `ros2 run isaac_nav grid_export_request --x -20 20 --y -20 20 --z 0 5`.
- `reach_radius` must stay above 1 m: NavRL stops commanding velocity within 1 m.
- `use_goal_height` must match navigation_node's parameter (z is checked only then).
- Regenerate goals: `ros2 topic pub --once /goal_manager/regenerate std_msgs/msg/Empty "{}"`.
- Tested: goal logic with synthetic odometry and grid (no goal in walls or a sealed
  room over 300 seeds); display and grid export in headless Isaac Sim with a toy
  scene. NOT tested inside the Pegasus extension, the Full Warehouse scene or a
  live drone.

### Global path planning (goal_manager_node, `use_planner`, default on)

With a `map_file` the goal manager plans an A* path on the exported grid (distance
transform cost keeps it away from the inflated boundary, line-of-sight smoothing)
and sends NavRL a moving carrot point `lookahead` metres ahead along the path
(5 Hz on `/goal_pose`) instead of the far goal. The carrot is pulled back to the
farthest point in line of sight, so it never sits behind a corner. The path is
published on `/goal_manager/path`. The drone is re-planned from where it is when it
gets more than `replan_distance` off the path. Reach detection still uses the final
goal. Without a path (or without `map_file`) it falls back to flying straight to the
goal. `goal_list:="x,y,z;x,y,z"` replaces the random goals with fixed ones.
With `use_goal_height:=false` the path stays in the drone's current z layer (NavRL
ignores the goal height then).

Tested: synthetic wall (detour round the end, sealed goal -> no path, carrot-following
simulated drone reaches both goals). NOT tested with a live drone. Run tests in their
own ROS domain (`ROS_DOMAIN_ID=88 ROS_LOCALHOST_ONLY=1`), the node publishes
/goal_pose and tests publish /drone0/odom, which would disturb a running system.
