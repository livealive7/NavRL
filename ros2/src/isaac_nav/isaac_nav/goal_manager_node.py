"""Generates random flight goals and feeds them to NavRL one at a time.

The node owns all the logic; Isaac Sim only draws the goals (see
``isaac_scripts/goal_display.py``).

  * N random 3D goals are sampled (parameter ``num_goals``, ``seed``).
  * With ``map_file`` (a static occupancy grid exported from the scene, see
    ``isaac_scripts/export_occupancy.py``) a goal is only accepted if its cell is
    free AND connected to the drone's start cell (flood fill), so goals never
    land inside obstacles or in enclosed areas the drone cannot reach.
  * Without ``map_file`` the only obstacle knowledge is NavRL's live occupancy
    map, which contains just what the depth camera has already seen. Goals are
    then checked against it when they become active, and bad ones are replaced.
  * Goal states: pending -> next -> current -> reached / failed. The current
    goal is published on /goal_pose (repeated at 1 Hz) and the next one is
    activated as soon as the drone is within ``reach_radius`` for ``dwell_s``.
  * States go out as visualization_msgs/MarkerArray (Foxglove / RViz) and as JSON
    over UDP to the Isaac Sim display script.
"""

import json
import math
import socket
from collections import deque

import numpy as np
import rclpy
from geometry_msgs.msg import Pose, PoseArray, PoseStamped
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from scipy import ndimage
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Empty
from visualization_msgs.msg import Marker, MarkerArray

from isaac_nav.path_planner import GridPlanner, PathFollower

try:
    from map_manager.srv import CheckPosCollision, RayCast
except ImportError:  # map_manager not built / not sourced
    CheckPosCollision = None
    RayCast = None

PENDING, NEXT, CURRENT, REACHED, FAILED = (
    'pending', 'next', 'current', 'reached', 'failed')

# RGBA per state; the Isaac Sim display uses the same table.
STATE_COLORS = {
    PENDING: (0.6, 0.6, 0.6, 0.8),
    NEXT: (0.0, 0.85, 0.95, 0.9),
    CURRENT: (1.0, 0.9, 0.0, 1.0),
    REACHED: (0.1, 0.8, 0.2, 0.8),
    FAILED: (0.9, 0.1, 0.1, 0.8),
}


def load_grid(path):
    """Loads an occupancy grid written by export_occupancy.py."""
    data = np.load(path)
    return (data['occupied'].astype(bool),
            np.asarray(data['origin'], dtype=float),
            float(data['resolution']))


def reachable_free_cells(occupied, start_idx, max_snap_cells=6):
    """Boolean mask of free cells 6-connected to the cell nearest ``start_idx``.

    The start cell itself can be marked occupied (the drone sits on the floor),
    so it is snapped to the closest free cell within ``max_snap_cells``.
    Returns None if no free cell is close enough.
    """
    from scipy import ndimage

    free = ~occupied
    labels, _ = ndimage.label(free, structure=ndimage.generate_binary_structure(3, 1))
    start = np.asarray(start_idx)
    lo = np.maximum(start - max_snap_cells, 0)
    hi = np.minimum(start + max_snap_cells + 1, occupied.shape)
    best, best_d = None, None
    for idx in np.argwhere(free[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]):
        idx = idx + lo
        d = np.linalg.norm(idx - start)
        if best_d is None or d < best_d:
            best, best_d = idx, d
    if best is None:
        return None
    return labels == labels[tuple(best)]


def sample_goals(candidates, count, min_spacing, rng):
    """Greedy random pick of ``count`` points at least ``min_spacing`` apart."""
    order = rng.permutation(len(candidates))
    chosen = []
    for i in order:
        p = candidates[i]
        if all(np.linalg.norm(p - q) >= min_spacing for q in chosen):
            chosen.append(p)
            if len(chosen) == count:
                break
    return chosen


class GoalManager(Node):

    def __init__(self):
        super().__init__('goal_manager')

        p = self.declare_parameter
        p('num_goals', 5)
        p('seed', 0)
        # x/y bounds only apply without map_file (the grid already limits them)
        p('x_min', -20.0)
        p('x_max', 20.0)
        p('y_min', -20.0)
        p('y_max', 20.0)
        p('z_min', 1.0)
        p('z_max', 3.0)
        p('min_goal_spacing', 3.0)
        p('min_start_distance', 3.0)
        p('reach_radius', 0.8)  # keep above navigation_node's goal_stop_distance (0.5)
        p('marker_radius', 0.25)  # sphere radius shown in Isaac Sim (m)
        p('dwell_s', 0.5)
        p('goal_timeout_s', 120.0)
        p('use_goal_height', False)  # must match navigation_node's parameter
        p('map_file', '')
        p('odom_topic', '/drone0/odom')
        p('goal_topic', '/goal_pose')
        p('frame_id', 'map')
        p('udp_host', '127.0.0.1')
        p('udp_port', 9870)
        p('collision_service', '/occupancy_map/check_pos_collision')
        # Global path planning on the map_file grid. NavRL is sent a moving "carrot"
        # point on the path instead of the far goal, so it goes round rack rows.
        # Fixed goals instead of random ones, e.g. "-4.25,18.75,1.25;3,0,1.5" (repeatable tests)
        p('goal_list', '')
        p('use_planner', True)
        # Where the planner's grid comes from: 'static' = map_file (exported from Isaac Sim),
        # 'live' = built while flying from NavRL's own occupancy map (unknown space counts as
        # free, so the path is replanned as obstacles are discovered). In live mode map_file is
        # only used to pick goals, if given.
        p('planner_source', 'static')
        p('live_map_topic', '/occupancy_map/voxel_map')
        p('live_resolution', 0.5)      # planning grid cell (m)
        p('live_inflate', 0.6)         # obstacles are grown by this much (m), like the static grid
        p('live_origin', [-30.0, -30.0, 0.0])   # same box as map_manager's map_size
        p('live_size', [60.0, 60.0, 6.0])
        p('live_window', [8.0, 8.0, 3.0])       # extent map_manager publishes around the drone
        p('live_replan_s', 1.0)
        # Points of the live map closer than this to a goal marker are dropped (the marker sphere
        # is seen by the depth camera). Same name and meaning as navigation_node's parameter.
        p('goal_ignore_radius', 0.0)
        p('status_interval_s', 2.0)    # log goal, distance and flight speed this often (0 = off)
        p('live_dump_dir', '')          # debug: save the grid, drone and path here after every replan
        p('lookahead', 3.0)        # carrot distance ahead of the drone along the path (m)
        p('replan_distance', 1.5)  # replan when the drone is this far off the path (m)
        # Map-free alternative to the planner: when the drone makes no progress towards
        # the goal, fly to a temporary side waypoint first. Exclusive with use_planner.
        p('use_detour', False)
        p('stuck_window_s', 8.0)       # no progress for this long counts as stuck
        p('stuck_progress', 0.5)       # ... if the goal got closer by less than this (m)
        p('detour_min_altitude', 1.0)  # stuck check starts after the drone first reaches this z (first take-off only)
        p('detour_offset', 2.0)        # sideways distance of the first waypoint (m)
        p('detour_growth', 2.0)        # the offset is multiplied by this after both sides failed (2 -> 4 -> 8 -> 16 m)
        p('detour_max', 16.0)          # largest offset; beyond it the goal is given up (m)
        p('detour_reach', 0.8)         # waypoint counts as reached within this (m)
        p('detour_timeout_s', 20.0)    # give up one waypoint after this long
        p('raycast_service', '/occupancy_map/raycast')

        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.num_goals = int(g('num_goals'))
        self.seed = int(g('seed'))
        self.bounds = (g('x_min'), g('x_max'), g('y_min'), g('y_max'))
        self.z_range = (float(g('z_min')), float(g('z_max')))
        self.spacing = float(g('min_goal_spacing'))
        self.min_start = float(g('min_start_distance'))
        self.reach_radius = float(g('reach_radius'))
        self.dwell = float(g('dwell_s'))
        self.marker_radius = float(g('marker_radius'))
        self.timeout = float(g('goal_timeout_s'))
        self.use_goal_height = bool(g('use_goal_height'))
        self.map_file = str(g('map_file'))
        self.frame_id = str(g('frame_id'))
        self.udp_addr = (str(g('udp_host')), int(g('udp_port')))
        self.goal_list = str(g('goal_list')).strip()
        self.use_planner = bool(g('use_planner'))
        self.lookahead = float(g('lookahead'))
        self.planner_source = str(g('planner_source')).lower()
        if self.planner_source not in ('static', 'live'):
            raise ValueError("planner_source must be 'static' or 'live'")
        self.live = self.use_planner and self.planner_source == 'live'
        self.live_res = float(g('live_resolution'))
        self.live_inflate = float(g('live_inflate'))
        self.live_origin = np.array(g('live_origin'), dtype=float)
        self.live_size = np.array(g('live_size'), dtype=float)
        self.live_window = np.array(g('live_window'), dtype=float)
        self.live_replan_s = float(g('live_replan_s'))
        self.goal_ignore_radius = float(g('goal_ignore_radius'))
        self.live_dump_dir = str(g('live_dump_dir'))
        self.live_dumps = 0
        self.status_interval = float(g('status_interval_s'))
        self.last_status = -1e9
        self.replan_distance = float(g('replan_distance'))
        self.use_detour = bool(g('use_detour'))
        if self.use_detour and self.use_planner:
            raise ValueError(
                'use_detour and use_planner are mutually exclusive: '
                'launch with use_planner:=false use_detour:=true')
        self.stuck_window = float(g('stuck_window_s'))
        self.stuck_progress = float(g('stuck_progress'))
        self.detour_min_alt = float(g('detour_min_altitude'))
        self.detour_offset = float(g('detour_offset'))
        self.detour_growth = max(1.1, float(g('detour_growth')))
        self.detour_max = float(g('detour_max'))
        self.detour_reach = float(g('detour_reach'))
        self.detour_timeout = float(g('detour_timeout_s'))

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        self.odom = None
        self.goals = []        # list of dict(pos=np.array, state=str)
        self.current = -1
        self.current_since = 0.0
        self.inside_since = None
        self.generation = 0
        self.last_goal_pub = 0.0
        self.finished = False
        self.verify_future = None
        self.verify_deadline = 0.0
        self.verify_tries = 0
        self.planner = None
        self.follower = None
        self.last_replan = -1e9
        self.last_path_pub = 0.0
        self.path_warned = False
        self.airborne = False   # set once, when the first take-off climb is over; never reset
        self.z_history = deque()  # (time, z) while waiting for the take-off to finish
        self.reset_detour()
        self.raycast_client = None
        self.raycast_warned = False

        self.goal_pub = self.create_publisher(PoseStamped, g('goal_topic'), 10)
        self.marker_pub = self.create_publisher(
            MarkerArray, 'goal_manager/markers', 1)
        self.path_pub = self.create_publisher(Path, 'goal_manager/path', 1)
        # Positions of all goal markers, so NavRL can ignore the lidar hits they cause.
        self.points_pub = self.create_publisher(
            PoseArray, 'goal_manager/goal_points',
            QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                       durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.create_subscription(Odometry, g('odom_topic'), self.odom_cb, 10)
        self.create_subscription(
            Empty, 'goal_manager/regenerate', lambda _: self.reset(), 1)

        if self.live:
            shape = np.ceil(self.live_size / self.live_res).astype(int)
            self.live_occ = np.zeros(shape, dtype=bool)   # obstacles seen so far
            self.live_dirty = False
            self.live_msgs = 0
            self.create_subscription(
                PointCloud2, g('live_map_topic'), self.live_map_cb,
                QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT))
            self.create_timer(self.live_replan_s, self.live_update)

        self.collision_client = None
        if CheckPosCollision is not None and not self.map_file:
            self.collision_client = self.create_client(
                CheckPosCollision, g('collision_service'))

        if self.use_detour and RayCast is not None:
            self.raycast_client = self.create_client(RayCast, g('raycast_service'))
        elif self.use_detour:
            self.get_logger().warn('use_detour needs map_manager (RayCast); it is disabled')
            self.use_detour = False

        self.create_timer(0.1, self.tick)
        self.create_timer(0.5, self.publish_vis)
        self.get_logger().info('Waiting for odometry before generating goals ...')

    # ------------------------------------------------------------ generation

    def odom_cb(self, msg):
        self.odom = msg

    def drone_pos(self):
        p = self.odom.pose.pose.position
        return np.array([p.x, p.y, p.z])

    def now_s(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def reset(self):
        self.goals = []
        self.current = -1
        self.finished = False
        self.follower = None
        self.generation += 1
        self.get_logger().info('Regenerating goals')

    def generate(self):
        rng = np.random.default_rng(self.seed + self.generation)
        start = self.drone_pos()

        if self.map_file:
            occupied, origin, res = load_grid(self.map_file)
            start_idx = np.floor((start - origin) / res).astype(int)
            start_idx = np.clip(start_idx, 0, np.array(occupied.shape) - 1)
            if self.use_planner and not self.live:
                self.planner = GridPlanner(occupied, origin, res)
            mask = reachable_free_cells(occupied, start_idx)
            if mask is None:
                self.get_logger().error(
                    'No free cell near the drone in map_file; is the grid '
                    'aligned with the scene?')
                return []
            idx = np.argwhere(mask)
            centers = origin + (idx + 0.5) * res
            ok = ((centers[:, 2] >= self.z_range[0])
                  & (centers[:, 2] <= self.z_range[1])
                  & (np.linalg.norm(centers - start, axis=1) >= self.min_start))
            candidates = centers[ok]
            self.get_logger().info(
                f'{int(mask.sum())} reachable free cells, '
                f'{len(candidates)} usable as goals')
        else:
            if self.use_planner and not self.live:
                self.get_logger().warn(
                    'use_planner needs a map_file; flying straight to each goal')
            n = 4000
            candidates = np.stack([
                rng.uniform(self.bounds[0], self.bounds[1], n),
                rng.uniform(self.bounds[2], self.bounds[3], n),
                rng.uniform(self.z_range[0], self.z_range[1], n)], axis=1)
            candidates = candidates[
                np.linalg.norm(candidates - start, axis=1) >= self.min_start]
            self.get_logger().warn(
                'No map_file: goals are random inside the bounds and only '
                'checked against the live occupancy map when they become '
                'active. Obstacles and unreachable areas the camera has not '
                'seen yet are NOT excluded.')

        if self.goal_list:
            chosen = [np.array([float(v) for v in item.split(',')])
                      for item in self.goal_list.split(';') if item.strip()]
            return [{'pos': c, 'state': PENDING} for c in chosen]
        chosen = sample_goals(candidates, self.num_goals, self.spacing, rng)
        if len(chosen) < self.num_goals:
            self.get_logger().warn(
                f'Only found {len(chosen)} of {self.num_goals} goals; '
                'relax spacing / bounds')
        return [{'pos': np.array(c), 'state': PENDING} for c in chosen]

    # ----------------------------------------------------------- state logic

    def request_check(self, pos):
        req = CheckPosCollision.Request()
        req.x, req.y, req.z = float(pos[0]), float(pos[1]), float(pos[2])
        req.inflated = True
        return self.collision_client.call_async(req)

    def resample_goal(self, i):
        rng = np.random.default_rng(
            self.seed + self.generation + 1000 * (i + 1) + self.verify_tries)
        self.goals[i]['pos'] = np.array([
            rng.uniform(self.bounds[0], self.bounds[1]),
            rng.uniform(self.bounds[2], self.bounds[3]),
            rng.uniform(*self.z_range)])

    def activate(self, i):
        """Makes goal ``i`` current once the live map (if any) accepts it."""
        self.current = i
        self.goals[i]['state'] = CURRENT
        if i + 1 < len(self.goals):
            self.goals[i + 1]['state'] = NEXT
        self.verify_tries = 0
        if self.collision_client is not None and self.collision_client.service_is_ready():
            self.verify_future = self.request_check(self.goals[i]['pos'])
            self.verify_deadline = self.now_s() + 1.0
            return  # tick() finishes the activation
        self.verify_future = None
        self.start_goal()

    def poll_verify(self):
        """Handles the pending live-map check; returns True while waiting."""
        if self.verify_future is None:
            return False
        if not self.verify_future.done() and self.now_s() < self.verify_deadline:
            return True
        future, self.verify_future = self.verify_future, None
        occupied = future.done() and future.result().occupied
        if occupied and self.verify_tries < 20:
            self.verify_tries += 1
            self.get_logger().warn(
                f'Goal {self.current + 1} is occupied in the live map, resampling')
            self.resample_goal(self.current)
            self.verify_future = self.request_check(self.goals[self.current]['pos'])
            self.verify_deadline = self.now_s() + 1.0
            return True
        self.start_goal()
        return False

    def start_goal(self):
        self.current_since = self.now_s()
        self.inside_since = None
        self.follower = None
        self.path_warned = False
        self.reset_detour()
        self.replan()
        self.publish_goal()
        self.publish_vis()
        pos = self.goals[self.current]['pos']
        self.get_logger().info(
            f'Goal {self.current + 1}/{len(self.goals)} -> '
            f'({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})')

    # --------------------------------------------------------------- planning

    def live_map_cb(self, msg):
        """Folds a map_manager snapshot into the persistent grid.

        map_manager only publishes a box around the drone (``live_window``). Inside that box
        the snapshot is the truth (cells missing from it are free again); outside it the
        earlier knowledge is kept.
        """
        if self.odom is None:
            return
        n = msg.point_step // 4
        if len(msg.fields) < 3 or [f.name for f in msg.fields[:3]] != ['x', 'y', 'z'] \
                or [f.offset for f in msg.fields[:3]] != [0, 4, 8]:
            return
        pts = np.frombuffer(bytes(msg.data), dtype=np.float32).reshape(-1, n)[:, :3].astype(float)
        pos = self.drone_pos()
        # A little inside the published box: its edge moves between snapshots.
        half = self.live_window / 2.0 - 0.5   # live_window is the full size (map_manager halves it)
        lo = np.array([pos[0] - half[0], pos[1] - half[1], self.live_origin[2]])
        hi = np.array([pos[0] + half[0], pos[1] + half[1], pos[2] + half[2]])
        a = np.clip(np.floor((lo - self.live_origin) / self.live_res).astype(int), 0, self.live_occ.shape)
        b = np.clip(np.ceil((hi - self.live_origin) / self.live_res).astype(int), 0, self.live_occ.shape)
        self.live_occ[a[0]:b[0], a[1]:b[1], a[2]:b[2]] = False
        if len(pts) and self.goal_ignore_radius > 0.0 and self.goals:
            marks = np.array([gl['pos'] for gl in self.goals])
            near = np.linalg.norm(pts[:, None, :] - marks[None, :, :], axis=2).min(axis=1)
            pts = pts[near > self.goal_ignore_radius]
        if len(pts):
            idx = np.floor((pts - self.live_origin) / self.live_res).astype(int)
            ok = np.all((idx >= 0) & (idx < np.array(self.live_occ.shape)), axis=1)
            idx = idx[ok]
            self.live_occ[idx[:, 0], idx[:, 1], idx[:, 2]] = True
        self.live_dirty = True
        self.live_msgs += 1

    def live_update(self):
        """Rebuilds the planner from the grid and replans (once per ``live_replan_s``)."""
        if not self.live_dirty or self.odom is None:
            return
        self.live_dirty = False
        # Same idea as the exported grid: obstacles grown by drone radius plus margin.
        dist = ndimage.distance_transform_edt(~self.live_occ) * self.live_res
        self.planner = GridPlanner(dist <= self.live_inflate, self.live_origin, self.live_res)
        if self.goals and 0 <= self.current < len(self.goals) and not self.finished \
                and self.detour_wp is None:
            self.replan(quiet=True)
        if self.live_dump_dir:
            import os
            os.makedirs(self.live_dump_dir, exist_ok=True)
            np.savez_compressed(
                os.path.join(self.live_dump_dir, f'{self.live_dumps:04d}.npz'),
                occ=self.live_occ, origin=self.live_origin, res=self.live_res,
                pos=self.drone_pos(),
                goal=self.goals[self.current]['pos'] if 0 <= self.current < len(self.goals) else np.zeros(3),
                path=self.follower.w if self.follower is not None else np.zeros((0, 3)))
            self.live_dumps += 1


    def replan(self, quiet=False):
        """Plans from the drone's position to the current goal (follower=None if not possible)."""
        self.last_replan = self.now_s()
        if self.planner is None:
            return
        goal = self.goals[self.current]['pos']
        path = self.planner.plan(self.drone_pos(), goal,
                                 fixed_height=not self.use_goal_height)
        if path is None:
            self.follower = None
            if not self.path_warned:
                self.path_warned = True
                self.get_logger().warn(
                    f'No path to goal {self.current + 1} in the grid; flying straight to it')
            return
        self.path_warned = False
        self.follower = PathFollower(path)
        straight = float(np.linalg.norm(goal[:2] - self.drone_pos()[:2]))
        self.publish_path()
        if quiet:
            return
        self.get_logger().info(
            f'Path to goal {self.current + 1}: {self.follower.length:.1f} m, '
            f'{len(path)} waypoints (straight line {straight:.1f} m)')

    def publish_path(self):
        if self.follower is None:
            return
        msg = Path()
        msg.header.frame_id = self.frame_id
        msg.header.stamp = self.get_clock().now().to_msg()
        for w in self.follower.w:
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = (
                float(w[0]), float(w[1]), float(w[2]))
            ps.pose.orientation.w = 1.0
            msg.poses.append(ps)
        self.path_pub.publish(msg)
        self.last_path_pub = self.now_s()

    def target_pos(self):
        """What NavRL is told to fly to: the carrot on the path, or the goal itself."""
        goal = self.goals[self.current]['pos']
        if self.detour_wp is not None:
            return self.detour_wp
        # No path yet (e.g. the drone starts outside the grid): keep retrying, it may
        # have flown into the mapped area by now.
        if (self.follower is None and self.planner is not None
                and self.now_s() - self.last_replan > 2.0):
            self.replan()
        if self.follower is None:
            return goal
        carrot, off_path, remaining = self.follower.carrot(
            self.drone_pos(), self.lookahead, self.planner.visible)
        if off_path > self.replan_distance and self.now_s() - self.last_replan > 2.0:
            self.replan()
            if self.follower is None:
                return goal
            carrot, off_path, remaining = self.follower.carrot(
                self.drone_pos(), self.lookahead, self.planner.visible)
        # The last stretch goes straight to the goal. In live mode the goal can sit in a cell the
        # grid marks as blocked (inflated obstacle), so the path ends beside it and the line of
        # sight test would never pass; NavRL's own avoidance covers the last metres.
        if remaining <= self.lookahead and (self.live or self.planner.visible(self.drone_pos(), goal)):
            return goal
        return carrot

    # ----------------------------------------------------------------- detour

    def reset_detour(self):
        self.detour_wp = None
        self.detour_since = 0.0
        self.detour_side = 0          # first side tried for this goal: +1 left of the goal direction, -1 right
        self.detour_anchor = None     # where the drone first got stuck on this goal; offsets are measured from here
        self.detour_points = None     # world points of the last lidar snapshot (36 x 4 beams)
        self.detour_attempts = 0
        self.detour_start = None
        self.wp_progress = deque()    # (time, distance to the detour waypoint)
        self.detour_future = None
        self.detour_deadline = 0.0
        self.detour_retry_at = 0.0
        self.progress = deque()       # (time, distance to goal)

    def max_attempts(self):
        """Left and right once per distance level, levels = offset, offset * growth, ... detour_max."""
        levels = int(math.floor(math.log(self.detour_max / self.detour_offset, self.detour_growth) + 1e-9)) + 1
        return 2 * max(1, levels)

    def horizontal(self, v):
        v = np.array(v, dtype=float)
        if not self.use_goal_height:
            v[2] = 0.0
        return float(np.linalg.norm(v))

    def request_clearance(self):
        req = RayCast.Request()
        p = self.drone_pos()
        req.position.x, req.position.y, req.position.z = (float(p[0]), float(p[1]), float(p[2]))
        req.start_angle = 0.0
        req.range = 4.0           # same lidar as NavRL: 36 columns x 4 vertical beams
        req.vfov_min, req.vfov_max, req.vbeams, req.hres = -10.0, 20.0, 4, 10.0
        req.visualize = False
        return self.raycast_client.call_async(req)

    def column_clearance(self, response):
        """Free distance (m) per 10 degree column of the lidar, nearest of its 4 beams."""
        pts = np.array(response.points, dtype=float).reshape(-1, 3)
        if len(pts) != 144:
            return None
        self.detour_points = pts
        d = np.linalg.norm(pts - self.drone_pos(), axis=1).reshape(36, 4)
        return d.min(axis=1)

    def corridor_length(self, theta, half_width=0.45, lidar_range=4.0):
        """Free length (m) of a 2 x half_width wide corridor leaving the drone towards theta.

        Uses the lidar hit points of the last snapshot (no hit at the lidar range counts as
        free). Unlike a cone, a narrow aisle does not look blocked because of its side walls.
        """
        pos = self.drone_pos()
        rel = self.detour_points[:, :2] - pos[:2]
        hit = np.linalg.norm(self.detour_points - pos, axis=1) < lidar_range - 0.05
        u = np.array([math.cos(theta), math.sin(theta)])
        along = rel @ u
        across = np.abs(rel[:, 0] * u[1] - rel[:, 1] * u[0])
        sel = hit & (along > 0.05) & (across < half_width)
        return float(along[sel].min()) if sel.any() else lidar_range

    def start_detour(self):
        """Asks for a lidar snapshot; the waypoint is chosen when it arrives."""
        if not self.raycast_client.service_is_ready():
            if not self.raycast_warned:
                self.raycast_warned = True
                self.get_logger().warn('Stuck, but the raycast service is not available yet')
            return
        self.detour_future = self.request_clearance()
        self.detour_deadline = self.now_s() + 1.0

    def poll_detour(self):
        """Finishes a pending snapshot request; returns True while waiting."""
        if self.detour_future is None:
            return False
        now = self.now_s()
        if not self.detour_future.done():
            if now < self.detour_deadline:
                return True
            self.detour_future = None
            self.detour_retry_at = now + 2.0
            return False
        future, self.detour_future = self.detour_future, None
        clear = self.column_clearance(future.result()) if future.result() else None
        if clear is None:
            self.detour_retry_at = now + 2.0
            return False
        self.choose_detour(clear)
        return False

    def choose_detour(self, clear):
        """Picks the next waypoint: left and right once at each distance, then farther.

        Offsets are measured from the place where the drone first got stuck on this goal
        (the anchor), perpendicular to the anchor-goal line, so "right" really is the other
        side of that line. The side with the longer free corridor goes first: e.g. 2 m better
        side, 2 m other side, 4 m better side, 4 m other side, 8 m, 16 m ... up to detour_max. The lidar is only used
        to pick the first side; the waypoint itself is sent at full distance and NavRL avoids
        what is in the way (a waypoint without progress is dropped by update_detour).
        """
        now = self.now_s()
        pos = self.drone_pos()
        goal = self.goals[self.current]['pos']
        start = self.detour_attempts
        total = self.max_attempts()
        if start >= total:
            self.get_logger().warn(
                f'Goal {self.current + 1}: detours exhausted, giving up on it')
            self.advance(FAILED)
            return
        if self.detour_anchor is None:
            self.detour_anchor = pos.copy()
        anchor = self.detour_anchor
        to_goal = goal[:2] - anchor[:2]
        heading = math.atan2(to_goal[1], to_goal[0])
        fwd = np.array([math.cos(heading), math.sin(heading)])
        left = np.array([-fwd[1], fwd[0]])

        if self.detour_side == 0:  # the freer side goes first, and stays the first one
            free_left = self.corridor_length(heading + math.pi / 2)
            free_right = self.corridor_length(heading - math.pi / 2)
            self.detour_side = 1 if free_left >= free_right else -1
        for k in range(start, total):
            sign = self.detour_side if k % 2 == 0 else -self.detour_side
            wanted = min(self.detour_offset * self.detour_growth ** (k // 2), self.detour_max)
            wp = pos.copy()
            wp[:2] = anchor[:2] + sign * wanted * left + 0.5 * wanted * fwd
            vec = wp[:2] - pos[:2]
            side_name = 'left' if sign > 0 else 'right'
            if float(np.linalg.norm(vec)) < 0.5:
                continue
            if self.use_goal_height:
                wp[2] = goal[2]
            self.detour_attempts = k + 1
            self.detour_wp = wp
            self.detour_since = now
            self.detour_start = pos.copy()
            self.wp_progress.clear()
            self.get_logger().info(
                f'Stuck on goal {self.current + 1}: detour {k + 1}/{total} to '
                f'({wp[0]:.1f}, {wp[1]:.1f}, {wp[2]:.1f}), {wanted:.0f} m {side_name} '
                f'of the line from where it got stuck; {float(np.linalg.norm(wp[:2] - pos[:2])):.1f} m from here')
            self.publish_goal()
            return
        # Nothing fits in this snapshot: count one attempt and look again in a moment.
        self.detour_attempts = start + 1
        self.detour_retry_at = now + 3.0
        self.get_logger().info('No detour has room, trying again in 3 s')

    def update_detour(self, now, dist):
        """Stuck detection and waypoint hand-over. Returns True if tick() should stop."""
        if self.detour_wp is not None:
            wp_dist = self.horizontal(self.drone_pos() - self.detour_wp)
            self.wp_progress.append((now, wp_dist))
            while self.wp_progress and now - self.wp_progress[0][0] > self.stuck_window:
                self.wp_progress.popleft()
            t0, d0 = self.wp_progress[0]
            reached = wp_dist <= self.detour_reach
            # No progress towards the waypoint for stuck_window_s: it is behind something.
            blocked = now - t0 >= 0.95 * self.stuck_window and d0 - wp_dist < self.stuck_progress
            # Hard cap, long enough to fly the whole way at about 0.5 m/s.
            slow = now - self.detour_since > max(self.detour_timeout, 2.0 * math.hypot(
                *(self.detour_wp[:2] - self.detour_start[:2])) + 5.0)
            if reached or blocked or slow:
                why = 'reached' if reached else ('blocked' if blocked else 'timed out')
                self.get_logger().info(f'Detour waypoint {why}, back to the goal')
                self.detour_wp = None
                self.progress.clear()
                self.publish_goal()
            return False
        if self.poll_detour():
            return True
        if not self.airborne:
            # Only the first take-off is skipped. The bridge climbs a fixed height above the
            # ground (cmd_vel is ignored meanwhile), so a plain altitude threshold would fire
            # in the middle of the climb: wait for the altitude AND for the climb to stop.
            z = self.drone_pos()[2]
            self.z_history.append((now, z))
            while self.z_history and now - self.z_history[0][0] > 1.0:
                self.z_history.popleft()
            t0, z0 = self.z_history[0]
            if z < self.detour_min_alt or now - t0 < 0.9 or abs(z - z0) > 0.15:
                return False
            self.airborne = True
            self.progress.clear()
        self.progress.append((now, dist))
        while self.progress and now - self.progress[0][0] > self.stuck_window:
            self.progress.popleft()
        if now < self.detour_retry_at or dist <= self.reach_radius + 0.5:
            self.progress.clear()
            return False
        t0, d0 = self.progress[0]
        if now - t0 >= 0.95 * self.stuck_window and d0 - dist < self.stuck_progress:
            self.progress.clear()
            self.start_detour()
            return self.poll_detour()
        return False

    def publish_goal(self):
        pos = self.target_pos()
        msg = PoseStamped()
        msg.header.frame_id = self.frame_id
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = (
            float(pos[0]), float(pos[1]), float(pos[2]))
        msg.pose.orientation.w = 1.0
        self.goal_pub.publish(msg)
        self.last_goal_pub = self.now_s()

    def advance(self, state):
        self.goals[self.current]['state'] = state
        nxt = self.current + 1
        if nxt < len(self.goals):
            self.activate(nxt)
        else:
            self.finished = True
            self.current = -1
            self.publish_vis()
            self.get_logger().info('All goals done')

    def tick(self):
        if self.odom is None or self.finished:
            return
        if not self.goals:
            self.goals = self.generate()
            if not self.goals:
                self.finished = True
                return
            self.activate(0)
            return
        if self.poll_verify():
            return

        now = self.now_s()
        pos = self.goals[self.current]['pos']
        delta = self.drone_pos() - pos
        if not self.use_goal_height:
            delta[2] = 0.0  # NavRL flies at its own height, ignore z
        dist = float(np.linalg.norm(delta))

        if self.status_interval > 0.0 and now - self.last_status >= self.status_interval:
            self.last_status = now
            v = self.odom.twist.twist.linear   # body frame; the norm is the flight speed
            speed = math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)
            d = self.drone_pos()
            self.get_logger().info(
                f'[status] goal {self.current + 1}/{len(self.goals)} dist {dist:.2f} m  '
                f'speed {speed:.2f} m/s  pos ({d[0]:.1f}, {d[1]:.1f}, {d[2]:.1f})')

        if dist <= self.reach_radius:
            if self.inside_since is None:
                self.inside_since = now
            if now - self.inside_since >= self.dwell:
                self.get_logger().info(f'Goal {self.current + 1} reached')
                self.advance(REACHED)
                return
        else:
            self.inside_since = None

        if now - self.current_since > self.timeout:
            self.get_logger().warn(f'Goal {self.current + 1} timed out')
            self.advance(FAILED)
            return

        if self.use_detour and self.update_detour(now, dist):
            return
        if self.current < 0 or self.finished:
            return  # a detour gave up on the last goal

        # The carrot moves with the drone, so refresh it quickly; a fixed goal only
        # needs a keep-alive.
        period = 0.2 if self.follower is not None else 1.0
        if now - self.last_goal_pub >= period:
            self.publish_goal()
        if self.follower is not None and now - self.last_path_pub >= 1.0:
            self.publish_path()

    # --------------------------------------------------------- visualisation

    def publish_vis(self):
        stamp = self.get_clock().now().to_msg()
        array = MarkerArray()
        payload = []
        for i, goal in enumerate(self.goals):
            r, g, b, a = STATE_COLORS[goal['state']]
            m = Marker()
            m.header.frame_id = self.frame_id
            m.header.stamp = stamp
            m.ns = 'goals'
            m.id = i
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x, m.pose.position.y, m.pose.position.z = (
                float(goal['pos'][0]), float(goal['pos'][1]), float(goal['pos'][2]))
            m.pose.orientation.w = 1.0
            size = 2.0 * self.marker_radius * (1.6 if goal['state'] == CURRENT else 1.0)
            m.scale.x = m.scale.y = m.scale.z = size
            m.color.r, m.color.g, m.color.b, m.color.a = r, g, b, a
            array.markers.append(m)
            payload.append({'id': i, 'x': float(goal['pos'][0]),
                            'y': float(goal['pos'][1]),
                            'z': float(goal['pos'][2]),
                            'state': goal['state']})
        self.marker_pub.publish(array)
        points = PoseArray()
        points.header.frame_id = self.frame_id
        points.header.stamp = stamp
        for goal in self.goals:
            pose = Pose()
            pose.position.x, pose.position.y, pose.position.z = (
                float(goal['pos'][0]), float(goal['pos'][1]), float(goal['pos'][2]))
            pose.orientation.w = 1.0
            points.poses.append(pose)
        self.points_pub.publish(points)
        try:
            self.sock.sendto(json.dumps(
                {'generation': self.generation, 'radius': self.marker_radius,
                 'goals': payload}).encode(),
                self.udp_addr)
        except OSError:
            pass


def main(args=None):
    rclpy.init(args=args)
    try:
        node = GoalManager()
    except ValueError as e:
        print(f'[goal_manager] {e}')
        rclpy.shutdown()
        return
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
