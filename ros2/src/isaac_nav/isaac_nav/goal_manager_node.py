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
import socket

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from std_msgs.msg import Empty
from visualization_msgs.msg import Marker, MarkerArray

from isaac_nav.path_planner import GridPlanner, PathFollower

try:
    from map_manager.srv import CheckPosCollision
except ImportError:  # map_manager not built / not sourced
    CheckPosCollision = None

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
        p('lookahead', 3.0)        # carrot distance ahead of the drone along the path (m)
        p('replan_distance', 1.5)  # replan when the drone is this far off the path (m)

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
        self.replan_distance = float(g('replan_distance'))

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

        self.goal_pub = self.create_publisher(PoseStamped, g('goal_topic'), 10)
        self.marker_pub = self.create_publisher(
            MarkerArray, 'goal_manager/markers', 1)
        self.path_pub = self.create_publisher(Path, 'goal_manager/path', 1)
        self.create_subscription(Odometry, g('odom_topic'), self.odom_cb, 10)
        self.create_subscription(
            Empty, 'goal_manager/regenerate', lambda _: self.reset(), 1)

        self.collision_client = None
        if CheckPosCollision is not None and not self.map_file:
            self.collision_client = self.create_client(
                CheckPosCollision, g('collision_service'))

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
            if self.use_planner:
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
            if self.use_planner:
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
        self.replan()
        self.publish_goal()
        self.publish_vis()
        pos = self.goals[self.current]['pos']
        self.get_logger().info(
            f'Goal {self.current + 1}/{len(self.goals)} -> '
            f'({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})')

    # --------------------------------------------------------------- planning

    def replan(self):
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
        self.follower = PathFollower(path)
        straight = float(np.linalg.norm(goal[:2] - self.drone_pos()[:2]))
        self.get_logger().info(
            f'Path to goal {self.current + 1}: {self.follower.length:.1f} m, '
            f'{len(path)} waypoints (straight line {straight:.1f} m)')
        self.publish_path()

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
        if remaining <= self.lookahead and self.planner.visible(self.drone_pos(), goal):
            return goal
        return carrot

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
        try:
            self.sock.sendto(json.dumps(
                {'generation': self.generation, 'radius': self.marker_radius,
                 'goals': payload}).encode(),
                self.udp_addr)
        except OSError:
            pass


def main(args=None):
    rclpy.init(args=args)
    node = GoalManager()
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
