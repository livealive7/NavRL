"""Grid path planner used by goal_manager_node (plain numpy, no ROS).

Plans on the static occupancy grid exported from Isaac Sim (``occupied``, ``origin``,
``resolution``). The grid is already inflated by the drone radius plus margin, so a
path through free cells keeps that clearance from every collider.

  * A* over 6/26-connected free cells, restricted to a band of z layers (a single
    layer when NavRL flies at a constant height).
  * Line-of-sight smoothing removes the staircase of the grid path.
  * ``PathFollower`` turns the path into a moving "carrot": the point a fixed
    distance ahead of the drone along the path. NavRL flies towards the carrot
    instead of the far goal, which makes it go round rack rows instead of nosing
    into them.
"""

import heapq

import numpy as np
from scipy import ndimage

_NEIGHBOURS_3D = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                  for dz in (-1, 0, 1) if (dx, dy, dz) != (0, 0, 0)]
_NEIGHBOURS_2D = [(dx, dy, 0) for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                  if (dx, dy) != (0, 0)]


class GridPlanner:

    def __init__(self, occupied, origin, resolution, soft_margin=0.6):
        self.occ = np.asarray(occupied, dtype=bool)
        self.origin = np.asarray(origin, dtype=float)
        self.res = float(resolution)
        self.shape = np.array(self.occ.shape)
        # Distance (m) of every free cell to the nearest occupied cell. The grid is
        # already inflated, so cells hugging the boundary are legal but leave no room
        # for NavRL's own avoidance: paths are pushed ``soft_margin`` away from it
        # where the space allows, and shortcuts may not cut closer than half of it.
        self.soft_margin = float(soft_margin)
        self.edt = ndimage.distance_transform_edt(~self.occ) * self.res

    # ------------------------------------------------------------ helpers

    def to_cell(self, p):
        return tuple(np.floor((np.asarray(p, dtype=float) - self.origin) / self.res).astype(int))

    def to_world(self, c):
        return self.origin + (np.asarray(c, dtype=float) + 0.5) * self.res

    def inside(self, c):
        return all(0 <= c[i] < self.shape[i] for i in range(3))

    def point_free(self, p):
        c = self.to_cell(p)
        return self.inside(c) and not self.occ[c]

    def segment_free(self, a, b, min_clear=0.0):
        """True if the segment a-b only crosses free cells at least min_clear m from
        occupied ones (the two end cells are exempt from the clearance test)."""
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        n = max(2, int(np.ceil(np.linalg.norm(b - a) / (self.res * 0.5))))
        for i in range(n + 1):
            c = self.to_cell(a + (b - a) * i / n)
            if not self.inside(c) or self.occ[c]:
                return False
            if min_clear > 0.0 and 0 < i < n and self.edt[c] < min_clear:
                return False
        return True

    def visible(self, a, b, skip=0.4):
        """Line of sight a -> b over free cells. The first ``skip`` metres are not
        checked: the drone may sit inside the inflated boundary."""
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        L = float(np.linalg.norm(b - a))
        n = max(2, int(np.ceil(L / (self.res * 0.5))))
        for i in range(n + 1):
            if i * L / n < skip:
                continue
            if not self.point_free(a + (b - a) * i / n):
                return False
        return True

    def snap_free(self, p, max_dist=1.5, zband=None):
        """Nearest free cell to p within max_dist (None if there is none)."""
        c0 = np.array(self.to_cell(p))
        r = int(np.ceil(max_dist / self.res))
        best, best_d = None, None
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                for dz in range(-r, r + 1):
                    c = (c0[0] + dx, c0[1] + dy, c0[2] + dz)
                    if not self.inside(c) or self.occ[c]:
                        continue
                    if zband is not None and not (zband[0] <= c[2] <= zband[1]):
                        continue
                    d = np.linalg.norm([dx, dy, dz]) * self.res
                    if d <= max_dist and (best_d is None or d < best_d):
                        best, best_d = c, d
        return best

    # ------------------------------------------------------------- planning

    def _astar(self, start, goal, zband):
        flat = zband[0] == zband[1]
        nbrs = _NEIGHBOURS_2D if flat else _NEIGHBOURS_3D
        res = self.res
        margin = max(self.soft_margin, 1e-6)
        gx, gy, gz = goal

        def h(c):
            return res * np.sqrt((c[0] - gx) ** 2 + (c[1] - gy) ** 2 + (c[2] - gz) ** 2)

        best = {start: 0.0}
        prev = {}
        heap = [(h(start), 0.0, start)]
        while heap:
            _, g, c = heapq.heappop(heap)
            if c == goal:
                path = [c]
                while c in prev:
                    c = prev[c]
                    path.append(c)
                return path[::-1]
            if g > best.get(c, 1e18):
                continue
            for dx, dy, dz in nbrs:
                q = (c[0] + dx, c[1] + dy, c[2] + dz)
                if not self.inside(q) or self.occ[q] or not (zband[0] <= q[2] <= zband[1]):
                    continue
                step = res * np.sqrt(dx * dx + dy * dy + dz * dz)
                near = max(0.0, (margin - self.edt[q]) / margin)
                ng = g + step * (1.0 + 2.0 * near)
                if ng < best.get(q, 1e18):
                    best[q] = ng
                    prev[q] = c
                    heapq.heappush(heap, (ng + h(q), ng, q))
        return None

    def plan(self, start, goal, fixed_height, z_margin=1.0):
        """Waypoints (N x 3, world frame) from start to goal, or None if unreachable.

        fixed_height: stay in the z layer of the start (NavRL ignores the goal z when
        ``use_goal_height`` is off). Otherwise z may vary between the two heights
        plus ``z_margin``.
        """
        start = np.asarray(start, dtype=float)
        goal = np.asarray(goal, dtype=float)
        if fixed_height:
            goal = np.array([goal[0], goal[1], start[2]])
            kz = self.to_cell(start)[2]
            zband = (kz, kz)
        else:
            lo = min(start[2], goal[2]) - z_margin
            hi = max(start[2], goal[2]) + z_margin
            zband = (max(0, self.to_cell([0, 0, lo])[2]),
                     min(self.shape[2] - 1, self.to_cell([0, 0, hi])[2]))
        cs = self.snap_free(start, zband=zband) or self.snap_free(start, max_dist=2.5, zband=None)
        cg = self.snap_free(goal, zband=zband) or self.snap_free(goal, max_dist=2.5, zband=None)
        if cs is None or cg is None:
            return None
        if not (zband[0] <= cs[2] <= zband[1] and zband[0] <= cg[2] <= zband[1]):
            zband = (min(zband[0], cs[2], cg[2]), max(zband[1], cs[2], cg[2]))
        cells = self._astar(cs, cg, zband)
        if cells is None:
            return None
        pts = [self.to_world(c) for c in cells]
        pts[0] = start if self.point_free(start) else pts[0]
        pts[-1] = goal if self.point_free(goal) else pts[-1]
        path = np.array(self.smooth(pts))
        if fixed_height:  # NavRL keeps its own height, do not wobble between layers
            path[:, 2] = start[2]
        return path

    def smooth(self, pts):
        """Greedy line-of-sight shortcutting."""
        out = [pts[0]]
        i = 0
        while i < len(pts) - 1:
            j = len(pts) - 1
            while j > i + 1 and not self.segment_free(pts[i], pts[j], self.soft_margin * 0.5):
                j -= 1
            out.append(pts[j])
            i = j
        return out


class PathFollower:
    """Carrot point and progress along a polyline."""

    def __init__(self, waypoints):
        self.w = np.asarray(waypoints, dtype=float)
        seg = np.linalg.norm(np.diff(self.w, axis=0), axis=1)
        self.cum = np.concatenate([[0.0], np.cumsum(seg)])
        self.length = float(self.cum[-1])

    def project(self, p):
        """(arc length of the closest path point, distance to the path)."""
        p = np.asarray(p, dtype=float)
        best_s, best_d = 0.0, 1e18
        for i in range(len(self.w) - 1):
            a, b = self.w[i], self.w[i + 1]
            ab = b - a
            L2 = float(ab @ ab)
            t = 0.0 if L2 == 0.0 else float(np.clip((p - a) @ ab / L2, 0.0, 1.0))
            q = a + t * ab
            d = float(np.linalg.norm(p - q))
            if d < best_d:
                best_d, best_s = d, float(self.cum[i] + t * np.sqrt(L2))
        return best_s, best_d

    def point_at(self, s):
        s = float(np.clip(s, 0.0, self.length))
        i = int(np.searchsorted(self.cum, s, side='right') - 1)
        i = min(i, len(self.w) - 2)
        seg = self.cum[i + 1] - self.cum[i]
        t = 0.0 if seg == 0.0 else (s - self.cum[i]) / seg
        return self.w[i] + t * (self.w[i + 1] - self.w[i])

    def carrot(self, p, lookahead, visible=None, step=0.5):
        """(carrot point, distance of p to the path, remaining path length).

        The carrot is the point ``lookahead`` metres ahead along the path. With a
        ``visible(a, b)`` check it is pulled back to the farthest point that is in
        line of sight, so it never sits behind a corner the drone is about to round.
        """
        s, d = self.project(p)
        offset = lookahead
        while visible is not None and offset > 0.0:
            if visible(p, self.point_at(s + offset)):
                break
            offset -= step
        return self.point_at(s + max(offset, 0.0)), d, self.length - s
