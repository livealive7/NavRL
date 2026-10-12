"""Isaac Sim side of the goal manager: draws goals and exports the scene grid.

Loaded automatically by the Pegasus ``multi_backend_example`` extension (see
``start()``), so nothing has to be run by hand. Everything is driven by UDP
messages on port 9870, no ROS2 is needed in Isaac Sim:

  * ``{"goals": [{"id", "x", "y", "z", "state"}, ...]}`` from
    ``goal_manager_node``: coloured spheres (radius from its ``marker_radius``)
    without collision. They are real meshes, so the drone's depth camera sees them
    and NavRL treats them as obstacles; keep the radius small.
  * The 3D occupancy grid is exported automatically once per loaded scene (drone
    spawned and simulation playing), with bounds taken from the scene's colliders.
  * ``{"cmd": "export_grid", ...}`` from ``grid_export_request``: writes the 3D
    occupancy grid used by ``goal_manager_node`` (``map_file``). The simulation
    must be playing, since the PhysX scene is queried.
"""


import json
import os
import socket
import time

import numpy as np
import omni.kit.app
import omni.timeline
import omni.usd
from carb import Float3
from omni.physx import get_physx_scene_query_interface
from pxr import Gf, Sdf, UsdGeom, UsdShade

UDP_PORT = 9870
ROOT_PATH = '/World/GoalMarkers'
RADIUS = 0.25          # default sphere radius (m); goal_manager_node's marker_radius overrides
CURRENT_SCALE = 1.6    # the current goal is drawn this many times larger

# Must match STATE_COLORS in goal_manager_node.py (RGB).
COLORS = {
    'pending': (0.6, 0.6, 0.6),      # grey    - not reached yet
    'next': (0.0, 0.85, 0.95),       # cyan    - the goal after the current one
    'current': (1.0, 0.9, 0.0),      # yellow  - being flown to now
    'reached': (0.1, 0.8, 0.2),      # green   - done
    'failed': (0.9, 0.1, 0.1),       # red     - timed out
}


class GoalDisplay:

    def __init__(self, port=UDP_PORT, auto_export=True, auto_cfg=None):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(('0.0.0.0', port))
        self.sock.setblocking(False)
        self.goals = []
        self.radius = RADIUS
        self.dirty = False
        self.auto_export = auto_export
        self.auto_cfg = auto_cfg or {}
        self.exported_for = None  # root layer of the stage already exported
        self.playing_frames = 0
        self.sub = omni.kit.app.get_app().get_update_event_stream() \
            .create_subscription_to_pop(self._on_update, name='goal_display')
        print(f'[goal_display] listening on UDP {port}')

    def close(self):
        self.sub = None
        self.sock.close()
        stage = omni.usd.get_context().get_stage()
        if stage and stage.GetPrimAtPath(ROOT_PATH):
            stage.RemovePrim(ROOT_PATH)

    # ------------------------------------------------------------------

    def _on_update(self, _event):
        if self.auto_export:
            self._auto_export_check()
        latest = None
        while True:  # keep only the newest goal packet, run every command
            try:
                packet = self.sock.recv(65535)
            except BlockingIOError:
                break
            except OSError:
                return
            try:
                data = json.loads(packet)
            except ValueError:
                continue
            if 'cmd' in data:
                self._command(data)
            else:
                latest = data
        if latest is not None and 'goals' in latest:
            self.goals = latest['goals']
            self.radius = float(latest.get('radius', RADIUS))
            self.dirty = True
        if self.dirty:
            stage = omni.usd.get_context().get_stage()
            if stage is not None:
                self._draw(stage)
                self.dirty = False

    def _auto_export_check(self):
        """Exports the grid once per loaded scene, after the drone spawned and Play."""
        stage = omni.usd.get_context().get_stage()
        if stage is None or not stage.GetPrimAtPath(VEHICLE_PATH):
            self.playing_frames = 0
            return
        layer = stage.GetRootLayer().identifier
        if layer == self.exported_for:
            return
        if not omni.timeline.get_timeline_interface().is_playing():
            self.playing_frames = 0
            return
        self.playing_frames += 1
        if self.playing_frames < 60:  # let physics settle for a moment
            return
        self.exported_for = layer
        try:
            cfg = dict(self.auto_cfg)
            cfg.setdefault('out_file', AUTO_OUT_FILE)
            bounds = scene_bounds(stage)
            if bounds is None:
                print('[goal_display] auto export: no colliders found, skipping')
                return
            cfg.update(bounds)
            print(f"[goal_display] auto export, x {cfg['x_range']} "
                  f"y {cfg['y_range']} z {cfg['z_range']}")
            export_grid(**cfg)
        except Exception as exc:
            print(f'[goal_display] auto export failed: {exc}')

    def _command(self, data):
        if data['cmd'] == 'probe':
            try:
                probe(data['x'], data['y'], data['z'], data.get('radius', 1.0))
            except Exception as exc:
                print(f'[goal_display] probe failed: {exc}')
        elif data['cmd'] == 'export_grid':
            try:
                export_grid(**{k: v for k, v in data.items() if k != 'cmd'})
            except Exception as exc:  # keep the sim alive whatever happens
                print(f'[goal_display] export failed: {exc}')

    def _material(self, stage, state):
        path = f'{ROOT_PATH}/Looks/{state}'
        if stage.GetPrimAtPath(path):
            return UsdShade.Material(stage.GetPrimAtPath(path))
        mat = UsdShade.Material.Define(stage, path)
        shader = UsdShade.Shader.Define(stage, path + '/shader')
        shader.CreateIdAttr('UsdPreviewSurface')
        rgb = Gf.Vec3f(*COLORS[state])
        shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(rgb)
        # Emissive so the colour stays readable in dark corners of the scene.
        shader.CreateInput('emissiveColor', Sdf.ValueTypeNames.Color3f).Set(rgb * 0.8)
        mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
        return mat

    def _draw(self, stage):
        UsdGeom.Xform.Define(stage, ROOT_PATH)
        keep = set()
        for goal in self.goals:
            path = f"{ROOT_PATH}/goal_{goal['id']}"
            keep.add(path)
            state = goal['state'] if goal['state'] in COLORS else 'pending'
            sphere = UsdGeom.Sphere.Define(stage, path)
            sphere.GetRadiusAttr().Set(
                self.radius * (CURRENT_SCALE if state == 'current' else 1.0))
            xform = UsdGeom.Xformable(sphere.GetPrim())
            xform.ClearXformOpOrder()
            xform.AddTranslateOp().Set(Gf.Vec3d(goal['x'], goal['y'], goal['z']))
            sphere.GetDisplayColorAttr().Set([Gf.Vec3f(*COLORS[state])])
            UsdShade.MaterialBindingAPI.Apply(sphere.GetPrim()).Bind(
                self._material(stage, state))
        root = stage.GetPrimAtPath(ROOT_PATH)
        for child in root.GetChildren():  # drop goals that no longer exist
            if child.GetName().startswith('goal_') and str(child.GetPath()) not in keep:
                stage.RemovePrim(child.GetPath())


# Colliders under these paths are not obstacles: the drone, the goal spheres and
# plain ground planes.
IGNORE_PREFIXES = ('/World/quadrotor', '/World/GoalMarkers', '/World/ground',
                   '/World/GroundPlane', '/World/defaultGroundPlane')
VEHICLE_PATH = '/World/quadrotor'
AUTO_OUT_FILE = '/Volume/NavRL/maps/scene_grid.npz'
MAX_HALF_EXTENT = 40.0   # auto export never scans more than +-40 m around the drone
AUTO_Z_RANGE = (0.0, 5.0)


def scene_bounds(stage, margin=1.0):
    """x/y/z ranges that cover every collider of the scene (None if there are none).

    The drone, goal markers and ground planes are left out. A very large scene is
    cut to +-MAX_HALF_EXTENT around the drone so the export stays quick.
    """
    from pxr import Usd, UsdPhysics
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        if str(prim.GetPath()).startswith(IGNORE_PREFIXES):
            continue
        box = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        if box.IsEmpty():
            continue
        lo = np.minimum(lo, np.array(box.GetMin()))
        hi = np.maximum(hi, np.array(box.GetMax()))
    if not np.all(np.isfinite(lo)):
        return None
    center = np.zeros(2)
    drone = stage.GetPrimAtPath(VEHICLE_PATH)
    if drone:
        t = UsdGeom.Xformable(drone).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        center = np.array(t.ExtractTranslation())[:2]
    xy_lo = np.maximum(lo[:2] - margin, center - MAX_HALF_EXTENT)
    xy_hi = np.minimum(hi[:2] + margin, center + MAX_HALF_EXTENT)
    return {'x_range': (float(xy_lo[0]), float(xy_hi[0])),
            'y_range': (float(xy_lo[1]), float(xy_hi[1])),
            'z_range': AUTO_Z_RANGE}


def export_grid(x_range=(-20.0, 20.0), y_range=(-20.0, 20.0), z_range=(0.0, 5.0),
                resolution=0.5, clearance=0.6,
                out_file='/Volume/NavRL/maps/scene_grid.npz'):
    """Saves a 3D occupancy grid of the scene (needs the simulation playing).

    Every cell is tested with a sphere of ``clearance`` radius against all
    colliders, so a free cell is a place where the drone body fits (about 0.38 m
    with propellers plus a margin). The npz holds ``occupied`` (bool [nx,ny,nz]),
    ``origin`` (lower corner, world frame) and ``resolution``. Visual-only meshes
    without a collider cannot be found, the drone's physics cannot see them either.
    """
    if not omni.timeline.get_timeline_interface().is_playing():
        print('[goal_display] export needs the simulation playing (press Play)')
        return None
    query = get_physx_scene_query_interface()
    origin = np.array([x_range[0], y_range[0], z_range[0]], dtype=float)
    upper = np.array([x_range[1], y_range[1], z_range[1]], dtype=float)
    shape = np.ceil((upper - origin) / resolution).astype(int)
    occupied = np.zeros(shape, dtype=bool)

    def blocked(center):
        pos = Float3(*map(float, center))
        if not query.overlap_sphere_any(clearance, pos):
            return False
        hits = []

        def report(hit):
            hits.append(hit.collision)
            return True

        query.overlap_sphere(clearance, pos, report, False)
        return any(not h.startswith(IGNORE_PREFIXES) for h in hits)

    t0 = time.time()
    for ix in range(shape[0]):
        for iy in range(shape[1]):
            for iz in range(shape[2]):
                center = origin + (np.array([ix, iy, iz]) + 0.5) * resolution
                occupied[ix, iy, iz] = blocked(center)
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    np.savez(out_file, occupied=occupied, origin=origin, resolution=resolution)
    print(f'[goal_display] grid {shape.tolist()}, {occupied.mean() * 100:.1f}% '
          f'occupied, {time.time() - t0:.1f}s -> {out_file}')
    return occupied


def probe(x, y, z, radius=1.0):
    """Prints every prim whose bounding box comes within ``radius`` of a point.

    Tells what a goal sits in or next to, and whether that prim has a collider:
    prims without one are missing from the exported grid.
    """
    from pxr import Usd, UsdPhysics
    stage = omni.usd.get_context().get_stage()
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    point = np.array([x, y, z], dtype=float)
    rows = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Gprim) or str(prim.GetPath()).startswith(IGNORE_PREFIXES):
            continue
        box = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        if box.IsEmpty():
            continue
        lo, hi = np.array(box.GetMin()), np.array(box.GetMax())
        gap = np.linalg.norm(np.maximum(np.maximum(lo - point, point - hi), 0.0))
        if gap <= radius:
            rows.append((gap, str(prim.GetPath()), prim.HasAPI(UsdPhysics.CollisionAPI),
                         np.round(hi - lo, 2).tolist()))
    rows.sort()
    print(f'[goal_display] probe ({x:.2f}, {y:.2f}, {z:.2f}) r={radius}: {len(rows)} prims')
    for gap, path, collider, size in rows[:15]:
        print(f'[goal_display]   {gap:.2f} m  collider={collider}  size={size}  {path}')


_display = None


def start(port=UDP_PORT, auto_export=True, **export_options):
    """Starts (or restarts) the display; called by the extension.

    With ``auto_export`` the occupancy grid is exported by itself once per loaded
    scene (see ``_auto_export_check``); ``export_options`` (resolution, clearance,
    out_file) override the defaults of ``export_grid``.
    """
    global _display
    stop()
    _display = GoalDisplay(port, auto_export, export_options)
    return _display


def stop():
    global _display
    if _display is not None:
        _display.close()
        _display = None
