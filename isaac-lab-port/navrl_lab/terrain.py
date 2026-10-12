"""Static obstacle terrain used by NavRL.

Re-implements the forked ``obstacle_height_mode="range"`` of ``HfDiscreteObstaclesTerrainCfg`` from the
old Orbit (including the rejection rule that forbids narrow 0.2-1.0 m gaps between obstacles), which
stock Isaac Lab does not provide.
"""

from __future__ import annotations

import numpy as np
import trimesh

from isaaclab.terrains import TerrainGenerator, TerrainGeneratorCfg
from isaaclab.terrains.height_field.hf_terrains_cfg import HfTerrainBaseCfg
from isaaclab.terrains.height_field.utils import height_field_to_mesh
from isaaclab.utils.configclass import configclass


LAST_PLACED_OBSTACLES: list = []
"""(x, y, w, l, height) in pixels of every obstacle placed by the most recent terrain build (used by check_env.py)."""


@height_field_to_mesh
def obstacles_range_terrain(difficulty: float, cfg: "HfObstaclesRangeTerrainCfg") -> np.ndarray:
    """Pillars whose height is drawn from probability-weighted ranges. Returns a height field in vertical steps."""
    rng = np.random.default_rng(cfg.seed)
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    wmin = int(cfg.obstacle_width_range[0] / cfg.horizontal_scale)
    wmax = int(cfg.obstacle_width_range[1] / cfg.horizontal_scale)
    width_choices = np.arange(wmin, wmax, 4)
    x_choices = np.arange(0, width_pixels, 4)
    y_choices = np.arange(0, length_pixels, 4)
    edges = cfg.obstacle_height_range
    probs = np.asarray(cfg.obstacle_height_probability, dtype=np.float64)
    assert len(edges) == len(probs) + 1, "need one more height edge than probabilities"

    LAST_PLACED_OBSTACLES.clear()
    hf = np.zeros((width_pixels, length_pixels))
    history = np.zeros((0, 4))  # x, y, width, length in pixels
    bad_lo, bad_hi = cfg.gap_forbidden_range_pixels

    def gap_ok(x, y, w, l):
        if len(history) == 0:
            return True
        dx = np.abs(history[:, 0] - x) - w
        dy = np.abs(history[:, 1] - y) - l
        overlapping = (dx < 0) & (dy < 0)
        dist = np.sqrt(dx**2 + dy**2)
        bad = (~overlapping) & (dist >= bad_lo) & (dist <= bad_hi)
        return not bad.any()

    for _ in range(cfg.num_obstacles):
        n = rng.choice(len(probs), p=probs)
        height = rng.uniform(edges[n] / cfg.vertical_scale, edges[n + 1] / cfg.vertical_scale)
        for _attempt in range(100000):
            w, l = int(rng.choice(width_choices)), int(rng.choice(width_choices))
            x0, y0 = int(rng.choice(x_choices)), int(rng.choice(y_choices))
            x0 = min(x0, width_pixels - w)
            y0 = min(y0, length_pixels - l)
            if gap_ok(x0, y0, w, l):
                break
        history = np.vstack([history, [x0, y0, w, l]])
        hf[x0 : x0 + w, y0 : y0 + l] = height
        LAST_PLACED_OBSTACLES.append((x0, y0, w, l, height))

    # clear the (optional) start platform in the centre
    pw = int(cfg.platform_width / cfg.horizontal_scale)
    if pw > 0:
        a, b = (width_pixels - pw) // 2, (width_pixels + pw) // 2
        c, d = (length_pixels - pw) // 2, (length_pixels + pw) // 2
        hf[a:b, c:d] = 0
    return np.rint(hf).astype(np.int16)


@configclass
class HfObstaclesRangeTerrainCfg(HfTerrainBaseCfg):
    """Sub-terrain config for :func:`obstacles_range_terrain`."""

    function = obstacles_range_terrain

    obstacle_width_range: tuple = (0.4, 1.1)
    """Min/max obstacle footprint edge [m]."""
    obstacle_height_range: list = (1.0, 1.5, 2.0, 4.0, 6.0)
    """Height bin edges [m]; bin ``i`` is ``[edges[i], edges[i+1]]``."""
    obstacle_height_probability: list = (0.1, 0.15, 0.20, 0.55)
    """Probability of each height bin."""
    num_obstacles: int = 350
    platform_width: float = 0.0
    gap_forbidden_range_pixels: tuple = (2, 10)
    """Reject obstacles whose gap to any earlier obstacle is in this range [pixels] (0.2-1.0 m at 0.1 m/px)."""
    seed: int = 0


class _MeshCapturingGenerator(TerrainGenerator):
    """TerrainGenerator that remembers the final (centred) mesh so the LiDAR can ray-cast against it."""

    last_mesh: trimesh.Trimesh | None = None

    def __init__(self, cfg, device="cpu"):
        super().__init__(cfg, device)
        _MeshCapturingGenerator.last_mesh = self.terrain_mesh


@configclass
class NavTerrainGeneratorCfg(TerrainGeneratorCfg):
    """40x40 m obstacle field inside a 5 m flat border (50x50 m total), as in NavRL."""

    class_type: type = _MeshCapturingGenerator
    seed: int = 0
    size: tuple = (40.0, 40.0)
    border_width: float = 5.0
    num_rows: int = 1
    num_cols: int = 1
    horizontal_scale: float = 0.1
    vertical_scale: float = 0.1
    slope_threshold: float = 0.75
    use_cache: bool = False
    color_scheme: str = "height"
    sub_terrains: dict = {"obstacles": HfObstaclesRangeTerrainCfg()}
