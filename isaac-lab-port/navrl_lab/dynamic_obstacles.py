"""Moving obstacles (3D cuboids + infinite-ish 2D cylinders), simulated purely as tensors.

The original NavRL never ray-casts against them: the policy sees them through the analytic
"closest-N" observation and collisions are checked from their state, so they need no physics bodies.
All envs share the same obstacle set, exactly as in the original.
"""

from __future__ import annotations

import time

import numpy as np
import torch


class DynamicObstacles:
    N_W = 4  # width bins in (0, 1]
    N_H = 2  # 3D cuboids vs 2D cylinders
    MAX_WIDTH = 1.0
    MAX_3D_HEIGHT = 1.0
    MAX_2D_HEIGHT = 5.0

    def __init__(self, num_obstacles: int, map_range, vel_range, local_range, dt: float, device, seed: int = 0):
        self.device = device
        self.map_range = map_range
        self.vel_range = vel_range
        self.local_range = local_range
        self.dt = dt
        self.width_res = self.MAX_WIDTH / self.N_W
        categories = self.N_W * self.N_H
        self.per_category = int(num_obstacles / categories)
        self.num = self.per_category * categories  # in case of round-off
        self.half = self.num // 2  # first half: 3D cuboids, second half: 2D cylinders

        n = self.num
        self.state_pos = torch.zeros(n, 3, device=device)
        self.goal = torch.zeros(n, 3, device=device)
        self.origin = torch.zeros(n, 3, device=device)
        self.vel = torch.zeros(n, 3, device=device)
        self.size = torch.zeros(n, 3, device=device)
        self.step_count = 0
        self.vel_period = int(2.0 / dt)
        self._sample_origins_and_sizes(np.random.default_rng(seed))

    def _sample_origins_and_sizes(self, rng):
        mx, my, mz = self.map_range
        obs_dist = 2 * np.sqrt(mx * my / self.num)  # preferred spacing between obstacles
        cur_dist, prev = obs_dist, []
        cuboid_cats = self.N_W
        for cat in range(2 * cuboid_cats):
            for i in range(self.per_category):
                t0 = time.time()
                while True:
                    ox, oy = rng.uniform(-mx, mx), rng.uniform(-my, my)
                    oz = rng.uniform(0.0, mz) if cat < cuboid_cats else self.MAX_2D_HEIGHT / 2.0
                    cur = np.array([ox, oy])
                    ok = all(np.linalg.norm(cur - p) > cur_dist for p in prev)
                    if time.time() - t0 > 0.1:  # relax the spacing if it cannot be satisfied
                        cur_dist *= 0.8
                        t0 = time.time()
                    if ok:
                        prev.append(cur)
                        break
                cur_dist = obs_dist
                idx = i + cat * self.per_category
                self.origin[idx] = torch.tensor([ox, oy, oz], device=self.device)
                self.state_pos[idx] = self.origin[idx]
            sl = slice(cat * self.per_category, (cat + 1) * self.per_category)
            if cat < cuboid_cats:
                w, h = (cat + 1) * self.width_res, self.MAX_3D_HEIGHT
            else:
                w, h = 2 * ((cat - cuboid_cats + 1) * self.width_res / 2.0), self.MAX_2D_HEIGHT
            self.size[sl] = torch.tensor([w, w, h], device=self.device)

    def step(self):
        """Advance all obstacles by one env step (same 4 steps as the original ``move_dynamic_obstacle``)."""
        lr, mr = self.local_range, self.map_range
        # 1. new goals for obstacles that reached theirs
        dist = (self.state_pos - self.goal).norm(dim=1) if self.step_count != 0 else torch.zeros(self.num, device=self.device)
        need = dist < 0.5
        k = int(need.sum())
        rnd = torch.rand(k, 3, device=self.device)
        # NOTE: z offset uses local_range[1] as in the original code (asymmetric: [-ly, -ly + 2*lz])
        local = torch.stack(
            [-lr[0] + 2 * lr[0] * rnd[:, 0], -lr[1] + 2 * lr[1] * rnd[:, 1], -lr[1] + 2 * lr[2] * rnd[:, 2]], dim=1
        )
        self.goal[need] = self.origin[need] + local
        self.goal[:, 0].clamp_(-mr[0], mr[0])
        self.goal[:, 1].clamp_(-mr[1], mr[1])
        self.goal[:, 2].clamp_(0.0, mr[2])
        self.goal[self.half :, 2] = self.MAX_2D_HEIGHT / 2.0
        # 2. resample speeds roughly every 2 s
        if self.step_count % self.vel_period == 0:
            speed = self.vel_range[0] + (self.vel_range[1] - self.vel_range[0]) * torch.rand(self.num, 1, device=self.device)
            direction = self.goal - self.state_pos
            self.vel = speed * direction / direction.norm(dim=1, keepdim=True).clamp_min(1e-6)
        # 3. integrate
        self.state_pos += self.vel * self.dt
        self.step_count += 1

    def closest(self, drone_pos: torch.Tensor, k: int, lidar_range: float):
        """Closest-``k`` obstacles per drone (by 2D distance). Returns the pieces the observation/reward need."""
        n = drone_pos.shape[0]
        rpos = self.state_pos.unsqueeze(0) - drone_pos.unsqueeze(1)  # (N,M,3)
        rpos[:, self.half :, 2] = 0.0  # 2D obstacles: ignore z
        dist2d = rpos[..., :2].norm(dim=2)
        _, idx = torch.topk(dist2d, k, dim=1, largest=False)
        out_of_range = dist2d.gather(1, idx) > lidar_range
        return {
            "idx": idx,
            "out_of_range": out_of_range,
            "rpos": torch.gather(rpos, 1, idx.unsqueeze(-1).expand(-1, -1, 3)),
            "vel": self.vel[idx],
            "size": self.size[idx],
        }
