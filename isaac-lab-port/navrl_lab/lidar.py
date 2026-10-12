"""Yaw-aligned LiDAR ray-cast against the static terrain mesh.

Isaac Lab 3.0's ``RayCaster(ray_alignment="yaw")`` keeps ray *directions* fixed in the world frame, whereas
NavRL's LiDAR frame rotates with the drone heading (Orbit's ``attach_yaw_only``) -- like a real body-mounted
LiDAR. So the yaw rotation is done here and only the mesh query is delegated to Isaac Lab (``raycast_mesh``).
"""

from __future__ import annotations

import math

import torch
import trimesh
import warp as wp

from isaaclab.utils.warp import convert_to_warp_mesh, raycast_mesh


def bpearl_directions(hres_deg: float, vertical_angles_deg, device) -> torch.Tensor:
    """Ray directions (H*V, 3), index = h * V + v. Same formula as Orbit/Isaac Lab ``bpearl_pattern``."""
    h = torch.arange(-180.0, 180.0, hres_deg, device=device)
    v = torch.as_tensor(list(vertical_angles_deg), dtype=torch.float32, device=device)
    pitch, yaw = torch.meshgrid(v, h, indexing="xy")
    pitch, yaw = torch.deg2rad(pitch.reshape(-1)), torch.deg2rad(yaw.reshape(-1))
    pitch = pitch + math.pi / 2
    x = torch.sin(pitch) * torch.cos(yaw)
    y = torch.sin(pitch) * torch.sin(yaw)
    z = torch.cos(pitch)
    return -torch.stack([x, y, z], dim=1)


class TerrainLidar:
    def __init__(self, terrain_mesh: trimesh.Trimesh, device, lidar_range: float, vfov, vbeams: int, hres: float):
        self.device = device
        self.range = lidar_range
        self.hbeams = int(360 / hres)
        self.vbeams = vbeams
        self.local_dirs = bpearl_directions(hres, torch.linspace(*vfov, vbeams), device)  # (R,3)
        self.num_rays = self.local_dirs.shape[0]
        self.mesh = convert_to_warp_mesh(
            terrain_mesh.vertices.astype("float32"), terrain_mesh.faces.astype("int32"), device=device
        )

    def scan(self, pos_w: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
        """Returns ``range - distance`` shaped (N, 1, hbeams, vbeams), in the yaw-aligned LiDAR frame."""
        n = pos_w.shape[0]
        c, s = torch.cos(yaw), torch.sin(yaw)
        d = self.local_dirs.unsqueeze(0).expand(n, -1, -1)
        dx = c.unsqueeze(1) * d[..., 0] - s.unsqueeze(1) * d[..., 1]
        dy = s.unsqueeze(1) * d[..., 0] + c.unsqueeze(1) * d[..., 1]
        dirs = torch.stack([dx, dy, d[..., 2]], dim=-1).reshape(-1, 3).contiguous()
        starts = pos_w.unsqueeze(1).expand(-1, self.num_rays, -1).reshape(-1, 3).contiguous()
        hits = raycast_mesh(starts, dirs, self.mesh, max_dist=self.range)[0]
        dist = (hits - starts).norm(dim=-1).clamp_max(self.range)  # inf (miss) -> range
        return (self.range - dist).reshape(n, 1, self.hbeams, self.vbeams)
