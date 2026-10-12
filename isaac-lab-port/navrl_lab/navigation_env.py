"""NavRL navigation task as an Isaac Lab ``DirectRLEnv`` (PhysX).

Port of ``isaac-training/training/scripts/env.py``. Same MDP:

* action      : world-frame velocity target (m/s, 3); the Lee controller turns it into rotor thrusts
* observation : ``state`` (8) + ``lidar`` (1x36x4) + ``direction`` (3) + ``dynamic_obstacle`` (5x10), flattened
                into ``obs["policy"]`` (see :func:`split_observation`)
* reward      : goal-velocity + 1 + log-clearance (static, dynamic) - smoothness - height penalty
* termination : collision (LiDAR < 0.3 m or dynamic obstacle) / z < 0.2 / z > 4 ; timeout 2200 steps
"""

from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.configclass import configclass

from isaaclab_contrib.controllers import LeeVelControllerCfg
from isaaclab_physx.physics import PhysxCfg

from .dynamic_obstacles import DynamicObstacles
from .frames import vec_to_new_frame
from .hummingbird import ALLOCATION_MATRIX, HUMMINGBIRD_CFG
from .lidar import TerrainLidar
from .terrain import NavTerrainGeneratorCfg, _MeshCapturingGenerator

# ---- observation layout -------------------------------------------------------------------------------------
STATE_DIM = 8
LIDAR_H, LIDAR_V = 36, 4
DIRECTION_DIM = 3
DYN_OBS_NUM, DYN_OBS_DIM = 5, 10
OBS_DIM = STATE_DIM + LIDAR_H * LIDAR_V + DIRECTION_DIM + DYN_OBS_NUM * DYN_OBS_DIM  # 205


def split_observation(obs: torch.Tensor) -> dict[str, torch.Tensor]:
    """Split the flat ``obs["policy"]`` (N, 205) into the tensors the NavRL network consumes."""
    n = obs.shape[0]
    a = STATE_DIM
    b = a + LIDAR_H * LIDAR_V
    c = b + DIRECTION_DIM
    return {
        "state": obs[:, :a],
        "lidar": obs[:, a:b].reshape(n, 1, LIDAR_H, LIDAR_V),
        "direction": obs[:, b:c],
        "dynamic_obstacle": obs[:, c:].reshape(n, 1, DYN_OBS_NUM, DYN_OBS_DIM),
    }


@configclass
class NavSceneCfg(InteractiveSceneCfg):
    """Shared static terrain (global) + one Hummingbird per env. All drones live in the same map (env_spacing=0)."""

    terrain: TerrainImporterCfg = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=NavTerrainGeneratorCfg(),
        collision_group=-1,
        debug_vis=False,
    )
    robot = HUMMINGBIRD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2000.0))


@configclass
class NavigationEnvCfg(DirectRLEnvCfg):
    # ---- env ----
    decimation = 1
    episode_length_s = 2200 * 0.016  # 2200 steps
    action_space = 3
    observation_space = OBS_DIM
    state_space = 0

    sim: SimulationCfg = SimulationCfg(
        dt=0.016,
        render_interval=1,
        physics=PhysxCfg(),
    )

    # ---- scene ----
    scene: NavSceneCfg = NavSceneCfg(num_envs=64, env_spacing=0.0, replicate_physics=True)

    # ---- LiDAR (drone.yaml `sensor`) ----
    lidar_range = 4.0
    lidar_vfov = (-10.0, 20.0)
    lidar_vbeams = 4
    lidar_hres = 10.0

    # ---- static / dynamic obstacles (train.yaml `env`, `env_dyn`) ----
    num_static_obstacles = 350  # mirrored into `terrain_generator.num_obstacles` by the scripts (--num_static)
    num_dynamic_obstacles = 80  # 0 for a static-only environment
    dyn_vel_range = (0.5, 1.5)
    dyn_local_range = (5.0, 5.0, 4.5)
    map_range = (20.0, 20.0, 4.5)

    # ---- controller: OmniDrones lee_controller_hummingbird.yaml (no gain randomisation) ----
    controller: LeeVelControllerCfg = LeeVelControllerCfg(
        K_vel_range=((2.2, 2.2, 2.2), (2.2, 2.2, 2.2)),
        K_rot_range=((0.7, 0.7, 0.035), (0.7, 0.7, 0.035)),
        K_angvel_range=((0.1, 0.1, 0.025), (0.1, 0.1, 0.025)),
        max_inclination_angle_rad=1.0471975511965976,
        max_yaw_rate=1.0471975511965976,
    )

    # ---- task ----
    eval_mode: bool = False  # True: deterministic start/goal line (original `env.eval()`)
    collision_radius = 0.3
    seed_dynamic = 0


class NavigationEnv(DirectRLEnv):
    cfg: NavigationEnvCfg

    def __init__(self, cfg: NavigationEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        n, dev = self.num_envs, self.device
        self._robot = self.scene["robot"]
        self._below = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._above = torch.zeros_like(self._below)

        # controller + thrust allocation
        ctrl_cfg = self.cfg.controller
        self._controller = ctrl_cfg.class_type(ctrl_cfg, self._robot, n, str(dev))
        self._alloc_pinv = torch.linalg.pinv(torch.tensor(ALLOCATION_MATRIX, device=dev, dtype=torch.float32))

        # LiDAR against the static terrain mesh
        assert _MeshCapturingGenerator.last_mesh is not None, "terrain mesh was not captured"
        self._lidar = TerrainLidar(
            _MeshCapturingGenerator.last_mesh, str(dev), cfg.lidar_range, cfg.lidar_vfov, cfg.lidar_vbeams, cfg.lidar_hres
        )
        assert (self._lidar.hbeams, self._lidar.vbeams) == (LIDAR_H, LIDAR_V)

        # dynamic obstacles (shared by all envs)
        self._dyn = None
        if cfg.num_dynamic_obstacles > 0:
            self._dyn = DynamicObstacles(
                cfg.num_dynamic_obstacles, cfg.map_range, cfg.dyn_vel_range, cfg.dyn_local_range, self.physics_dt, dev,
                seed=cfg.seed_dynamic,
            )

        # per-env buffers
        self._target_pos = torch.zeros(n, 3, device=dev)
        self._target_dir = torch.zeros(n, 3, device=dev)  # start->goal vector, fixes the goal frame for the episode
        self._height_range = torch.zeros(n, 2, device=dev)
        self._prev_vel = torch.zeros(n, 3, device=dev)
        self._actions = torch.zeros(n, 3, device=dev)
        self._ep_return = torch.zeros(n, device=dev)
        self._reward = torch.zeros(n, device=dev)
        self._collision = torch.zeros(n, dtype=torch.bool, device=dev)
        self._static_collision = torch.zeros_like(self._collision)
        self._dynamic_collision = torch.zeros_like(self._collision)
        self._reach_goal = torch.zeros(n, dtype=torch.bool, device=dev)
        self._obs_cache: torch.Tensor | None = None
        self._obs_stale = True

    # ------------------------------------------------------------------ stepping
    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone()

    def _apply_action(self):
        yaw = math_utils.euler_xyz_from_quat(self._robot.data.root_quat_w.torch)[2]
        yaw_q = math_utils.quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)
        cmd = torch.zeros(self.num_envs, 4, device=self.device)
        cmd[:, :3] = math_utils.quat_apply_inverse(yaw_q, self._actions)  # world vel -> yaw-aligned body frame
        wrench = self._controller.compute(cmd)
        thrust = torch.nan_to_num(wrench @ self._alloc_pinv.T, 0.0).clamp(min=0.0)
        self._robot.set_thrust_target(thrust)

    # ------------------------------------------------------------------ MDP terms
    def _compute_state(self, bookkeeping: bool = True):
        """LiDAR, observation, reward and termination flags from the current drone / obstacle state.

        With ``bookkeeping=False`` only the observation cache is refreshed (used after resets).
        """
        d = self._robot.data
        pos = d.root_pos_w.torch
        vel = d.root_lin_vel_w.torch
        yaw = math_utils.euler_xyz_from_quat(d.root_quat_w.torch)[2]
        cfg, R = self.cfg, self.cfg.lidar_range
        n = self.num_envs

        # -- LiDAR (range - distance), yaw-aligned
        lidar_scan = self._lidar.scan(pos, yaw)  # (N,1,H,V)

        # -- internal state in the goal frame
        rpos = self._target_pos - pos
        distance = rpos.norm(dim=-1, keepdim=True)
        distance_2d = rpos[:, :2].norm(dim=-1, keepdim=True)
        distance_z = rpos[:, 2:3]
        target_dir_2d = self._target_dir.clone()
        target_dir_2d[:, 2] = 0.0
        rpos_unit_g = vec_to_new_frame(rpos / distance.clamp_min(1e-6), target_dir_2d)
        vel_g = vec_to_new_frame(vel, target_dir_2d)
        state = torch.cat([rpos_unit_g, distance_2d, distance_z, vel_g], dim=-1)  # (N,8)

        # -- dynamic obstacles
        if self._dyn is not None:
            c = self._dyn.closest(pos, DYN_OBS_NUM, R)
            oor = c["out_of_range"]
            rpos_o = c["rpos"]  # (N,K,3)
            rpos_o_g = vec_to_new_frame(rpos_o, target_dir_2d)
            rpos_o_g[oor] = 0.0
            dist_o = rpos_o.norm(dim=-1, keepdim=True)
            dist_o_2d = rpos_o_g[..., :2].norm(dim=-1, keepdim=True)
            dist_o_z = rpos_o_g[..., 2:3]
            rpos_o_gn = rpos_o_g / dist_o.clamp_min(1e-6)
            vel_o_g = vec_to_new_frame(c["vel"], target_dir_2d)
            vel_o_g[oor] = 0.0
            width = c["size"][..., 0:1]
            width_cat = width / self._dyn.width_res - 1.0
            width_cat[oor] = 0.0
            height = c["size"][..., 2:3]
            height_cat = torch.where(height > self._dyn.MAX_3D_HEIGHT, torch.zeros_like(height), height)
            height_cat[oor] = 0.0
            dyn_obs = torch.cat([rpos_o_gn, dist_o_2d, dist_o_z, vel_o_g, width_cat, height_cat], dim=-1)  # (N,K,10)

            d2 = rpos_o[..., :2].norm(dim=-1, keepdim=True)
            d2[oor] = float("inf")
            dz = rpos_o[..., 2:3].abs()
            dz[oor] = float("inf")
            m = cfg.collision_radius
            dynamic_collision = ((d2 <= width / 2.0 + m) & (dz <= height / 2.0 + m)).any(dim=1).squeeze(-1)
            dist_reward = rpos_o.norm(dim=-1) - c["size"][..., 0] / 2.0
            dist_reward[oor] = R
            reward_safety_dynamic = torch.log(dist_reward.clamp(min=1e-6, max=R)).mean(dim=-1)
        else:
            dyn_obs = torch.zeros(n, DYN_OBS_NUM, DYN_OBS_DIM, device=self.device)
            dynamic_collision = torch.zeros(n, dtype=torch.bool, device=self.device)
            reward_safety_dynamic = torch.zeros(n, device=self.device)

        # -- observation (flat)
        self._obs_cache = torch.cat(
            [state, lidar_scan.reshape(n, -1), target_dir_2d, dyn_obs.reshape(n, -1)], dim=-1
        )
        self._obs_stale = False

        # -- reward
        if not bookkeeping:
            return
        reward_safety_static = torch.log((R - lidar_scan).clamp(min=1e-6, max=R)).mean(dim=(1, 2, 3))
        vel_dir = rpos / distance.clamp_min(1e-6)
        reward_vel = (vel * vel_dir).sum(-1)
        penalty_smooth = (vel - self._prev_vel).norm(dim=-1)
        z = pos[:, 2]
        hi, lo = self._height_range[:, 1], self._height_range[:, 0]
        penalty_height = torch.zeros(n, device=self.device)
        above, below = z > hi + 0.2, z < lo - 0.2
        penalty_height[above] = ((z - hi - 0.2) ** 2)[above]
        penalty_height[below] = ((lo - 0.2 - z) ** 2)[below]
        self._reward = (
            reward_vel + 1.0 + reward_safety_static + reward_safety_dynamic - penalty_smooth * 0.1 - penalty_height * 8.0
        )

        # -- termination flags
        static_collision = lidar_scan.amax(dim=(1, 2, 3)) > (R - cfg.collision_radius)
        self._static_collision, self._dynamic_collision = static_collision, dynamic_collision
        self._collision = static_collision | dynamic_collision
        self._reach_goal = distance.squeeze(-1) < 0.5
        self._below = z < 0.2
        self._above = z > 4.0
        self._prev_vel = vel.clone()
        self._ep_return += self._reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._dyn is not None:
            self._dyn.step()
        self._compute_state(bookkeeping=True)
        terminated = self._below | self._above | self._collision
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, time_out

    def _get_rewards(self) -> torch.Tensor:
        return self._reward

    def _get_observations(self) -> dict:
        if self._obs_stale:  # some drones were reset after the cached observation was computed
            self._compute_state(bookkeeping=False)
        return {"policy": self._obs_cache}

    # ------------------------------------------------------------------ reset
    def set_start_goal(self, env_ids, pos: torch.Tensor, target: torch.Tensor):
        """Teleport drones to ``pos`` (facing the goal, at rest) and set their goals. Also used by scripts/fly.py."""
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        n = len(env_ids)
        self._target_pos[env_ids] = target
        self._target_dir[env_ids] = target - pos

        diff = target - pos
        yaw = torch.atan2(diff[:, 1], diff[:, 0])
        zeros = torch.zeros_like(yaw)
        quat = math_utils.quat_from_euler_xyz(zeros, zeros, yaw)  # (x, y, z, w)
        self._robot.write_root_pose_to_sim_index(root_pose=torch.cat([pos, quat], dim=-1), env_ids=env_ids)
        self._robot.write_root_velocity_to_sim_index(root_velocity=torch.zeros(n, 6, device=self.device), env_ids=env_ids)

        self._prev_vel[env_ids] = 0.0
        self._ep_return[env_ids] = 0.0
        self._actions[env_ids] = 0.0
        self._height_range[env_ids, 0] = torch.minimum(pos[:, 2], target[:, 2])
        self._height_range[env_ids, 1] = torch.maximum(pos[:, 2], target[:, 2])
        self._controller.reset_idx(env_ids)
        self._obs_stale = True

    def set_goal(self, env_ids, target: torch.Tensor):
        """Give drones a new goal from their current position (the goal frame is re-anchored at the current position)."""
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        cur = self._robot.data.root_pos_w.torch[env_ids]
        self._target_pos[env_ids] = target
        self._target_dir[env_ids] = target - cur
        self._height_range[env_ids, 0] = torch.minimum(cur[:, 2], target[:, 2])
        self._height_range[env_ids, 1] = torch.maximum(cur[:, 2], target[:, 2])
        self._obs_stale = True

    def _sample_edge_positions(self, n: int) -> torch.Tensor:
        """Random point on one of the 4 map edges (x or y at +-24 m), height 0.5-2.5 m."""
        masks = torch.tensor([[1, 0, 1], [1, 0, 1], [0, 1, 1], [0, 1, 1]], dtype=torch.float, device=self.device)
        shifts = torch.tensor(
            [[0, 24, 0], [0, -24, 0], [24, 0, 0], [-24, 0, 0]], dtype=torch.float, device=self.device
        )
        side = torch.randint(0, 4, (n,), device=self.device)
        p = 48.0 * torch.rand(n, 3, device=self.device) - 24.0
        p[:, 2] = 0.5 + torch.rand(n, device=self.device) * 2.0
        return p * masks[side] + shifts[side]

    def _reset_idx(self, env_ids):
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        # episode statistics (logged as the mean over the envs that just finished)
        if len(env_ids) > 0 and self.episode_length_buf[env_ids].max() > 0:
            self.extras["log"] = {
                "return": self._ep_return[env_ids].mean().item(),
                "episode_len": self.episode_length_buf[env_ids].float().mean().item(),
                "reach_goal": self._reach_goal[env_ids].float().mean().item(),
                "collision": self._collision[env_ids].float().mean().item(),
            }
        super()._reset_idx(env_ids)
        n = len(env_ids)

        if self.cfg.eval_mode:
            frac = (env_ids.float() / self.num_envs - 0.5) * 32.0
            pos = torch.stack([frac, torch.full_like(frac, 24.0), torch.full_like(frac, 2.0)], dim=-1)
            tgt_x = torch.linspace(-0.5, 0.5, self.num_envs, device=self.device)[env_ids] * 32.0
            target = torch.stack([tgt_x, torch.full_like(tgt_x, -24.0), torch.full_like(tgt_x, 2.0)], dim=-1)
        else:
            target = self._sample_edge_positions(n)
            pos = self._sample_edge_positions(n)
        self.set_start_goal(env_ids, pos, target)
