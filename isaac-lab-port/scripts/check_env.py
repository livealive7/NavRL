"""Sanity checks for NavigationEnv (PhysX, headless).

    python scripts/check_env.py [--num_envs 64] [--steps 400]
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=400)
parser.add_argument("--num_static", type=int, default=350, help="static obstacles in the terrain")
parser.add_argument("--num_dynamic", type=int, default=80, help="dynamic obstacles (even; half cuboids, half cylinders)")
parser.add_argument("--no_dynamic", action="store_true")
parser.add_argument("--eval_mode", action="store_true")
add_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.physics = "isaacsim_physx"


def check(name, ok, detail=""):
    print(f"[check] {'PASS' if ok else 'FAIL'}  {name}  {detail}")
    return ok


def main():
    from isaaclab.physics import PhysicsCfg

    with launch_simulation(cfg=PhysicsCfg(), launcher_args=args_cli) as physics_cfg:
        import numpy as np
        import torch

        import isaaclab.utils.math as math_utils

        from navrl_lab.lidar import bpearl_directions
        from navrl_lab.navigation_env import OBS_DIM, NavigationEnv, NavigationEnvCfg, split_observation
        from navrl_lab.terrain import LAST_PLACED_OBSTACLES, _MeshCapturingGenerator

        cfg = NavigationEnvCfg()
        cfg.sim.physics = physics_cfg
        cfg.scene.num_envs = args_cli.num_envs
        cfg.eval_mode = args_cli.eval_mode
        cfg.num_static_obstacles = args_cli.num_static
        cfg.scene.terrain.terrain_generator.sub_terrains["obstacles"].num_obstacles = args_cli.num_static
        cfg.num_dynamic_obstacles = 0 if args_cli.no_dynamic else args_cli.num_dynamic
        cfg.sim.device = args_cli.device or "cuda:0"
        env = NavigationEnv(cfg)
        dev, N = env.device, env.num_envs
        results = []

        # ---- terrain / basic shapes
        mesh = _MeshCapturingGenerator.last_mesh
        lo, hi = mesh.bounds
        results.append(check("terrain centred, ~50x50 m, obstacles up to 6 m",
                             abs(lo[0] + 25) < 0.6 and abs(hi[0] - 25) < 0.6 and 1.0 < hi[2] <= 6.05,
                             f"bounds lo={lo.round(1)} hi={hi.round(1)}"))
        obs, _ = env.reset()
        o = obs["policy"]
        placed = LAST_PLACED_OBSTACLES
        results.append(check("static obstacle count == --num_static", len(placed) == args_cli.num_static
                             and env.cfg.scene.terrain.terrain_generator.sub_terrains["obstacles"].num_obstacles == args_cli.num_static,
                             f"placed {len(placed)} (requested {args_cli.num_static})"))
        results.append(check("obs shape/finite", o.shape == (N, OBS_DIM) and torch.isfinite(o).all().item(), f"{tuple(o.shape)}"))
        parts = split_observation(o)
        print("[info] obs parts:", {k: tuple(v.shape) for k, v in parts.items()})

        # ---- LiDAR right after reset must match an independent trimesh ray-cast at the *new* pose
        d = env._robot.data
        pos = d.root_pos_w.torch.cpu().numpy()
        yaw = math_utils.euler_xyz_from_quat(d.root_quat_w.torch)[2].cpu().numpy()
        scan = parts["lidar"].cpu().numpy()  # (N,1,36,4)
        local = bpearl_directions(cfg.lidar_hres, np.linspace(*cfg.lidar_vfov, cfg.lidar_vbeams), "cpu").numpy()
        worst, hits = 0.0, 0
        for i in range(min(N, 16)):
            c, s = np.cos(yaw[i]), np.sin(yaw[i])
            dirs = np.stack([c * local[:, 0] - s * local[:, 1], s * local[:, 0] + c * local[:, 1], local[:, 2]], 1)
            loc, ridx, _ = mesh.ray.intersects_location(np.repeat(pos[i][None], len(dirs), 0), dirs, multiple_hits=False)
            dist = np.full(len(dirs), cfg.lidar_range)
            dist[ridx] = np.minimum(np.linalg.norm(loc - pos[i], axis=1), cfg.lidar_range)
            gt = (cfg.lidar_range - dist).reshape(36, 4)
            worst = max(worst, np.abs(gt - scan[i, 0]).max())
            hits += int((dist < cfg.lidar_range).sum())
        results.append(check("LiDAR after reset == trimesh ground truth (yaw-aligned)", worst < 1e-3, f"max err {worst:.4f} m, {hits} hits"))

        # ---- start / goal geometry
        tp = env._target_pos
        results.append(check("start & goal on map edges, heights 0.5-2.5",
                             bool(((d.root_pos_w.torch[:, :2].abs().amax(1) - 24).abs() < 1e-3).all()
                                  and ((tp[:, :2].abs().amax(1) - 24).abs() < 1e-3).all()
                                  and (d.root_pos_w.torch[:, 2] >= 0.5).all() and (d.root_pos_w.torch[:, 2] <= 2.5).all())))

        # ---- hover: zero velocity command holds position
        z0 = d.root_pos_w.torch[:, 2].clone()
        for _ in range(100):
            obs, r, term, trunc, _ = env.step(torch.zeros(N, 3, device=dev))
        dz = (d.root_pos_w.torch[:, 2] - z0).abs().max().item()
        speed = d.root_lin_vel_w.torch.norm(dim=1).max().item()
        results.append(check("hover holds altitude", dz < 0.1 and speed < 0.2, f"max|dz|={dz:.3f} m max speed={speed:.3f} m/s"))


        # ---- drones must not collide with each other (all share one map): stack them on the same point
        env.reset()
        same = torch.tensor([0.0, 24.0, 2.0, 0.0, 0.0, 0.0, 1.0], device=dev).repeat(N, 1)
        env._robot.write_root_pose_to_sim_index(root_pose=same)
        env._robot.write_root_velocity_to_sim_index(root_velocity=torch.zeros(N, 6, device=dev))
        for _ in range(60):
            env._actions[:] = 0.0
            env.step(torch.zeros(N, 3, device=dev))
        sp = d.root_lin_vel_w.torch.norm(dim=1).max().item()
        spread = (d.root_pos_w.torch - d.root_pos_w.torch.mean(0)).norm(dim=1).max().item()
        results.append(check("stacked drones do not push each other", sp < 0.5 and spread < 0.5, f"max speed {sp:.3f} m/s, spread {spread:.3f} m"))

        # ---- dynamic obstacle: parking next to a 2D cylinder must register as observation + collision
        if env._dyn is not None:
            ok = False
            for j in range(env._dyn.half, env._dyn.num):
                target_xy = env._dyn.state_pos[j, :2]
                p = torch.cat([target_xy + torch.tensor([0.3, 0.0], device=dev), torch.tensor([2.0], device=dev)])
                env.reset()
                pose = torch.cat([p, torch.tensor([0, 0, 0, 1.0], device=dev)]).unsqueeze(0)
                env._robot.write_root_pose_to_sim_index(root_pose=pose, env_ids=torch.tensor([0], device=dev))
                env._robot.write_root_velocity_to_sim_index(root_velocity=torch.zeros(1, 6, device=dev), env_ids=torch.tensor([0], device=dev))
                obs, r, term, trunc, _ = env.step(torch.zeros(N, 3, device=dev))
                # the step above already reset env 0 if it collided -> inspect flags, not the (reset) observation
                static_hit = False
                if env._collision[0]:
                    ln = split_observation(obs["policy"])
                    # re-derive: was the cause dynamic?  collision flag set but LiDAR (pre-reset) unknown -> use closest()
                    c = env._dyn.closest(p.unsqueeze(0), 1, cfg.lidar_range)
                    dist2d = c["rpos"][0, 0, :2].norm().item()
                    ok = dist2d < 0.7 and term[0].item()
                    if ok:
                        break
            results.append(check("dynamic-obstacle collision terminates the episode", ok, f"cylinder #{j}"))

        # ---- fly straight at the goal (baseline policy) and collect stats
        env.reset()
        rets, ncol, nreach, neps, nan_seen = [], 0, 0, 0, False
        t0 = time.time()
        for _ in range(args_cli.steps):
            to_goal = env._target_pos - d.root_pos_w.torch
            act = 1.5 * to_goal / to_goal.norm(dim=1, keepdim=True).clamp_min(1e-6)
            obs, r, term, trunc, extras = env.step(act)
            nan_seen |= not (torch.isfinite(obs["policy"]).all() and torch.isfinite(r).all()).item()
            rets.append(r.mean().item())
            done = (term | trunc)
            neps += int(done.sum())
            ncol += int((term & env._collision).sum())
        fps = args_cli.steps * N / (time.time() - t0)
        speed_to_goal = float((d.root_lin_vel_w.torch * (to_goal / to_goal.norm(dim=1, keepdim=True))).sum(1).mean())
        results.append(check("no NaN/inf in obs/reward", not nan_seen))
        print(f"[info] baseline: mean step reward={np.mean(rets):.3f} episodes ended={neps} (collisions={ncol}) "
              f"mean speed toward goal={speed_to_goal:.2f} m/s  env-steps/s={fps:,.0f}")
        results.append(check("baseline reward is finite and drones make progress", np.isfinite(np.mean(rets)) and speed_to_goal > 0.5))

        # ---- random actions: robust to violent commands
        env.reset()
        for _ in range(200):
            obs, r, *_ = env.step(4.0 * (torch.rand(N, 3, device=dev) - 0.5))
        results.append(check("random actions stay finite", bool(torch.isfinite(obs["policy"]).all() and torch.isfinite(r).all())))

        print(f"[check] {sum(results)}/{len(results)} passed")
        env.close()


if __name__ == "__main__":
    main()
