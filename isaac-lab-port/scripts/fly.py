"""Fly one drone with a trained policy (inference only) from a start point through one or more goals.

    # window (needs a desktop session):
    python scripts/fly.py --official --visualizer kit --realtime --num_static 350 --num_dynamic 80
    # headless, writes a top-view PNG of the flight:
    python scripts/fly.py --official --plot /tmp/flight.png
    # custom route (x,y,z; map is +-24 m, heights 0.5-2.5 m):
    python scripts/fly.py --checkpoint logs/navrl/<run>/checkpoint_2000.pt --start -20,22,2 --goals 0,0,2 18,-22,2

Dynamic obstacles are drawn as markers (cuboids / cylinders) when a visualizer is on or --plot is given.
"""

import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--official", action="store_true")
parser.add_argument("--start", type=str, default="-18,22,2")
parser.add_argument("--goals", type=str, nargs="+", default=["18,-22,2"], help="one or more x,y,z waypoints, flown in order")
parser.add_argument("--num_static", type=int, default=350)
parser.add_argument("--num_dynamic", type=int, default=80)
parser.add_argument("--no_dynamic", action="store_true")
parser.add_argument("--max_steps", type=int, default=2190, help="env steps (0.016 s each); episode limit is 2199")
parser.add_argument("--stochastic", action="store_true", help="sample actions instead of the Beta mean")
parser.add_argument("--realtime", action="store_true", help="sleep so that 1 sim second ~ 1 wall second")
parser.add_argument("--plot", type=str, default=None, help="save a top-view + altitude PNG of the flight")
add_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.physics = "isaacsim_physx"
assert args_cli.checkpoint or args_cli.official, "give --checkpoint or --official"


def xyz(s):
    v = [float(t) for t in s.split(",")]
    assert len(v) == 3, f"expected x,y,z, got {s}"
    return v


def main():
    from isaaclab.physics import PhysicsCfg

    with launch_simulation(cfg=PhysicsCfg(), launcher_args=args_cli) as physics_cfg:
        import torch

        from navrl_lab.navigation_env import NavigationEnv, NavigationEnvCfg, split_observation
        from navrl_lab.ppo import PPO
        from navrl_lab.terrain import _MeshCapturingGenerator

        cfg = NavigationEnvCfg()
        cfg.sim.physics = physics_cfg
        cfg.sim.device = args_cli.device or "cuda:0"
        cfg.scene.num_envs = 1
        cfg.eval_mode = False
        cfg.num_static_obstacles = args_cli.num_static
        cfg.scene.terrain.terrain_generator.sub_terrains["obstacles"].num_obstacles = args_cli.num_static
        cfg.num_dynamic_obstacles = 0 if args_cli.no_dynamic else args_cli.num_dynamic
        env = NavigationEnv(cfg)
        dev = env.device
        ckpt = os.path.join(os.path.dirname(HERE), "quick-demos", "ckpts", "navrl_checkpoint.pt") if args_cli.official else args_cli.checkpoint
        policy = PPO(device=dev)
        policy.load_state_dict(torch.load(ckpt, map_location=dev))
        print(f"[fly] checkpoint: {ckpt}")

        start = torch.tensor([xyz(args_cli.start)], device=dev)
        goals = [torch.tensor([xyz(g)], device=dev) for g in args_cli.goals]

        # ---- visual markers (best effort: a failure here must not stop the flight) ----
        markers = goal_marker = None
        try:
            import isaaclab.sim as sim_utils
            from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

            if env._dyn is not None:
                markers = VisualizationMarkers(VisualizationMarkersCfg(prim_path="/Visuals/dyn_obstacles", markers={
                    "cuboid": sim_utils.CuboidCfg(size=(1.0, 1.0, 1.0), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.9, 0.1))),
                    "cylinder": sim_utils.CylinderCfg(radius=0.5, height=1.0, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.9, 0.1))),
                }))
            goal_marker = VisualizationMarkers(VisualizationMarkersCfg(prim_path="/Visuals/goal", markers={
                "goal": sim_utils.SphereCfg(radius=0.4, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.1, 0.1))),
            }))
        except Exception as e:  # noqa: BLE001
            print(f"[fly] markers disabled: {type(e).__name__}: {e}")
        try:
            env.sim.set_camera_view(eye=(0.0, -45.0, 40.0), target=(0.0, 0.0, 0.0))
        except Exception as e:  # noqa: BLE001
            print(f"[fly] camera not set: {type(e).__name__}: {e}")

        def draw(goal):
            if goal_marker is not None:
                goal_marker.visualize(translations=goal)
            if markers is not None:
                dyn = env._dyn
                idx = torch.zeros(dyn.num, dtype=torch.int32, device=dev)
                idx[dyn.half :] = 1
                markers.visualize(translations=dyn.state_pos, scales=dyn.size, marker_indices=idx)

        # ---- place the drone and fly ----
        env.reset()
        env.set_start_goal([0], start, goals[0])
        zero = torch.zeros(1, 3, device=dev)
        obs, *_ = env.step(zero)  # one idle step so the observation is computed from the new pose
        gi, traj, t_reach, outcome = 0, [], {}, "max_steps reached"
        dt = env.physics_dt
        for step in range(args_cli.max_steps):
            t0 = time.time()
            o = split_observation(obs["policy"])
            act = policy.act(o, deterministic=not args_cli.stochastic)["action"]
            obs, r, term, trunc, extras = env.step(act)
            ended = bool((term | trunc).any())
            if not ended:  # on the final step the env has already auto-reset the drone: its pose is no longer part of the flight
                traj.append(env._robot.data.root_pos_w.torch[0].cpu().tolist())
            draw(goals[gi])
            if bool(env._reach_goal[0]) and not ended:
                t_reach[gi] = step * dt
                print(f"[fly] reached goal {gi + 1}/{len(goals)} {goals[gi][0].tolist()} at t={step * dt:.1f}s")
                if gi + 1 == len(goals):
                    outcome = "all goals reached"
                    break
                gi += 1
                env.set_goal([0], goals[gi])
            if ended:
                outcome = "collision" if bool(env._collision[0]) else ("timeout" if bool(trunc.any()) else "left height bounds")
                break
            if args_cli.realtime:
                time.sleep(max(0.0, dt - (time.time() - t0)))
        print(f"[fly] result: {outcome} after {len(traj)} steps ({len(traj) * dt:.1f}s); final pos {[round(v, 2) for v in traj[-1]]}")

        if len(traj) > 1:
            import numpy as _np

            _tr = _np.array(traj)
            _path = float(_np.linalg.norm(_np.diff(_tr, axis=0), axis=1).sum())
            _speed = _np.linalg.norm(_np.diff(_tr, axis=0), axis=1) / dt
            print(f"[fly] path length {_path:.1f} m | mean speed {_path / (len(traj) * dt):.2f} m/s | max speed {_speed.max():.2f} m/s "
                  f"| distance to last goal at end {float(_np.linalg.norm(_tr[-1] - goals[-1][0].cpu().numpy())):.2f} m")
        if args_cli.plot:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np

            tr = np.array(traj)
            fig, (a, b) = plt.subplots(1, 2, figsize=(13, 6), gridspec_kw={"width_ratios": [3, 2]})
            v = _MeshCapturingGenerator.last_mesh.vertices
            m = v[:, 2] > 0.5
            a.scatter(v[m, 0], v[m, 1], s=1, c="gray", label="static obstacles (> 0.5 m)")
            if env._dyn is not None:
                p = env._dyn.state_pos.cpu().numpy()
                a.scatter(p[:, 0], p[:, 1], s=30, c="limegreen", marker="s", label="dynamic obstacles (final)")
            sc = a.scatter(tr[:, 0], tr[:, 1], c=np.arange(len(tr)) * dt, s=4, cmap="viridis")
            a.plot(*start[0, :2].cpu().tolist(), "b^", ms=10, label="start")
            for i, g in enumerate(goals):
                a.plot(*g[0, :2].cpu().tolist(), "r*", ms=14, label="goal" if i == 0 else None)
            a.set_aspect("equal"); a.set_xlim(-26, 26); a.set_ylim(-26, 26); a.legend(loc="upper right", fontsize=7)
            a.set_title(f"top view - {outcome}"); fig.colorbar(sc, ax=a, label="t [s]", shrink=0.7)
            b.plot(np.arange(len(tr)) * dt, tr[:, 2]); b.set_xlabel("t [s]"); b.set_ylabel("z [m]"); b.set_title("altitude")
            fig.tight_layout(); fig.savefig(args_cli.plot, dpi=110)
            print(f"[fly] saved {args_cli.plot}")
        env.close()


if __name__ == "__main__":
    main()
