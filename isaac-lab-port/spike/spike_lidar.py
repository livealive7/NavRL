"""Spike A: NavRL-style 4-beam LiDAR (RayCaster) vs. trimesh ground truth on an obstacle terrain.

Uses InteractiveScene so the layout matches training: terrain is global (/World/ground),
the sensor carrier is cloned per env.

    python spike_lidar.py --physics isaacsim_physx
    python spike_lidar.py --physics newton_mjwarp
"""

import argparse

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser()
parser.add_argument("--physics", default="isaacsim_physx")
parser.add_argument("--samples", type=int, default=64)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--debug", action="store_true")
add_launcher_args(parser)
args_cli = parser.parse_args()

LIDAR_RANGE, VFOV, VBEAMS, HRES = 4.0, (-10.0, 20.0), 4, 10.0
MAP = 20.0  # half extent, as in NavRL


def terrain_gen_cfg():
    from isaaclab.terrains import TerrainGeneratorCfg
    from isaaclab.terrains.height_field import HfDiscreteObstaclesTerrainCfg

    return TerrainGeneratorCfg(
        seed=0, size=(MAP * 2, MAP * 2), border_width=5.0, num_rows=1, num_cols=1,
        horizontal_scale=0.1, vertical_scale=0.1, slope_threshold=0.75, use_cache=False,
        sub_terrains={
            "obstacles": HfDiscreteObstaclesTerrainCfg(
                horizontal_scale=0.1, vertical_scale=0.1, border_width=0.0, num_obstacles=60,
                obstacle_height_mode="fixed", obstacle_width_range=(0.4, 1.1),
                obstacle_height_range=(4.0, 4.0), platform_width=0.0,
            )
        },
    )


def main():
    from isaaclab.physics import PhysicsCfg

    with launch_simulation(cfg=PhysicsCfg(), launcher_args=args_cli) as physics_cfg:
        # isaaclab modules must be imported only after Kit is up (they pull in pxr)
        import numpy as np
        import torch

        import isaaclab.sim as sim_utils
        from isaaclab.assets import RigidObjectCfg
        from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
        from isaaclab.sensors.ray_caster import RayCasterCfg, patterns
        from isaaclab.terrains import TerrainImporterCfg
        from isaaclab.utils.configclass import configclass

        pattern = patterns.BpearlPatternCfg(horizontal_res=HRES, vertical_ray_angles=torch.linspace(*VFOV, VBEAMS))

        @configclass
        class SceneCfg(InteractiveSceneCfg):
            terrain = TerrainImporterCfg(
                prim_path="/World/ground", terrain_type="generator", terrain_generator=terrain_gen_cfg(),
                collision_group=-1)
            carrier = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Carrier",
                spawn=sim_utils.CuboidCfg(
                    size=(0.1, 0.1, 0.1),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True, kinematic_enabled=True),
                    mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
                    collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
                ),
            )
            lidar = RayCasterCfg(
                prim_path="{ENV_REGEX_NS}/Carrier", mesh_prim_paths=["/World/ground"], ray_alignment="yaw",
                pattern_cfg=pattern, max_distance=LIDAR_RANGE, debug_vis=False,
                global_world_only=True)  # Newton only: ray-cast the global world (terrain) only

        sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.016, device=args_cli.device, physics=physics_cfg))
        sim_utils.DistantLightCfg(intensity=2000).func("/World/light", sim_utils.DistantLightCfg(intensity=2000))

        # ground-truth mesh: same seeds -> identical terrain to the one the scene builds
        np.random.seed(0)
        gt_mesh = terrain_gen_cfg().class_type(cfg=terrain_gen_cfg(), device=sim.device).terrain_mesh
        np.random.seed(0)
        scene = InteractiveScene(SceneCfg(num_envs=args_cli.num_envs, env_spacing=0.0))
        sim.reset()
        carrier, lidar = scene["carrier"], scene["lidar"]
        print(f"[spike] sensor class: {type(lidar).__name__}")

        _, local_dirs = patterns.bpearl_pattern(pattern, device=sim.device)
        local_dirs = local_dirs.cpu().numpy()
        n_rays = local_dirs.shape[0]
        print(f"[spike] rays per sensor: {n_rays} (expect {int(360 / HRES) * VBEAMS})")

        rng = np.random.default_rng(1)
        n_bad, n_hit, n_total, max_err = 0, 0, 0, 0.0
        for it in range(args_cli.samples // args_cli.num_envs):
            pos = np.stack([rng.uniform(-MAP + 2, MAP - 2, args_cli.num_envs),
                            rng.uniform(-MAP + 2, MAP - 2, args_cli.num_envs),
                            rng.uniform(1.0, 3.0, args_cli.num_envs)], axis=1)
            pose = torch.tensor(np.concatenate([pos, np.tile([0, 0, 0, 1.0], (args_cli.num_envs, 1))], 1),
                                dtype=torch.float32, device=sim.device)
            carrier.write_root_pose_to_sim_index(root_pose=pose)
            sim.step()
            scene.update(sim.get_physics_dt())

            hits_all = lidar.data.ray_hits_w.torch.cpu().numpy()
            for e in range(args_cli.num_envs):
                d_sim = np.linalg.norm(hits_all[e] - pos[e], axis=1)
                d_sim[~np.isfinite(d_sim) | (d_sim > LIDAR_RANGE)] = LIDAR_RANGE
                origins = np.repeat(pos[e][None], n_rays, 0)
                loc, ray_idx, _ = gt_mesh.ray.intersects_location(origins, local_dirs, multiple_hits=False)
                d_gt = np.full(n_rays, LIDAR_RANGE)
                d_gt[ray_idx] = np.minimum(np.linalg.norm(loc - pos[e], axis=1), LIDAR_RANGE)
                if args_cli.debug and it == 0 and e == 0:
                    print(f"[debug] pos={pos[e].round(2)} sensor pos_w={lidar.data.pos_w.torch[e].cpu().numpy().round(2)}")
                    print(f"[debug] d_sim min/mean/max={d_sim.min():.2f}/{d_sim.mean():.2f}/{d_sim.max():.2f} "
                          f"d_gt min/mean/max={d_gt.min():.2f}/{d_gt.mean():.2f}/{d_gt.max():.2f}")
                err = np.abs(d_sim - d_gt)
                max_err = max(max_err, err.max())
                n_bad += int((err > 0.15).sum())
                n_hit += int((d_gt < LIDAR_RANGE).sum())
                n_total += err.size

        print(f"[spike] physics={args_cli.physics} rays={n_total} obstacle_hits(gt)={n_hit} "
              f"mismatch(>0.15m)={n_bad} max_err={max_err:.3f}m")
        print("[spike] LIDAR PASS" if n_bad <= 0.01 * n_total and n_hit > 0 else "[spike] LIDAR FAIL")


if __name__ == "__main__":
    main()
