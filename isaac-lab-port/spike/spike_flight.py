"""Spike B: Multirotor + LeeVelController velocity tracking (PhysX), several envs in parallel.

Checks: hover holds altitude, then body-frame velocity steps are tracked. Uses ARL Robot 1 as a stand-in
for the Hummingbird (custom USD + thruster params come later).

    python spike_flight.py
"""

import argparse

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=16)
add_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.physics = "isaacsim_physx"

DT = 0.016


def main():
    from isaaclab.physics import PhysicsCfg

    with launch_simulation(cfg=PhysicsCfg(), launcher_args=args_cli) as physics_cfg:
        import torch

        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg
        from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
        from isaaclab.utils.configclass import configclass

        from isaaclab_contrib.assets import MultirotorCfg
        from isaaclab_contrib.controllers import LeeVelControllerCfg

        from isaaclab_assets.robots.arl_robot_1 import ARL_ROBOT_1_CFG

        robot_cfg: MultirotorCfg = ARL_ROBOT_1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        robot_cfg.actuators["thrusters"].dt = DT
        robot_cfg.init_state.pos = (0.0, 0.0, 2.0)

        @configclass
        class SceneCfg(InteractiveSceneCfg):
            ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
            robot: MultirotorCfg = robot_cfg

        sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=DT, device=args_cli.device, physics=physics_cfg))
        sim_utils.DistantLightCfg(intensity=2000).func("/World/light", sim_utils.DistantLightCfg(intensity=2000))
        scene = InteractiveScene(SceneCfg(num_envs=args_cli.num_envs, env_spacing=4.0))
        sim.reset()
        robot = scene["robot"]
        N, dev = args_cli.num_envs, str(sim.device)

        ctrl_cfg = LeeVelControllerCfg(
            K_vel_range=((2.5, 2.5, 1.5), (3.5, 3.5, 2.0)),
            K_rot_range=((1.6, 1.6, 0.25), (1.85, 1.85, 0.4)),
            K_angvel_range=((0.4, 0.4, 0.075), (0.5, 0.5, 0.09)),
            max_inclination_angle_rad=1.0471975511965976,
            max_yaw_rate=1.0471975511965976,
        )
        ctrl = ctrl_cfg.class_type(ctrl_cfg, robot, N, dev)
        alloc_pinv = torch.linalg.pinv(torch.tensor(robot_cfg.allocation_matrix, device=dev, dtype=torch.float32))

        import isaaclab.utils.math as math_utils

        def yaw_of():
            return math_utils.euler_xyz_from_quat(robot.data.root_quat_w.torch)[2]

        def run(cmd_vel, seconds):
            """Run and return (vel in yaw frame, pos, yaw) averaged over the last second."""
            cmd = torch.zeros(N, 4, device=dev)
            cmd[:, :3] = torch.tensor(cmd_vel, device=dev)
            steps, tail = int(seconds / DT), int(1.0 / DT)
            vs = []
            for i in range(steps):
                wrench = ctrl.compute(cmd)
                robot.set_thrust_target((wrench @ alloc_pinv.T).clamp(min=0.0))
                robot.write_data_to_sim()
                sim.step()
                scene.update(DT)
                if i >= steps - tail:
                    yaw = yaw_of()
                    q = math_utils.quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), -yaw)
                    vs.append(math_utils.quat_apply(q, robot.data.root_lin_vel_w.torch))
            return torch.stack(vs).mean(0), robot.data.root_pos_w.torch - scene.env_origins, yaw_of()

        # hover: let the initial thrust transient settle, then require a stationary, stable altitude
        v, p, yaw = run((0.0, 0.0, 0.0), 4.0)
        z_mean = p[:, 2].mean().item()
        v2, p2, _ = run((0.0, 0.0, 0.0), 2.0)
        drift = (p2[:, 2] - p[:, 2]).abs().max().item()
        print(f"[spike] hover: z={z_mean:.2f} m (spread {p[:, 2].std().item():.3f}), drift over 2s={drift:.3f} m, "
              f"max |v|={v2.abs().max().item():.3f} m/s, yaw range=[{yaw.min().item():.2f},{yaw.max().item():.2f}] rad")
        hover_ok = drift < 0.05 and v2.abs().max().item() < 0.1

        results = []
        for name, cmd in [("+x 1.0", (1.0, 0.0, 0.0)), ("+y 1.0", (0.0, 1.0, 0.0)), ("+z 0.5", (0.0, 0.0, 0.5)),
                          ("-x 1.0", (-1.0, 0.0, 0.0)), ("mix", (1.0, -1.0, 0.0))]:
            v, p, yaw = run(cmd, 3.0)
            err = (v - torch.tensor(cmd, device=dev)).abs().max().item()
            print(f"[spike] step {name:7s}: v(yaw frame) mean={v.mean(0).cpu().numpy().round(3)} "
                  f"max err over envs={err:.3f} m/s  yaw mean={yaw.mean().item():.2f} rad")
            results.append(err < 0.25)
        print("[spike] FLIGHT PASS" if hover_ok and all(results) else "[spike] FLIGHT FAIL")


if __name__ == "__main__":
    main()
