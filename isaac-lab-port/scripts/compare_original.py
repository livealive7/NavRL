"""Item-by-item numerical comparison against the ORIGINAL NavRL / OmniDrones code.

The original simulator (Isaac Sim 2023.1) cannot run on this machine, so we run the original *pure-PyTorch* pieces
verbatim (they are loaded from the repo files, not re-typed) on states taken from the port:

  B. Lee controller      original LeePositionController          vs  Isaac Lab LeeVelController + allocation
  C. rotor/motor model   original RotorGroup                     vs  Isaac Lab Thruster
  D. closed loop         same PhysX, same commands, three stacks (see below)
  E. observation/reward  original `_compute_state_and_obs` (exec'd from env.py)  vs  NavigationEnv

    python scripts/compare_original.py [--sections BCDE]
"""

import argparse
import ast
import importlib.util
import os
import sys
import textwrap
import types

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORIG = os.path.join(os.path.dirname(HERE), "isaac-training")
OMNI = os.path.join(ORIG, "third_party", "OmniDrones", "omni_drones")
sys.path.insert(0, HERE)

VARIANTS = ["port, damping 0.0 (before)", "port, damping 0.2 (now)", "ORIGINAL controller+rotors, damping 0.2"]
SCHEDULE = [((0, 0, 0), 1.0), ((2, 0, 0), 3.0), ((-2, 0, 0), 3.0), ((0, 0, 0.5), 1.5)]


def d_metrics(v, tilt):
    import numpy as np

    s1 = int(1.0 / 0.016)
    seg = v[s1 : s1 + int(3.0 / 0.016), 0]
    t90 = np.argmax(seg >= 1.8) * 0.016 if (seg >= 1.8).any() else float("nan")
    return {"t90 [s]": t90, "peak vx": seg.max(), "vx@end": seg[-1], "max tilt [deg]": tilt.max()}


def d_report(outdir):
    import numpy as np

    outs = {}
    for i, name in enumerate(VARIANTS):
        f = os.path.join(outdir, f"d_{i}.npz")
        if os.path.exists(f):
            z = np.load(f)
            outs[name] = (z["v"], z["tilt"])
            print(f"[cmp] {name:58s} " + " | ".join(f"{k} {val:.3f}" for k, val in d_metrics(*outs[name]).items()))
    ref = outs.get(VARIANTS[2])
    if ref is not None:
        for name in VARIANTS[:2]:
            if name in outs:
                dv = np.abs(outs[name][0] - ref[0]).max(axis=0)
                print(f"[cmp] max |v - v_orig_stack| [{name}]  {dv.max():.3f} m/s  per axis {dv.round(3)}")


if __name__ == "__main__" and "--report" in sys.argv:
    d_report(sys.argv[sys.argv.index("--outdir") + 1] if "--outdir" in sys.argv else "/tmp/navrl_cmp")
    sys.exit(0)

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser()
parser.add_argument("--sections", default="BCDE")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--variant", type=int, default=-1, help="section D: which stack to run (0,1,2); one env per process")
parser.add_argument("--outdir", default="/tmp/navrl_cmp")
parser.add_argument("--report", action="store_true", help="section D: print the table from saved variants (no simulator)")
add_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.physics = "isaacsim_physx"


def row(name, value, note=""):
    print(f"[cmp] {name:58s} {value:>14s}  {note}")


def load_by_path(name, path, extra_modules=None):
    for k, v in (extra_modules or {}).items():
        sys.modules[k] = v
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def function_source(path, name):
    src = open(path).read()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return textwrap.dedent("\n".join(src.splitlines()[node.lineno - 1 : node.end_lineno]))
    raise KeyError(name)


def main():
    from isaaclab.physics import PhysicsCfg

    with launch_simulation(cfg=PhysicsCfg(), launcher_args=args_cli) as physics_cfg:
        import numpy as np
        import torch
        import yaml

        import isaaclab.utils.math as mu

        from navrl_lab import hummingbird as hb
        from navrl_lab.navigation_env import NavigationEnv, NavigationEnvCfg, split_observation

        dev = args_cli.device or "cuda:0"

        # ---- load original pure-torch code from the repo ---------------------------------------------------
        pkg = types.ModuleType("omni_drones"); pkg.__path__ = []
        utils_pkg = types.ModuleType("omni_drones.utils"); utils_pkg.__path__ = []
        td_stub = types.ModuleType("tensordict"); td_stub.TensorDict = dict
        o_torch = load_by_path("omni_drones.utils.torch", os.path.join(OMNI, "utils", "torch.py"),
                               {"omni_drones": pkg, "omni_drones.utils": utils_pkg, "tensordict": td_stub})
        o_lee = load_by_path("omni_drones_lee", os.path.join(OMNI, "controllers", "lee_position_controller.py"))
        o_rotor = load_by_path("omni_drones_rotor", os.path.join(OMNI, "actuators", "rotor_group.py"))
        # controller yaml lives next to the original module
        o_lee.__file__ = os.path.join(OMNI, "controllers", "lee_position_controller.py")
        uav = yaml.safe_load(open(os.path.join(OMNI, "robots", "assets", "usd", "hummingbird.yaml")))

        def make_env(lin_damp, ang_damp, tau=None, n=None):
            cfg = NavigationEnvCfg()
            cfg.sim.physics = physics_cfg
            cfg.sim.device = dev
            cfg.scene.num_envs = n or args_cli.num_envs
            rp = cfg.scene.robot.spawn.rigid_props
            rp.linear_damping, rp.angular_damping = lin_damp, ang_damp
            if tau is not None:
                a = cfg.scene.robot.actuators["thrusters"]
                a.tau_inc_range = a.tau_dec_range = (tau, tau)
            return NavigationEnv(cfg)

        print("\n[cmp] ==== A. static parameters " + "=" * 40)
        rp = NavigationEnvCfg().scene.robot.spawn.rigid_props
        row("linear_damping   (port / original)", f"{rp.linear_damping} / 0.2", "<-- was 0.0 before this comparison" if rp.linear_damping == 0 else "")
        row("angular_damping  (port / original)", f"{rp.angular_damping} / 0.2")
        row("max_depenetration_velocity (port / original)", f"{rp.max_depenetration_velocity} / 10.0")
        row("mass [kg] (yaml / port constant)", f"{uav['mass']} / {hb.MASS}")
        row("max thrust per rotor [N] (KF*w^2)", f"{hb.MAX_THRUST:.3f}", f"orig {uav['rotor_configuration']['force_constants'][0]*uav['rotor_configuration']['max_rotation_velocities'][0]**2:.3f}")

        env = N = robot = d = None
        if any(c in args_cli.sections for c in "BE"):  # (only one env can be created per process)
            env = make_env(0.2, 0.2)
            N = env.num_envs
            robot = env._robot
            d = robot.data
        results = {}

        # =============================================================================== B. controller
        if "B" in args_cli.sections:
            print("\n[cmp] ==== B. Lee controller: same state, same velocity target " + "=" * 12)
            ctrl_o = o_lee.LeePositionController(9.81, uav).to(dev)
            inertia_il = env._controller.robot_inertia[0]
            row("inertia diag [orig yaml]", str([0.007, 0.007, 0.012]))
            row("inertia diag [port, aggregated about CoM]", str([round(inertia_il[i, i].item(), 5) for i in range(3)]))
            row("mass [port controller]", f"{env._controller.mass[0].item():.4f}")
            obs, _ = env.reset()
            g = torch.Generator(device=dev).manual_seed(0)
            cmd_w = torch.zeros(N, 3, device=dev)
            errs, rel = [], []
            for t in range(200):
                if t % 25 == 0:
                    cmd_w = 4.0 * (torch.rand(N, 3, device=dev, generator=g) - 0.5)
                # --- port
                yaw = mu.euler_xyz_from_quat(d.root_quat_w.torch)[2]
                yq = mu.quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)
                c4 = torch.zeros(N, 4, device=dev)
                c4[:, :3] = mu.quat_apply_inverse(yq, cmd_w)
                wrench = env._controller.compute(c4).clone()
                thr_il = wrench @ env._alloc_pinv.T
                # --- original (wxyz quaternion convention!)
                q = d.root_quat_w.torch
                q_wxyz = q[:, [3, 0, 1, 2]]
                state13 = torch.cat([d.root_pos_w.torch, q_wxyz, d.root_lin_vel_w.torch, d.root_ang_vel_w.torch], -1)
                cmd = ctrl_o(state13.unsqueeze(1), target_vel=cmd_w.unsqueeze(1)).squeeze(1)
                thr_o = (cmd + 1) / 2 * hb.MAX_THRUST
                e = (thr_il - thr_o).abs()
                errs.append(e.max().item())
                rel.append((e.mean() / thr_o.abs().mean().clamp_min(1e-6)).item())
                env.step(cmd_w)
            row("per-rotor thrust target |diff| max over run [N]", f"{max(errs):.4f}", f"(hover thrust/rotor = {hb.MASS*9.81/4:.3f} N)")
            row("mean relative difference", f"{np.mean(rel):.2%}")
            results["B"] = np.mean(rel)

        # =============================================================================== C. motor
        if "C" in args_cli.sections:
            print("\n[cmp] ==== C. rotor model: same thrust-target sequence " + "=" * 18)
            from isaaclab_contrib.actuators import Thruster
            from isaaclab_contrib.utils.types import MultiRotorActions

            tcfg = hb.HUMMINGBIRD_CFG.actuators["thrusters"]
            th = Thruster(tcfg, ["r0", "r1", "r2", "r3"], [0, 1, 2, 3], 1, dev, torch.full((1, 4), hb.HOVER_W, device=dev))
            rg = o_rotor.RotorGroup(uav["rotor_configuration"], 0.016).to(dev)
            t0 = hb.MASS * 9.81 / 4
            rg.throttle.data[:] = (torch.tensor(t0 / hb.MAX_THRUST)).sqrt()
            g = torch.Generator().manual_seed(1)
            tgt = torch.full((1, 4), t0, device=dev)
            worst, traj_il, traj_o = 0.0, [], []
            for t in range(300):
                if t % 20 == 0:
                    tgt = (torch.rand(1, 4, generator=g) * hb.MAX_THRUST).to(dev)
                a = th.compute(MultiRotorActions(thrusts=tgt.clone(), thruster_indices=[0, 1, 2, 3])).thrusts.clone()
                cmds = 2 * tgt[0] / hb.MAX_THRUST - 1
                thrusts_o, _ = rg(cmds)
                worst = max(worst, (a[0] - thrusts_o).abs().max().item())
            row("max |thrust_port - thrust_orig| over 300 steps [N]", f"{worst:.2e}")
            results["C"] = worst

        # =============================================================================== D. closed loop
        if "D" in args_cli.sections:
            print("\n[cmp] ==== D. closed-loop velocity response in PhysX (identical commands) " + "=" * 3)

            class OrigStackEnv(NavigationEnv):
                """Original Lee controller + original RotorGroup lag, applied through the same Multirotor/PhysX."""

                def __init__(self, cfg):
                    super().__init__(cfg)
                    self._ctrl_o = o_lee.LeePositionController(9.81, uav).to(self.device)
                    self._rg = [o_rotor.RotorGroup(uav["rotor_configuration"], 0.016).to(self.device) for _ in range(self.num_envs)]
                    for r in self._rg:
                        r.throttle.data[:] = (torch.tensor(hb.MASS * 9.81 / 4 / hb.MAX_THRUST)).sqrt()

                def _apply_action(self):
                    dd = self._robot.data
                    q = dd.root_quat_w.torch[:, [3, 0, 1, 2]]
                    st = torch.cat([dd.root_pos_w.torch, q, dd.root_lin_vel_w.torch, dd.root_ang_vel_w.torch], -1)
                    cmd = self._ctrl_o(st.unsqueeze(1), target_vel=self._actions.unsqueeze(1)).squeeze(1)
                    cmd = torch.nan_to_num(cmd, 0.0)
                    thr = torch.stack([self._rg[i](cmd[i])[0] for i in range(self.num_envs)])
                    self._robot.set_thrust_target(thr)

            def run(env_cls, lin, ang, tau=None):
                cfg = NavigationEnvCfg()
                cfg.sim.physics = physics_cfg
                cfg.sim.device = dev
                n = 8
                cfg.scene.num_envs = n
                rp = cfg.scene.robot.spawn.rigid_props
                rp.linear_damping, rp.angular_damping = lin, ang
                if tau is not None:
                    a = cfg.scene.robot.actuators["thrusters"]
                    a.tau_inc_range = a.tau_dec_range = (tau, tau)
                e = env_cls(cfg)
                e.reset()
                dd = e._robot.data
                pose = torch.tensor([0.0, 24.0, 2.0, 0, 0, 0, 1.0], device=dev).repeat(n, 1)
                e._robot.write_root_pose_to_sim_index(root_pose=pose)
                e._robot.write_root_velocity_to_sim_index(root_velocity=torch.zeros(n, 6, device=dev))
                vx, tilt = [], []
                for cmd, dur in SCHEDULE:
                    for _ in range(int(dur / 0.016)):
                        e.step(torch.tensor(cmd, device=dev, dtype=torch.float).repeat(n, 1))
                        vx.append(dd.root_lin_vel_w.torch.mean(0).cpu().numpy())
                        r, p, _ = mu.euler_xyz_from_quat(dd.root_quat_w.torch)
                        tilt.append(torch.rad2deg(torch.maximum(r.abs(), p.abs())).mean().item())
                return np.array(vx), np.array(tilt)

            if args_cli.variant < 0:
                print("[cmp] section D needs --variant 0|1|2 (one env per process); then run with --report")
            else:
                cfgs = [(NavigationEnv, 0.0, 0.0, None), (NavigationEnv, 0.2, 0.2, None), (OrigStackEnv, 0.2, 0.2, 1e-6)]
                cls, lin, ang, tau = cfgs[args_cli.variant]
                v, tl = run(cls, lin, ang, tau)
                os.makedirs(args_cli.outdir, exist_ok=True)
                np.savez(os.path.join(args_cli.outdir, f"d_{args_cli.variant}.npz"), v=v, tilt=tl)
                print(f"[cmp] variant {args_cli.variant} ({VARIANTS[args_cli.variant]}) saved")
                return

        # =============================================================================== E. obs / reward
        if "E" in args_cli.sections:
            print("\n[cmp] ==== E. observation / reward / termination: original code exec'd verbatim " + "=" * 0)
            try:
                import einops
            except ImportError:  # only `reduce(x, "n 1 w h -> n 1", "max")` is used
                einops = types.ModuleType("einops")
                einops.reduce = lambda x, pattern, op: x.amax(dim=(2, 3)).reshape(x.shape[0], 1)
            utils_src = open(os.path.join(ORIG, "training", "scripts", "utils.py")).read()
            uns = {"torch": torch}
            exec(function_source(os.path.join(ORIG, "training", "scripts", "utils.py"), "vec_to_new_frame"), uns)
            orig_fn_src = function_source(os.path.join(ORIG, "training", "scripts", "env.py"), "_compute_state_and_obs")
            ns = {"torch": torch, "einops": einops, "TensorDict": lambda d_, bs=None: d_, "vec_to_new_frame": uns["vec_to_new_frame"], "np": np}
            exec(orig_fn_src, ns)
            orig_fn = ns["_compute_state_and_obs"]

            class Stats(dict):
                def clone(self):
                    return Stats({k: v.clone() for k, v in self.items()})

            obs, _ = env.reset()
            g = torch.Generator(device=dev).manual_seed(2)
            worst = {"state": 0.0, "lidar": 0.0, "direction": 0.0, "dyn_obs": 0.0, "reward": 0.0}
            term_mismatch = coll_mismatch = compared = 0
            for t in range(150):
                a = 4.0 * (torch.rand(N, 3, device=dev, generator=g) - 0.5)
                prev_vel = env._prev_vel.clone()
                obs, rew, term, trunc, _ = env.step(a)
                done = (term | trunc)
                keep = ~done
                if keep.sum() == 0:
                    continue
                pos, vel = d.root_pos_w.torch, d.root_lin_vel_w.torch
                mine = split_observation(obs["policy"])
                R = env.cfg.lidar_range
                # feed the original code the same LiDAR distances (as ray hits along +x, same ordering/reshape)
                dist = (R - mine["lidar"]).reshape(N, -1)
                hits = pos.unsqueeze(1) + torch.stack([dist, torch.zeros_like(dist), torch.zeros_like(dist)], -1)
                dyn = env._dyn
                S = types.SimpleNamespace()
                S.num_envs, S.device = N, dev
                S.lidar_range, S.lidar_resolution = R, (36, 4)
                S.lidar = types.SimpleNamespace(data=types.SimpleNamespace(ray_hits_w=hits, pos_w=pos))
                rs = torch.zeros(N, 1, 23, device=dev)
                rs[:, 0, :3], rs[:, 0, 7:10] = pos, vel
                S.drone = types.SimpleNamespace(get_state=lambda env_frame=False: rs, vel_w=vel.unsqueeze(1), pos=pos.unsqueeze(1))
                S.info = {"drone_state": torch.zeros(N, 1, 13, device=dev)}
                S.stats = Stats({k: torch.zeros(N, 1, device=dev) for k in ("return", "episode_len", "reach_goal", "collision", "truncated")})
                S.progress_buf = env.episode_length_buf.clone()
                S.max_episode_length = env.max_episode_length
                S.target_pos, S.target_dir = env._target_pos.unsqueeze(1), env._target_dir.unsqueeze(1)
                S.height_range, S.prev_drone_vel_w = env._height_range.unsqueeze(1), prev_vel.unsqueeze(1)
                S.cfg = types.SimpleNamespace(env_dyn=types.SimpleNamespace(num_obstacles=dyn.num),
                                              algo=types.SimpleNamespace(feature_extractor=types.SimpleNamespace(dyn_obs_num=5)),
                                              sensor=types.SimpleNamespace(lidar_range=R), device=dev)
                S._should_render = lambda i: False
                st = torch.zeros(dyn.num, 13, device=dev); st[:, :3] = dyn.state_pos
                S.dyn_obs_state, S.dyn_obs_vel, S.dyn_obs_size = st, dyn.vel, dyn.size
                S.dyn_obs_width_res, S.max_obs_3d_height = dyn.width_res, dyn.MAX_3D_HEIGHT
                S.batch_size = [N]
                out = orig_fn(S)
                o = out["agents"]["observation"]
                worst["state"] = max(worst["state"], (o["state"] - mine["state"])[keep].abs().max().item())
                worst["lidar"] = max(worst["lidar"], (o["lidar"] - mine["lidar"])[keep].abs().max().item())
                worst["direction"] = max(worst["direction"], (o["direction"].squeeze(1) - mine["direction"])[keep].abs().max().item())
                worst["dyn_obs"] = max(worst["dyn_obs"], (o["dynamic_obstacle"] - mine["dynamic_obstacle"])[keep].abs().max().item())
                worst["reward"] = max(worst["reward"], (S.reward.squeeze(1) - rew)[keep].abs().max().item())
                term_o = S.terminated.squeeze(1)
                term_mismatch += int((term_o != term)[keep].sum())
                compared += int(keep.sum())
            for k, v in worst.items():
                row(f"max |orig - port|  {k}", f"{v:.2e}")
            row("termination flag mismatches", f"{term_mismatch} / {compared}", "env-steps compared")
            results["E"] = worst

        print("\n[cmp] done")
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
