"""Finer evaluation than reached/collision: per-env closest approach and final distance to the goal, deterministic vs sampled actions.

    python scripts/eval_detail.py --run logs/navrl/<run> --iters 1000,2000,3000 --num_envs 256 --num_static 0 --no_dynamic
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser()
parser.add_argument("--run", type=str, required=True)
parser.add_argument("--iters", type=str, required=True, help="comma separated checkpoint iterations")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--num_static", type=int, default=350)
parser.add_argument("--num_dynamic", type=int, default=80)
parser.add_argument("--no_dynamic", action="store_true")
parser.add_argument("--random_start", action="store_true")
add_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.physics = "isaacsim_physx"


def main():
    from isaaclab.physics import PhysicsCfg

    with launch_simulation(cfg=PhysicsCfg(), launcher_args=args_cli) as physics_cfg:
        import torch

        from navrl_lab.navigation_env import NavigationEnv, NavigationEnvCfg, split_observation
        from navrl_lab.ppo import PPO

        cfg = NavigationEnvCfg()
        cfg.sim.physics = physics_cfg
        cfg.sim.device = args_cli.device or "cuda:0"
        cfg.scene.num_envs = args_cli.num_envs
        cfg.eval_mode = not args_cli.random_start
        cfg.num_static_obstacles = args_cli.num_static
        cfg.scene.terrain.terrain_generator.sub_terrains["obstacles"].num_obstacles = args_cli.num_static
        cfg.num_dynamic_obstacles = 0 if args_cli.no_dynamic else args_cli.num_dynamic
        env = NavigationEnv(cfg)
        dev, N = env.device, env.num_envs
        policy = PPO(device=dev)
        q = lambda x, p: float(torch.quantile(x, p))  # noqa: E731
        print("[detail] it | mode | reached(<0.5m any time) | closest approach median/p90/max [m] | final dist median/p90 [m] | mean speed [m/s] | collision")
        for it in [int(i) for i in args_cli.iters.split(",")]:
            policy.load_state_dict(torch.load(os.path.join(args_cli.run, f"checkpoint_{it}.pt"), map_location=dev))
            for det in (True, False):
                torch.manual_seed(0)
                obs, _ = env.reset()
                finished = torch.zeros(N, dtype=torch.bool, device=dev)
                closest = torch.full((N,), 1e9, device=dev)
                final = torch.zeros(N, device=dev)
                speed_sum = torch.zeros(N, device=dev)
                steps = torch.zeros(N, device=dev)
                col = torch.zeros(N, dtype=torch.bool, device=dev)
                for _ in range(env.max_episode_length + 1):
                    o = split_observation(obs["policy"])
                    act = policy.act(o, deterministic=det)["action"]
                    obs, r, term, trunc, _ = env.step(act)
                    ended = (term | trunc) & ~finished
                    live = ~finished & ~ended  # on the ending step the env has already reset the drone
                    pos = env._robot.data.root_pos_w.torch
                    d = (env._target_pos - pos).norm(dim=-1)
                    closest = torch.where(live, torch.minimum(closest, d), closest)
                    final = torch.where(live, d, final)
                    speed_sum += torch.where(live, env._robot.data.root_lin_vel_w.torch.norm(dim=-1), torch.zeros_like(d))
                    steps += live.float()
                    col |= ended & term & env._collision
                    finished |= ended
                    if finished.all():
                        break
                reached = (closest < 0.5).float().mean().item()
                print(f"[detail] {it:6d} | {'mean  ' if det else 'sample'} | {reached * 100:5.1f}% | "
                      f"{q(closest, .5):5.2f} / {q(closest, .9):5.2f} / {closest.max().item():5.2f} | "
                      f"{q(final, .5):5.2f} / {q(final, .9):5.2f} | {(speed_sum / steps.clamp_min(1)).mean().item():4.2f} | {col.float().mean().item() * 100:4.1f}%",
                      flush=True)
        env.close()


if __name__ == "__main__":
    main()
