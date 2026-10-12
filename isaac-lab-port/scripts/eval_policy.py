"""Evaluate a checkpoint deterministically (Beta mean) in the eval setup: drones start on one edge, goals on the opposite edge.

    python scripts/eval_policy.py --checkpoint logs/navrl/<run>/checkpoint_final.pt [--num_envs 64] [--official]
    --official evaluates quick-demos/ckpts/navrl_checkpoint.pt (the pretrained NavRL policy) in this port.
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--official", action="store_true")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--episodes", type=int, default=1, help="rounds of full episodes per env")
parser.add_argument("--num_static", type=int, default=350, help="static obstacles in the terrain")
parser.add_argument("--num_dynamic", type=int, default=80, help="dynamic obstacles (even; half cuboids, half cylinders)")
parser.add_argument("--no_dynamic", action="store_true")
parser.add_argument("--random_start", action="store_true", help="training-style random edge starts instead of the eval line")
add_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.physics = "isaacsim_physx"
assert args_cli.checkpoint or args_cli.official, "give --checkpoint or --official"


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
        ckpt = os.path.join(os.path.dirname(HERE), "quick-demos", "ckpts", "navrl_checkpoint.pt") if args_cli.official else args_cli.checkpoint
        policy = PPO(device=dev)
        policy.load_state_dict(torch.load(ckpt, map_location=dev))
        print(f"[eval] checkpoint: {ckpt}")

        obs, _ = env.reset()
        finished = torch.zeros(N, dtype=torch.bool, device=dev)
        n_col = n_reach = n_out = n_timeout = 0
        o_reach = o_col = o_trunc = 0  # original-style: stats at each env's first done step
        for ep in range(args_cli.episodes):
            finished[:] = False
            reached = torch.zeros(N, dtype=torch.bool, device=dev)
            first_done_reason = {}
            for _ in range(env.max_episode_length + 1):
                o = split_observation(obs["policy"])
                act = policy.act(o, deterministic=True)["action"]
                obs, r, term, trunc, _ = env.step(act)
                reached |= env._reach_goal & ~finished
                newly = (term | trunc) & ~finished
                if newly.any():
                    o_reach += int((newly & env._reach_goal).sum())
                    o_col += int((newly & env._collision).sum())
                    o_trunc += int((newly & trunc).sum())
                    n_col += int((newly & term & env._collision).sum())
                    n_out += int((newly & term & ~env._collision).sum())
                    n_timeout += int((newly & trunc & ~term).sum())
                finished |= newly
                if finished.all():
                    break
            n_reach += int(reached.sum())
        total = N * args_cli.episodes
        print(f"[eval] {total} episodes | reached goal (at any time): {n_reach / total:.1%} | collision: {n_col / total:.1%} | "
              f"out of height bounds: {n_out / total:.1%} | timeout: {n_timeout / total:.1%}")
        print(f"[eval-orig] original utils.evaluate semantics (stats at the first done step, {total} episodes) | "
              f"reach_goal (within 0.5 m at that moment): {o_reach / total:.1%} | collision: {o_col / total:.1%} | truncated: {o_trunc / total:.1%}")
        env.close()


if __name__ == "__main__":
    main()
