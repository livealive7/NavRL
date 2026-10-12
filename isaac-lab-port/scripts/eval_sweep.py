"""Evaluate many checkpoints of one run in a single process (one env, weights swapped between checkpoints).

    python scripts/eval_sweep.py --run logs/navrl/<run> --num_envs 50 --num_static 80 --num_dynamic 10 --random_start [--every 1]
Writes <run>/eval_sweep.csv and prints one line per checkpoint (same metrics as eval_policy.py).
"""

import argparse
import glob
import os
import re
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser()
parser.add_argument("--run", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--episodes", type=int, default=1, help="rounds of full episodes per env")
parser.add_argument("--after", type=int, default=0, help="only checkpoints with iteration > AFTER (CSV is then eval_sweep_after<AFTER>.csv)")
parser.add_argument("--every", type=int, default=1, help="evaluate every N-th checkpoint")
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

        ckpts = sorted(glob.glob(os.path.join(args_cli.run, "checkpoint_*.pt")),
                       key=lambda p: (int(re.search(r"checkpoint_(\d+)", p).group(1)) if re.search(r"checkpoint_(\d+)", p) else 10**9))
        ckpts = [c for c in ckpts if "final" not in c and int(re.search(r"checkpoint_(\d+)", c).group(1)) > args_cli.after][:: args_cli.every]
        csv_path = os.path.join(args_cli.run, "eval_sweep.csv" if args_cli.after == 0 else f"eval_sweep_after{args_cli.after}.csv")
        with open(csv_path, "w") as f:
            f.write("iteration,reached,collision,out_of_bounds,timeout,orig_reach_goal,orig_collision\n")

        for ckpt in ckpts:
            it = int(re.search(r"checkpoint_(\d+)", ckpt).group(1))
            policy.load_state_dict(torch.load(ckpt, map_location=dev))
            obs, _ = env.reset()
            n_col = n_reach = n_out = n_timeout = 0
            o_reach = o_col = 0
            for _ in range(args_cli.episodes):
                finished = torch.zeros(N, dtype=torch.bool, device=dev)
                reached = torch.zeros(N, dtype=torch.bool, device=dev)
                for _ in range(env.max_episode_length + 1):
                    o = split_observation(obs["policy"])
                    act = policy.act(o, deterministic=True)["action"]
                    obs, r, term, trunc, _ = env.step(act)
                    reached |= env._reach_goal & ~finished
                    newly = (term | trunc) & ~finished
                    if newly.any():
                        o_reach += int((newly & env._reach_goal).sum())
                        o_col += int((newly & env._collision).sum())
                        n_col += int((newly & term & env._collision).sum())
                        n_out += int((newly & term & ~env._collision).sum())
                        n_timeout += int((newly & trunc & ~term).sum())
                    finished |= newly
                    if finished.all():
                        break
                n_reach += int(reached.sum())
            total = N * args_cli.episodes
            row = (it, n_reach / total, n_col / total, n_out / total, n_timeout / total, o_reach / total, o_col / total)
            with open(csv_path, "a") as f:
                f.write("%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n" % row)
            print(f"[sweep] it {it:6d} | reached {row[1]:.1%} | collision {row[2]:.1%} | bounds {row[3]:.1%} | timeout {row[4]:.1%} | orig-style reach {row[5]:.1%} col {row[6]:.1%}", flush=True)
        env.close()


if __name__ == "__main__":
    main()
