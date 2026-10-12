"""On-policy Beta entropy / concentration of several checkpoints in the same env.

    python scripts/policy_entropy.py --num_static 80 --num_dynamic 10 ckpt1.pt ckpt2.pt ...   (use the literal 'official' for the pretrained one)
Rolls each policy out with sampling (training-style) and averages entropy (sum over 3 action dims) over states visited.
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser()
parser.add_argument("ckpts", nargs="+")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--steps", type=int, default=500)
parser.add_argument("--num_static", type=int, default=350)
parser.add_argument("--num_dynamic", type=int, default=80)
add_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.physics = "isaacsim_physx"


def main():
    from isaaclab.physics import PhysicsCfg

    with launch_simulation(cfg=PhysicsCfg(), launcher_args=args_cli) as physics_cfg:
        import torch

        from navrl_lab.navigation_env import NavigationEnv, NavigationEnvCfg, split_observation
        from navrl_lab.ppo import PPO, independent_beta

        cfg = NavigationEnvCfg()
        cfg.sim.physics = physics_cfg
        cfg.sim.device = args_cli.device or "cuda:0"
        cfg.scene.num_envs = args_cli.num_envs
        cfg.eval_mode = False
        cfg.num_static_obstacles = args_cli.num_static
        cfg.scene.terrain.terrain_generator.sub_terrains["obstacles"].num_obstacles = args_cli.num_static
        cfg.num_dynamic_obstacles = args_cli.num_dynamic
        env = NavigationEnv(cfg)
        dev = env.device
        policy = PPO(device=dev)
        print("[entropy] checkpoint | entropy(sum of 3 dims) | mean alpha+beta | mean alpha/(alpha+beta) std | ended: reach / collision")
        for c in args_cli.ckpts:
            path = os.path.join(os.path.dirname(HERE), "quick-demos", "ckpts", "navrl_checkpoint.pt") if c == "official" else c
            policy.load_state_dict(torch.load(path, map_location=dev))
            torch.manual_seed(0)
            obs, _ = env.reset()
            ent, conc, n = 0.0, 0.0, 0
            n_end = n_reach = n_col = 0
            with torch.no_grad():
                for _ in range(args_cli.steps):
                    o = split_observation(obs["policy"])
                    feat = policy.features(o)
                    a_, b_ = policy.dist_params(feat)
                    d = independent_beta(a_, b_)
                    ent += d.entropy().mean().item()
                    conc += (a_ + b_).mean().item()
                    n += 1
                    act = policy.act(o, deterministic=False)["action"]
                    obs, r, term, trunc, _ = env.step(act)
                    done = term | trunc
                    if done.any():
                        n_end += int(done.sum())
                        n_reach += int((env._reach_goal & done).sum())
                        n_col += int((term & env._collision & done).sum())
            tag = "official" if c == "official" else os.path.basename(c)
            print(f"[entropy] {tag:22s} | {ent / n:7.3f} | {conc / n:8.2f} | ended {n_end}: reach {n_reach} col {n_col}", flush=True)
        env.close()


if __name__ == "__main__":
    main()
