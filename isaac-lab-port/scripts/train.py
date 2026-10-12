"""Train NavRL (PPO) on Isaac Lab / PhysX.

    python scripts/train.py --num_envs 1024 --max_iterations 2000
    python scripts/train.py --resume logs/navrl/<run>/checkpoint_500.pt

One iteration = `training_frame_num` (32) env steps in every env, then one PPO update (4 epochs x 16 minibatches).
Checkpoints are plain `state_dict`s that the original torchrl `ppo.PPO` (ros1/ros2/quick-demos) loads strictly.
"""

import argparse
import datetime
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--max_iterations", type=int, default=2000)
parser.add_argument("--save_interval", type=int, default=100)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--num_static", type=int, default=350, help="static obstacles in the terrain")
parser.add_argument("--num_dynamic", type=int, default=80, help="dynamic obstacles (even; half cuboids, half cylinders)")
parser.add_argument("--lr", type=float, default=None, help="override all three learning rates (default 5e-4 as in the original)")
parser.add_argument("--entropy_coef", type=float, default=None, help="override entropy_loss_coefficient (default 1e-3)")
parser.add_argument("--num_minibatches", type=int, default=None, help="override num_minibatches (default 16)")
parser.add_argument("--randomize_episode_start", action="store_true",
                    help="random initial episode_length_buf per env so episodes (and mission phases) are not synchronised")
parser.add_argument("--no_dynamic", action="store_true", help="static obstacles only")
parser.add_argument("--resume", type=str, default=None, help="checkpoint (.pt state_dict) to start from")
parser.add_argument("--log_dir", type=str, default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", "navrl"))
parser.add_argument("--run_name", type=str, default=None)
add_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.physics = "isaacsim_physx"


def main():
    from isaaclab.physics import PhysicsCfg

    with launch_simulation(cfg=PhysicsCfg(), launcher_args=args_cli) as physics_cfg:
        import numpy as np
        import torch
        from torch.utils.tensorboard import SummaryWriter

        from navrl_lab.navigation_env import NavigationEnv, NavigationEnvCfg, split_observation
        from navrl_lab.rollout import collect_rollout
        from navrl_lab.ppo import PPO, PPOConfig

        torch.manual_seed(args_cli.seed)
        np.random.seed(args_cli.seed)

        cfg = NavigationEnvCfg()
        cfg.sim.physics = physics_cfg
        cfg.sim.device = args_cli.device or "cuda:0"
        cfg.scene.num_envs = args_cli.num_envs
        cfg.compute_final_obs = True  # terminal observations for correct value bootstrapping
        cfg.seed = args_cli.seed
        cfg.num_static_obstacles = args_cli.num_static
        cfg.scene.terrain.terrain_generator.sub_terrains["obstacles"].num_obstacles = args_cli.num_static
        cfg.num_dynamic_obstacles = 0 if args_cli.no_dynamic else args_cli.num_dynamic
        env = NavigationEnv(cfg)
        dev, N = env.device, env.num_envs

        pcfg = PPOConfig()
        if args_cli.lr is not None:
            pcfg.feature_extractor_lr = pcfg.actor_lr = pcfg.critic_lr = args_cli.lr
        if args_cli.entropy_coef is not None:
            pcfg.entropy_loss_coefficient = args_cli.entropy_coef
        if args_cli.num_minibatches is not None:
            pcfg.num_minibatches = args_cli.num_minibatches
        print(f"[NavRL] lr={pcfg.actor_lr} entropy_coef={pcfg.entropy_loss_coefficient} minibatches={pcfg.num_minibatches} "
              f"randomize_episode_start={args_cli.randomize_episode_start}")
        policy = PPO(pcfg, device=dev)
        if args_cli.resume:
            policy.load_state_dict(torch.load(args_cli.resume, map_location=dev))
            print(f"[NavRL] resumed from {args_cli.resume}")

        run = args_cli.run_name or datetime.datetime.now().strftime("%m-%d_%H-%M")
        log_dir = os.path.join(args_cli.log_dir, run)
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir)
        print(f"[NavRL] logging to {log_dir}")

        T = pcfg.training_frame_num
        obs, _ = env.reset()
        if args_cli.randomize_episode_start:
            env.episode_length_buf = torch.randint_like(env.episode_length_buf, high=env.max_episode_length)
        obs = split_observation(obs["policy"])
        frames, t_start = 0, time.time()

        for it in range(args_cli.max_iterations + 1):
            t0 = time.time()
            data, obs, ep = collect_rollout(env, policy.act, obs, T, split_observation)
            t_collect = time.time() - t0

            losses = policy.train_on_rollout(data)

            frames += N * T
            fps = N * T / (time.time() - t0)
            info = {"train/" + k: v for k, v in losses.items()}
            info["train/reward_step_mean"] = data["reward"].mean().item()
            for k, v in ep.items():
                if v:
                    info["episode/" + k] = float(np.mean(v))
            info["perf/fps"] = fps
            for k, v in info.items():
                writer.add_scalar(k, v, it)

            if it % 10 == 0:
                print(
                    f"[NavRL] it {it:5d} | frames {frames:>10,d} | fps {fps:>8,.0f} (collect {t_collect:.1f}s) | "
                    f"step-r {info['train/reward_step_mean']:6.2f} | "
                    f"ep-len {info.get('episode/episode_len', float('nan')):7.1f} | "
                    f"collision {info.get('episode/collision', float('nan')):.2f} | "
                    f"reach {info.get('episode/reach_goal', float('nan')):.2f} | "
                    f"actor {losses['actor_loss']:.3f} critic {losses['critic_loss']:.3f} ev {losses['explained_var']:.2f}",
                    flush=True,
                )
            if it % args_cli.save_interval == 0 and it > 0:
                torch.save(policy.state_dict(), os.path.join(log_dir, f"checkpoint_{it}.pt"))
        torch.save(policy.state_dict(), os.path.join(log_dir, "checkpoint_final.pt"))
        print(f"[NavRL] done. {frames:,d} frames in {(time.time() - t_start) / 60:.1f} min -> {log_dir}")
        writer.close()
        env.close()


if __name__ == "__main__":
    main()
