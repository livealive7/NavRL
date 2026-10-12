"""Gradient check: in which direction would PPO push a given policy?

Collects K independent rollouts with a FIXED policy (no update), and for each one computes the full-batch gradient of the three PPO
loss terms (policy-gradient, entropy bonus, critic) with respect to the feature extractor and the actor head.
If the policy is already at a good optimum, the gradients should be pure noise: different batches point in unrelated directions
(cosine ~ 0, ||mean g||^2 / mean ||g||^2 ~ 1/K).  A consistent direction (cosine >> 0, ratio >> 1/K) is a systematic push.

    python scripts/grad_check.py --official --num_static 80 --num_dynamic 8
"""

import argparse
import itertools
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--official", action="store_true")
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--batches", type=int, default=8, help="independent rollouts (K)")
parser.add_argument("--warmup", type=int, default=300, help="env steps before the first batch (leave the reset state)")
parser.add_argument("--num_static", type=int, default=80)
parser.add_argument("--num_dynamic", type=int, default=8)
parser.add_argument("--no_dynamic", action="store_true")
add_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.physics = "isaacsim_physx"
assert args_cli.checkpoint or args_cli.official, "give --checkpoint or --official"


def main():
    from isaaclab.physics import PhysicsCfg

    with launch_simulation(cfg=PhysicsCfg(), launcher_args=args_cli) as physics_cfg:
        import torch
        import torch.nn.functional as F

        from navrl_lab.navigation_env import NavigationEnv, NavigationEnvCfg, split_observation
        from navrl_lab.ppo import PPO, independent_beta
        from navrl_lab.rollout import collect_rollout

        cfg = NavigationEnvCfg()
        cfg.sim.physics = physics_cfg
        cfg.sim.device = args_cli.device or "cuda:0"
        cfg.scene.num_envs = args_cli.num_envs
        cfg.compute_final_obs = True
        cfg.num_static_obstacles = args_cli.num_static
        cfg.scene.terrain.terrain_generator.sub_terrains["obstacles"].num_obstacles = args_cli.num_static
        cfg.num_dynamic_obstacles = 0 if args_cli.no_dynamic else args_cli.num_dynamic
        env = NavigationEnv(cfg)
        dev = env.device
        ckpt = os.path.join(os.path.dirname(HERE), "quick-demos", "ckpts", "navrl_checkpoint.pt") if args_cli.official else args_cli.checkpoint
        policy = PPO(device=dev)
        policy.load_state_dict(torch.load(ckpt, map_location=dev))
        c, T = policy.cfg, policy.cfg.training_frame_num
        print(f"[grad] checkpoint {ckpt} | static {args_cli.num_static} dynamic {cfg.num_dynamic_obstacles}")

        obs, _ = env.reset()
        obs = split_observation(obs["policy"])
        for _ in range(args_cli.warmup):
            out = policy.act(obs)
            o, *_ = env.step(out["action"])
            obs = split_observation(o["policy"])

        groups = {"feature_extractor": list(policy.feature_extractor.parameters()),
                  "actor_head": list(policy.actor.parameters())}
        all_params = groups["feature_extractor"] + groups["actor_head"] + list(policy.critic.parameters())
        grads = {}  # (group, term) -> list of flat vectors over batches
        scal = {"ev": [], "critic_loss": [], "ratio_dev": [], "entropy_term": [], "vn_mean": [], "vn_std": [], "ret_std": []}

        for k in range(args_cli.batches):
            data, obs, _ = collect_rollout(env, policy.act, obs, T, split_observation)
            N = data["reward"].shape[0]
            flat = lambda x: x.reshape(N * T, *x.shape[2:])  # noqa: E731
            with torch.no_grad():
                nxt = {kk: flat(v) for kk, v in data["next_obs"].items()}
                next_values = policy.value(policy.features(nxt)).reshape(N, T, 1)
            values = policy.value_norm.denormalize(data["value"])
            next_values = policy.value_norm.denormalize(next_values)
            adv, ret = policy.gae(data["reward"], data["terminated"], values, next_values)
            scal["ret_std"].append(ret.std().item())
            adv = (adv - adv.mean()) / adv.std().clip(1e-7)
            policy.value_norm.update(ret)
            ret = policy.value_norm.normalize(ret)
            m, v = policy.value_norm.running_mean_var()
            scal["vn_mean"].append(m.item()); scal["vn_std"].append(v.sqrt().item())

            o = {kk: flat(v) for kk, v in data["obs"].items()}
            a, old_lp, b_val = flat(data["action_normalized"]), flat(data["log_prob"]), flat(data["value"])
            adv, ret = flat(adv), flat(ret)

            feat = policy.features(o)
            alpha, beta = policy.dist_params(feat)
            dist = independent_beta(alpha, beta)
            lp = dist.log_prob(a)
            ratio = torch.exp(lp - old_lp).unsqueeze(-1)
            actor_loss = -torch.mean(torch.min(adv * ratio, adv * ratio.clamp(1 - c.actor_clip_ratio, 1 + c.actor_clip_ratio))) * 3
            entropy_loss = -c.entropy_loss_coefficient * dist.entropy().mean()
            value = policy.value(feat)
            value_clipped = b_val + (value - b_val).clamp(-c.critic_clip_ratio, c.critic_clip_ratio)
            critic_loss = torch.max(F.huber_loss(value_clipped, ret, delta=c.huber_delta), F.huber_loss(value, ret, delta=c.huber_delta))
            scal["ev"].append((1 - F.mse_loss(value, ret) / ret.var()).item())
            scal["critic_loss"].append(critic_loss.item()); scal["ratio_dev"].append((ratio - 1).abs().mean().item())
            scal["entropy_term"].append(entropy_loss.item())

            for term, loss in (("policy_gradient", actor_loss), ("entropy_bonus", entropy_loss), ("critic", critic_loss)):
                g = torch.autograd.grad(loss, all_params, retain_graph=True, allow_unused=True)
                gd = dict(zip(all_params, g))
                for gname, ps in groups.items():
                    vec = torch.cat([(gd[p] if gd[p] is not None else torch.zeros_like(p)).flatten() for p in ps])
                    grads.setdefault((gname, term), []).append(vec)
            print(f"[grad] batch {k + 1}/{args_cli.batches} done | ev {scal['ev'][-1]:.3f} critic_loss {scal['critic_loss'][-1]:.4f}", flush=True)

        K = args_cli.batches
        print("\n=== scalars (mean over batches) ===")
        for kk, v in scal.items():
            print(f"  {kk:13s} {sum(v) / len(v):10.4f}   (min {min(v):.4f} max {max(v):.4f})")
        print(f"\n=== gradients over K={K} independent batches (pure noise: pairwise cosine ~0, mean-ratio ~{1 / K:.3f}; consistent push: both large) ===")
        print(f"  {'group':18s} {'term':16s} {'mean |g|':>10s} {'pair cos':>9s} {'|mean g|^2/mean|g|^2':>22s}")
        for (gname, term), vs in grads.items():
            G = torch.stack(vs)
            norms = G.norm(dim=1)
            cos = [F.cosine_similarity(G[i], G[j], dim=0).item() for i, j in itertools.combinations(range(K), 2)]
            ratio_m = (G.mean(0).norm() ** 2 / (norms ** 2).mean()).item()
            print(f"  {gname:18s} {term:16s} {norms.mean().item():10.4f} {sum(cos) / len(cos):9.3f} {ratio_m:22.3f}")
        env.close()


if __name__ == "__main__":
    main()
