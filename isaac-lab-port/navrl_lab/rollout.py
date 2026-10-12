"""Rollout collection (the part of the original ``SyncDataCollector`` + env step that PPO depends on).

Kept free of Isaac imports so it can be unit-tested against torchrl (scripts/check_collector.py).
"""

from __future__ import annotations

import torch

_KEYS = ("obs", "next_obs", "action_normalized", "log_prob", "value", "reward", "terminated")


def collect_rollout(env, act, obs: dict, T: int, split):
    """Run ``T`` env steps.

    ``env.step`` follows the Isaac Lab convention (auto-reset: the returned observation is the post-reset one,
    ``extras["final_obs"]`` the pre-reset one when ``compute_final_obs`` is on). ``act(obs)`` must return a dict with
    ``action_normalized``, ``log_prob``, ``value``, ``action``. ``split`` turns ``obs["policy"]`` into the policy's dict.

    Returns ``data`` in (N, T, ...) layout as expected by ``PPO.train_on_rollout``, the observation to continue from,
    and the per-finished-episode statistics collected from ``extras["log"]``.
    """
    buf = {k: [] for k in _KEYS}
    ep = {"return": [], "episode_len": [], "reach_goal": [], "collision": []}
    for _ in range(T):
        out = act(obs)
        nxt, reward, terminated, truncated, extras = env.step(out["action"])
        nxt = split(nxt["policy"])
        done = terminated | truncated
        next_for_value = nxt
        if "final_obs" in extras and bool(done.any()):
            # env auto-resets: the true successor of a finished step is the pre-reset observation
            final = split(extras["final_obs"]["policy"])
            next_for_value = {k: torch.where(done.view(-1, *[1] * (v.dim() - 1)), final[k], v) for k, v in nxt.items()}
        buf["obs"].append({k: v.clone() for k, v in obs.items()})
        buf["next_obs"].append({k: v.clone() for k, v in next_for_value.items()})
        buf["action_normalized"].append(out["action_normalized"])
        buf["log_prob"].append(out["log_prob"])
        buf["value"].append(out["value"])
        buf["reward"].append(reward.unsqueeze(-1))
        buf["terminated"].append(terminated.unsqueeze(-1))
        if "log" in extras and bool(done.any()):
            for k in ep:
                ep[k].append(extras["log"][k])
        obs = nxt

    # (N, T, ...) layout, time along dim 1
    data = {
        "obs": {k: torch.stack([b[k] for b in buf["obs"]], 1) for k in buf["obs"][0]},
        "next_obs": {k: torch.stack([b[k] for b in buf["next_obs"]], 1) for k in buf["next_obs"][0]},
        **{k: torch.stack(buf[k], 1) for k in ("action_normalized", "log_prob", "value", "reward", "terminated")},
    }
    data["terminated"] = data["terminated"].float()
    return data, obs, ep
