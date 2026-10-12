"""Rollout-collection check: our ``collect_rollout`` vs the torchrl ``SyncDataCollector`` the original NavRL uses.

Both run the same deterministic toy simulator (random-looking terminations / truncations, per-env episode lengths,
re-sampled start states) with the same deterministic policy, over 3 consecutive batches.

  python scripts/check_collector.py --mode mine --out /tmp/coll   # any env with torch (no Isaac needed)
  python scripts/check_collector.py --mode ref  --out /tmp/coll   # env with torchrl 0.6 / tensordict 0.6: compares
"""

import argparse
import importlib.util
import os

import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
p = argparse.ArgumentParser()
p.add_argument("--mode", choices=["mine", "ref"], required=True)
p.add_argument("--out", required=True)
args = p.parse_args()
os.makedirs(args.out, exist_ok=True)

N, T, BATCHES = 5, 6, 4


class Sim:
    """Toy simulator with the same bookkeeping surface as a drone env: reset sampling, termination, truncation."""

    def __init__(self):
        self.x = torch.zeros(N)
        self.t = torch.zeros(N)
        self.ep = torch.zeros(N)
        self.L = 5.0 + (torch.arange(N) % 3).float()  # per-env time limit
        self.ids = torch.arange(N).float()

    def reset_idx(self, ids):
        self.ep[ids] += 1
        self.x[ids] = 10 * self.ids[ids] + self.ep[ids]
        self.t[ids] = 0

    def step(self, a):
        self.x += 0.5 * a.sum(-1) + 0.3
        self.t += 1
        reward = 0.1 * self.x - 0.01 * self.t
        terminated = (self.x * 3.7).frac() < 0.15
        truncated = (self.t >= self.L) & ~terminated  # a step can be both in torchrl terms; keep them exclusive here
        return reward, terminated, truncated

    def obs(self):
        return torch.stack([self.x, self.t], -1).clone()


def policy_fn(s):  # deterministic stand-in for the network
    a = torch.sigmoid(0.1 * s[:, 0:1] + 0.1 * torch.arange(3).float())
    return {"action_normalized": a, "log_prob": -(a**2).sum(-1), "value": s.sum(-1, keepdim=True), "action": 2 * a - 1}


def run_mine():
    spec = importlib.util.spec_from_file_location("rollout", os.path.join(HERE, "navrl_lab", "rollout.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    sim = Sim()

    class MineEnv:  # Isaac Lab DirectRLEnv convention: auto reset, post-reset obs returned, final_obs in extras
        def reset(self):
            sim.reset_idx(torch.arange(N))
            return {"policy": sim.obs()}, {}

        def step(self, action):
            r, term, trunc = sim.step(action)
            final = sim.obs()
            done = term | trunc
            if done.any():
                sim.reset_idx(done.nonzero().squeeze(-1))
            return {"policy": sim.obs()}, r, term, trunc, ({} if os.environ.get("NO_FINAL") else {"final_obs": {"policy": final}})

    env = MineEnv()
    o, _ = env.reset()
    obs = {"state": o["policy"]}
    out = []
    for _ in range(BATCHES):
        data, obs, _ = mod.collect_rollout(env, lambda ob: policy_fn(ob["state"]), obs, T, lambda o: {"state": o})
        out.append(data)
    torch.save(out, os.path.join(args.out, "mine.pt"))
    print("saved", len(out), "batches")


def run_ref():
    from tensordict import TensorDict
    from tensordict.nn import TensorDictModule
    from torchrl.collectors import SyncDataCollector
    from torchrl.data import CompositeSpec, DiscreteTensorSpec, UnboundedContinuousTensorSpec
    from torchrl.envs import EnvBase
    from torchrl.envs.utils import ExplorationType

    sim = Sim()

    class RefEnv(EnvBase):
        def __init__(self):
            super().__init__(device="cpu", batch_size=[N])
            self.observation_spec = CompositeSpec(state=UnboundedContinuousTensorSpec((N, 2)), shape=(N,))
            self.action_spec = UnboundedContinuousTensorSpec((N, 3))
            self.reward_spec = UnboundedContinuousTensorSpec((N, 1))
            b = lambda: DiscreteTensorSpec(2, (N, 1), dtype=torch.bool)  # noqa: E731
            self.done_spec = CompositeSpec(done=b(), terminated=b(), truncated=b(), shape=(N,))

        def _reset(self, td=None, **kw):  # same structure as IsaacEnv._reset
            mask = td.get("_reset").reshape(N) if td is not None and "_reset" in td.keys() else torch.ones(N, dtype=bool)
            sim.reset_idx(mask.nonzero().squeeze(-1))
            z = torch.zeros(N, 1, dtype=torch.bool)
            return TensorDict({"state": sim.obs(), "done": z, "terminated": z.clone(), "truncated": z.clone()}, [N])

        def _step(self, td):  # same structure as IsaacEnv._step: no reset inside
            r, term, trunc = sim.step(td["action"])
            return TensorDict(
                {"state": sim.obs(), "reward": r.unsqueeze(-1), "done": (term | trunc).unsqueeze(-1),
                 "terminated": term.unsqueeze(-1), "truncated": trunc.unsqueeze(-1)}, [N])

        def _set_seed(self, seed):
            pass

    env = RefEnv()

    def fn(s):
        o = policy_fn(s)
        return o["action"], o["log_prob"], o["value"], o["action_normalized"]

    pol = TensorDictModule(fn, in_keys=["state"], out_keys=["action", "sample_log_prob", "state_value", "action_normalized"])
    coll = SyncDataCollector(env, policy=pol, frames_per_batch=N * T, total_frames=N * T * BATCHES, device="cpu",
                             return_same_td=True, exploration_type=ExplorationType.RANDOM)
    mine = torch.load(os.path.join(args.out, "mine.pt"))
    bad = 0
    for bi, td in enumerate(coll):
        m = mine[bi]
        pairs = {
            "obs": (td["state"], m["obs"]["state"]),
            "next_obs (terminal obs at done steps)": (td["next", "state"], m["next_obs"]["state"]),
            "action_normalized": (td["action_normalized"], m["action_normalized"]),
            "log_prob": (td["sample_log_prob"], m["log_prob"]),
            "value": (td["state_value"], m["value"]),
            "reward": (td["next", "reward"], m["reward"]),
            "terminated": (td["next", "terminated"].float(), m["terminated"]),
        }
        for k, (a, b) in pairs.items():
            same = a.shape == b.shape and torch.allclose(a.float(), b.float(), atol=1e-6)
            bad += not same
            print(f"[batch {bi}] {'PASS' if same else 'FAIL'}  {k}  shapes {tuple(a.shape)} vs {tuple(b.shape)}"
                  + ("" if same else f"  max diff {(a.float() - b.float()).abs().max().item() if a.shape == b.shape else 'n/a'}"))
        n_done = int(td["next", "done"].sum())
        print(f"[batch {bi}] info: {n_done} done steps ({int(td['next','terminated'].sum())} terminated) in this batch")
    print("ALL PASS" if bad == 0 else f"{bad} FAILED")


run_mine() if args.mode == "mine" else run_ref()
