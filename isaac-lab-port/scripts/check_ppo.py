"""PPO port checks.

  # 1) in the Isaac Lab venv (plain torch): strict-load the official checkpoint, dump outputs, save a fresh checkpoint
  python scripts/check_ppo.py --mode mine --out /tmp/ppo_check
  # 2) in an env with torchrl 0.6 / tensordict 0.6 (the original implementation): compare + deployment compatibility
  python scripts/check_ppo.py --mode ref --out /tmp/ppo_check
"""

import argparse
import os
import sys

import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEMOS = os.path.join(os.path.dirname(HERE), "quick-demos")
CKPT = os.path.join(DEMOS, "ckpts", "navrl_checkpoint.pt")

p = argparse.ArgumentParser()
p.add_argument("--mode", choices=["mine", "ref"], required=True)
p.add_argument("--out", required=True)
args = p.parse_args()
os.makedirs(args.out, exist_ok=True)

N = 5  # not 3: the original vec_to_new_frame uses torch.cross without `dim`, which is ambiguous when N == 3


def make_inputs():
    g = torch.Generator().manual_seed(0)
    direction = torch.randn(N, 3, generator=g)
    direction[:, 2] = 0.0
    direction = 10 * direction / direction.norm(dim=1, keepdim=True)
    return {
        "mb_action": 0.05 + 0.9 * torch.rand(N, 3, generator=g),
        "mb_logp_old": 0.5 * torch.randn(N, generator=g),
        "mb_adv": torch.randn(N, 1, generator=g),
        "mb_ret": torch.randn(N, 1, generator=g),
        "mb_value": 0.3 * torch.randn(N, 1, generator=g),
        "state": torch.randn(N, 8, generator=g),
        "lidar": 4 * torch.rand(N, 1, 36, 4, generator=g),
        "direction": direction,
        "dynamic_obstacle": torch.randn(N, 1, 5, 10, generator=g),
    }


def load_or_make_inputs():
    """Inputs are generated once (mine mode) and re-used by the reference run, so RNG differences between torch versions cannot matter."""
    path = os.path.join(args.out, "inputs.pt")
    if args.mode == "mine":
        torch.save(make_inputs(), path)
    return torch.load(path)


def report(name, a, b, tol=1e-4):
    err = (a - b).abs().max().item()
    print(f"[check] {'PASS' if err < tol else 'FAIL'}  {name}  max|diff|={err:.2e}")
    return err < tol


if args.mode == "mine":
    sys.path.insert(0, HERE)
    from navrl_lab.ppo import PPO

    torch.manual_seed(1)
    res = []
    policy = PPO()
    sd = torch.load(CKPT, map_location="cpu")
    try:
        policy.load_state_dict(sd, strict=True)
        print("[check] PASS  official checkpoint loads with strict=True")
        res.append(True)
    except Exception as e:  # noqa: BLE001
        print("[check] FAIL  official checkpoint:", str(e)[:400])
        res.append(False)
    assert list(policy.state_dict().keys()) == list(sd.keys()), "key order differs"
    print("[check] PASS  state_dict keys identical (order too)")

    obs = load_or_make_inputs()

    def outputs(pol):
        with torch.no_grad():
            feat = pol.features(obs)
            alpha, beta = pol.dist_params(feat)
            a = alpha / (alpha + beta)
            return {"alpha": alpha, "beta": beta, "value": pol.value(feat), "mean_action": a,
                    "world_action": pol.to_world_action(a, obs["direction"])}

    torch.save(outputs(policy), os.path.join(args.out, "mine_official.pt"))

    # one PPO gradient step from the official weights on a fixed minibatch
    upd = PPO()
    upd.load_state_dict(sd)
    o_ = {k: obs[k] for k in ("state", "lidar", "direction", "dynamic_obstacle")}
    mb = {**{f"obs/{k}": v for k, v in o_.items()}, "action_normalized": obs["mb_action"], "log_prob": obs["mb_logp_old"],
          "value": obs["mb_value"], "adv": obs["mb_adv"], "ret": obs["mb_ret"]}
    info = upd._update(mb)
    torch.save({"info": {k: v.reshape(-1) for k, v in info.items()}, "sd": upd.state_dict()}, os.path.join(args.out, "mine_update.pt"))

    fresh = PPO()
    fresh.value_norm.update(torch.randn(64, 1) * 3 + 1)  # non-trivial running stats
    torch.save(fresh.state_dict(), os.path.join(args.out, "fresh.pt"))
    torch.save(outputs(fresh), os.path.join(args.out, "mine_fresh.pt"))
    print(f"[check] {sum(res)}/{len(res)} passed (mine); now run --mode ref")

else:
    sys.path.insert(0, DEMOS)
    from tensordict.tensordict import TensorDict
    from torchrl.data import CompositeSpec, UnboundedContinuousTensorSpec
    from torchrl.envs.utils import ExplorationType, set_exploration_type
    from ppo import PPO  # the ORIGINAL torchrl implementation used by the deployment code

    dev = "cpu"
    obs_spec = CompositeSpec({"agents": CompositeSpec({"observation": CompositeSpec({
        "state": UnboundedContinuousTensorSpec((8,), device=dev),
        "lidar": UnboundedContinuousTensorSpec((1, 36, 4), device=dev),
        "direction": UnboundedContinuousTensorSpec((1, 3), device=dev),
        "dynamic_obstacle": UnboundedContinuousTensorSpec((1, 5, 10), device=dev)})}).expand(N)}, shape=[N], device=dev)
    act_spec = CompositeSpec({"agents": CompositeSpec({"action": UnboundedContinuousTensorSpec((3,), device=dev)})}).expand(N, 3).to(dev)

    def ref_outputs(pol):
        o = load_or_make_inputs()
        td = TensorDict({"agents": TensorDict({"observation": TensorDict({
            "state": o["state"], "lidar": o["lidar"], "direction": o["direction"].unsqueeze(1),
            "dynamic_obstacle": o["dynamic_obstacle"]}, [N])}, [N])}, [N])
        with torch.no_grad(), set_exploration_type(ExplorationType.MEAN):
            out = pol(td)
        return {"alpha": out["alpha"], "beta": out["beta"], "value": out["state_value"],
                "mean_action": out["agents", "action_normalized"], "world_action": out["agents", "action"].reshape(N, 3)}

    res = []
    for tag, sd_path, mine_file in [("official", CKPT, "mine_official.pt"), ("fresh (deployment compat)", os.path.join(args.out, "fresh.pt"), "mine_fresh.pt")]:
        pol = PPO(obs_spec, act_spec, dev)
        try:
            pol.load_state_dict(torch.load(sd_path, map_location=dev), strict=True)
            print(f"[check] PASS  original PPO loads {tag} checkpoint strictly")
            res.append(True)
        except Exception as e:  # noqa: BLE001
            print(f"[check] FAIL  original PPO loading {tag}:", str(e)[:400])
            res.append(False)
            continue
        ref, mine = ref_outputs(pol), torch.load(os.path.join(args.out, mine_file))
        for k in ["alpha", "beta", "value", "mean_action", "world_action"]:
            res.append(report(f"{tag}: {k}", ref[k], mine[k]))
    # one PPO gradient step with the ORIGINAL TRAINING implementation (training/scripts/ppo.py)
    import types

    for name in ("ppo", "utils"):
        sys.modules.pop(name, None)
    sys.path.remove(DEMOS)
    TRAIN = os.path.join(os.path.dirname(HERE), "isaac-training", "training", "scripts")
    sys.path.insert(0, TRAIN)
    for stub in ("wandb", "omni_drones", "omni_drones.utils", "omni_drones.utils.torchrl"):  # only used for logging/rendering
        sys.modules[stub] = types.ModuleType(stub)
    sys.modules["omni_drones.utils.torchrl"].RenderCallback = object
    from ppo import PPO as TrainPPO

    ns = types.SimpleNamespace
    algo = ns(feature_extractor=ns(learning_rate=5e-4, dyn_obs_num=5), actor=ns(learning_rate=5e-4, clip_ratio=0.1, action_limit=2.0),
              critic=ns(learning_rate=5e-4, clip_ratio=0.1), entropy_loss_coefficient=1e-3, training_epoch_num=4, num_minibatches=16)
    pol = TrainPPO(algo, obs_spec, act_spec, dev)
    pol.load_state_dict(torch.load(CKPT, map_location=dev))
    o = load_or_make_inputs()
    tdu = TensorDict({
        "agents": TensorDict({"observation": TensorDict({
            "state": o["state"], "lidar": o["lidar"], "direction": o["direction"].unsqueeze(1),
            "dynamic_obstacle": o["dynamic_obstacle"]}, [N]), "action_normalized": o["mb_action"]}, [N]),
        "sample_log_prob": o["mb_logp_old"], "state_value": o["mb_value"], "adv": o["mb_adv"], "ret": o["mb_ret"]}, [N])
    ref_info = pol._update(tdu)
    mine_u = torch.load(os.path.join(args.out, "mine_update.pt"))
    for k in ["actor_loss", "critic_loss", "entropy", "actor_grad_norm", "critic_grad_norm", "explained_var"]:
        res.append(report(f"update step: {k}", ref_info[k].reshape(-1), mine_u["info"][k], tol=1e-4))
    ref_sd = pol.state_dict()
    worst = max((ref_sd[k].float() - mine_u["sd"][k].float()).abs().max().item() for k in ref_sd)
    print(f"[check] {'PASS' if worst < 1e-5 else 'FAIL'}  update step: all parameters after the step  max|diff|={worst:.2e}")
    res.append(worst < 1e-5)
    print(f"[check] {sum(res)}/{len(res)} passed (ref)")
