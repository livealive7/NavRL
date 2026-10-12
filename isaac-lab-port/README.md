# NavRL on Isaac Lab 3.0 / Isaac Sim 6.0.1 (PhysX)

Port of `isaac-training/training/scripts/env.py`. Run everything from the Isaac Lab venv:

```bash
source ~/Volume/IsaacLab/.venv/bin/activate
python ~/Volume/NavRL/isaac-lab-port/scripts/check_env.py                 # 10 sanity checks, headless
python ~/Volume/NavRL/isaac-lab-port/scripts/check_env.py --no_dynamic    # static obstacles only
python ~/Volume/NavRL/isaac-lab-port/scripts/check_env.py --eval_mode     # deterministic start/goal line
```

| file | what |
|---|---|
| `navrl_lab/navigation_env.py` | `NavigationEnv` (`DirectRLEnv`) + `NavigationEnvCfg`, `split_observation()` |
| `navrl_lab/hummingbird.py` | Hummingbird as `Multirotor` (OmniDrones' mass/thrust/allocation/rotor lag) + USD |
| `navrl_lab/terrain.py` | obstacle terrain incl. the forked `range` height mode |
| `navrl_lab/lidar.py` | yaw-aligned 36x4 LiDAR on the terrain mesh |
| `navrl_lab/dynamic_obstacles.py` | 80 moving cuboids/cylinders (tensor-only, shared by all envs) |
| `navrl_lab/frames.py` | goal-frame helpers `vec_to_new_frame` / `vec_to_world` |
| `spike/` | the two feasibility spikes (LiDAR, flight) |

Interface: `env.step(a)` takes a **world-frame velocity target (N,3) in m/s**; `obs["policy"]` is (N,205) flat,
`split_observation` gives `state(8) / lidar(1,36,4) / direction(3) / dynamic_obstacle(1,5,10)`.
The policy must output goal-frame velocities and convert with `vec_to_world` before `env.step`, as in the original `ppo.py`.

## Deliberate differences from the original
* LiDAR is our own (`ray_alignment="yaw"` in Isaac Lab 3.0 does not rotate ray directions with yaw).
* Lee controller is Isaac Lab's `LeeVelController` with the OmniDrones hummingbird gains; yaw is not held (same as before).
* Dynamic obstacles have no USD prims (the original never ray-casts them either); add markers if you need to see them.
* Timeout is `episode_length_buf >= max_episode_length - 1` (Isaac Lab convention, 1 step earlier).

## PPO (plain PyTorch) - `navrl_lab/ppo.py`
Same network/loss/hyper-parameters as `training/scripts/ppo.py`, without torchrl/tensordict. The module tree reproduces the
old `state_dict` keys, so **checkpoints are interchangeable with the original code in both directions**: the official
`quick-demos/ckpts/navrl_checkpoint.pt` loads with `strict=True`, and ours load strictly into the original torchrl `ppo.PPO`
used by `ros1/`, `ros2/`, `quick-demos/`.

`scripts/check_ppo.py` verifies this against the original implementation. It needs an env with torchrl 0.6 / tensordict 0.6
(e.g. `uv venv --python 3.12 && uv pip install torch==2.5.1 tensordict==0.6.2 torchrl==0.6.0 einops "numpy<2"`).
19 checks: forward outputs match to <=4e-5, and one full PPO update (losses, grad norms, all parameters) matches to <=2.4e-7.

Gotcha found on the way: torchrl's `CatTensors` sorts its keys, so the original network concatenates
`[cnn(128), dyn_obs(64), state(8)]`, not the `[cnn, state, dyn_obs]` written in the source. `ppo.py` uses the real order;
with the wrong order the official checkpoint produces garbage.

## Training / evaluation
```bash
source ~/Volume/IsaacLab/.venv/bin/activate
python scripts/train.py --num_envs 1024 --max_iterations 1500      # logs/navrl/<run>/ (tensorboard + checkpoint_*.pt)
python scripts/eval_policy.py --checkpoint logs/navrl/<run>/checkpoint_final.pt
python scripts/eval_policy.py --official                              # pretrained NavRL policy inside this port
```
Fly one drone with a policy (inference only, optional window and PNG of the path):
```bash
python scripts/fly.py --official --visualizer kit --realtime        # window; needs a desktop session (not tested here)
python scripts/fly.py --official --plot /tmp/flight.png              # headless + top-view PNG
python scripts/fly.py --checkpoint logs/navrl/<run>/checkpoint_2000.pt --start=-20,22,2 --goals 0,0,2 18,-22,2
```
Obstacle counts (defaults = original 350 / 80): `--num_static N`, `--num_dynamic N` (even; half cuboids, half cylinders), `--no_dynamic`;
same flags on `train.py`, `eval_policy.py`, `check_env.py`. A policy trained with one obstacle setup should be evaluated with the same one.
Evaluation numbers fluctuate a lot between runs (64 envs; dynamic obstacles use unseeded torch RNG): average several runs.
GAE keeps the original semantics (the trace is cut on `terminated` only); terminal observations come from `extras["final_obs"]`.

## Comparison against the original code - `scripts/compare_original.py`
The original simulator (Isaac Sim 2023.1) does not run on RTX 5080, so the original *pure-PyTorch* code is loaded / `exec`'d
verbatim from this repo and fed the same states as the port:

| item | result |
|---|---|
| rigid-body damping | **was wrong** (port used ARL's 0.0/0.0/1.0; original `robots/config.py` is linear 0.2, angular 0.2, depenetration 10). Fixed in `hummingbird.py` |
| Lee controller (original `LeePositionController`) | thrust targets differ 0.29 % on average |
| rotor lag (original `RotorGroup` vs `Thruster`) | max 1.4e-6 N |
| closed-loop velocity response in PhysX | with damping 0.2 identical (x axis 0.000 m/s) to "original controller + original rotors"; with 0.0: t90 0.78 s vs 1.14 s, steady 2.00 vs 1.83 m/s for a 2 m/s command |
| observation / reward / termination (`_compute_state_and_obs` exec'd from `env.py`) | <= 1e-6 over 9596 env-steps, 0 termination mismatches |

```bash
python scripts/compare_original.py --sections BCE                          # controller, rotors, obs/reward
for v in 0 1 2; do python scripts/compare_original.py --sections D --variant $v; done   # closed loop, one env per process
python scripts/compare_original.py --report                                # table for section D (no simulator needed)
```
Only one `NavigationEnv` can be created per process (a second one fails inside `cfg.validate()`).

Not comparable: PhysX itself (2023.1 vs 6.0.1) and the unseeded motion of the dynamic obstacles.

## Results so far (256 envs, random edge starts, deterministic actions, 2 runs each)
| policy | reached | collision |
|---|---|---|
| official `navrl_checkpoint.pt` | 59-64 % | 34 % |
| this port, 3000 iterations (~1e8 frames) | 46-50 % | 49-53 % |

The original recipe uses 1.2e9 frames; longer runs are in `logs/navrl/run3` (resumed from run2).

## Not done yet
Visualisation of dynamic obstacles; wandb logging (TensorBoard is used); numerical comparison of the drone dynamics
against OmniDrones.
