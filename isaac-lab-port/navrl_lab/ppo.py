"""NavRL PPO in plain PyTorch (no torchrl / tensordict).

Ported from ``isaac-training/training/scripts/ppo.py`` + ``utils.py``. The module tree is deliberately built with
the same nesting the torchrl version produced (``feature_extractor.module.0.module.0.weight`` ...), so:

* the official checkpoints (``quick-demos/ckpts/navrl_checkpoint.pt``) load with ``strict=True``, and
* checkpoints written here load into the original ``ppo.PPO`` used by the ``ros1``/``ros2`` deployment code.

Network: LiDAR (1x36x4) CNN -> 128 | state (8) | dynamic obstacles (5x10) MLP -> 64 -> MLP 256 -> Beta actor / value.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from .frames import vec_to_world


# ------------------------------------------------------------------------------------------------ config
@dataclass
class PPOConfig:
    """Defaults = ``training/cfg/ppo.yaml``."""

    feature_extractor_lr: float = 5e-4
    actor_lr: float = 5e-4
    critic_lr: float = 5e-4  # the original builds the critic optimiser from cfg.actor.learning_rate
    dyn_obs_num: int = 5
    actor_clip_ratio: float = 0.1
    critic_clip_ratio: float = 0.1
    action_limit: float = 2.0  # m/s, per axis in the goal frame
    entropy_loss_coefficient: float = 1e-3
    training_frame_num: int = 32  # rollout length T per update
    training_epoch_num: int = 4
    num_minibatches: int = 16
    gamma: float = 0.99
    lmbda: float = 0.95
    huber_delta: float = 10.0
    max_grad_norm: float = 5.0


# ------------------------------------------------------------------------------------------------ building blocks
class _Wrap(nn.Module):
    """Holds ``module`` as an attribute; reproduces the ``TensorDictModule`` key prefix (``<name>.module``)."""

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module


class _Seq(nn.Module):
    """Holds ``module`` as a ModuleList; reproduces ``TensorDictSequential`` / ``ProbabilisticActor`` prefixes."""

    def __init__(self, *modules: nn.Module):
        super().__init__()
        self.module = nn.ModuleList(modules)


def make_mlp(in_dim: int, num_units: list[int]) -> nn.Sequential:
    layers, d = [], in_dim
    for n in num_units:
        layers += [nn.Linear(d, n), nn.LeakyReLU(), nn.LayerNorm(n)]
        d = n
    return nn.Sequential(*layers)


class BetaActor(nn.Module):
    def __init__(self, in_dim: int, action_dim: int):
        super().__init__()
        self.alpha_layer = nn.Linear(in_dim, action_dim)
        self.beta_layer = nn.Linear(in_dim, action_dim)
        self.alpha_softplus = nn.Softplus()
        self.beta_softplus = nn.Softplus()

    def forward(self, features: torch.Tensor):
        alpha = 1.0 + self.alpha_softplus(self.alpha_layer(features)) + 1e-6
        beta = 1.0 + self.beta_softplus(self.beta_layer(features)) + 1e-6
        return alpha, beta


def independent_beta(alpha: torch.Tensor, beta: torch.Tensor) -> torch.distributions.Independent:
    return torch.distributions.Independent(torch.distributions.Beta(alpha, beta), 1)


class ValueNorm(nn.Module):
    """Running mean/var of returns (PopArt-style normalisation, no output-layer correction), as in the original."""

    def __init__(self, input_shape: int = 1, beta: float = 0.995, epsilon: float = 1e-5):
        super().__init__()
        self.input_shape = torch.Size((input_shape,))
        self.epsilon, self.beta = epsilon, beta
        self.register_buffer("running_mean", torch.zeros(input_shape))
        self.register_buffer("running_mean_sq", torch.zeros(input_shape))
        self.register_buffer("debiasing_term", torch.tensor(0.0))

    def running_mean_var(self):
        mean = self.running_mean / self.debiasing_term.clamp(min=self.epsilon)
        mean_sq = self.running_mean_sq / self.debiasing_term.clamp(min=self.epsilon)
        return mean, (mean_sq - mean**2).clamp(min=1e-2)

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        dim = tuple(range(x.dim() - 1))
        w = self.beta
        self.running_mean.mul_(w).add_(x.mean(dim=dim) * (1.0 - w))
        self.running_mean_sq.mul_(w).add_((x**2).mean(dim=dim) * (1.0 - w))
        self.debiasing_term.mul_(w).add_(1.0 - w)

    def normalize(self, x):
        mean, var = self.running_mean_var()
        return (x - mean) / torch.sqrt(var)

    def denormalize(self, x):
        mean, var = self.running_mean_var()
        return x * torch.sqrt(var) + mean


class GAE(nn.Module):
    def __init__(self, gamma: float, lmbda: float):
        super().__init__()
        self.register_buffer("gamma", torch.tensor(gamma))
        self.register_buffer("lmbda", torch.tensor(lmbda))

    def forward(self, reward, terminated, value, next_value):
        """All inputs (N, T, 1). Time runs along dim 1 (one trajectory per env)."""
        adv = torch.zeros_like(reward)
        not_done = 1 - terminated.float()
        gae = 0
        for t in reversed(range(reward.shape[1])):
            delta = reward[:, t] + self.gamma * next_value[:, t] * not_done[:, t] - value[:, t]
            adv[:, t] = gae = delta + self.gamma * self.lmbda * not_done[:, t] * gae
        return adv, adv + value


# ------------------------------------------------------------------------------------------------ the policy
class PPO(nn.Module):
    """Actor-critic + PPO update.

    Observation dict keys (as produced by ``navrl_lab.navigation_env.split_observation``):
    ``state`` (N,8), ``lidar`` (N,1,36,4), ``direction`` (N,3), ``dynamic_obstacle`` (N,1,5,10).
    """

    def __init__(self, cfg: PPOConfig | None = None, device="cpu", action_dim: int = 3):
        super().__init__()
        self.cfg = cfg or PPOConfig()
        self.device = torch.device(device)
        self.action_dim = action_dim
        c = self.cfg

        cnn = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=(5, 3), padding=(2, 1)), nn.ELU(),
            nn.Conv2d(4, 16, kernel_size=(5, 3), stride=(2, 1), padding=(2, 1)), nn.ELU(),
            nn.Conv2d(16, 16, kernel_size=(5, 3), stride=(2, 2), padding=(2, 1)), nn.ELU(),
            nn.Flatten(1),
            nn.Linear(16 * 9 * 2, 128), nn.LayerNorm(128),
        )
        dyn = nn.Sequential(nn.Flatten(1), make_mlp(c.dyn_obs_num * 10, [128, 64]))
        self.feature_extractor = _Seq(
            _Wrap(cnn), _Wrap(dyn), nn.Identity(), _Wrap(make_mlp(128 + 8 + 64, [256, 256]))
        )
        self.actor = _Seq(_Wrap(BetaActor(256, action_dim)))
        self.critic = _Wrap(nn.Linear(256, 1))
        self.value_norm = ValueNorm(1)
        self.gae = GAE(c.gamma, c.lmbda)

        # orthogonal(0.01) init for actor and critic Linear layers (the feature extractor keeps torch defaults)
        def init_(m):
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, 0.01)
                nn.init.constant_(m.bias, 0.0)

        self.actor.apply(init_)
        self.critic.apply(init_)
        self.to(self.device)

        self.feature_extractor_optim = torch.optim.Adam(self.feature_extractor.parameters(), lr=c.feature_extractor_lr)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=c.actor_lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=c.critic_lr)

    # ---------------------------------------------------------------- forward pieces
    def features(self, obs: dict) -> torch.Tensor:
        fe = self.feature_extractor.module
        cnn_feat = fe[0].module(obs["lidar"])
        dyn_feat = fe[1].module(obs["dynamic_obstacle"])
        # NOTE order: torchrl's CatTensors sorts its keys, so the original network really sees [cnn, dyn, state]
        # (string keys first, tuple key last) -- not the [cnn, state, dyn] written in the original code.
        x = torch.cat([cnn_feat, dyn_feat, obs["state"]], dim=-1)
        return fe[3].module(x)

    def dist_params(self, feat: torch.Tensor):
        return self.actor.module[0].module(feat)

    def value(self, feat: torch.Tensor) -> torch.Tensor:
        return self.critic.module(feat)

    def to_world_action(self, action_normalized: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
        """Beta sample in [0,1]^3 -> goal-frame velocity in [-limit, limit] -> world-frame velocity."""
        lim = self.cfg.action_limit
        return vec_to_world(2 * action_normalized * lim - lim, direction)

    @torch.no_grad()
    def act(self, obs: dict, deterministic: bool = False) -> dict:
        """Policy step for data collection / evaluation. ``deterministic`` uses the Beta mean (ExplorationType.MEAN)."""
        feat = self.features(obs)
        alpha, beta = self.dist_params(feat)
        dist = independent_beta(alpha, beta)
        a = dist.mean if deterministic else dist.sample()
        return {
            "action_normalized": a,
            "log_prob": dist.log_prob(a),
            "value": self.value(feat),
            "action": self.to_world_action(a, obs["direction"]),
        }

    def state_dict_for_deployment(self):
        return self.state_dict()

    # ---------------------------------------------------------------- PPO update
    def train_on_rollout(self, data: dict) -> dict:
        """One PPO update.

        ``data`` holds tensors shaped (N_envs, T, ...): ``obs`` and ``next_obs`` (dicts as above),
        ``action_normalized`` (3), ``log_prob`` (), ``value`` (1), ``reward`` (1), ``terminated`` (1).
        """
        c = self.cfg
        N, T = data["reward"].shape[:2]
        flat = lambda x: x.reshape(N * T, *x.shape[2:])  # noqa: E731

        with torch.no_grad():
            next_obs = {k: flat(v) for k, v in data["next_obs"].items()}
            next_values = self.value(self.features(next_obs)).reshape(N, T, 1)
        values = self.value_norm.denormalize(data["value"])
        next_values = self.value_norm.denormalize(next_values)

        adv, ret = self.gae(data["reward"], data["terminated"], values, next_values)
        adv = (adv - adv.mean()) / adv.std().clip(1e-7)
        self.value_norm.update(ret)
        ret = self.value_norm.normalize(ret)

        batch = {
            **{f"obs/{k}": flat(v) for k, v in data["obs"].items()},
            "action_normalized": flat(data["action_normalized"]),
            "log_prob": flat(data["log_prob"]),
            "value": flat(data["value"]),
            "adv": flat(adv),
            "ret": flat(ret),
        }

        infos = []
        for _ in range(c.training_epoch_num):
            perm = torch.randperm((N * T // c.num_minibatches) * c.num_minibatches, device=self.device)
            for idx in perm.reshape(c.num_minibatches, -1):
                infos.append(self._update({k: v[idx] for k, v in batch.items()}))
        return {k: float(torch.stack([i[k] for i in infos]).mean()) for k in infos[0]}

    def _update(self, mb: dict) -> dict:
        c = self.cfg
        obs = {k[4:]: v for k, v in mb.items() if k.startswith("obs/")}
        feat = self.features(obs)
        alpha, beta = self.dist_params(feat)
        dist = independent_beta(alpha, beta)
        log_probs = dist.log_prob(mb["action_normalized"])

        entropy_loss = -c.entropy_loss_coefficient * dist.entropy().mean()

        adv = mb["adv"]
        ratio = torch.exp(log_probs - mb["log_prob"]).unsqueeze(-1)
        surr1 = adv * ratio
        surr2 = adv * ratio.clamp(1.0 - c.actor_clip_ratio, 1.0 + c.actor_clip_ratio)
        actor_loss = -torch.mean(torch.min(surr1, surr2)) * self.action_dim

        b_value, ret = mb["value"], mb["ret"]
        value = self.value(feat)
        value_clipped = b_value + (value - b_value).clamp(-c.critic_clip_ratio, c.critic_clip_ratio)
        critic_loss = torch.max(
            F.huber_loss(value_clipped, ret, delta=c.huber_delta), F.huber_loss(value, ret, delta=c.huber_delta)
        )

        loss = entropy_loss + actor_loss + critic_loss
        self.feature_extractor_optim.zero_grad()
        self.actor_optim.zero_grad()
        self.critic_optim.zero_grad()
        loss.backward()
        actor_gn = nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=c.max_grad_norm)
        critic_gn = nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=c.max_grad_norm)
        self.feature_extractor_optim.step()
        self.actor_optim.step()
        self.critic_optim.step()

        explained_var = 1 - F.mse_loss(value, ret) / ret.var()
        with torch.no_grad():  # policy-drift diagnostics (not part of the original)
            approx_kl = ((ratio - 1.0) - torch.log(ratio)).mean()
            clip_fraction = ((ratio - 1.0).abs() > c.actor_clip_ratio).float().mean()
        return {
            "approx_kl": approx_kl,
            "clip_fraction": clip_fraction,
            "actor_loss": actor_loss.detach(),
            "critic_loss": critic_loss.detach(),
            "entropy": entropy_loss.detach(),
            "actor_grad_norm": actor_gn.detach(),
            "critic_grad_norm": critic_gn.detach(),
            "explained_var": explained_var.detach(),
        }
