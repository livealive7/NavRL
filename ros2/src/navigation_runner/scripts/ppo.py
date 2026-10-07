"""NavRL policy for deployment, in plain PyTorch.

This replaces the torchrl / tensordict implementation (CompositeSpec,
UnboundedContinuousTensorSpec, TensorDictSequential, ProbabilisticActor, ...),
which only works with one specific old torchrl release. The module tree is
deliberately nested exactly like the torchrl version, e.g.
``feature_extractor.module.0.module.0.weight``, so the existing checkpoints
(``ckpts/navrl_checkpoint.pt``) load with ``strict=True`` and no conversion.

Network: LiDAR (1 x H x V) CNN -> 128, dynamic obstacles (N x 10) MLP -> 64,
concatenated with the 8-dim drone state, MLP 256 -> 256, then a Beta actor.
The Beta mean (what ``ExplorationType.MEAN`` returned) is the deterministic
action.
"""

import torch
import torch.nn as nn

from utils import GAE, ValueNorm, vec_to_world


class _Wrap(nn.Module):
    """Holds ``module``; reproduces the ``TensorDictModule`` state_dict prefix."""

    def __init__(self, module):
        super().__init__()
        self.module = module


class _Seq(nn.Module):
    """Holds ``module`` as a ModuleList; reproduces the ``TensorDictSequential`` prefix."""

    def __init__(self, *modules):
        super().__init__()
        self.module = nn.ModuleList(modules)


def make_mlp(in_dim, num_units):
    layers = []
    for n in num_units:
        layers += [nn.Linear(in_dim, n), nn.LeakyReLU(), nn.LayerNorm(n)]
        in_dim = n
    return nn.Sequential(*layers)


class BetaActor(nn.Module):
    def __init__(self, in_dim, action_dim):
        super().__init__()
        self.alpha_layer = nn.Linear(in_dim, action_dim)
        self.beta_layer = nn.Linear(in_dim, action_dim)
        self.alpha_softplus = nn.Softplus()
        self.beta_softplus = nn.Softplus()

    def forward(self, features):
        alpha = 1. + self.alpha_softplus(self.alpha_layer(features)) + 1e-6
        beta = 1. + self.beta_softplus(self.beta_layer(features)) + 1e-6
        return alpha, beta


class PPO(nn.Module):
    """Actor-critic policy used by the deployment node.

    Args:
        cfg: the ``algo`` config node (needs ``feature_extractor.dyn_obs_num``
            and ``actor.action_limit``).
        device: torch device.
        lidar_shape: (channels, horizontal beams, vertical beams) of the LiDAR image.
        state_dim: size of the drone state vector.
        dyn_obs_dim: number of features per dynamic obstacle.
        action_dim: size of the action.
    """

    def __init__(self, cfg, device, lidar_shape=(1, 36, 4), state_dim=8,
                 dyn_obs_dim=10, action_dim=3):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device(device)
        self.action_dim = action_dim
        self.lidar_shape = tuple(lidar_shape)
        self.state_dim = state_dim
        self.dyn_obs_num = cfg.feature_extractor.dyn_obs_num
        self.dyn_obs_dim = dyn_obs_dim

        # LiDAR feature extractor. The flattened size depends on the number of
        # beams, so it is measured with a dummy forward pass.
        conv = nn.Sequential(
            nn.Conv2d(lidar_shape[0], 4, kernel_size=(5, 3), padding=(2, 1)), nn.ELU(),
            nn.Conv2d(4, 16, kernel_size=(5, 3), stride=(2, 1), padding=(2, 1)), nn.ELU(),
            nn.Conv2d(16, 16, kernel_size=(5, 3), stride=(2, 2), padding=(2, 1)), nn.ELU(),
            nn.Flatten(1),
        )
        with torch.no_grad():
            flat_dim = conv(torch.zeros(1, *lidar_shape)).shape[-1]
        cnn = nn.Sequential(*conv, nn.Linear(flat_dim, 128), nn.LayerNorm(128))

        # Dynamic obstacle feature extractor.
        dyn = nn.Sequential(
            nn.Flatten(1),
            make_mlp(self.dyn_obs_num * dyn_obs_dim, [128, 64]),
        )

        self.feature_extractor = _Seq(
            _Wrap(cnn),
            _Wrap(dyn),
            nn.Identity(),  # keeps the index of the (parameter-free) concat step
            _Wrap(make_mlp(128 + state_dim + 64, [256, 256])),
        )
        self.actor = _Seq(_Wrap(BetaActor(256, action_dim)))
        self.critic = _Wrap(nn.Linear(256, 1))
        # Not used for inference, but part of the checkpoint.
        self.value_norm = ValueNorm(1)
        self.gae = GAE(0.99, 0.95)

        self.to(self.device)

    def features(self, obs):
        fe = self.feature_extractor.module
        cnn_feature = fe[0].module(obs["lidar"])
        dyn_feature = fe[1].module(obs["dynamic_obstacle"])
        # The torchrl CatTensors sorted its keys, so the trained network sees
        # [lidar, dynamic obstacles, state], not the order it was written in.
        x = torch.cat([cnn_feature, dyn_feature, obs["state"]], dim=-1)
        return fe[3].module(x)

    @torch.no_grad()
    def forward(self, obs, deterministic=True):
        """Run the policy.

        Args:
            obs: dict with ``state`` (N, 8), ``lidar`` (N, 1, H, V),
                ``dynamic_obstacle`` (N, 1, K, 10) and ``direction`` (3,) or (N, 3).
            deterministic: use the Beta mean (default) instead of sampling.

        Returns:
            dict with ``action_normalized`` (N, 3) in [0, 1] (goal frame),
            ``action`` (world-frame velocity scaled by ``actor.action_limit``)
            and ``state_value``.
        """
        feature = self.features(obs)
        alpha, beta = self.actor.module[0].module(feature)
        dist = torch.distributions.Beta(alpha, beta)
        action_normalized = dist.mean if deterministic else dist.sample()

        limit = self.cfg.actor.action_limit
        action_local = 2 * action_normalized * limit - limit
        return {
            "action_normalized": action_normalized,
            "action": vec_to_world(action_local, obs["direction"]),
            "state_value": self.critic.module(feature),
        }
