# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL PPO configuration for one-ball KUKA-Allegro juggling."""

import torch
from rsl_rl.modules.distribution import GaussianDistribution
from torch.distributions import Normal

from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


class JuggleGaussianDistribution(GaussianDistribution):
    """Bounded Gaussian exploration with separate arm and hand initial scales."""

    def __init__(
        self,
        output_dim: int,
        init_std: float = 0.12,
        hand_init_std: float = 0.15,
        arm_action_dim: int = 7,
        std_range: tuple[float, float] = (0.06, 0.50),
        std_type: str = "scalar",
        **kwargs,
    ) -> None:
        if not 0 < arm_action_dim < output_dim:
            raise ValueError("arm_action_dim must split the continuous arm and hand actions.")
        if len(std_range) != 2 or std_range[0] <= 0.0 or std_range[0] >= std_range[1]:
            raise ValueError("std_range must contain positive, increasing bounds.")
        if not std_range[0] < init_std < std_range[1] or not std_range[0] < hand_init_std < std_range[1]:
            raise ValueError("Initial arm and hand standard deviations must lie inside std_range.")
        if std_type != "scalar":
            raise ValueError("JuggleGaussianDistribution supports scalar standard deviations.")
        super().__init__(output_dim, init_std=init_std, std_range=std_range, std_type=std_type, **kwargs)
        self._bounded_std_range = (float(std_range[0]), float(std_range[1]))
        initial_std = torch.full_like(self.std_param, hand_init_std)
        initial_std[:arm_action_dim] = init_std
        initial_fraction = (initial_std - std_range[0]) / (std_range[1] - std_range[0])
        with torch.no_grad():
            self.std_param.copy_(torch.logit(initial_fraction))

    def update(self, mlp_output: torch.Tensor) -> None:
        """Update the diagonal Gaussian through differentiable standard-deviation bounds."""
        minimum_std, maximum_std = self._bounded_std_range
        std = minimum_std + (maximum_std - minimum_std) * torch.sigmoid(self.std_param)
        self._distribution = Normal(mlp_output, std)


@configclass
class JuggleGaussianDistributionCfg(RslRlMLPModelCfg.GaussianDistributionCfg):
    """Configuration for bounded 7-arm plus 16-hand exploration."""

    class_name: str = (
        "isaaclab_tasks.contrib.juggle.config.kuka_allegro.agents.rsl_rl_ppo_cfg:JuggleGaussianDistribution"
    )
    hand_init_std: float = 0.15
    arm_action_dim: int = 7
    std_range: tuple[float, float] = (0.06, 0.50)


@configclass
class KukaAllegroJugglePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO hyperparameters for the sparse, phase-reset juggling task."""

    num_steps_per_env = 32
    max_iterations = 10_000
    save_interval = 50
    init_at_random_ep_len = False
    empirical_normalization = False
    experiment_name = "kuka_allegro_juggle"
    wandb_project = "kuka_allegro_juggle"
    clip_actions = 1.0
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=JuggleGaussianDistributionCfg(init_std=0.12),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=1.0e-4,
        num_learning_epochs=5,
        num_mini_batches=8,
        learning_rate=5.0e-5,
        schedule="fixed",
        gamma=0.998,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
