# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PPO configuration for full randomized Franka RJ45 pick-and-insert training."""

from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from .rsl_rl_ppo_cfg import RJ45GaussianBernoulliDistributionCfg


@configclass
class FrankaRJ45PickInsertPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Long-horizon policy with continuous arm and true binary-gripper exploration."""

    num_steps_per_env = 96
    max_iterations = 8000
    init_at_random_ep_len = False
    save_interval = 25
    clip_actions = 1.0
    logger = "tensorboard"
    obs_groups = {"actor": ["policy"], "critic": ["policy", "privileged"]}
    experiment_name = "franka_rj45_pick_insert"
    run_name = "six_stage_randomized"
    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RJ45GaussianBernoulliDistributionCfg(init_std=0.45),
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
        entropy_coef=1.0e-3,
        num_learning_epochs=5,
        num_mini_batches=8,
        learning_rate=1.0e-4,
        schedule="fixed",
        gamma=0.99 ** (1.0 / 3.0),
        lam=0.95 ** (1.0 / 3.0),
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


__all__ = ["FrankaRJ45PickInsertPPORunnerCfg"]
