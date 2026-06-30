# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class FrankaPourPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 32
    max_iterations = 3000
    save_interval = 10
    experiment_name = "franka_pour"
    run_name = "pour"
    logger = "tensorboard"
    # Asymmetric actor-critic: the actor sees proprioception + EE/bowl poses; the critic additionally
    # sees privileged sim state (per-bowl particle fractions).
    obs_groups = {"actor": ["policy"], "critic": ["policy", "privileged"]}
    actor = RslRlMLPModelCfg(
        hidden_dims=[256, 256, 128],
        activation="elu",
        obs_normalization=True,
        # The 0.35 baseline commanded 3.5 cm random relative targets at 60 Hz and learned to flee
        # the cup. Eight-millimetre exploration preserves contact while still random-walking across
        # the full lift/carry workspace over an episode.
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.08),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[256, 256, 128],
        activation="elu",
        obs_normalization=True,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
