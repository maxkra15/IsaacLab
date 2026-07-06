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
    save_interval = 50
    experiment_name = "franka_pour"
    run_name = "pour"
    clip_actions = 1.0
    # W&B uses the active CLI login (or WANDB_API_KEY); set WANDB_MODE=offline
    # to keep run data local without changing the task config.
    logger = "wandb"
    wandb_project = "franka-pour-mpm"
    obs_groups = {"actor": ["policy"], "critic": ["policy", "privileged"]}
    actor = RslRlMLPModelCfg(
        hidden_dims=[256, 128, 64],
        activation="elu",
        # Curriculum stages intentionally change reset offsets and geometry. Running empirical
        # normalization makes the first unseen stage arbitrarily out-of-distribution because
        # stage-constant features have zero variance. Observations are physically scaled in the
        # environment instead, matching the standard Franka Lift policy setup.
        obs_normalization=False,
        # The action terms already normalize phase, joint residual, and gripper commands into
        # contact-safe physical ranges. Use Isaac Lab's standard state-independent Gaussian with
        # modest exploration rather than maintaining a task-specific distribution implementation.
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.1, std_type="log"),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[256, 128, 64],
        activation="elu",
        obs_normalization=False,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        # Retain modest exploration when the curriculum first exposes unseen reset geometry.
        entropy_coef=1.0e-3,
        # Match Isaac Lab's Franka manipulation defaults. The earlier one-fifth-rate, two-epoch
        # update left the actor statistically unchanged after millions of transitions and could
        # not adapt the nominal trajectory to randomized cup poses.
        num_learning_epochs=5,
        num_mini_batches=4,
        # The state-independent, low-noise residual policy produces very small KL even after a
        # meaningful update. RSL-RL's adaptive schedule therefore ramps to destructive rates;
        # hold the standard Franka base rate fixed instead.
        learning_rate=1.0e-4,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
