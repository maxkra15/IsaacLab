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
        # Relative joint increments and the continuous gripper command already map normalized
        # actions into bounded physical ranges. Use the standard state-independent Gaussian with
        # modest initial exploration around the contact-sensitive backward resets.
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
        # Match Isaac Lab's Franka manipulation update depth. The earlier one-fifth-rate,
        # two-epoch update left the actor statistically unchanged after millions of transitions.
        num_learning_epochs=5,
        num_mini_batches=4,
        # The contact-sensitive, low-noise policy produces very small KL early in the backward
        # curriculum. Keep the standard Franka base rate fixed so an adaptive schedule cannot
        # amplify it during those initially short episodes.
        learning_rate=1.0e-4,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class FrankaPourResetMixturePPORunnerCfg(FrankaPourPPORunnerCfg):
    """PPO runner calibrated for the stationary four-region reset mixture."""

    @configclass
    class ExplorationDistributionCfg(RslRlMLPModelCfg.HeteroscedasticGaussianDistributionCfg):
        """State-dependent exploration bounded for independently sampled action noise."""

        # RSL-RL samples this noise independently at every 30 Hz policy step. The task-side FIR
        # suppresses high-frequency chatter while retaining the exploration range that discovered
        # the grasp-and-pour transition in the proven 10 Hz run.
        std_range: tuple[float, float] = (0.05, 1.0)

    # Match OmniReset's 3.2-second physical PPO horizon while retaining 30 Hz control. Reach,
    # align, close, and lift transitions routinely cross the former 1.067-second rollout boundary.
    num_steps_per_env = 96
    save_interval = 25
    # Keep incompatible 7-action, history-stacked checkpoints separate from the earlier direct-
    # joint experiment so automatic checkpoint discovery cannot warm-start across semantics.
    experiment_name = "franka_pour_reset_mixture_diffik"
    run_name = "omnireset_diffik"
    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128, 64],
        activation="elu",
        # Every reset region is represented from the first rollout, so running statistics cannot
        # acquire the sequential-stage zero-variance bias that motivated disabling normalization
        # in the reverse curriculum.
        obs_normalization=True,
        # Mainline RSL-RL has no temporally correlated gSDE. Retain state-dependent exploration,
        # but use the standard Isaac Lab entropy scale below so independently sampled noise does
        # not grow merely because the environment clips actions.
        distribution_cfg=ExplorationDistributionCfg(
            # Mainline RSL-RL samples this head independently at every 30 Hz step, unlike
            # OmniReset's temporally correlated gSDE. Start near the proven policy's learned
            # 0.43 standard deviation instead of injecting 0.8-IID contact chatter.
            init_std=0.4,
            std_type="log",
        ),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128, 64],
        activation="elu",
        obs_normalization=True,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=1.0e-3,
        # Each rank already collects 196,608 transitions per rollout. One pass avoids repeatedly
        # fitting the same contact trajectories, which erased the transferred broad-reset skill.
        num_learning_epochs=1,
        # At 2,048 environments per rank, 24 splits retain 8,192 samples per optimizer minibatch
        # after restoring the three-times-longer physical rollout.
        num_mini_batches=24,
        learning_rate=1.0e-4,
        # Adaptive KL scheduling caused repeated learning-rate spikes in the 30 Hz run. Keep the
        # update scale fixed and preserve the earlier 10 Hz discount horizons per physical second.
        schedule="fixed",
        gamma=0.99 ** (1.0 / 3.0),
        lam=0.95 ** (1.0 / 3.0),
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
