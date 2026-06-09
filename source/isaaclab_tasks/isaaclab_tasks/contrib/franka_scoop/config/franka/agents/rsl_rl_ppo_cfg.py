# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class FrankaScoopPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 32
    max_iterations = 3000
    save_interval = 50
    experiment_name = "franka_scoop_pile"
    run_name = "scoop_pile"
    # Weights & Biases logging. rsl_rl's WandbSummaryWriter auto-logs all scalars (Episode_Reward/*,
    # Curriculum/stage/*, Episode_Termination/*, losses), the train+env config (wandb.config), and saves
    # model checkpoints. Entity comes from the WANDB_USERNAME env var if set, else the logged-in default
    # (~/.netrc). The wandb run name is "<timestamp>_<run_name>". Set WANDB_MODE=offline to log without network.
    logger = "wandb"
    wandb_project = "franka-scoop-mpm"
    # asymmetric actor-critic: actor sees realistic obs (heightfield + proprio);
    # critic additionally sees privileged sim state (particle counts).
    obs_groups = {"actor": ["policy"], "critic": ["policy", "privileged"]}
    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
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
        entropy_coef=0.006,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
