# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Single-GPU experimental sparse-event SuperMimic configuration."""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlMLPModelCfg,
    RslRlSuperMimicActorCfg,
    RslRlSuperMimicAlgorithmCfg,
    RslRlSuperMimicRunnerCfg,
    RslRlSuperMimicTaskCfg,
    RslRlWorldModelCfg,
)


@configclass
class SuctionSuperMimicRunnerCfg(RslRlSuperMimicRunnerCfg):
    """Fit sparse event probabilities; evaluate success exclusively in native physics."""

    num_steps_per_env = 1
    max_iterations = 10000
    save_interval = 200
    experiment_name = "conveyor_suction_sparse_supermimic"
    clip_actions = None
    obs_groups = {
        "state": ["state"],
        "goal": ["supermimic_goal"],
        "context": ["supermimic_goal"],
        "labels": ["supermimic_labels"],
        "prev_action": ["supermimic_action"],
        "prev_prev_action": ["supermimic_previous_action"],
    }
    actor = RslRlSuperMimicActorCfg(
        class_name="isaaclab_tasks.contrib.conveyor_suction.supermimic:SuctionActor",
        encoder_hidden_dims=(256, 128),
        decoder_hidden_dims=(512, 256, 128),
    )
    world_model = RslRlWorldModelCfg(
        class_name="isaaclab_tasks.contrib.conveyor_suction.supermimic:SuctionWorldModel",
        hidden_dims=(512, 512, 512),
        label_hidden_dims=(256, 128),
    )
    critic = RslRlMLPModelCfg(hidden_dims=[256, 128], activation="elu", obs_normalization=False)
    algorithm = RslRlSuperMimicAlgorithmCfg(
        class_name="isaaclab_tasks.contrib.conveyor_suction.supermimic:EventMonitoredSuperMimic",
        task_cfg=RslRlSuperMimicTaskCfg(class_name="isaaclab_tasks.contrib.conveyor_suction.supermimic:SuctionTask"),
        replay_buffer_size=1_048_576,
        max_replay_memory_fraction=0.25,
        world_model_batch_size=2048,
        policy_batch_size=256,
        world_model_warmup_iterations=500,
        action_noise=0.1,
        gamma=0.999,
        gae_lambda=0.99,
        max_step_reward=10.0,
    )
