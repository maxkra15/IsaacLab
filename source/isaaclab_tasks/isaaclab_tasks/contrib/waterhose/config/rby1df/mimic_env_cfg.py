# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RBY1DF waterhose task using direct bimanual actions for Isaac Lab Mimic."""

from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.utils.configclass import configclass

from ...waterhose_env_cfg import (
    TerminationsCfg,
    WaterhoseNewtonDirectBimanualActionsCfg,
    WaterhoseProxyIkEnvCfg,
)


@configclass
class WaterhoseMimicEnvCfg(WaterhoseProxyIkEnvCfg, MimicEnvCfg):
    """Mimic configuration for the direct 20D RBY1DF action interface."""

    actions: WaterhoseNewtonDirectBimanualActionsCfg = WaterhoseNewtonDirectBimanualActionsCfg()
    annotation_replay_action_key: str = "processed_actions"
    annotation_reset_sim_buffer_each_episode: bool = False
    skillgen_unsupported_reason: str = (
        "the task uses bimanual Newton IK targets and deformable cable contact, but has no task-specific "
        "cuRobo planner or SkillGen start-signal annotations. Run standard MimicGen without --use_skillgen."
    )

    def __post_init__(self) -> None:
        super().__post_init__()

        # The scripted demo config disables termination so it can inspect retention after insertion.
        # Mimic needs the real task-success predicate to classify generated episodes.
        self.terminations = TerminationsCfg()

        # This is a replay/generation task, not an XR input task.
        self.xr = None
        self.isaac_teleop = None

        self.datagen_config.name = "rby1_waterhose_bimanual_direct_D0"
        self.datagen_config.generation_guarantee = True
        self.datagen_config.generation_keep_failed = False
        self.datagen_config.generation_num_trials = 10
        self.datagen_config.generation_select_src_per_subtask = False
        self.datagen_config.generation_select_src_per_arm = False
        self.datagen_config.generation_transform_first_robot_pose = False
        self.datagen_config.generation_interpolate_from_last_target_pose = True
        self.datagen_config.max_num_failures = 25
        self.datagen_config.seed = 1

        # Start with one complete, synchronized trajectory per arm. Once robust phase signals are
        # available, grasp and insertion can be split into finer object-centric subtasks.
        final_subtask = SubTaskConfig(
            object_ref="socket",
            subtask_term_signal=None,
            selection_strategy="nearest_neighbor_object",
            selection_strategy_kwargs={"nn_k": 3},
            # Establish a replay-correct baseline before perturbing a socket
            # corridor whose radial success tolerance is only 1 mm.
            action_noise=0.0,
            num_interpolation_steps=3,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        self.subtask_configs = {
            "right": [final_subtask],
            "left": [
                SubTaskConfig(
                    object_ref="socket",
                    subtask_term_signal=None,
                    selection_strategy="nearest_neighbor_object",
                    selection_strategy_kwargs={"nn_k": 3},
                    action_noise=0.0,
                    num_interpolation_steps=3,
                    num_fixed_steps=0,
                    apply_noise_during_interpolation=False,
                )
            ],
        }
