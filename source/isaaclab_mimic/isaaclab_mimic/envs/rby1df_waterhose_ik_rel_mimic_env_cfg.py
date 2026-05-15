# Copyright (c) 2024-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.waterhose.waterhose_env_cfg import RBY1DFWaterhoseEnvCfg


@configclass
class RBY1DFWaterhoseIKRelMimicEnvCfg(RBY1DFWaterhoseEnvCfg, MimicEnvCfg):
    """Mimic configuration for the RBY1 waterhose task."""

    def __post_init__(self):
        super().__post_init__()
        self.datagen_config.name = "demo_src_waterhose_rby1df_task_D0"
        self.datagen_config.generation_guarantee = True
        self.datagen_config.generation_keep_failed = False
        self.datagen_config.generation_num_trials = 10
        self.datagen_config.generation_select_src_per_subtask = True
        self.datagen_config.generation_transform_first_robot_pose = False
        self.datagen_config.generation_interpolate_from_last_target_pose = True
        self.datagen_config.generation_relative = True
        self.datagen_config.max_num_failures = 25
        self.datagen_config.seed = 1

        self.subtask_configs["rby1df_right"] = [
            SubTaskConfig(
                object_ref="hose_plug",
                subtask_term_signal="approach",
                subtask_term_offset_range=(5, 10),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.02,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Approach the waterhose plug",
                next_subtask_description="Grasp the waterhose",
            ),
            SubTaskConfig(
                object_ref="hose_plug",
                subtask_term_signal="grasp",
                subtask_term_offset_range=(5, 10),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.02,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                next_subtask_description="Move the hose to the socket",
            ),
            SubTaskConfig(
                object_ref="socket",
                subtask_term_signal="align",
                subtask_term_offset_range=(5, 10),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.015,
                num_interpolation_steps=8,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                next_subtask_description="Insert the hose tip into the socket",
            ),
            SubTaskConfig(
                object_ref="socket",
                subtask_term_signal="insert",
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.01,
                num_interpolation_steps=8,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
            ),
        ]
