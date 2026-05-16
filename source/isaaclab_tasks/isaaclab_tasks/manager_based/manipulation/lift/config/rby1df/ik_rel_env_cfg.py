# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.utils import configclass

from . import joint_pos_env_cfg
from .joint_pos_env_cfg import RBY1DF_TCP_OFFSET_POS, RBY1DF_TCP_OFFSET_ROT, RIGHT_ARM_JOINTS

##
# Pre-defined configs
##
from isaaclab_assets.robots.rby1df import RBY1DF_HIGH_PD_CFG  # isort: skip


@configclass
class RBY1DFCubeLiftEnvCfg(joint_pos_env_cfg.RBY1DFCubeLiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Use stiffer PD gains for task-space IK tracking.
        self.scene.robot = RBY1DF_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Relative DLS IK drives the right-arm TCP while the inherited gripper action closes on the cube.
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=RIGHT_ARM_JOINTS,
            body_name="right_arm_tool_flange",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
            scale=(0.04, 0.04, 0.04, 0.25, 0.25, 0.25),
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(
                pos=RBY1DF_TCP_OFFSET_POS,
                rot=RBY1DF_TCP_OFFSET_ROT,
            ),
        )


@configclass
class RBY1DFCubeLiftEnvCfg_PLAY(RBY1DFCubeLiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
