# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sim-free frame-contract tests for waterhose scripted and AVP control."""

import pytest

from isaaclab_tasks.contrib.waterhose.geometry import (
    RIGHT_GRIPPER_EE_FRAME_POS,
    RIGHT_GRIPPER_EE_FRAME_QUAT_XYZW,
)
from isaaclab_tasks.contrib.waterhose.waterhose_env_cfg import (
    WaterhoseNewtonBimanualIkActionsCfg,
    WaterhoseNewtonIkActionsCfg,
)


def _pose_objective(actions_cfg, name: str):
    """Return one named pose objective from an action configuration."""

    return next(
        objective for objective in actions_cfg.arm_action.objectives if getattr(objective, "name", None) == name
    )


def test_bimanual_avp_ik_targets_gripper_base_frames_without_tool_offsets():
    """Tracked AVP wrists must command the corresponding robot wrist frames directly."""

    actions_cfg = WaterhoseNewtonBimanualIkActionsCfg()

    for side in ("right", "left"):
        objective = _pose_objective(actions_cfg, f"{side}_ee")
        assert objective.body_name == f"{side}_gripper_base"
        assert objective.body_offset_pos == (0.0, 0.0, 0.0)
        assert objective.body_offset_rot == (0.0, 0.0, 0.0, 1.0)


def test_scripted_insertion_keeps_the_right_gripper_contact_frame_offset():
    """The scripted insertion target must remain at the finger-pad contact frame."""

    actions_cfg = WaterhoseNewtonIkActionsCfg()
    objective = _pose_objective(actions_cfg, "right_ee")

    assert objective.body_name == "right_gripper_base"
    assert objective.body_offset_pos == RIGHT_GRIPPER_EE_FRAME_POS
    assert objective.body_offset_rot == RIGHT_GRIPPER_EE_FRAME_QUAT_XYZW
    assert objective.body_offset_pos != (0.0, 0.0, 0.0)
    assert objective.body_offset_rot != (0.0, 0.0, 0.0, 1.0)


def test_bimanual_avp_retargeters_do_not_apply_hardcoded_rotation_offsets():
    """Robot-neutral calibration, rather than fixed Euler angles, must align each wrist."""

    pytest.importorskip("isaacteleop")
    from isaaclab_tasks.contrib.waterhose.teleop_pipelines import build_waterhose_bimanual_teleop_pipeline

    _, retargeters = build_waterhose_bimanual_teleop_pipeline()

    assert len(retargeters) == 2
    for retargeter in retargeters:
        assert retargeter._config.target_offset_roll == 0.0
        assert retargeter._config.target_offset_pitch == 0.0
        assert retargeter._config.target_offset_yaw == 0.0
