# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""IsaacTeleop pipeline configuration for the waterhose robot demo."""

from __future__ import annotations

import logging
from typing import Any

from isaaclab_teleop.xr_cfg import XrCfg


logger = logging.getLogger(__name__)


def build_waterhose_teleop_pipeline():
    """Build an IsaacTeleop retargeting pipeline for the waterhose task.

    The task action is a relative 7D command:
    ``[dx, dy, dz, droll, dpitch, dyaw, gripper]``.  The pipeline uses the
    right hand for relative end-effector motion and right-hand pinch/controller
    trigger for the binary gripper command.
    """

    from isaacteleop.retargeters import (
        GripperRetargeter,
        GripperRetargeterConfig,
        Se3RelRetargeter,
        Se3RetargeterConfig,
        TensorReorderer,
    )
    from isaacteleop.retargeting_engine.deviceio_source_nodes import ControllersSource, HandsSource
    from isaacteleop.retargeting_engine.interface import OutputCombiner, ValueInput
    from isaacteleop.retargeting_engine.tensor_types import TransformMatrix

    controllers = ControllersSource(name="controllers")
    hands = HandsSource(name="hands")

    transform_input = ValueInput("world_T_anchor", TransformMatrix())
    transformed_hands = hands.transformed(transform_input.output(ValueInput.VALUE))

    se3_cfg = Se3RetargeterConfig(
        input_device=HandsSource.RIGHT,
        zero_out_xy_rotation=True,
        use_wrist_rotation=True,
        use_wrist_position=True,
        delta_pos_scale_factor=15.0,
        delta_rot_scale_factor=2.0,
        alpha_pos=0.5,
        alpha_rot=0.5,
    )
    se3 = Se3RelRetargeter(se3_cfg, name="ee_delta")
    connected_se3 = se3.connect({HandsSource.RIGHT: transformed_hands.output(HandsSource.RIGHT)})

    gripper_cfg = GripperRetargeterConfig(hand_side="right")
    gripper = GripperRetargeter(gripper_cfg, name="gripper")
    connected_gripper = gripper.connect(
        {
            "hand_right": hands.output(HandsSource.RIGHT),
            "controller_right": controllers.output(ControllersSource.RIGHT),
        }
    )

    delta_elements = ["dx", "dy", "dz", "droll", "dpitch", "dyaw"]
    gripper_elements = ["gripper"]

    reorderer = TensorReorderer(
        input_config={
            "ee_delta": delta_elements,
            "gripper": gripper_elements,
        },
        output_order=delta_elements + gripper_elements,
        name="action_reorderer",
        input_types={
            "ee_delta": "array",
            "gripper": "scalar",
        },
    )
    connected_reorderer = reorderer.connect(
        {
            "ee_delta": connected_se3.output("ee_delta"),
            "gripper": connected_gripper.output("gripper_command"),
        }
    )

    return OutputCombiner({"action": connected_reorderer.output("output")})


def make_waterhose_isaac_teleop_cfg(sim_device: str, xr_cfg: XrCfg) -> Any | None:
    """Create the IsaacTeleop config when the optional runtime is installed."""

    try:
        import isaacteleop  # noqa: F401
        from isaaclab_teleop import IsaacTeleopCfg
    except ImportError as exc:
        logger.debug("IsaacTeleop runtime is unavailable; XR teleoperation is disabled: %s", exc)
        return None

    return IsaacTeleopCfg(
        pipeline_builder=build_waterhose_teleop_pipeline,
        sim_device=sim_device,
        xr_cfg=xr_cfg,
        teleoperation_active_default=False,
        app_name="WaterhoseRobotDemoTeleop",
    )

