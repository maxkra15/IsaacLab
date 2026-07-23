# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bimanual IsaacTeleop retargeting for the waterhose task."""

from __future__ import annotations


def build_waterhose_bimanual_teleop_pipeline():
    """Build the 16D Apple Vision Pro pipeline for both RBY1 wrists and grippers.

    Absolute wrist poses preserve the tracked orientation exactly, including all three rotation
    axes. ``IsaacTeleopCfg.target_frame_prim_path`` rebases these poses into the robot root frame
    before they reach the Newton IK action. The output order is right wrist, left wrist, then the
    independent binary right- and left-hand pinch commands.
    """

    import numpy as np
    from isaacteleop.retargeters import (
        GripperRetargeter,
        GripperRetargeterConfig,
        Se3AbsRetargeter,
        Se3RetargeterConfig,
        TensorReorderer,
    )
    from isaacteleop.retargeting_engine.deviceio_source_nodes import ControllersSource, HandsSource
    from isaacteleop.retargeting_engine.interface import OutputCombiner, ValueInput
    from isaacteleop.retargeting_engine.tensor_types import TransformMatrix

    class WaterhoseSe3AbsRetargeter(Se3AbsRetargeter):
        """Emit an invalid sentinel on dropout so the action can re-clutch safely."""

        _INVALID_POSE = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)

        def _compute_fn(self, inputs, outputs, context) -> None:
            if inputs[self._config.input_device].is_none:
                self._last_pose = self._INVALID_POSE.copy()
                outputs["ee_pose"][0] = self._last_pose
                return
            super()._compute_fn(inputs, outputs, context)

    controllers = ControllersSource(name="controllers")
    hands = HandsSource(name="hands")
    transform_input = ValueInput("world_T_anchor", TransformMatrix())
    transformed_hands = hands.transformed(transform_input.output(ValueInput.VALUE))

    # The action term calibrates each first valid pose onto the corresponding
    # robot wrist, so no guessed tracker-to-tool rotation belongs here.
    right_cfg = Se3RetargeterConfig(
        input_device=HandsSource.RIGHT,
        zero_out_xy_rotation=False,
        use_wrist_rotation=True,
        use_wrist_position=True,
        target_offset_roll=0.0,
        target_offset_pitch=0.0,
        target_offset_yaw=0.0,
    )
    right_se3 = WaterhoseSe3AbsRetargeter(right_cfg, name="right_ee_pose")
    right_output = right_se3.connect({HandsSource.RIGHT: transformed_hands.output(HandsSource.RIGHT)}).output("ee_pose")

    left_cfg = Se3RetargeterConfig(
        input_device=HandsSource.LEFT,
        zero_out_xy_rotation=False,
        use_wrist_rotation=True,
        use_wrist_position=True,
        target_offset_roll=0.0,
        target_offset_pitch=0.0,
        target_offset_yaw=0.0,
    )
    left_se3 = WaterhoseSe3AbsRetargeter(left_cfg, name="left_ee_pose")
    left_output = left_se3.connect({HandsSource.LEFT: transformed_hands.output(HandsSource.LEFT)}).output("ee_pose")

    right_gripper = GripperRetargeter(GripperRetargeterConfig(hand_side="right"), name="right_gripper")
    right_gripper_output = right_gripper.connect(
        {
            HandsSource.RIGHT: hands.output(HandsSource.RIGHT),
            ControllersSource.RIGHT: controllers.output(ControllersSource.RIGHT),
        }
    ).output("gripper_command")
    left_gripper = GripperRetargeter(GripperRetargeterConfig(hand_side="left"), name="left_gripper")
    left_gripper_output = left_gripper.connect(
        {
            HandsSource.LEFT: hands.output(HandsSource.LEFT),
            ControllersSource.LEFT: controllers.output(ControllersSource.LEFT),
        }
    ).output("gripper_command")

    right_elements = ["r_pos_x", "r_pos_y", "r_pos_z", "r_quat_x", "r_quat_y", "r_quat_z", "r_quat_w"]
    left_elements = ["l_pos_x", "l_pos_y", "l_pos_z", "l_quat_x", "l_quat_y", "l_quat_z", "l_quat_w"]
    right_gripper_elements = ["right_gripper"]
    left_gripper_elements = ["left_gripper"]
    reorderer = TensorReorderer(
        input_config={
            "right_ee_pose": right_elements,
            "left_ee_pose": left_elements,
            "right_gripper": right_gripper_elements,
            "left_gripper": left_gripper_elements,
        },
        output_order=right_elements + left_elements + right_gripper_elements + left_gripper_elements,
        name="action_reorderer",
        input_types={
            "right_ee_pose": "array",
            "left_ee_pose": "array",
            "right_gripper": "scalar",
            "left_gripper": "scalar",
        },
    )
    output = reorderer.connect(
        {
            "right_ee_pose": right_output,
            "left_ee_pose": left_output,
            "right_gripper": right_gripper_output,
            "left_gripper": left_gripper_output,
        }
    )

    pipeline = OutputCombiner({"action": output.output("output")})
    return pipeline, [right_se3, left_se3]
