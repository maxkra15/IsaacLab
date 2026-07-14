# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""IsaacTeleop retargeting pipelines for the waterhose task.

These follow the framework's idiomatic conventions (retargeters keyed by the
device source constants ``ControllersSource.RIGHT`` / ``HandsSource.RIGHT``,
matching the reference manipulation teleop tasks). The previous, known-working
variants are preserved verbatim in :mod:`.teleop_pipelines_legacy`; if a change
here ever regresses the live XR session, switch the imports in
:mod:`.waterhose_env_cfg` back to that module to restore the old behavior.
"""

from __future__ import annotations


def build_waterhose_bimanual_teleop_pipeline():
    """Build the 15D Apple Vision Pro pipeline for both RBY1 wrists and the right gripper.

    Absolute wrist poses preserve the tracked orientation exactly, including all three rotation
    axes. ``IsaacTeleopCfg.target_frame_prim_path`` rebases these poses into the robot root frame
    before they reach the Newton IK action. The output order is right wrist, left wrist, then the
    right-hand pinch command.
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

    gripper = GripperRetargeter(GripperRetargeterConfig(hand_side="right"), name="gripper")
    gripper_output = gripper.connect(
        {
            HandsSource.RIGHT: hands.output(HandsSource.RIGHT),
            ControllersSource.RIGHT: controllers.output(ControllersSource.RIGHT),
        }
    ).output("gripper_command")

    right_elements = ["r_pos_x", "r_pos_y", "r_pos_z", "r_quat_x", "r_quat_y", "r_quat_z", "r_quat_w"]
    left_elements = ["l_pos_x", "l_pos_y", "l_pos_z", "l_quat_x", "l_quat_y", "l_quat_z", "l_quat_w"]
    gripper_elements = ["gripper"]
    reorderer = TensorReorderer(
        input_config={
            "right_ee_pose": right_elements,
            "left_ee_pose": left_elements,
            "gripper": gripper_elements,
        },
        output_order=right_elements + left_elements + gripper_elements,
        name="action_reorderer",
        input_types={"right_ee_pose": "array", "left_ee_pose": "array", "gripper": "scalar"},
    )
    output = reorderer.connect(
        {
            "right_ee_pose": right_output,
            "left_ee_pose": left_output,
            "gripper": gripper_output,
        }
    )

    pipeline = OutputCombiner({"action": output.output("output")})
    return pipeline, [right_se3, left_se3]


def build_waterhose_teleop_pipeline():
    """Build the IsaacTeleop pipeline for the absolute Waterhose IK action space.

    Currently unused: the registered teleop task drives the *relative* action space via
    :func:`build_waterhose_relative_teleop_pipeline`. This absolute builder is retained for the
    absolute Newton-IK action variant (:class:`WaterhoseNewtonIkActionsCfg`) should a teleop task
    bind to it.

    The end-effector pose is driven from the right HAND wrist so the pipeline works under Apple
    Vision Pro (which streams hand tracking, not controllers). The gripper is wired to both the hand
    and controller sources, so the same pipeline also works with Quest/Pico controllers: the
    gripper retargeter prefers the controller trigger when present and falls back to the hand pinch.
    This mirrors the framework's hand-tracking single-end-effector teleop pipelines.
    """

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

    controllers = ControllersSource(name="controllers")
    hands = HandsSource(name="hands")
    transform_input = ValueInput("world_T_anchor", TransformMatrix())
    transformed_hands = hands.transformed(transform_input.output(ValueInput.VALUE))

    # End-effector pose from the right hand wrist. ``use_wrist_position``/``use_wrist_rotation`` make
    # the gripper follow the hand's position and orientation; the target offset keeps the gripper's
    # side grip aligned. Tune ``zero_out_xy_rotation``/``target_offset_*`` for the desired feel.
    se3_cfg = Se3RetargeterConfig(
        input_device=HandsSource.RIGHT,
        zero_out_xy_rotation=False,
        use_wrist_rotation=True,
        use_wrist_position=True,
        target_offset_roll=90.0,
        target_offset_pitch=0.0,
        target_offset_yaw=0.0,
    )
    se3 = Se3AbsRetargeter(se3_cfg, name="ee_pose")
    connected_se3 = se3.connect(
        {
            HandsSource.RIGHT: transformed_hands.output(HandsSource.RIGHT),
        }
    )

    gripper_cfg = GripperRetargeterConfig(hand_side="right")
    gripper = GripperRetargeter(gripper_cfg, name="gripper")
    connected_gripper = gripper.connect(
        {
            HandsSource.RIGHT: hands.output(HandsSource.RIGHT),
            ControllersSource.RIGHT: controllers.output(ControllersSource.RIGHT),
        }
    )

    ee_pose_elements = ["pos_x", "pos_y", "pos_z", "quat_x", "quat_y", "quat_z", "quat_w"]
    gripper_elements = ["gripper_value"]
    reorderer = TensorReorderer(
        input_config={
            "ee_pose": ee_pose_elements,
            "gripper_command": gripper_elements,
        },
        output_order=ee_pose_elements + gripper_elements,
        name="action_reorderer",
        input_types={"ee_pose": "array", "gripper_command": "scalar"},
    )
    connected_reorderer = reorderer.connect(
        {
            "ee_pose": connected_se3.output("ee_pose"),
            "gripper_command": connected_gripper.output("gripper_command"),
        }
    )

    return OutputCombiner({"action": connected_reorderer.output("output")})


def build_waterhose_relative_teleop_pipeline():
    """Build the IsaacTeleop pipeline for the relative Waterhose IK teleop action space."""

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
        zero_out_xy_rotation=False,
        use_wrist_rotation=True,
        use_wrist_position=True,
        delta_pos_scale_factor=15.0,
        # Wrist rotation is a per-frame delta angle, unlike SpaceMouse's normalized puck deflection.
        # Keep enough gain for deliberate AVP wrist roll to visibly twist the gripper.
        delta_rot_scale_factor=2.0,
        alpha_pos=0.5,
        alpha_rot=0.5,
    )
    se3 = Se3RelRetargeter(se3_cfg, name="ee_delta")
    ee_delta_output = se3.connect({HandsSource.RIGHT: transformed_hands.output(HandsSource.RIGHT)}).output("ee_delta")

    gripper_cfg = GripperRetargeterConfig(hand_side="right")
    gripper = GripperRetargeter(gripper_cfg, name="gripper")
    # Key the gripper inputs by the device source constants (idiomatic; matches the
    # reference manipulation teleop tasks) rather than free-form alias strings.
    connected_gripper = gripper.connect(
        {
            HandsSource.RIGHT: hands.output(HandsSource.RIGHT),
            ControllersSource.RIGHT: controllers.output(ControllersSource.RIGHT),
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
            "ee_delta": ee_delta_output,
            "gripper": connected_gripper.output("gripper_command"),
        }
    )

    return OutputCombiner({"action": connected_reorderer.output("output")})
