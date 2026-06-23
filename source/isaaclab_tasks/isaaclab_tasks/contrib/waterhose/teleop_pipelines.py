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


def build_waterhose_teleop_pipeline():
    """Build the IsaacTeleop pipeline for the absolute Waterhose IK action space.

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

    import numpy as np
    from isaacteleop.retargeters import (
        GripperRetargeter,
        GripperRetargeterConfig,
        Se3RelRetargeter,
        Se3RetargeterConfig,
        TensorReorderer,
    )
    from isaacteleop.retargeting_engine.deviceio_source_nodes import ControllersSource, HandsSource
    from isaacteleop.retargeting_engine.interface import BaseRetargeter, OutputCombiner, TensorGroupType, ValueInput
    from isaacteleop.retargeting_engine.tensor_types import DLDataType, NDArrayType, TransformMatrix

    class WaterhoseDeltaFrameRemapper(BaseRetargeter):
        """Adapt AVP wrist deltas to the waterhose relative IK action semantics."""

        def input_spec(self):
            return {
                "ee_delta": TensorGroupType(
                    "ee_delta",
                    [NDArrayType("delta", shape=(6,), dtype=DLDataType.FLOAT, dtype_bits=32)],
                )
            }

        def output_spec(self):
            return {
                "ee_delta": TensorGroupType(
                    "ee_delta",
                    [NDArrayType("delta", shape=(6,), dtype=DLDataType.FLOAT, dtype_bits=32)],
                )
            }

        def _compute_fn(self, inputs, outputs, context) -> None:
            # Keep only the first rotation-vector component (the twist the operator uses to
            # line the connector up with the bore) and zero the other two, so wrist wobble
            # does not fight the insertion. The mask is applied to the retargeter's anchor-
            # frame rotvec; the relative IK action then conjugates it into the end-effector
            # frame (see WaterhoseLocalFrameNewtonInverseKinematicsAction). Translation
            # (indices 0:3) passes through unchanged.
            delta = np.asarray(inputs["ee_delta"][0], dtype=np.float32).flatten()
            remapped = delta.copy()
            remapped[4] = 0.0
            remapped[5] = 0.0
            outputs["ee_delta"][0] = remapped

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
        delta_rot_scale_factor=2.0,
        alpha_pos=0.5,
        alpha_rot=0.5,
    )
    se3 = Se3RelRetargeter(se3_cfg, name="ee_delta")
    connected_se3 = se3.connect({HandsSource.RIGHT: transformed_hands.output(HandsSource.RIGHT)})
    delta_remapper = WaterhoseDeltaFrameRemapper(name="waterhose_delta_frame")
    connected_delta = delta_remapper.connect({"ee_delta": connected_se3.output("ee_delta")})

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
            "ee_delta": connected_delta.output("ee_delta"),
            "gripper": connected_gripper.output("gripper_command"),
        }
    )

    return OutputCombiner({"action": connected_reorderer.output("output")})
