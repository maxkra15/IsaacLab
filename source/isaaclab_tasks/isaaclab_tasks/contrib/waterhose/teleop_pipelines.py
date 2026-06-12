# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""IsaacTeleop retargeting pipelines for the waterhose task."""

from __future__ import annotations


def build_waterhose_teleop_pipeline():
    """Build the IsaacTeleop pipeline for the absolute Waterhose IK action space."""

    from isaacteleop.retargeters import (
        GripperRetargeter,
        GripperRetargeterConfig,
        Se3AbsRetargeter,
        Se3RetargeterConfig,
        TensorReorderer,
    )
    from isaacteleop.retargeting_engine.deviceio_source_nodes import ControllersSource
    from isaacteleop.retargeting_engine.interface import OutputCombiner, ValueInput
    from isaacteleop.retargeting_engine.tensor_types import TransformMatrix

    controllers = ControllersSource(name="controllers")
    transform_input = ValueInput("world_T_anchor", TransformMatrix())
    transformed_controllers = controllers.transformed(transform_input.output(ValueInput.VALUE))

    se3_cfg = Se3RetargeterConfig(
        input_device=ControllersSource.RIGHT,
        zero_out_xy_rotation=False,
        use_wrist_rotation=False,
        use_wrist_position=False,
        target_offset_roll=90.0,
        target_offset_pitch=0.0,
        target_offset_yaw=0.0,
    )
    se3 = Se3AbsRetargeter(se3_cfg, name="ee_pose")
    connected_se3 = se3.connect(
        {
            ControllersSource.RIGHT: transformed_controllers.output(ControllersSource.RIGHT),
        }
    )

    gripper_cfg = GripperRetargeterConfig(hand_side="right")
    gripper = GripperRetargeter(gripper_cfg, name="gripper")
    connected_gripper = gripper.connect(
        {
            ControllersSource.RIGHT: transformed_controllers.output(ControllersSource.RIGHT),
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
            delta = np.asarray(inputs["ee_delta"][0], dtype=np.float32).flatten()
            remapped = delta.copy()
            remapped[3] = delta[3]
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
            "ee_delta": connected_delta.output("ee_delta"),
            "gripper": connected_gripper.output("gripper_command"),
        }
    )

    return OutputCombiner({"action": connected_reorderer.output("output")})
