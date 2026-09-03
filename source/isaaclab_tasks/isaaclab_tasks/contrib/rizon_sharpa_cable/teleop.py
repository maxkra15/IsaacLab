# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Right-hand Apple Vision Pro retargeting for the Sharpa task."""

from __future__ import annotations

import numpy as np

SHARPA_THUMB_RETARGETING_GAINS = {
    "right_thumb_CMC_FE": 1.55,
    "right_thumb_CMC_AA": 1.00,
    "right_thumb_MCP_FE": 1.55,
    "right_thumb_MCP_AA": 1.00,
    "right_thumb_IP": 1.75,
}


def apply_sharpa_thumb_retargeting_gain(
    joint_positions: np.ndarray,
    joint_names: list[str] | tuple[str, ...],
    joint_limits: list[tuple[float, float]] | tuple[tuple[float, float], ...],
) -> np.ndarray:
    """Amplify Sharpa thumb articulation and clamp every result to its joint limit [rad]."""
    if len(joint_positions) != len(joint_names) or len(joint_names) != len(joint_limits):
        raise ValueError("Sharpa thumb remapping requires one position, name, and limit per joint.")
    remapped = np.asarray(joint_positions, dtype=np.float64).copy()
    for index, (name, limits) in enumerate(zip(joint_names, joint_limits, strict=True)):
        gain = SHARPA_THUMB_RETARGETING_GAINS.get(name, 1.0)
        remapped[index] = np.clip(remapped[index] * gain, limits[0], limits[1])
    return remapped


def build_rizon_sharpa_teleop_pipeline():
    """Build an absolute right-wrist pose plus 22-DoF hand action pipeline."""
    from isaacteleop.retargeters import (
        DexHandRetargeter,
        DexHandRetargeterConfig,
        Se3AbsRetargeter,
        Se3RetargeterConfig,
        TensorReorderer,
    )
    from isaacteleop.retargeting_engine.deviceio_source_nodes import HandsSource
    from isaacteleop.retargeting_engine.interface import OutputCombiner, ValueInput
    from isaacteleop.retargeting_engine.tensor_types import TransformMatrix

    from .robot_asset import RIZON_SHARPA_RIGHT_HAND_JOINT_NAMES, RIZON_SHARPA_RIGHT_HAND_LIMITS_RAD
    from .sharpa_hand_retargeting import (
        SHARPA_HANDTRACKING_TO_BASELINK,
        SHARPA_OPENXR_TO_CANONICAL_PALM_RPY_DEG,
        sharpa_dexpilot_config_path,
        sharpa_dexpilot_urdf_path,
    )

    class DropoutAwareSe3AbsRetargeter(Se3AbsRetargeter):
        """Emit the clutch sentinel on tracking loss instead of a stale pose."""

        _INVALID_POSE = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)

        def _compute_fn(self, inputs, outputs, context) -> None:
            if inputs[self._config.input_device].is_none:
                self._last_pose = self._INVALID_POSE.copy()
                outputs["ee_pose"][0] = self._last_pose
                return
            super()._compute_fn(inputs, outputs, context)

    class SharpaThumbDexHandRetargeter(DexHandRetargeter):
        """Keep DexPilot's independent fingers while restoring full thumb travel."""

        def __init__(self, config, name: str) -> None:
            super().__init__(config, name)
            limits_by_name = dict(
                zip(RIZON_SHARPA_RIGHT_HAND_JOINT_NAMES, RIZON_SHARPA_RIGHT_HAND_LIMITS_RAD, strict=True)
            )
            self._output_joint_limits = tuple(limits_by_name[joint_name] for joint_name in self._dof_names)

        def _compute_hand(self, poses):
            joint_positions = super()._compute_hand(poses)
            return apply_sharpa_thumb_retargeting_gain(
                joint_positions,
                tuple(self._dof_names),
                self._output_joint_limits,
            )

    hands = HandsSource(name="hands")
    transform_input = ValueInput("world_T_anchor", TransformMatrix())
    transformed_hands = hands.transformed(transform_input.output(ValueInput.VALUE))

    # Pose is absolute in the XR anchor frame, using the exact OpenXR wrist ->
    # Fabrics-Sim canonical palm transform derived from NVIDIA's Sharpa mapping
    # and r_palm_ctrl joint. Tracking dropout is converted to a hold sentinel.
    palm = DropoutAwareSe3AbsRetargeter(
        Se3RetargeterConfig(
            input_device=HandsSource.RIGHT,
            zero_out_xy_rotation=False,
            use_wrist_rotation=True,
            use_wrist_position=True,
            target_offset_roll=SHARPA_OPENXR_TO_CANONICAL_PALM_RPY_DEG[0],
            target_offset_pitch=SHARPA_OPENXR_TO_CANONICAL_PALM_RPY_DEG[1],
            target_offset_yaw=SHARPA_OPENXR_TO_CANONICAL_PALM_RPY_DEG[2],
        ),
        name="right_palm",
    )
    palm_output = palm.connect({HandsSource.RIGHT: transformed_hands.output(HandsSource.RIGHT)}).output("ee_pose")
    # NVIDIA's Sharpa reference uses DexPilot on the raw OpenXR hand. It
    # recenters the skeleton at the wrist and owns the exact tracker-to-MANO
    # basis change internally. Feeding world/root-transformed joints here
    # double-applies frame semantics and reverses several visible finger axes.
    hand = SharpaThumbDexHandRetargeter(
        DexHandRetargeterConfig(
            hand_retargeting_config=str(sharpa_dexpilot_config_path()),
            hand_urdf=str(sharpa_dexpilot_urdf_path()),
            hand_joint_names=list(RIZON_SHARPA_RIGHT_HAND_JOINT_NAMES),
            handtracking_to_baselink_frame_transform=SHARPA_HANDTRACKING_TO_BASELINK,
            hand_side="right",
        ),
        name="right_sharpa_dexpilot",
    )
    hand_output = hand.connect({HandsSource.RIGHT: hands.output(HandsSource.RIGHT)}).output("hand_joints")

    pose_elements = ["pos_x", "pos_y", "pos_z", "quat_x", "quat_y", "quat_z", "quat_w"]
    reorderer = TensorReorderer(
        input_config={
            "right_palm": pose_elements,
            "right_hand": list(RIZON_SHARPA_RIGHT_HAND_JOINT_NAMES),
        },
        output_order=pose_elements + list(RIZON_SHARPA_RIGHT_HAND_JOINT_NAMES),
        name="rizon_sharpa_action",
        input_types={"right_palm": "array", "right_hand": "scalar"},
    )
    output = reorderer.connect({"right_palm": palm_output, "right_hand": hand_output})
    return OutputCombiner({"action": output.output("output")})


__all__ = [
    "SHARPA_THUMB_RETARGETING_GAINS",
    "apply_sharpa_thumb_retargeting_gain",
    "build_rizon_sharpa_teleop_pipeline",
]
