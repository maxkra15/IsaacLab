# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for selecting native RJ45 apertures on a SimReady GB300 rack."""

from __future__ import annotations

import math
from typing import Literal

from isaaclab.utils.configclass import configclass

from .dual_rack_env_cfg import FrankaRJ45DualRackInsertEnvCfg
from .gb300_workcell import (
    GB300_DEFAULT_TARGET_TASK_TRANSLATION,
    GB300_ROBOT_POSITION_E,
    GB300_ROBOT_ROTATION_XYZW,
    GB300_STUDIO_FLOOR_HEIGHT_M,
    GB300_TARGET_TASK_TRANSLATIONS,
    GB300_TASK_ROTATION_XYZW,
    GB300_WORKCELL_CFG,
    GB300_WORKCELL_CONTRACT_VERSION,
    gb300_workcell_contract,
)
from .pick_insert_env_cfg import FrankaRJ45PickInsertEnvCfg, pick_insert_reset_dataset_task_contract


def _component_bounds(values: tuple[tuple[float, float, float], ...]) -> tuple[tuple[float, ...], tuple[float, ...]]:
    return (
        tuple(min(value[axis] for value in values) for axis in range(3)),
        tuple(max(value[axis] for value in values) for axis in range(3)),
    )


_TARGET_TASK_LOWER, _TARGET_TASK_UPPER = _component_bounds(GB300_TARGET_TASK_TRANSLATIONS)
_PICKUP_LOWER, _PICKUP_UPPER = (0.34, -0.20, 0.0105), (0.55, -0.025, 0.0145)
_TASK_WORKSPACE_LOWER, _TASK_WORKSPACE_UPPER = (0.25, -0.30, -0.02), (0.78, 0.22, 0.42)
_TASK_BODY_WORKSPACE_LOWER, _TASK_BODY_WORKSPACE_UPPER = (0.20, -0.34, -1.04), (0.90, 0.30, 0.58)
GB300_RJ45_ENTRY_BODY_PATTERNS = (r"/World/envs/env_[^/]+/Rj45Assembly",)


@configclass
class FrankaRJ45Gb300InsertEnvCfg(FrankaRJ45DualRackInsertEnvCfg):
    """Insert one free end into a uniformly selected native GB300 SN2201 jack."""

    task_translation: tuple[float, float, float] = GB300_DEFAULT_TARGET_TASK_TRANSLATION
    task_rotation_xyzw: tuple[float, float, float, float] = GB300_TASK_ROTATION_XYZW
    rj45_entry_body_patterns: tuple[str, ...] = GB300_RJ45_ENTRY_BODY_PATTERNS
    socket_position_lower: tuple[float, float, float] = _TARGET_TASK_LOWER
    socket_position_upper: tuple[float, float, float] = _TARGET_TASK_UPPER
    socket_yaw_range: tuple[float, float] = (0.0, 0.0)
    pickup_position_lower: tuple[float, float, float] = _PICKUP_LOWER
    pickup_position_upper: tuple[float, float, float] = _PICKUP_UPPER
    pickup_yaw_range: tuple[float, float] = (-math.radians(70.0), math.radians(70.0))
    task_workspace_lower: tuple[float, float, float] = _TASK_WORKSPACE_LOWER
    task_workspace_upper: tuple[float, float, float] = _TASK_WORKSPACE_UPPER
    task_body_workspace_lower: tuple[float, float, float] = _TASK_BODY_WORKSPACE_LOWER
    task_body_workspace_upper: tuple[float, float, float] = _TASK_BODY_WORKSPACE_UPPER

    reset_dataset_path: str = "datasets/franka_rj45_gb300_insert/reset_dataset.pt"
    reset_validation_report_path: str = "logs/rsl_rl/franka_rj45_gb300_insert/validation/reset_validation.json"

    def __post_init__(self) -> None:
        super().__post_init__()
        from isaaclab_visualizers.kit import KitVisualizerCfg

        self.scene.robot.init_state.pos = GB300_ROBOT_POSITION_E
        self.scene.robot.init_state.rot = GB300_ROBOT_ROTATION_XYZW
        self.scene.table.spawn = None
        self.scene.table_contact_surface.spawn = None
        self.scene.ground.init_state.pos = (0.0, 0.0, GB300_STUDIO_FLOOR_HEIGHT_M)
        self.scene.ground.spawn.color = (0.96, 0.97, 0.99)
        self.scene.light.spawn.intensity = 1450.0
        self.sim.default_visualizer_cfg = KitVisualizerCfg(
            eye=(2.45, -10.0, 0.20),
            lookat=(2.45, 0.34, -0.70),
            focal_length=8.0,
            origin_type="env",
            origin_env_index=0,
        )

    def validated_task_rotation_xyzw(self) -> tuple[float, float, float, float]:
        """Admit the identity task frame aligned to the rotated GB300 front."""
        return GB300_TASK_ROTATION_XYZW

    def validated_rj45_entry_body_patterns(self) -> tuple[str, ...]:
        """Exclude the deliberately absent table-contact body from VBD ownership."""
        return GB300_RJ45_ENTRY_BODY_PATTERNS

    def validated_table_scene_policy(self) -> Literal["seattle-contact", "absent"]:
        """Keep the GB300 showroom free of Seattle-table visuals and contacts."""
        return "absent"

    def validate_config(self) -> None:
        """Validate the base task without imposing the dual-AS4610 fixed port."""
        FrankaRJ45PickInsertEnvCfg.validate_config(self)
        if tuple(self.task_translation) != GB300_DEFAULT_TARGET_TASK_TRANSLATION:
            raise ValueError("GB300 canonical task translation must use the pinned default candidate.")
        if tuple(self.task_rotation_xyzw) != GB300_TASK_ROTATION_XYZW:
            raise ValueError("GB300 native SN2201 sockets require the pinned front-facing workcell rotation.")
        if tuple(self.socket_position_lower) != _TARGET_TASK_LOWER or tuple(self.socket_position_upper) != (
            _TARGET_TASK_UPPER
        ):
            raise ValueError("GB300 socket bounds must exactly enclose the native CAD candidate bank.")
        if tuple(self.socket_yaw_range) != (0.0, 0.0):
            raise ValueError("GB300 native SN2201 ports require the pinned front-facing socket yaw.")
        if not math.isfinite(float(self.anchored_cable_endpoint_tolerance_m)) or not (
            0.0 < float(self.anchored_cable_endpoint_tolerance_m) <= 0.01
        ):
            raise ValueError("anchored_cable_endpoint_tolerance_m must lie in (0, 0.01] m.")
        if GB300_WORKCELL_CONTRACT_VERSION != 3 or GB300_WORKCELL_CFG.presentation_kind != "gb300":
            raise ValueError("Unsupported GB300 workcell contract.")


def gb300_reset_dataset_task_contract(cfg: FrankaRJ45Gb300InsertEnvCfg) -> dict[str, object]:
    """Return the complete reset-bank contract for the native-port GB300 task."""
    contract = pick_insert_reset_dataset_task_contract(cfg)
    base_version = int(contract["contract_version"])
    base_semantics = int(contract["pick_insert"]["semantics_version"])
    contract["contract_version"] = 3
    contract["base_pick_insert_contract_version"] = base_version
    contract["task_variant"] = "franka-rj45-gb300-insert"
    contract["gb300"] = {
        "semantics_version": 3,
        "base_pick_insert_semantics_version": base_semantics,
        "cable_count": 1,
        "physical_plug_count": 2,
        "active_physical_socket_count_per_world": 2,
        "target_socket_selection": "one-resettable-hidden-exact-sdf-registered-to-eight-native-sn2201-jacks",
        "anchored_end": "one-static-seated-exact-sdf-socket-plug-latch",
        "inactive_candidate_ports": "native-simready-sn2201-cad-only-no-collision",
        "anchored_cable_endpoint_tolerance_m": float(cfg.anchored_cable_endpoint_tolerance_m),
        "workcell": gb300_workcell_contract(),
        "observations": {
            "selected_target_socket": "inherited-role-stable-pick-insert-terms",
            "anchored_socket_pose": "actor",
            "anchored_plug_pose": "actor",
            "anchored_cable_endpoint_error": "actor-anchored-plug-frame",
        },
    }
    contract["pick_insert"]["socket_pose_policy"] = "uniform-discrete-native-gb300-sn2201-rj45-jack"
    contract["pick_insert"]["whole_cable_state_policy"] = (
        "exact-segment-length-route-from-free-plug-prefix-to-static-gb300-anchored-plug"
    )
    diversity = contract["pick_insert"]["full_pick_diversity"]
    diversity["minimum_unique_socket_rows"] = len(GB300_TARGET_TASK_TRANSLATIONS)
    diversity["minimum_socket_span_fraction"] = 0.95
    return contract


__all__ = [
    "GB300_RJ45_ENTRY_BODY_PATTERNS",
    "FrankaRJ45Gb300InsertEnvCfg",
    "gb300_reset_dataset_task_contract",
]
