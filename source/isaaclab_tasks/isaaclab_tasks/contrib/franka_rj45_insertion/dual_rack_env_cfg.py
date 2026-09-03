# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for routing one two-ended RJ45 cable between stacked racks."""

from __future__ import annotations

import math
from typing import Literal

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass

from . import mdp
from .dual_rack_workcell import (
    DUAL_RACK_TARGET_TASK_TRANSLATION,
    DUAL_RACK_TASK_ROTATION_XYZW,
    DUAL_RACK_WORKCELL_CFG,
    DUAL_RACK_WORKCELL_CONTRACT_VERSION,
    dual_rack_workcell_contract,
)
from .pick_insert_env_cfg import (
    PICK_INSERT_RJ45_ENTRY_BODY_PATTERNS,
    FrankaRJ45PickInsertEnvCfg,
    PickInsertObservationsCfg,
    PickInsertTerminationsCfg,
    pick_insert_reset_dataset_task_contract,
)

# Every Franka collision link is copied into the VBD task view so the arm,
# gripper, cable, frame, and rack cuboids share one contact pipeline.  The
# original pick-insert task keeps its narrower hand/finger-only selector.
DUAL_RACK_ROBOT_PROXY_BODY_PATTERNS = (r"/World/envs/env_[^/]+/Robot/Geometry/.*panda_.*",)


@configclass
class DualRackObservationsCfg(PickInsertObservationsCfg):
    """Original free-end observations plus the occupied cable-end geometry."""

    @configclass
    class PolicyCfg(PickInsertObservationsCfg.PolicyCfg):
        anchored_socket_pose = ObsTerm(func=mdp.anchored_socket_pose_obs)
        anchored_plug_pose = ObsTerm(func=mdp.anchored_plug_pose_obs)
        anchored_cable_endpoint_error = ObsTerm(func=mdp.anchored_cable_endpoint_error_obs, scale=20.0)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class DualRackTerminationsCfg(PickInsertTerminationsCfg):
    """Pick-insert failures plus integrity of the permanently connected end."""

    anchored_cable_disconnected = DoneTerm(func=mdp.anchored_cable_disconnected)


@configclass
class FrankaRJ45DualRackInsertEnvCfg(FrankaRJ45PickInsertEnvCfg):
    """Insert the free end of a cable whose opposite end occupies a second rack."""

    observations: DualRackObservationsCfg = DualRackObservationsCfg()
    terminations: DualRackTerminationsCfg = DualRackTerminationsCfg()

    task_translation: tuple[float, float, float] = DUAL_RACK_TARGET_TASK_TRANSLATION
    task_rotation_xyzw: tuple[float, float, float, float] = DUAL_RACK_TASK_ROTATION_XYZW
    rj45_entry_body_patterns: tuple[str, ...] = PICK_INSERT_RJ45_ENTRY_BODY_PATTERNS
    proxy_body_patterns: tuple[str, ...] = DUAL_RACK_ROBOT_PROXY_BODY_PATTERNS

    reset_dataset_path: str = "datasets/franka_rj45_dual_rack_insert/reset_dataset.pt"
    reset_validation_report_path: str = (
        "logs/rsl_rl/franka_rj45_dual_rack_insert/validation/reset_validation.json"
    )
    reset_source: Literal["dataset"] = "dataset"

    # The target socket is mechanically fixed in the upper rack.  Diversity
    # comes from cable/free-plug/robot configurations, never decor motion.
    socket_position_lower: tuple[float, float, float] = DUAL_RACK_TARGET_TASK_TRANSLATION
    socket_position_upper: tuple[float, float, float] = DUAL_RACK_TARGET_TASK_TRANSLATION
    socket_yaw_range: tuple[float, float] = (0.0, 0.0)
    pickup_position_lower: tuple[float, float, float] = (0.34, -0.20, 0.0105)
    pickup_position_upper: tuple[float, float, float] = (0.55, -0.025, 0.0145)
    pickup_yaw_range: tuple[float, float] = (-math.radians(70.0), math.radians(70.0))
    minimum_pickup_socket_distance: float = 0.14

    task_workspace_lower: tuple[float, float, float] = (0.25, -0.30, -0.02)
    task_workspace_upper: tuple[float, float, float] = (0.78, 0.22, 0.42)
    task_body_workspace_lower: tuple[float, float, float] = (0.20, -0.34, -0.01)
    task_body_workspace_upper: tuple[float, float, float] = (0.90, 0.30, 0.58)
    max_cable_socket_offset: float = 0.70
    max_cable_goal_offset: float = 0.70
    anchored_cable_endpoint_tolerance_m: float = 0.003

    def __post_init__(self) -> None:
        super().__post_init__()
        from isaaclab_visualizers.kit import KitVisualizerCfg

        self.sim.default_visualizer_cfg = KitVisualizerCfg(
            eye=(1.18, -0.92, 0.76),
            lookat=(0.56, 0.06, 0.10),
            origin_type="env",
            origin_env_index=0,
        )

    def validated_proxy_body_patterns(self) -> tuple[str, ...]:
        """Use every Franka collision link around the physical rack frame."""
        return DUAL_RACK_ROBOT_PROXY_BODY_PATTERNS

    def play_mode(self) -> None:
        """Inspect exact bank rows one at a time without outcome-driven sampling."""
        ManagerBasedRLEnvCfg.play_mode(self)
        self.scene.num_envs = 1
        self.reset_source = "dataset"
        self.reset_dataset_sampling_mode = "uniform"
        self.curriculum_freeze = True

    def validate_config(self) -> None:
        super().validate_config()
        if tuple(self.task_translation) != DUAL_RACK_TARGET_TASK_TRANSLATION:
            raise ValueError("Dual-rack target assembly must remain aligned to the upper rack opening.")
        if tuple(self.task_rotation_xyzw) != DUAL_RACK_TASK_ROTATION_XYZW:
            raise ValueError("Dual-rack workcell currently requires identity connector orientation.")
        if tuple(self.socket_position_lower) != tuple(self.socket_position_upper):
            raise ValueError("Dual-rack target socket bounds must describe one fixed rack opening.")
        if tuple(self.socket_position_lower) != DUAL_RACK_TARGET_TASK_TRANSLATION:
            raise ValueError("Dual-rack fixed socket bounds must equal the target assembly translation.")
        if tuple(self.socket_yaw_range) != (0.0, 0.0):
            raise ValueError("Dual-rack target socket yaw must remain fixed at zero.")
        if not math.isfinite(float(self.anchored_cable_endpoint_tolerance_m)) or not (
            0.0 < float(self.anchored_cable_endpoint_tolerance_m) <= 0.01
        ):
            raise ValueError("anchored_cable_endpoint_tolerance_m must lie in (0, 0.01] m.")
        # Constructing the immutable value also runs every box/path invariant.
        if DUAL_RACK_WORKCELL_CFG.contract_version != DUAL_RACK_WORKCELL_CONTRACT_VERSION:
            raise ValueError("Unsupported dual-rack workcell contract version.")


def dual_rack_reset_dataset_task_contract(cfg: FrankaRJ45DualRackInsertEnvCfg) -> dict[str, object]:
    """Return the complete reset-bank contract for the two-ended cable task."""
    contract = pick_insert_reset_dataset_task_contract(cfg)
    base_version = int(contract["contract_version"])
    base_semantics = int(contract["pick_insert"]["semantics_version"])
    contract["contract_version"] = 1
    contract["base_pick_insert_contract_version"] = base_version
    contract["task_variant"] = "franka-rj45-dual-rack-insert"
    contract["dual_rack"] = {
        "semantics_version": 1,
        "base_pick_insert_semantics_version": base_semantics,
        "free_end": "dynamic-plug-latch-plus-four-plug-relative-cable-anchors",
        "anchored_end": "static-seated-socket-plug-latch-plus-four-pinned-strain-relief-segments",
        "cable_count": 1,
        "physical_socket_count": 2,
        "physical_plug_count": 2,
        "learned_target": "free-plug-into-upper-target-socket",
        "anchored_cable_endpoint_tolerance_m": float(cfg.anchored_cable_endpoint_tolerance_m),
        "workcell": dual_rack_workcell_contract(),
        "observations": {
            "free_plug_and_target_socket": "inherited-role-stable-pick-insert-terms",
            "anchored_socket_pose": "actor",
            "anchored_plug_pose": "actor",
            "anchored_cable_endpoint_error": "actor-anchored-plug-frame",
        },
    }
    contract["pick_insert"]["socket_pose_policy"] = "fixed-upper-rack-opening"
    contract["pick_insert"]["whole_cable_state_policy"] = (
        "exact-segment-length-route-from-free-plug-prefix-to-static-anchored-plug"
    )
    diversity = contract["pick_insert"]["full_pick_diversity"]
    diversity["minimum_unique_socket_rows"] = 1
    diversity["minimum_socket_span_fraction"] = 0.0
    return contract


__all__ = [
    "DUAL_RACK_ROBOT_PROXY_BODY_PATTERNS",
    "DualRackObservationsCfg",
    "FrankaRJ45DualRackInsertEnvCfg",
    "dual_rack_reset_dataset_task_contract",
]
