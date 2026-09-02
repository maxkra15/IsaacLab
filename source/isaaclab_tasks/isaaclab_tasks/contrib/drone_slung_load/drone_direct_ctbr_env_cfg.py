# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Rigid-drone Direct-CTBR waypoint task sharing the FLARE route objective."""

from __future__ import annotations

from isaaclab_newton.physics import NewtonCfg

from isaaclab.assets import AssetBaseCfg, CableObjectCfg, RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass

import isaaclab_tasks.contrib.drone_slung_load.mdp as mdp
from isaaclab_tasks.contrib.drone_slung_load.drone_slung_load_env_cfg import (
    DirectCTBRCurriculumCfg,
    DirectCTBRObservationsCfg,
    DirectCTBRPolicyCfg,
    DirectCTBRPrivilegedCfg,
    DirectCTBRRewardsCfg,
    DroneSlungLoadSceneCfg,
    DroneSlungLoadWaypointDirectCTBREnvCfg,
    EnhancedTerminationsCfg,
    EventCfg,
)


@configclass
class DroneDirectCTBRSceneCfg(DroneSlungLoadSceneCfg):
    """Reuse the FLARE rigid drone and world without suspended-system assets."""

    payload: RigidObjectCfg | None = None
    cable: CableObjectCfg | None = None
    drone_cable_attach: AssetBaseCfg | None = None
    cable_payload_attach: AssetBaseCfg | None = None


@configclass
class DroneDirectCTBRPolicyCfg(DirectCTBRPolicyCfg):
    """Direct-CTBR actor state with only measurements available on a rigid drone."""

    swing_angles: ObsTerm | None = None
    swing_angular_velocity: ObsTerm | None = None


@configclass
class DroneDirectCTBRObservationsCfg(DirectCTBRObservationsCfg):
    """Symmetric 43-value rigid-drone observations with no load-only group."""

    policy: DroneDirectCTBRPolicyCfg = DroneDirectCTBRPolicyCfg()
    privileged: DirectCTBRPrivilegedCfg | None = None


@configclass
class DroneDirectCTBREventCfg(EventCfg):
    """Reset only the rigid drone; no cable/payload state exists."""

    reset_slung_load: EventTerm | None = None


@configclass
class DroneDirectCTBRRewardsCfg(DirectCTBRRewardsCfg):
    """Shared Direct-CTBR route objective with load-only costs disabled."""

    swing_safety: RewTerm | None = None
    swing_magnitude: RewTerm | None = None
    transverse_speed: RewTerm | None = None
    crash = RewTerm(
        func=mdp.unsafe_termination_impulse,
        weight=-100.0,
        params={
            "unsafe_term_names": (
                "drone_crash",
                "illegal_drone",
                "illegal_action",
                "drone_out_of_workspace",
                "path_corridor",
            )
        },
    )


@configclass
class DroneDirectCTBRTerminationsCfg(EnhancedTerminationsCfg):
    """Rigid-drone safety and route-success guards."""

    payload_crash: DoneTerm | None = None
    illegal_payload: DoneTerm | None = None
    illegal_cable: DoneTerm | None = None
    cable_integrity: DoneTerm | None = None
    payload_out_of_workspace: DoneTerm | None = None


@configclass
class DroneWaypointDirectCTBREnvCfg(DroneSlungLoadWaypointDirectCTBREnvCfg):
    """Policy-owned collective thrust/body rates for the rigid FLARE drone.

    Route generation, path observations, reward geometry, curriculum, action
    mapping, rate PID, rotor mixer, and 100 Hz control cadence are inherited
    from the slung-load Direct-CTBR task. Only unavailable suspended-system
    assets and MDP terms are removed, and Newton's rigid MJWarp solver replaces
    the coupled VBD solve.
    """

    scene: DroneDirectCTBRSceneCfg = DroneDirectCTBRSceneCfg(num_envs=32, env_spacing=0.0)
    observations: DroneDirectCTBRObservationsCfg = DroneDirectCTBRObservationsCfg()
    rewards: DroneDirectCTBRRewardsCfg = DroneDirectCTBRRewardsCfg()
    terminations: DroneDirectCTBRTerminationsCfg = DroneDirectCTBRTerminationsCfg()
    events: DroneDirectCTBREventCfg = DroneDirectCTBREventCfg()
    curriculum: DirectCTBRCurriculumCfg | None = DirectCTBRCurriculumCfg()

    def _newton_physics_cfg(self) -> NewtonCfg:
        """Build the rigid-only default MJWarp solver configuration."""
        return NewtonCfg()

    def __post_init__(self):
        super().__post_init__()
        # The inherited simulation/action dt remain exactly 0.01 s.
        self.commands.route.record_slung_load_metrics = False
        # The rigid drone can brake and corner twice as aggressively as the
        # suspended system while keeping the exact same waypoint geometry.
        self.episode_length_s = 15.0
        self.commands.route.maximum_lateral_acceleration = 6.0
        self.commands.route.maximum_braking_acceleration = 6.0
        self.rewards.path_progress.params["maximum_lateral_acceleration"] = 6.0

    def evaluation_mode(self):
        """Use the shared hard-route suite with the rigid-drone horizon."""
        super().evaluation_mode()
        self.episode_length_s = 15.0
