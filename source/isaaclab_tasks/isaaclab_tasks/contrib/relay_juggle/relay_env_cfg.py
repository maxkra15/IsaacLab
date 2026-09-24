# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Manager-based rigid-ball outbound transfer from KUKA-Allegro to Fourier GR1T2."""

from pathlib import Path

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg, NewtonShapeCfg
from isaaclab_newton.sim.schemas import NewtonCollisionCfg, NewtonMaterialPropertiesCfg

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.visualizers import VisualizerCfg

from isaaclab_tasks.contrib.juggle.mdp.reset import (
    BALL_MASS,
    BALL_RADIUS,
    JUGGLE_SPHERE_PRELOAD_HAND_POSITION,
    KUKA_ALLEGRO_JUGGLE_ARM_WORKSPACE_LOWER,
    KUKA_ALLEGRO_JUGGLE_ARM_WORKSPACE_UPPER,
    KUKA_ARM_JOINT_NAMES,
)
from isaaclab_tasks.contrib.stack.mdp.actions_cfg import WorkspaceBoundedRelativeJointPositionActionCfg
from isaaclab_tasks.contrib.stack.mdp.kuka_allegro_reset import KUKA_ALLEGRO_ALL_HAND_JOINT_NAMES

from isaaclab_assets.robots import KUKA_ALLEGRO_CFG
from isaaclab_assets.robots.fourier import GR1T2_HIGH_PD_CFG

from . import mdp

TASK_ID = "IsaacContrib-RelayJuggle-KukaAllegro-GR1T2"
"""Gym task identifier for the rigid-ball relay."""

KUKA_BASE_POSITION = (-0.90, 0.0, 0.35)
"""KUKA root position in each environment [m]."""

KUKA_ARM_START = (2.09426120, 1.15379100, -2.19415376, 1.48760476, -2.65542972, 1.64765074, -2.88137803)
"""Calibrated cradle pose for the KUKA arm [rad]."""

GR1_RIGHT_ARM_JOINT_NAMES = (
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_pitch_joint",
    "right_wrist_yaw_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
)
"""GR1T2 right-arm action order."""

GR1_RIGHT_HAND_JOINT_NAMES = (
    "R_index_proximal_joint",
    "R_middle_proximal_joint",
    "R_pinky_proximal_joint",
    "R_ring_proximal_joint",
    "R_thumb_proximal_yaw_joint",
    "R_index_intermediate_joint",
    "R_middle_intermediate_joint",
    "R_pinky_intermediate_joint",
    "R_ring_intermediate_joint",
    "R_thumb_proximal_pitch_joint",
    "R_thumb_distal_joint",
)
"""GR1T2 right-hand action order."""

_GR1_NEUTRAL_JOINTS = (
    "(?!right_elbow_pitch_joint$|left_elbow_pitch_joint$|R_thumb_proximal_yaw_joint$|L_thumb_proximal_yaw_joint$).*"
)

_GALLERY_USD = Path(__file__).parent / "assets" / "relay_gallery.usda"


def _kuka_cfg() -> ArticulationCfg:
    """Configure the calibrated KUKA cradle on the left side of the lane."""
    joint_pos = {
        **dict(zip(KUKA_ARM_JOINT_NAMES, KUKA_ARM_START, strict=True)),
        **dict(zip(KUKA_ALLEGRO_ALL_HAND_JOINT_NAMES, JUGGLE_SPHERE_PRELOAD_HAND_POSITION, strict=True)),
    }
    cfg = KUKA_ALLEGRO_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Kuka",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=KUKA_BASE_POSITION,
            rot=(0.0, 0.0, 1.0, 0.0),
            joint_pos=joint_pos,
        ),
    )
    cfg.spawn.fix_root_link = True
    hand_expression = "(index|middle|ring|thumb)_joint_(0|1|2|3)"
    actuator = cfg.actuators["kuka_allegro_actuators"]
    actuator.stiffness[hand_expression] = 20.0
    actuator.damping[hand_expression] = 0.5
    # The one-metre Juggle task established this simulated arm-effort tier.
    actuator.joint_effort_limit = {
        **actuator.joint_effort_limit,
        "iiwa7_joint_(1|2)": 352.0,
        "iiwa7_joint_(3|4|5)": 220.0,
        "iiwa7_joint_(6|7)": 80.0,
    }
    return cfg


def _gr1_cfg() -> ArticulationCfg:
    """Configure a fixed-base GR1T2 receiver with both hands in view."""
    cfg = GR1T2_HIGH_PD_CFG.replace(
        prim_path="{ENV_REGEX_NS}/GR1T2",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.35, 0.0, 0.95),
            rot=(0.0, 0.0, 1.0, 0.0),
            joint_pos={
                _GR1_NEUTRAL_JOINTS: 0.0,
                "right_elbow_pitch_joint": -1.40,
                "left_elbow_pitch_joint": -1.40,
                "R_thumb_proximal_yaw_joint": -1.20,
                "L_thumb_proximal_yaw_joint": -1.20,
            },
            joint_vel={".*": 0.0},
        ),
    )
    cfg.spawn.fix_root_link = True
    cfg.actuators["head"] = ImplicitActuatorCfg(joint_names_expr=["head_.*"], stiffness=100.0, damping=10.0)
    cfg.actuators["legs"] = ImplicitActuatorCfg(
        joint_names_expr=[".*_hip_.*", ".*_knee_.*", ".*_ankle_.*"], stiffness=400.0, damping=20.0
    )
    return cfg


@configclass
class RelaySceneCfg(InteractiveSceneCfg):
    """Two articulated performers, one live ball, and a visual-only gallery."""

    kuka: ArticulationCfg = _kuka_cfg()
    gr1: ArticulationCfg = _gr1_cfg()
    ball: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Ball",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.40, 0.0, 0.35)),
        spawn=sim_utils.SphereCfg(
            radius=BALL_RADIUS,
            rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(rigid_body_enabled=True, kinematic_enabled=False),
            collision_props=[
                sim_utils.UsdPhysicsCollisionCfg(collision_enabled=True),
                NewtonCollisionCfg(contact_margin=0.0, contact_gap=0.0),
            ],
            mass_props=sim_utils.MassCfg(mass=BALL_MASS),
            physics_material=NewtonMaterialPropertiesCfg(
                static_friction=1.0,
                dynamic_friction=0.8,
                restitution=0.0,
                contact_stiffness=1.0e4,
                contact_damping=120.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.28, 0.08),
                metallic=0.15,
                roughness=0.30,
            ),
            semantic_tags=[("class", "relay_ball")],
        ),
    )
    ground: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        collision_group=-1,
    )
    gallery: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Gallery",
        spawn=sim_utils.UsdFileCfg(usd_path=str(_GALLERY_USD)),
    )
    key_light: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/RelayKeyLight",
        spawn=sim_utils.DistantLightCfg(color=(1.0, 0.91, 0.80), intensity=1800.0),
    )
    sky_light: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/RelaySkyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.72, 0.82, 1.0), intensity=1700.0),
    )


@configclass
class ActionsCfg:
    """Gravity-compensated KUKA deltas and default-relative hand/GR1 targets [rad]."""

    kuka_arm = WorkspaceBoundedRelativeJointPositionActionCfg(
        asset_name="kuka",
        joint_names=list(KUKA_ARM_JOINT_NAMES),
        preserve_order=True,
        scale=1.0,
        max_delta=0.10,
        workspace_lower=KUKA_ALLEGRO_JUGGLE_ARM_WORKSPACE_LOWER,
        workspace_upper=KUKA_ALLEGRO_JUGGLE_ARM_WORKSPACE_UPPER,
        gravity_compensation=True,
    )
    kuka_hand = base_mdp.JointPositionActionCfg(
        asset_name="kuka", joint_names=list(KUKA_ALLEGRO_ALL_HAND_JOINT_NAMES), preserve_order=True, scale=1.0
    )
    gr1_arm = base_mdp.JointPositionActionCfg(
        asset_name="gr1", joint_names=list(GR1_RIGHT_ARM_JOINT_NAMES), preserve_order=True, scale=1.0
    )
    gr1_hand = base_mdp.JointPositionActionCfg(
        asset_name="gr1", joint_names=list(GR1_RIGHT_HAND_JOINT_NAMES), preserve_order=True, scale=1.0
    )


@configclass
class ObservationsCfg:
    """Proprioception and measured ball/hand geometry for state-based PPO."""

    @configclass
    class PolicyCfg(ObsGroup):
        kuka_joint_pos = ObsTerm(func=base_mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("kuka")})
        kuka_joint_vel = ObsTerm(func=base_mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("kuka")})
        gr1_joint_pos = ObsTerm(func=base_mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("gr1")})
        gr1_joint_vel = ObsTerm(func=base_mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("gr1")})
        ball_position = ObsTerm(func=mdp.ball_position_local)
        ball_velocity = ObsTerm(func=mdp.ball_velocity_world)
        kuka_to_ball = ObsTerm(
            func=mdp.ball_relative_to_hand,
            params={
                "hand_cfg": SceneEntityCfg("kuka", body_names=["palm_link"]),
                "offset": (0.02790133, -0.03190392, 0.03965311),
            },
        )
        gr1_to_ball = ObsTerm(
            func=mdp.ball_relative_to_hand,
            params={"hand_cfg": SceneEntityCfg("gr1", body_names=["right_hand_pitch_link"])},
        )
        direction = ObsTerm(func=mdp.relay_target_direction)
        actions = ObsTerm(func=base_mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """Small, state-only PPO shaping objective for an outbound physical transfer."""

    receiver_proximity = RewTerm(
        func=mdp.receiver_proximity,
        weight=0.5,
        params={
            "kuka_hand_cfg": SceneEntityCfg("kuka", body_names=["palm_link"]),
            "gr1_hand_cfg": SceneEntityCfg("gr1", body_names=["right_hand_pitch_link"]),
        },
    )
    outbound_flight = RewTerm(
        func=mdp.outbound_flight_progress,
        weight=2.0,
        params={
            "kuka_hand_cfg": SceneEntityCfg("kuka", body_names=["palm_link"]),
            "gr1_hand_cfg": SceneEntityCfg("gr1", body_names=["right_hand_pitch_link"]),
        },
    )
    gr1_stability_proxy = RewTerm(
        func=mdp.gr1_stable_ball_proximity,
        weight=1.5,
        params={"gr1_hand_cfg": SceneEntityCfg("gr1", body_names=["right_hand_pitch_link"])},
    )
    action_rate = RewTerm(func=base_mdp.action_rate_l2, weight=-1.0e-4)


@configclass
class TerminationsCfg:
    """Reset after a drop, an escaped ball, or a finite episode horizon."""

    dropped = DoneTerm(func=mdp.ball_dropped, params={"minimum_height": 0.10})
    escaped = DoneTerm(func=mdp.ball_out_of_bounds, params={"horizontal_limit": 2.0})
    invalid_state = DoneTerm(
        func=mdp.nonfinite_relay_state,
        params={
            "kuka_cfg": SceneEntityCfg("kuka", joint_names=list(KUKA_ARM_JOINT_NAMES), body_names=["palm_link"]),
            "gr1_cfg": SceneEntityCfg(
                "gr1", joint_names=list(GR1_RIGHT_ARM_JOINT_NAMES), body_names=["right_hand_pitch_link"]
            ),
        },
    )
    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)


@configclass
class EventsCfg:
    """Restore both robots, then seat the ball in the calibrated KUKA cradle."""

    reset_scene = EventTerm(func=base_mdp.reset_scene_to_default, mode="reset", params={"reset_joint_targets": True})
    seat_ball = EventTerm(
        func=mdp.reset_ball_in_kuka_hand,
        mode="reset",
        params={"arm_joint_names": KUKA_ARM_JOINT_NAMES, "kuka_base_position": KUKA_BASE_POSITION},
    )


@configclass
class RelayJuggleEnvCfg(ManagerBasedRLEnvCfg):
    """Script an outbound deflection or train its outbound-transfer PPO baseline."""

    record_relay_invalid_details: bool = False
    """Record per-environment invalid-state fields for the scripted demo only."""

    decimation = 2
    episode_length_s = 12.0
    scene: RelaySceneCfg = RelaySceneCfg(num_envs=1, env_spacing=6.0, replicate_physics=True)
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=decimation,
        gravity=(0.0, 0.0, -9.81),
        physics=NewtonCfg(
            solver_cfg=MJWarpSolverCfg(
                solver="newton",
                integrator="implicitfast",
                iterations=100,
                ls_iterations=30,
                ccd_iterations=50,
                use_mujoco_contacts=False,
            ),
            num_substeps=4,
            collision_decimation=1,
            collision_cfg=NewtonCollisionPipelineCfg(
                broad_phase="explicit", reduce_contacts=True, rigid_contact_max=250000
            ),
            default_shape_cfg=NewtonShapeCfg(margin=0.0, gap=0.0),
        ),
        default_visualizer_cfg=VisualizerCfg(eye=(2.6, -3.2, 2.0), lookat=(0.0, 0.0, 0.9)),
    )
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()
    commands = None
    curriculum = None
