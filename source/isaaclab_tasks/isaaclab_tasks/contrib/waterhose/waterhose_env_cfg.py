# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Waterhose manipulation environment.

MDP config groups are copied from the Franka cable-plug task
(:mod:`isaaclab_tasks.core.lift_franka_soft`), whose
``mdp`` functions are reused by import.
"""

from __future__ import annotations

import logging
import math
import os

from isaaclab_newton.physics import (
    AdmmContactPairCfg,
    AdmmCouplingCfg,
    CoupledProxyCfg,
    CoupledSolverCfg,
    CoupledSolverEntryCfg,
    MJWarpSolverCfg,
    NewtonCollisionPipelineCfg,
    ProxyCouplingCfg,
)
from isaaclab_newton.envs.mdp.actions.newton_ik_actions_cfg import NewtonInverseKinematicsActionCfg
from isaaclab_newton.ik.newton_ik_manager_cfg import NewtonIKManagerCfg
from isaaclab_newton.sim.spawners.materials.physics_materials_cfg import NewtonCableMaterialCfg
from isaaclab_visualizers.kit.kit_visualizer_cfg import KitVisualizerCfg
from isaaclab_visualizers.newton.newton_visualizer_cfg import NewtonVisualizerCfg

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.assets.rigid_object.rigid_object_cfg import RigidObjectCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.devices.device_base import DevicesCfg
from isaaclab.devices.keyboard import Se3KeyboardCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
import isaaclab.envs.mdp as mdp
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass
from isaaclab_contrib.cable.cable_object_cfg import CableAttachmentCfg, CableObjectCfg
from isaaclab_contrib.deformable.newton_manager_cfg import (
    CoupledNewtonCfg,
    NewtonModelCfg,
    VBDSolverCfg,
)
from isaaclab_teleop import IsaacTeleopCfg, XrCfg

from .geometry import (
    ANCHOR_POS,
    CABLE_HEAD_TO_PLUG_ORIGIN_LOCAL_Z,
    FRIDGE_POS,
    RIGHT_GRIPPER_EE_FRAME_POS,
    RIGHT_GRIPPER_EE_FRAME_QUAT_XYZW,
    SOCKET_COLLISION_MESH_PATTERN,
)
from .mdp.actions import WaterhoseGripperPositionActionCfg
from .teleop import WaterhoseSpaceMouseCfg
from .teleop_pipelines import build_waterhose_relative_teleop_pipeline, build_waterhose_teleop_pipeline

WATERHOSE_ASSETS_DIR = os.environ.get(
    "WATERHOSE_ASSETS_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets"),
)

_FRIDGE_USD = os.path.join(WATERHOSE_ASSETS_DIR, "fridge", "fridge_waterhose.usda")
_RBY1_USD = os.path.join(WATERHOSE_ASSETS_DIR, "rby1df", "rby1df_waterhose.usda")
_PLUG_USD = os.path.join(WATERHOSE_ASSETS_DIR, "fridge", "cable", "plug.usda")
_CABLE1_USD = os.path.join(WATERHOSE_ASSETS_DIR, "fridge", "cable", "cable001.usda")

# Hand-authored initial visualizer cameras. These used to live in
# scripts/environments/waterhose/run_robot_demo.py; keep them in the task config
# so IsaacLab's official --visualizer/--viz path uses the same starting views.
# KitVisualizer uses eye/lookat; this lookat is one meter along camera-local -Z
# from the authored Kit transform translate=(-0.9, 0.6, 0.3), rotateXYZ=(73.32259, 0, -112.30437).
_KIT_CAMERA_EYE = (-0.9, 0.6, 0.3)
_KIT_CAMERA_LOOKAT = (-0.013736291, 0.236437794, 0.013017143)
_NEWTON_CAMERA_EYE = (-2.55, -7.1, 2.3)
_NEWTON_CAMERA_LOOKAT = (0.55, -0.42, 0.9)
_ROBOT_BASE_PRIM_PATH_ENV0 = "/World/envs/env_0/Robot/Geometry/origin"


_RBY1_GRIPPER_MIMIC_JOINT_TOKENS = (
    "gripper_left_finger_joint",
    "gripper_right_finger_joint",
)
_GRIPPER_PAD_STATIC_FRICTION = 1.5
_GRIPPER_PAD_DYNAMIC_FRICTION = 1.0
_GRIPPER_DRIVER_STIFFNESS = 1.0e4
_GRIPPER_DRIVER_DAMPING = 1.0e3
_GRIPPER_DRIVER_EFFORT_LIMIT = 150.0
_GRIPPER_FINGER_STIFFNESS = 4.0e4
_GRIPPER_FINGER_DAMPING = 2.0e3
_GRIPPER_FINGER_EFFORT_LIMIT = 120.0
_VBD_CONTACT_STIFFNESS = 1.0e5
_VBD_CONTACT_DAMPING = 5.0e-2
_VBD_DEFAULT_SHAPE_FRICTION = 0.8
_VBD_SOFT_CONTACT_FRICTION = 0.6
_VBD_GRIPPER_PROXY_FRICTION = 5.0e6
_VBD_GRIPPER_PROXY_MARGIN = 0.00075
_RIGHT_GRIPPER_JOINT_NAMES = [
    "right_gripper_finger_joint_1",
    "right_gripper_left_finger_joint",
    "right_gripper_right_finger_joint",
]
_RIGHT_GRIPPER_OPEN_COMMAND = {
    "right_gripper_finger_joint_1": 0.09,
    "right_gripper_left_finger_joint": -0.045,
    "right_gripper_right_finger_joint": 0.045,
}
_RIGHT_GRIPPER_CLOSE_COMMAND = {
    # Shallow preload on the ~14.6 mm plug flange. A 14.8 mm target had too
    # little normal force to grip; the older 13 mm target visibly drove the
    # fingers through the plug. This asks for ~0.3 mm compression per side.
    "right_gripper_finger_joint_1": 0.0142,
    "right_gripper_left_finger_joint": -0.0071,
    "right_gripper_right_finger_joint": 0.0071,
}


def _disable_rby1df_gripper_mimic_constraints(_payload=None) -> None:
    """Disable imported RBY1 gripper mimic constraints before Newton finalizes the model."""

    from isaaclab_newton.physics import NewtonManager

    builder = getattr(NewtonManager, "_builder", None)
    if builder is None:
        raise RuntimeError("Newton builder is unavailable while disabling RBY1 gripper mimic constraints.")

    labels = getattr(builder, "constraint_mimic_label", None)
    enabled = getattr(builder, "constraint_mimic_enabled", None)
    if labels is None or enabled is None:
        raise RuntimeError("Newton builder is missing gripper mimic constraint arrays.")

    disabled = 0
    for index, label in enumerate(labels):
        if any(token in str(label) for token in _RBY1_GRIPPER_MIMIC_JOINT_TOKENS):
            enabled[index] = False
            disabled += 1

    if disabled == 0:
        raise RuntimeError(
            "No RBY1 gripper mimic constraints matched the expected right/left finger joint labels."
        )

    logging.debug("Disabled %d RBY1 gripper mimic constraints for explicit finger control.", disabled)


def _register_rby1df_gripper_mimic_override() -> None:
    """Match Newton's waterhose examples by using explicit gripper drives, not mimic equality."""

    from isaaclab.physics import PhysicsEvent
    from isaaclab_newton.physics import NewtonManager

    NewtonManager.register_callback(
        _disable_rby1df_gripper_mimic_constraints,
        PhysicsEvent.MODEL_INIT,
        order=10,
        name="waterhose_disable_rby1df_gripper_mimics",
        wrap_weak_ref=False,
    )


def _make_proxy_collision_pipeline(model):
    """Build the destination-view collision pipeline used by Newton proxy coupling."""
    from newton import CollisionPipeline

    return CollisionPipeline(
        model,
        broad_phase="explicit",
        rigid_contact_max=30000,
        # "sticky" replays previous-frame contact geometry; that is useful for
        # hard-contact demos, but makes the hose feel glued to the fingers here.
        contact_matching="latest",
        contact_matching_pos_threshold=0.005,
        contact_matching_normal_dot_threshold=0.95,
    )


_RBY1_IK_INITIAL_JOINT_POS = {
    "torso_joint_1": 0.0,
    "torso_joint_2": 0.872664213180542,
    "torso_joint_3": -1.5707811117172241,
    "torso_joint_4": 0.6981245279312134,
    "torso_joint_5": 3.796982127823867e-06,
    "torso_joint_6": 0.0,
    "right_arm_joint_1": 0.3021828234195709,
    "right_arm_joint_2": -0.013802030123770237,
    "right_arm_joint_3": -0.09509921818971634,
    "right_arm_joint_4": -2.2242417335510254,
    "right_arm_joint_5": -0.7117632627487183,
    "right_arm_joint_6": 0.14113007485866547,
    "right_arm_joint_7": 0.5137608647346497 + math.pi / 2.0,
    "left_arm_joint_1": -0.4555884897708893,
    "left_arm_joint_2": 0.2500312626361847,
    "left_arm_joint_3": -0.665743887424469,
    "left_arm_joint_4": -1.3314952850341797,
    "left_arm_joint_5": -0.19328542053699493,
    "left_arm_joint_6": -0.5307496786117554,
    "left_arm_joint_7": 0.6565361022949219 - math.pi / 2.0,
    "right_gripper_finger_joint_1": 0.09138019700534642,
    "right_gripper_left_finger_joint": -0.09138019700534642 / 2.0,
    "right_gripper_right_finger_joint": 0.09138019700534642 / 2.0,
    "left_gripper_finger_joint_1": 0.09098683297634125,
    "left_gripper_left_finger_joint": -0.09098683297634125 / 2.0,
    "left_gripper_right_finger_joint": 0.09098683297634125 / 2.0,
}


##
# Scene
##


@configclass
class WaterhoseSceneCfg(InteractiveSceneCfg):
    """Cable + plug with the cable tail welded to a kinematic anchor; sky light and ground."""

    ### Static fridge body
    fridge = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Fridge",
        spawn=sim_utils.UsdFileCfg(usd_path=_FRIDGE_USD),
        init_state=AssetBaseCfg.InitialStateCfg(pos=FRIDGE_POS),
    )

    ### rby1df robot (28-DOF, fixed base). Gripper drives are force-limited so VBD contacts can stop the fingers.
    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(usd_path=_RBY1_USD),
        articulation_root_prim_path="/Geometry/origin",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 1.0, -1.0),
            rot=(0.0, 0.0, -0.70710678, 0.70710678),  # 90 deg about +Z (x, y, z, w)
        ),
        actuators={
            "body": ImplicitActuatorCfg(
                joint_names_expr=["torso_joint_.*", "left_arm_joint_.*", "right_arm_joint_.*", "head_joint_.*"],
                stiffness=120000.0,
                damping=12000.0,
                effort_limit_sim=10000.0,
                armature=0.2,
            ),
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=[".*_gripper_finger_joint_1"],
                stiffness=_GRIPPER_DRIVER_STIFFNESS,
                damping=_GRIPPER_DRIVER_DAMPING,
                effort_limit_sim=_GRIPPER_DRIVER_EFFORT_LIMIT,
                armature=0.5,
            ),
            "fingers": ImplicitActuatorCfg(
                joint_names_expr=[".*_gripper_(left|right)_finger_joint"],
                stiffness=_GRIPPER_FINGER_STIFFNESS,
                damping=_GRIPPER_FINGER_DAMPING,
                effort_limit_sim=_GRIPPER_FINGER_EFFORT_LIMIT,
                armature=0.5,
            ),
        },
    )

    ### Cable 1 (graspable plug welded to the head; tail welded to a kinematic anchor sphere)

    plug1 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Plug1",
        spawn=sim_utils.UsdFileCfg(usd_path=_PLUG_USD),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(-0.38398558, 0.34585292, 0.5 - 0.36874688),
            rot=(0.0, -0.57096256, 0.0, 0.8209761),
        ),
    )

    # Cable tail anchor. The cable tail is pinned to a 1 mm per-env kinematic sphere
    # so it does not fall under gravity while the robot grasps and inserts the head.
    # The cable tail node is welded to this body with a Newton `add_joint_fixed`
    # constraint (see the `CableAttachmentCfg` on `cable1`). A per-env body is used
    # rather than the global world body (-1): a fixed joint to the shared world body
    # corrupts the multi-env coupled MJWarp+VBD solve (robot joints go NaN at step 0).
    anchor1 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Anchor1",
        spawn=sim_utils.SphereCfg(
            radius=0.001,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.1, 0.1)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=ANCHOR_POS),
    )

    # The deformable cable, simulated as a Cosserat rod by the VBD solver. These values track the
    # Newton waterhose success demo closely; the previous 1e8 stretch stiffness made the hose behave
    # like a rigid rod during plug motion.
    cable1 = CableObjectCfg(
        prim_path="/World/envs/env_.*/Cable1",
        spawn=sim_utils.UsdFileCfg(
            usd_path=_CABLE1_USD,
            physics_material=NewtonCableMaterialCfg(
                stretch_stiffness=1.0e6,
                bend_stiffness=20.0,
                stretch_damping=1.0e-5,
                bend_damping=1e0,
                density=1000.0,
            ),
        ),
        init_state=CableObjectCfg.InitialStateCfg(
            pos=FRIDGE_POS,
        ),
        # Keep the authored node spacing: the head plug weld's 22 mm offset is authored against
        # the original segment-0 frame, so resampling would invalidate it. None = no resampling.
        resample_segment_length=None,
        attachments=[
            # Head weld: cable segment-0 head node -> graspable Plug1 rigid body.
            CableAttachmentCfg(
                target_prim_path="/World/envs/env_.*/Plug1",
                cable_anchor=0,
                cable_local_pos=(0.0, 0.0, CABLE_HEAD_TO_PLUG_ORIGIN_LOCAL_Z),
            ),
            # Tail weld: cable last-segment start node (42) -> kinematic Anchor1 sphere.
            CableAttachmentCfg(
                target_prim_path="/World/envs/env_.*/Anchor1",
                cable_anchor=42,  # last segment start node; Anchor1 sits exactly there
            ),
        ],
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    ground: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.0, -1.05]),
        spawn=GroundPlaneCfg(),
    )


##
# MDP overrides
##


@configclass
class ActionsCfg:
    """Joint-position control of the rby1df robot.

    Actions are offsets from the default joint pose (``use_default_offset=True``), so a
    zero action holds the rest configuration. The right gripper is commanded explicitly
    through driver and follower finger joints because the imported USD mimic signs are
    disabled for Newton.
    """

    body_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["torso_joint_.*", "left_arm_joint_.*", "right_arm_joint_.*", "head_joint_.*"],
        scale=0.1,
        use_default_offset=True,
    )
    gripper_action = WaterhoseGripperPositionActionCfg(
        asset_name="robot",
        joint_names=_RIGHT_GRIPPER_JOINT_NAMES,
        open_command_expr=dict(_RIGHT_GRIPPER_OPEN_COMMAND),
        close_command_expr=dict(_RIGHT_GRIPPER_CLOSE_COMMAND),
    )


@configclass
class ObservationsCfg:
    """Policy observations for the cable plug task."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Reset events for the cable plug task."""

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (1.0, 1.0), "velocity_range": (0.0, 0.0)},
    )
    gripper_finger_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["right_gripper_leftfinger", "right_gripper_rightfinger"],
            ),
            "static_friction_range": (_GRIPPER_PAD_STATIC_FRICTION, _GRIPPER_PAD_STATIC_FRICTION),
            "dynamic_friction_range": (_GRIPPER_PAD_DYNAMIC_FRICTION, _GRIPPER_PAD_DYNAMIC_FRICTION),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 1,
        },
    )


@configclass
class RewardsCfg:
    """Reach-and-track reward shaping for the rigid plug."""

    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-1e-2)
    joint_torque = RewTerm(func=mdp.joint_torques_l2, weight=-1e-4)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-1e-4)


@configclass
class TerminationsCfg:
    """Time-out only; the cable is anchored so the plug cannot escape the workspace."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


##
# Environment configuration
##


@configclass
class WaterhoseEnvCfg(ManagerBasedRLEnvCfg):
    """Waterhose environment reusing the cable-plug MDP on an externally loaded scene."""

    scene: WaterhoseSceneCfg = WaterhoseSceneCfg(num_envs=8, env_spacing=2.5, replicate_physics=True)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self) -> None:
        _register_rby1df_gripper_mimic_override()

        # general settings
        self.decimation = 1
        self.episode_length_s = 1.0

        # simulation settings
        self.sim.dt = 1 / 100.0
        self.sim.render_interval = self.decimation
        self.sim.gravity = (0.0, 0.0, -9.81)

        kit_view = dict(
            eye=_KIT_CAMERA_EYE,
            lookat=_KIT_CAMERA_LOOKAT,
            window_width=1600,
            window_height=1600,
        )
        newton_view = dict(
            eye=_NEWTON_CAMERA_EYE,
            lookat=_NEWTON_CAMERA_LOOKAT,
            window_width=1600,
            window_height=1600,
            show_static=True,
        )
        self.sim.visualizer_cfgs = [KitVisualizerCfg(**kit_view), NewtonVisualizerCfg(**newton_view)]

        # Resolution of `--video` recordings (independent of the on-screen visualizer windows above).
        self.video_recorder.window_width = 1600
        self.video_recorder.window_height = 1600

        # Coupled physics: MuJoCo-Warp solves the articulated rby1df robot, VBD solves the
        # deformable cable (and its welded plug/anchor bodies). The two are joined by "proxy"
        # coupling: the gripper bodies are mirrored as proxy bodies in the VBD solver so the cable
        # collides against them.
        #
        # NOTE on coupling direction: proxies are fully dynamic finite-mass bodies
        # (mass = ``mass_scale * effective mass``) that exchange force with the cable in both
        # directions, so the robot feels cable/plug contacts. Keep this finite-mass proxy path as
        # the only client-facing demo path.
        self.sim.physics = CoupledNewtonCfg(
            scene_cfg=self.scene,
            use_cuda_graph=True,
            solver_cfg=CoupledSolverCfg(
                coupling_type="proxy",
                scene_cfg=self.scene,
                entries=[
                    CoupledSolverEntryCfg(
                        name="mjc",
                        solver_cfg=MJWarpSolverCfg(
                            cone="elliptic",
                            ls_parallel=True,
                            ls_iterations=20,
                            integrator="implicitfast",
                            use_mujoco_contacts=False,
                        ),
                        body_entities=[SceneEntityCfg("robot")],
                    ),
                    CoupledSolverEntryCfg(
                        name="vbd",
                        # Keep plug-era cable welds compliant: the cable head->Plug1 and
                        # tail->Anchor1 fixed joints have small authored offsets, so hard
                        # AVBD joints can inject a large startup impulse and explode the cable.
                        solver_cfg=VBDSolverCfg(
                            iterations=15,  # matches the Newton waterhose success demo for the softer cable
                            friction_epsilon=0.1,
                            rigid_contact_hard=False,
                            rigid_joint_hard=False,
                            rigid_avbd_beta=1.0e5,
                            rigid_avbd_gamma=0.999,
                            rigid_contact_history=False,
                            rigid_contact_k_start=1.0e2,
                            # The generic cable_robot example uses 128, but the authored
                            # cable/gripper/socket contacts can exceed 1000 contacts on one body,
                            # so smaller buffers overflow and poison the solve.
                            rigid_body_contact_buffer_size=4096,
                            rigid_joint_linear_ke=1.0e5,
                            rigid_joint_angular_ke=1.0e5,
                            rigid_joint_linear_k_start=1.0e4,
                            rigid_joint_angular_k_start=1.0e1,
                            rigid_joint_linear_kd=0.0,
                            rigid_joint_angular_kd=0.0,
                        ),
                        solver_class="newton.solvers:SolverVBD",
                        body_entities=[
                            SceneEntityCfg("cable1"),
                            # Weld targets must live in the VBD model so the cable fixed joint
                            # (head -> Plug1, tail -> Anchor1) can be created against them.
                            SceneEntityCfg("plug1"),
                            SceneEntityCfg("anchor1"),
                        ],
                        all_particles=True,
                        include_static_shapes=False,
                        # Pull ONLY the embedded static socket collider into the VBD solver so the
                        # VBD-owned plug collides with the bore (include_static_shapes=False keeps
                        # the cable from colliding with the ground and the rest of the fridge).
                        shape_label_patterns=[SOCKET_COLLISION_MESH_PATTERN],
                    ),
                ],
                proxy_coupling=ProxyCouplingCfg(
                    proxies=[
                        CoupledProxyCfg(
                            source="mjc",
                            destination="vbd",
                            body_entities=[
                                SceneEntityCfg(
                                    "robot",
                                    body_names=[
                                        "right_gripper_leftfinger",
                                        "right_gripper_rightfinger",
                                    ],
                                )
                            ],
                            # Use the source solver's end-of-step pose and velocity together.
                            # The lagged path copies begin pose + end velocity, which can inject
                            # inconsistent proxy motion while the gripper closes on the plug.
                            mode="staggered",
                            mass_scale=1.0,
                            collision_pipeline_factory=_make_proxy_collision_pipeline,
                            collide_interval=1,
                            # Finite rubber-pad proxy material. The standalone Newton success demo
                            # uses mu=1e6, but that script also custom-filters proxy feedback; in
                            # IsaacLab it makes the cable/plug behave glued to the fingers.
                            shape_material_ke=_VBD_CONTACT_STIFFNESS,
                            shape_material_kd=_VBD_CONTACT_DAMPING,
                            shape_material_mu=_VBD_GRIPPER_PROXY_FRICTION,
                            shape_margin=_VBD_GRIPPER_PROXY_MARGIN,
                        )
                    ],
                    iterations=1,
                ),
            ),
            num_substeps=10,
            collision_cfg=NewtonCollisionPipelineCfg(rigid_contact_max=65536),
            model_cfg=NewtonModelCfg(
                shape_material_ke=_VBD_CONTACT_STIFFNESS,
                shape_material_kd=_VBD_CONTACT_DAMPING,
                soft_contact_mu=_VBD_SOFT_CONTACT_FRICTION,
                shape_material_mu=_VBD_DEFAULT_SHAPE_FRICTION,
            ),
        )


@configclass
class WaterhoseIkActionsCfg:
    """Absolute right end-effector pose plus normalized right-gripper action for scripted demos."""

    arm_action = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["right_arm_joint_.*"],
        body_name="right_gripper_base",
        controller=DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=False,
            ik_method="dls",
            ik_params={"lambda_val": 0.05},
        ),
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(
            pos=RIGHT_GRIPPER_EE_FRAME_POS,
            rot=RIGHT_GRIPPER_EE_FRAME_QUAT_XYZW,
        ),
    )
    gripper_action = WaterhoseGripperPositionActionCfg(
        asset_name="robot",
        joint_names=_RIGHT_GRIPPER_JOINT_NAMES,
        open_command_expr=dict(_RIGHT_GRIPPER_OPEN_COMMAND),
        close_command_expr=dict(_RIGHT_GRIPPER_CLOSE_COMMAND),
    )


@configclass
class WaterhoseIkEnvCfg(WaterhoseEnvCfg):
    """Waterhose variant with an IK action space for the scripted RBY1 demo."""

    actions: WaterhoseIkActionsCfg = WaterhoseIkActionsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.episode_length_s = 30.0
        self.scene.robot.init_state.joint_pos = _RBY1_IK_INITIAL_JOINT_POS


@configclass
class WaterhoseNewtonIkActionsCfg(WaterhoseIkActionsCfg):
    """Newton-native absolute right end-effector pose plus normalized right-gripper action."""

    arm_action = NewtonInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["torso_joint_.*", "left_arm_joint_.*", "right_arm_joint_.*"],
        body_name="right_gripper_base",
        controller=NewtonIKManagerCfg(
            command_type="pose",
            use_relative_mode=False,
            iterations=24,
            lambda_initial=0.1,
            jacobian_mode="analytic",
            joint_limit_weight=10.0,
            use_persistent_seed=True,
        ),
        ik_model_source="asset_usd",
        fixed_body_names=["left_gripper_base", "torso_hip_yaw"],
        fixed_body_weights=[1.0, 50.0],
        body_offset=NewtonInverseKinematicsActionCfg.OffsetCfg(
            pos=RIGHT_GRIPPER_EE_FRAME_POS,
            rot=RIGHT_GRIPPER_EE_FRAME_QUAT_XYZW,
        ),
    )


@configclass
class WaterhoseNewtonRelativeIkActionsCfg(WaterhoseIkActionsCfg):
    """Newton-native relative end-effector delta pose plus normalized right-gripper action."""

    arm_action = NewtonInverseKinematicsActionCfg(
        class_type="isaaclab_tasks.contrib.waterhose.mdp.actions:WaterhoseLocalFrameNewtonInverseKinematicsAction",
        asset_name="robot",
        joint_names=["torso_joint_.*", "left_arm_joint_.*", "right_arm_joint_.*"],
        body_name="right_gripper_base",
        controller=NewtonIKManagerCfg(
            command_type="pose",
            use_relative_mode=True,
            iterations=24,
            lambda_initial=0.1,
            jacobian_mode="analytic",
            joint_limit_weight=10.0,
            use_persistent_seed=True,
        ),
        ik_model_source="asset_usd",
        fixed_body_names=["left_gripper_base", "torso_hip_yaw"],
        fixed_body_weights=[1.0, 50.0],
        body_offset=NewtonInverseKinematicsActionCfg.OffsetCfg(
            pos=RIGHT_GRIPPER_EE_FRAME_POS,
            rot=RIGHT_GRIPPER_EE_FRAME_QUAT_XYZW,
        ),
    )


@configclass
class WaterhoseProxyIkEnvCfg(WaterhoseEnvCfg):
    """Waterhose task with Newton proxy coupling and the scripted IK action space."""

    actions: WaterhoseNewtonIkActionsCfg = WaterhoseNewtonIkActionsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.episode_length_s = 30.0
        self.scene.robot.init_state.joint_pos = _RBY1_IK_INITIAL_JOINT_POS
        self.xr = XrCfg(
            anchor_pos=(0.0, 0.9, -1),
            # XrCfg quaternions are xyzw. Rotate the simulation 180 deg
            # around world up so the headset initially faces the fridge.
            anchor_rot=(0.0, 0.0, 1.0, 0.0),
        )
        self.isaac_teleop = IsaacTeleopCfg(
            pipeline_builder=build_waterhose_teleop_pipeline,
            sim_device=self.sim.device,
            xr_cfg=self.xr,
            target_frame_prim_path=_ROBOT_BASE_PRIM_PATH_ENV0,
            teleoperation_active_default=True,
            control_channel_uuid=None,
        )


@configclass
class WaterhoseAdmmIkEnvCfg(WaterhoseProxyIkEnvCfg):
    """Waterhose task variant using Newton ADMM contact coupling instead of proxy bodies."""

    def __post_init__(self) -> None:
        super().__post_init__()
        solver_cfg = self.sim.physics.solver_cfg
        solver_cfg.coupling_type = "admm"
        solver_cfg.use_collision_pipeline = False
        # NOTE: ADMM contact coupling is fragile for this plug grasp. With rho=50 the cable
        # explodes the instant the fingers close on the plug (each augmented-Lagrangian iteration
        # overshoots the stiff contact, so *more* iterations make it worse). rho=5 lets the grasp
        # itself succeed, but the scene still destabilizes later during transport (the plug blows
        # up while the gripper carries it toward the socket). Prefer the finite-mass proxy task
        # (Isaac-Waterhose-Coupled-v0) for a stable demo; treat this ADMM variant as experimental
        # until the ADMM contact
        # solve is hardened (lower/auto rho, contact damping, or a robust complementarity step).
        solver_cfg.admm_coupling = AdmmCouplingCfg(
            iterations=5,
            rho=5.0,
            gamma=0.1,
            baumgarte=0.01,
            contact_pairs=[
                AdmmContactPairCfg(
                    source="mjc",
                    destination="vbd",
                    detection_margin=0.005,
                )
            ],
        )


@configclass
class WaterhoseProxyTeleopEnvCfg(WaterhoseProxyIkEnvCfg):
    """Waterhose task variant for native IsaacLab keyboard and SpaceMouse teleoperation."""

    actions: WaterhoseNewtonRelativeIkActionsCfg = WaterhoseNewtonRelativeIkActionsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.isaac_teleop = IsaacTeleopCfg(
            pipeline_builder=build_waterhose_relative_teleop_pipeline,
            sim_device=self.sim.device,
            xr_cfg=self.xr,
            app_name="WaterhoseTeleop",
            target_frame_prim_path=_ROBOT_BASE_PRIM_PATH_ENV0,
            teleoperation_active_default=True,
            control_channel_uuid=None,
        )
        self.teleop_devices = DevicesCfg(
            devices={
                "keyboard": Se3KeyboardCfg(
                    pos_sensitivity=0.02,
                    rot_sensitivity=0.05,
                    sim_device=self.sim.device,
                ),
                "spacemouse": WaterhoseSpaceMouseCfg(
                    pos_sensitivity=0.05,
                    rot_sensitivity=0.15,
                    sim_device=self.sim.device,
                ),
            }
        )
