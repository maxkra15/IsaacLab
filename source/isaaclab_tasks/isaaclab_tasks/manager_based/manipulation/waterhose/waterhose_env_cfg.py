# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Waterhose manipulation environment.

MDP config groups are copied from the Franka cable-plug task
(:mod:`isaaclab_tasks.manager_based.manipulation.lift_franka_soft`), whose
``mdp`` functions are reused by import.
"""

from __future__ import annotations

import math
import os

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCollisionPipelineCfg
from isaaclab_newton.sim.spawners.materials.physics_materials_cfg import NewtonCableMaterialCfg
from isaaclab_visualizers.kit.kit_visualizer_cfg import KitVisualizerCfg
from isaaclab_visualizers.newton.newton_visualizer_cfg import NewtonVisualizerCfg

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.assets.rigid_object.rigid_object_cfg import RigidObjectCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.actions.actions_cfg import BinaryJointPositionActionCfg, DifferentialInverseKinematicsActionCfg
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
from isaaclab_contrib.coupling.coupled_manager_cfg import CoupledAdmmSolverCfg, CoupledProxySolverCfg
from isaaclab_contrib.deformable.newton_manager_cfg import (
    CoupledNewtonCfg,
    NewtonModelCfg,
    VBDSolverCfg,
)

import isaaclab.envs.mdp as mdp

WATERHOSE_ASSETS_DIR = os.environ.get(
    "WATERHOSE_ASSETS_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets"),
)

# rby1df robot: URDF converted to USD (scripts/tools/convert_urdf.py) then flattened
# into a single self-contained asset.
_RBY1_USD = os.path.join(WATERHOSE_ASSETS_DIR, "rby1df", "rby1df.usda")

# add_rod_graph places each segment's body frame at the edge's start node u (edge
# (u, v), +Z from u->v), so cable_local_pos=(0, 0, 0) welds at u. The anchor sits
# at that start node: cable001's last segment is edge (42, 43) -> u=42; cable002's
# first segment is edge (0, 1) -> u=0.
_FRIDGE_POS = (0.0, 0.0, 0.5)
_CABLE1_TAIL_NODE_42 = (-0.18810473382472992, 0.3453156650066376, -0.25986239314079285)
_CABLE1_TAIL_NODE_43 = (-0.18807558715343475, 0.3453156650066376, -0.2473306804895401)
_CABLE2_HEAD_NODE_0 = (-0.18045400083065033, 0.3453156650066376, -0.24754305183887482)
_CABLE2_HEAD_NODE_1 = (-0.18038532137870789, 0.3453156650066376, -0.25747784972190857)
_CABLE1_ANCHOR_NODE = _CABLE1_TAIL_NODE_42
_CABLE2_ANCHOR_NODE = _CABLE2_HEAD_NODE_1

# World positions of the cable fixed-end nodes = per-env kinematic anchor bodies. The
# cable welds to these (a per-env body) rather than the shared static world body: a
# fixed joint to the global world body (-1) corrupts the multi-env coupled MJWarp+VBD
# solve (robot joints go NaN at step 0).
_ANCHOR_POS = tuple(p + n for p, n in zip(_FRIDGE_POS, _CABLE1_ANCHOR_NODE))
_ANCHOR2_POS = tuple(p + n for p, n in zip(_FRIDGE_POS, _CABLE2_ANCHOR_NODE))

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
    """Cable + plug with the cable tail pinned to static anchors; sky light and ground."""

    ### Static fridge body
    fridge = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Fridge",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(WATERHOSE_ASSETS_DIR, "fridge", "fridge.usda"),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=_FRIDGE_POS),
    )

    ### Per-env kinematic anchors the cable fixed ends weld to (see _ANCHOR_POS note).
    anchor1 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Anchor1",
        spawn=sim_utils.SphereCfg(
            radius=0.001,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.1, 0.1)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=_ANCHOR_POS),
    )
    anchor2 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Anchor2",
        spawn=sim_utils.SphereCfg(
            radius=0.001,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.1, 0.1)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=_ANCHOR2_POS),
    )

    ### rby1df robot (28-DOF, fixed base). Drive gains match the reference example.
    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(usd_path=_RBY1_USD),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 1.0, -1.0),
            rot=(0.0, 0.0, -0.70710678, 0.70710678),  # 90 deg about +Z (x, y, z, w)
        ),
        actuators={
            "body": ImplicitActuatorCfg(
                joint_names_expr=["torso_joint_.*", "left_arm_joint_.*", "right_arm_joint_.*", "head_joint_.*"],
                stiffness=120000.0,
                damping=12000.0,
                effort_limit=10000.0,
                armature=0.2,
            ),
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=[".*_gripper_finger_joint_1"],
                stiffness=10000.0,
                damping=1000.0,
                effort_limit=100000.0,
                armature=0.5,
            ),
            "fingers": ImplicitActuatorCfg(
                joint_names_expr=[".*_gripper_(left|right)_finger_joint"],
                stiffness=500000.0,
                damping=10000.0,
                effort_limit=500000.0,
                armature=0.5,
            ),
        },
    )

    ### Cable 1
    plug1 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Plug1",
        spawn=sim_utils.UsdFileCfg(usd_path=os.path.join(WATERHOSE_ASSETS_DIR, "fridge", "cable", "plug.usda")),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(-0.38398558, 0.34585292, 0.5 - 0.36874688),
            rot=(0.0, -0.57096256, 0.0, 0.8209761),
        ),
    )

    cable1 = CableObjectCfg(
        prim_path="/World/envs/env_.*/Cable1",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(WATERHOSE_ASSETS_DIR, "fridge", "cable", "cable001.usda"),
            physics_material=NewtonCableMaterialCfg(
                stretch_stiffness=1e6,
                bend_stiffness=2e1,
                stretch_damping=1e-5,
                bend_damping=1e0,
                density=1000.0,
            ),
        ),
        init_state=CableObjectCfg.InitialStateCfg(
            pos=_FRIDGE_POS,
        ),
        attachments=[
            CableAttachmentCfg(
                target_prim_path="/World/envs/env_.*/Plug1",
                cable_anchor=0,
                cable_local_pos=(0.0, 0.0, 0.022),  # the head node is 22mm along +Z from the head body center
            ),
            CableAttachmentCfg(
                target_prim_path="/World/envs/env_.*/Anchor1",
                cable_anchor=42,  # last segment start node; Anchor1 sits exactly there
            ),
        ],
    )

    ### Cable 2
    plug2 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Plug2",
        spawn=sim_utils.UsdFileCfg(usd_path=os.path.join(WATERHOSE_ASSETS_DIR, "fridge", "cable", "plug.usda")),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.00921878, 0.34529759, 0.5 - 0.37485825),
            rot=(0.0, 0.52994014, 0.0, 0.84803505),
        ),
    )

    cable2 = CableObjectCfg(
        prim_path="/World/envs/env_.*/Cable2",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(WATERHOSE_ASSETS_DIR, "fridge", "cable", "cable002.usda"),
            physics_material=NewtonCableMaterialCfg(
                stretch_stiffness=1e6,
                bend_stiffness=2e1,
                stretch_damping=1e-5,
                bend_damping=1e0,
                density=1000.0,
            ),
        ),
        init_state=CableObjectCfg.InitialStateCfg(
            pos=_FRIDGE_POS,
        ),
        attachments=[
            CableAttachmentCfg(
                target_prim_path="/World/envs/env_.*/Plug2",
                cable_anchor=-1,
                cable_local_pos=(0.0, 0.0, 0.022),  # the head node is 22mm along +Z from the head body center
            ),
            CableAttachmentCfg(
                target_prim_path="/World/envs/env_.*/Anchor2",
                cable_anchor=1,  # head segment start node; Anchor2 sits exactly there
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
    zero action holds the rest configuration. Only the gripper *driver* joints
    (``*_gripper_finger_joint_1``) are actuated; the left/right finger joints follow
    them via the USD mimic joints.
    """

    body_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["torso_joint_.*", "left_arm_joint_.*", "right_arm_joint_.*", "head_joint_.*"],
        scale=0.1,
        use_default_offset=True,
    )
    gripper_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*_gripper_finger_joint_1"],
        scale=1.0,
        use_default_offset=True,
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
        params={"position_range": (0.9, 1.1), "velocity_range": (0.0, 0.0)},
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
        # general settings
        self.decimation = 1
        self.episode_length_s = 1.0

        # simulation settings
        self.sim.dt = 1 / 60.0
        self.sim.render_interval = self.decimation
        self.sim.gravity = (0.0, 0.0, -9.81)

        view = dict(eye=(-2.0, 1.5, 0.8), lookat=(0.0, 0.35, 0.2), window_width=1600, window_height=1600)
        self.sim.visualizer_cfgs = [KitVisualizerCfg(**view), NewtonVisualizerCfg(**view)]

        # Resolution of `--video` recordings (independent of the on-screen visualizer windows above).
        self.video_recorder.window_width = 1600
        self.video_recorder.window_height = 1600

        # Coupled MJWarp (articulated rby1df robot) + VBD (cables/plugs).
        self.sim.physics = CoupledNewtonCfg(
            scene_cfg=self.scene,
            use_cuda_graph=False,
            solver_cfg=CoupledAdmmSolverCfg(
                src_solver_cfg=MJWarpSolverCfg(
                    cone="elliptic",
                    ls_parallel=True,
                    ls_iterations=20,
                    integrator="implicitfast",
                ),
                dst_solver_cfg=VBDSolverCfg(
                    iterations=20,
                    rigid_avbd_beta=1e2,
                    rigid_contact_k_start=1e1,
                    rigid_body_contact_buffer_size=1024,
                ),
                src_bodies=[SceneEntityCfg("robot")],
                dst_bodies=[
                    SceneEntityCfg("cable1"),
                    SceneEntityCfg("cable2"),
                    SceneEntityCfg("plug1"),
                    SceneEntityCfg("plug2"),
                    SceneEntityCfg("anchor1"),
                    SceneEntityCfg("anchor2"),
                ],
                iterations=5,
                rho=30.0,
                gamma=0.1,
                baumgarte=0.005,
                contact_distance=0.003,
            ),
            num_substeps=8,
            collision_cfg=NewtonCollisionPipelineCfg(rigid_contact_max=65536),
            model_cfg=NewtonModelCfg(
                shape_material_ke=1.0e3,
                shape_material_kd=1.0e0,
                shape_material_mu=1.0,
            ),
        )


@configclass
class WaterhoseProxyEnvCfg(WaterhoseEnvCfg):
    """Waterhose variant using Newton's standard proxy-coupled solver."""

    def __post_init__(self) -> None:
        super().__post_init__()

        self.sim.physics.solver_cfg = CoupledProxySolverCfg(
            src_solver_cfg=MJWarpSolverCfg(
                cone="elliptic",
                ls_parallel=True,
                ls_iterations=20,
                integrator="implicitfast",
            ),
            dst_solver_cfg=VBDSolverCfg(
                iterations=20,
                rigid_avbd_beta=1e2,
                rigid_contact_k_start=1e1,
                rigid_body_contact_buffer_size=1024,
            ),
            src_bodies=[SceneEntityCfg("robot")],
            dst_bodies=[
                SceneEntityCfg("cable1"),
                SceneEntityCfg("cable2"),
                SceneEntityCfg("plug1"),
                SceneEntityCfg("plug2"),
                SceneEntityCfg("anchor1"),
                SceneEntityCfg("anchor2"),
            ],
            proxy_bodies=[
                SceneEntityCfg(
                    "robot",
                    body_names=[
                        "right_gripper_leftfinger",
                        "right_gripper_rightfinger",
                        "left_gripper_leftfinger",
                        "left_gripper_rightfinger",
                    ],
                )
            ],
            proxy_mode="lagged",
            proxy_iterations=1,
            proxy_collide_interval=1,
            proxy_mass_scale=1.0,
        )


@configclass
class WaterhoseIkActionsCfg:
    """Absolute right end-effector pose plus binary gripper action for scripted demos."""

    arm_action = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["right_arm_joint_.*"],
        body_name="right_gripper_dummy",
        controller=DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=False,
            ik_method="dls",
            ik_params={"lambda_val": 0.05},
        ),
    )
    gripper_action = BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "right_gripper_finger_joint_1",
            "right_gripper_left_finger_joint",
            "right_gripper_right_finger_joint",
        ],
        open_command_expr={
            "right_gripper_finger_joint_1": 0.09,
            "right_gripper_left_finger_joint": -0.045,
            "right_gripper_right_finger_joint": 0.045,
        },
        close_command_expr={
            "right_gripper_finger_joint_1": 0.008,
            "right_gripper_left_finger_joint": -0.004,
            "right_gripper_right_finger_joint": 0.004,
        },
    )


@configclass
class WaterhoseIkEnvCfg(WaterhoseEnvCfg):
    """Waterhose variant with an IK action space for the scripted RBY1 demo."""

    actions: WaterhoseIkActionsCfg = WaterhoseIkActionsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.episode_length_s = 30.0
        self.scene.num_envs = 1
        self.scene.robot.init_state.joint_pos = _RBY1_IK_INITIAL_JOINT_POS
        self.events.reset_robot_joints = None
