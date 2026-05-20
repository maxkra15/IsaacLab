# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonShapeCfg
from isaaclab_newton.sim.schemas import NewtonCollisionPropertiesCfg, NewtonMaterialPropertiesCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg import LiftEnvCfg

from . import mdp as rby1df_mdp

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab_assets.robots.rby1df import RBY1DF_FIXED_BASE_CFG  # isort: skip


RIGHT_ARM_JOINTS = ["right_arm_joint_[1-7]"]
RIGHT_GRIPPER_JOINTS = [
    "right_gripper_finger_joint_1",
    "right_gripper_left_finger_joint",
    "right_gripper_right_finger_joint",
]
SUPPORT_JOINTS = [
    "torso_joint_[1-6]",
    "head_joint_[1-2]",
    "left_arm_joint_[1-7]",
    "left_gripper_.*finger_joint.*",
]
RBY1DF_RIGHT_GRIPPER_OPEN_JOINT_POS = (0.10, -0.05, 0.05)
RBY1DF_RIGHT_GRIPPER_CLOSE_JOINT_POS = (0.055, -0.0275, 0.0275)
RBY1DF_RIGHT_GRIPPER_OPEN_COMMAND = dict(zip(RIGHT_GRIPPER_JOINTS, RBY1DF_RIGHT_GRIPPER_OPEN_JOINT_POS))
RBY1DF_RIGHT_GRIPPER_CLOSE_COMMAND = dict(zip(RIGHT_GRIPPER_JOINTS, RBY1DF_RIGHT_GRIPPER_CLOSE_JOINT_POS))
RIGHT_TASK_JOINTS = [*RIGHT_ARM_JOINTS, *RIGHT_GRIPPER_JOINTS]

# URDF chain: right_arm_tool_flange -> attach_right_hand -> right_gripper_end_effector_joint.
BASE_PRIM = "{ENV_REGEX_NS}/Robot/Geometry/origin"
RIGHT_TOOL_FLANGE_PRIM = "/".join(
    [
        BASE_PRIM,
        "torso_anckle_roll",
        "torso_anckle_pitch",
        "torso_knee_pitch",
        "torso_hip_pitch",
        "torso_hip_roll",
        "torso_hip_yaw",
        "right_arm_shoulder_pitch",
        "right_arm_shoulder_roll",
        "right_arm_shoulder_yaw",
        "right_arm_elbow_pitch",
        "right_arm_wrist_yaw",
        "right_arm_wrist_pitch",
        "right_arm_tool_flange",
    ]
)
RBY1DF_TCP_OFFSET_POS = (0.0, 0.0, -0.2596)
RBY1DF_TCP_OFFSET_ROT = (0.70710678, 0.70710678, 0.0, 0.0)
RBY1DF_ROBOT_INIT_POS = (-0.60, 0.0, -1.05)
RBY1DF_TABLE_SIZE = (1.0, 1.2, 0.04)
RBY1DF_CUBE_SIZE = (0.05, 0.05, 0.05)
RBY1DF_CUBE_MASS = 0.05
RBY1DF_TABLE_TOP_Z = 0.03
RBY1DF_CUBE_RESET_CLEARANCE = 0.01
RBY1DF_CUBE_INIT_POS = (
    0.5,
    0.0,
    RBY1DF_TABLE_TOP_Z + RBY1DF_CUBE_SIZE[2] / 2.0 + RBY1DF_CUBE_RESET_CLEARANCE,
)
RBY1DF_TABLE_INIT_POS = (0.5, 0.0, RBY1DF_TABLE_TOP_Z - RBY1DF_TABLE_SIZE[2] / 2.0)
RBY1DF_OBJECT_DROP_HEIGHT = RBY1DF_TABLE_TOP_Z
RBY1DF_OBJECT_LIFT_HEIGHT = RBY1DF_TABLE_TOP_Z + 0.04
RBY1DF_CONTACT_MARGIN = 1.0e-4
RBY1DF_CONTACT_GAP = 1.0e-3
RBY1DF_NEWTON_SUBSTEPS = 10
RBY1DF_ARM_ACTION_SCALE = 0.5
RBY1DF_REACHING_STD = 1.0
RBY1DF_SUPPORT_STIFFNESS = 45000.0
RBY1DF_SUPPORT_DAMPING = 4500.0
RBY1DF_SUPPORT_EFFORT_LIMIT = 1000.0
RBY1DF_SUPPORT_ARMATURE = 0.2
RBY1DF_CUBE_RESET_RANGE = {
    "x": (-0.1, 0.1),
    "y": (-0.25, 0.25),
    "z": (RBY1DF_CUBE_RESET_CLEARANCE, RBY1DF_CUBE_RESET_CLEARANCE),
}
RBY1DF_PREGRASP_JOINT_POS = {
    # Torso joints are not policy-controlled; this posture simply puts the fixed body near the tabletop grasp.
    "torso_joint_1": -0.1208304,
    "torso_joint_2": 0.4030448,
    "torso_joint_3": -0.0539504,
    "torso_joint_4": 0.4444314,
    "torso_joint_5": 0.0492970,
    "torso_joint_6": -0.1618403,
    "right_arm_joint_1": -0.5883926,
    "right_arm_joint_2": -1.0626243,
    "right_arm_joint_3": 0.0486137,
    "right_arm_joint_4": -1.7432801,
    "right_arm_joint_5": 2.4583492,
    "right_arm_joint_6": 0.9467512,
    "right_arm_joint_7": -1.0541205,
}


def _controlled_joints() -> SceneEntityCfg:
    return SceneEntityCfg("robot", joint_names=RIGHT_TASK_JOINTS)


def _right_gripper_joints() -> SceneEntityCfg:
    return SceneEntityCfg("robot", joint_names=RIGHT_GRIPPER_JOINTS, preserve_order=True)


def _support_joints() -> SceneEntityCfg:
    return SceneEntityCfg("robot", joint_names=SUPPORT_JOINTS)


def _rby1df_robot_cfg(base_cfg):
    robot_cfg = base_cfg.copy()
    robot_cfg.prim_path = "{ENV_REGEX_NS}/Robot"
    # The shared lift scene is a tabletop scene. Spawn RBY1DF on the floor in front of it.
    robot_cfg.init_state.pos = RBY1DF_ROBOT_INIT_POS
    robot_cfg.init_state.joint_pos = {
        **robot_cfg.init_state.joint_pos,
        **RBY1DF_PREGRASP_JOINT_POS,
        **RBY1DF_RIGHT_GRIPPER_OPEN_COMMAND,
    }
    for actuator_name in ("torso", "left_arm", "head"):
        actuator = robot_cfg.actuators[actuator_name]
        actuator.stiffness = RBY1DF_SUPPORT_STIFFNESS
        actuator.damping = RBY1DF_SUPPORT_DAMPING
        actuator.effort_limit_sim = RBY1DF_SUPPORT_EFFORT_LIMIT
        actuator.armature = RBY1DF_SUPPORT_ARMATURE
    return robot_cfg


def _table_cfg() -> AssetBaseCfg:
    return AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=RBY1DF_TABLE_INIT_POS, rot=(0.0, 0.0, 0.0, 1.0)),
        spawn=sim_utils.CuboidCfg(
            size=RBY1DF_TABLE_SIZE,
            collision_props=NewtonCollisionPropertiesCfg(
                collision_enabled=True,
                contact_margin=RBY1DF_CONTACT_MARGIN,
                contact_gap=RBY1DF_CONTACT_GAP,
            ),
            physics_material=NewtonMaterialPropertiesCfg(
                static_friction=0.8,
                dynamic_friction=0.8,
                torsional_friction=0.005,
                rolling_friction=0.0001,
            ),
        ),
    )


def _cube_cfg() -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=RBY1DF_CUBE_INIT_POS, rot=[0, 0, 0, 1]),
        spawn=sim_utils.CuboidCfg(
            size=RBY1DF_CUBE_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            collision_props=NewtonCollisionPropertiesCfg(
                collision_enabled=True,
                contact_margin=RBY1DF_CONTACT_MARGIN,
                contact_gap=RBY1DF_CONTACT_GAP,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=RBY1DF_CUBE_MASS),
            physics_material=NewtonMaterialPropertiesCfg(
                static_friction=0.8,
                dynamic_friction=0.8,
                torsional_friction=0.005,
                rolling_friction=0.0001,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.45, 0.8)),
        ),
    )


def _right_tcp_frame_cfg() -> FrameTransformerCfg:
    marker_cfg = FRAME_MARKER_CFG.copy()
    marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    marker_cfg.prim_path = "/Visuals/FrameTransformer"
    return FrameTransformerCfg(
        prim_path=BASE_PRIM,
        debug_vis=False,
        visualizer_cfg=marker_cfg,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path=RIGHT_TOOL_FLANGE_PRIM,
                name="end_effector",
                offset=OffsetCfg(pos=RBY1DF_TCP_OFFSET_POS, rot=RBY1DF_TCP_OFFSET_ROT),
            ),
        ],
    )


@configclass
class RBY1DFCubeLiftEnvCfg(LiftEnvCfg):
    """RBY1DF cube-lift scene and shared MDP setup.

    The default registered RBY1DF task uses joint-position control to mirror the
    canonical Franka cube-lift training setup. The IK variant is kept as a
    separate task in ``ik_rel_env_cfg``.
    """

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set RBY1DF as robot.
        self.scene.num_envs = 4096
        # Newton multi-world simulation is more stable when replicated worlds are not physically offset far apart.
        self.scene.env_spacing = 0.0
        self.scene.robot = _rby1df_robot_cfg(RBY1DF_FIXED_BASE_CFG)
        # Keep the tabletop as a simple static cuboid for deterministic cube contact.
        self.scene.table = _table_cfg()

        # Use Newton's MJWarp manager/solver for the rigid robot/object simulation.
        self.sim.physics = NewtonCfg(
            solver_cfg=MJWarpSolverCfg(
                solver="newton",
                integrator="implicitfast",
                njmax=2000,
                nconmax=1000,
                impratio=1000.0,
                cone="elliptic",
                update_data_interval=2,
                iterations=20,
                ls_iterations=100,
                use_mujoco_contacts=False,
                ccd_iterations=5000,
            ),
            num_substeps=RBY1DF_NEWTON_SUBSTEPS,
            default_shape_cfg=NewtonShapeCfg(margin=RBY1DF_CONTACT_MARGIN, gap=RBY1DF_CONTACT_GAP),
            debug_mode=False,
        )
        self.sim.use_newton_actuators = True

        # Expose only the right-arm task joints to the policy and velocity penalty.
        self.observations.policy.joint_pos.params = {"asset_cfg": _controlled_joints()}
        self.observations.policy.joint_vel.params = {"asset_cfg": _controlled_joints()}
        self.rewards.joint_vel.params["asset_cfg"] = _controlled_joints()
        self.rewards.reaching_object.params["std"] = RBY1DF_REACHING_STD
        self.rewards.lifting_object.params["minimal_height"] = RBY1DF_OBJECT_LIFT_HEIGHT
        self.rewards.object_goal_tracking.params["minimal_height"] = RBY1DF_OBJECT_LIFT_HEIGHT
        self.rewards.object_goal_tracking_fine_grained.params["minimal_height"] = RBY1DF_OBJECT_LIFT_HEIGHT
        self.terminations.object_dropping.params["minimum_height"] = RBY1DF_OBJECT_DROP_HEIGHT
        self.events.reset_all.func = rby1df_mdp.reset_articulations_to_default
        # Keep the non-task joints servo-held at their default posture without adding them to the policy action space.
        self.events.reset_all.params["reset_joint_state"] = True
        self.events.reset_all.params["reset_joint_targets"] = True
        self.events.reset_right_gripper_open = EventTerm(
            func=rby1df_mdp.reset_joints_to_positions,
            mode="reset",
            params={
                "joint_positions": RBY1DF_RIGHT_GRIPPER_OPEN_JOINT_POS,
                "asset_cfg": _right_gripper_joints(),
            },
        )
        self.events.lock_support_joints = EventTerm(
            func=rby1df_mdp.reset_articulations_to_default,
            mode="interval",
            interval_range_s=(0.0, 0.0),
            params={
                "reset_joint_state": True,
                "reset_joint_targets": True,
                "asset_cfg": _support_joints(),
            },
        )

        # Keep Franka's action authority but use a lower policy std for gentler initial exploration.
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=RIGHT_ARM_JOINTS,
            scale=RBY1DF_ARM_ACTION_SCALE,
            use_default_offset=True,
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=RIGHT_GRIPPER_JOINTS,
            open_command_expr=RBY1DF_RIGHT_GRIPPER_OPEN_COMMAND,
            close_command_expr=RBY1DF_RIGHT_GRIPPER_CLOSE_COMMAND,
        )

        # The command term needs a physical body; the TCP itself is represented by the frame offset below.
        self.commands.object_pose.body_name = "right_arm_tool_flange"
        self.commands.object_pose.ranges.pos_x = (
            0.4 - RBY1DF_ROBOT_INIT_POS[0],
            0.6 - RBY1DF_ROBOT_INIT_POS[0],
        )
        self.commands.object_pose.ranges.pos_y = (-0.25, 0.25)
        self.commands.object_pose.ranges.pos_z = (
            0.25 - RBY1DF_ROBOT_INIT_POS[2],
            0.5 - RBY1DF_ROBOT_INIT_POS[2],
        )

        # Set Cube at the same world-space initial pose as the Franka cube task.
        self.scene.object = _cube_cfg()
        self.events.reset_object_position.params["pose_range"] = RBY1DF_CUBE_RESET_RANGE

        # Listen to the right-arm TCP, expressed as an offset from the robust physical flange body.
        self.scene.ee_frame = _right_tcp_frame_cfg()


@configclass
class RBY1DFCubeLiftEnvCfg_PLAY(RBY1DFCubeLiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
