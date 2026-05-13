# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg import LiftEnvCfg

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab_assets.robots.rby1df import RBY1DF_CFG  # isort: skip


CONTROLLED_JOINTS = ["right_arm_joint_[1-7]", "right_gripper_finger_joint_1"]
RIGHT_ARM_JOINTS = ["right_arm_joint_[1-7]"]
RIGHT_GRIPPER_JOINTS = ["right_gripper_finger_joint_1"]

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


def _controlled_joints() -> SceneEntityCfg:
    return SceneEntityCfg("robot", joint_names=CONTROLLED_JOINTS)


def _cube_cfg() -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.55, -0.25, 0.055], rot=[0, 0, 0, 1]),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            scale=(0.8, 0.8, 0.8),
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
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
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set RBY1DF as robot.  The base is fixed at floor height while the table top remains near z=0.
        self.scene.robot = RBY1DF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Use Newton's MJWarp manager/solver for the rigid robot/object simulation.
        self.sim.physics = NewtonCfg(
            solver_cfg=MJWarpSolverCfg(
                solver="newton",
                integrator="implicitfast",
                njmax=300,
                nconmax=120,
                impratio=10.0,
                cone="elliptic",
                update_data_interval=2,
                iterations=100,
                ls_iterations=15,
                use_mujoco_contacts=True,
                ccd_iterations=5000,
            ),
            num_substeps=2,
            debug_mode=False,
        )

        # Keep the first task narrow: train only the right arm and one gripper command joint.
        self.observations.policy.joint_pos.params = {"asset_cfg": _controlled_joints()}
        self.observations.policy.joint_vel.params = {"asset_cfg": _controlled_joints()}
        self.rewards.joint_vel.params["asset_cfg"] = _controlled_joints()

        # Set actions for right-arm cube manipulation.
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=RIGHT_ARM_JOINTS,
            scale=0.35,
            use_default_offset=True,
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=RIGHT_GRIPPER_JOINTS,
            open_command_expr={"right_gripper_finger_joint_1": 0.08},
            close_command_expr={"right_gripper_finger_joint_1": 0.0},
        )

        # The command term needs a physical body.  The actual TCP is represented as an offset below.
        self.commands.object_pose.body_name = "right_arm_tool_flange"
        self.commands.object_pose.ranges.pos_x = (0.45, 0.65)
        self.commands.object_pose.ranges.pos_y = (-0.4, -0.1)
        self.commands.object_pose.ranges.pos_z = (1.12, 1.35)

        # Set Cube as object in front of the right gripper.
        self.scene.object = _cube_cfg()
        self.events.reset_object_position.params["pose_range"] = {
            "x": (-0.05, 0.05),
            "y": (-0.1, 0.1),
            "z": (0.0, 0.0),
        }

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
