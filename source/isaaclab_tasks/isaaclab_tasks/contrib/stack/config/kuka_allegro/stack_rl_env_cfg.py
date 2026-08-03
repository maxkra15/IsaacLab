# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reinforcement-learning configuration for KUKA-Allegro cube stacking."""

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg, NewtonShapeCfg

from isaaclab.assets import ArticulationCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.contrib.stack import mdp
from isaaclab_tasks.contrib.stack.config.franka.stack_rl_env_cfg import (
    CurriculumCfg,
    FrankaCubeStackRLEnvCfg,
    RewardsCfg,
)
from isaaclab_tasks.contrib.stack.config.franka.stack_rl_env_cfg import (
    EventCfg as FrankaEventCfg,
)
from isaaclab_tasks.contrib.stack.mdp.kuka_allegro_reset import (
    KUKA_ALLEGRO_ALL_HAND_JOINT_NAMES,
    KUKA_ALLEGRO_CLOSED_PINCH_POSE,
    KUKA_ALLEGRO_DIVERSE_ARM_WORKSPACE_LOWER,
    KUKA_ALLEGRO_DIVERSE_ARM_WORKSPACE_UPPER,
    KUKA_ALLEGRO_FULL_HAND_GRASP_PAIR_CONTACT_POSES,
    KUKA_ALLEGRO_FULL_HAND_GRASP_PAIR_OPEN_COMMANDS,
    KUKA_ALLEGRO_FULL_HAND_GRASP_PAIR_OPEN_POSES,
    KUKA_ALLEGRO_FULL_HAND_GRASP_PAIR_PRELOAD_COMMANDS,
    KUKA_ALLEGRO_FULL_HAND_GRASP_PAIR_TOOL_OFFSETS,
    KUKA_ALLEGRO_GRASP_PAIR_BODY_NAMES,
    KUKA_ALLEGRO_GRASP_PAIR_CLOSED_POSES,
    KUKA_ALLEGRO_GRASP_PAIR_JOINT_NAMES,
    KUKA_ALLEGRO_GRASP_PAIR_OPEN_POSES,
    KUKA_ALLEGRO_GRASP_PAIR_TOOL_OFFSETS,
    KUKA_ALLEGRO_LARGE_CUBE_EDGE_LENGTH,
    KUKA_ALLEGRO_LARGE_CUBE_GRASP_PAIR_CLOSED_POSES,
    KUKA_ALLEGRO_LARGE_CUBE_GRASP_PAIR_OPEN_POSES,
    KUKA_ALLEGRO_LARGE_CUBE_GRASP_PAIR_TOOL_OFFSETS,
    KUKA_ALLEGRO_LARGE_CUBE_RESTING_HEIGHT,
    KUKA_ALLEGRO_OPEN_PINCH_POSE,
    KUKA_ALLEGRO_PARKED_FINGER_POSE,
    KUKA_ALLEGRO_STACK_ARM_POSES,
    KUKA_ALLEGRO_STACK_ARM_WORKSPACE_LOWER,
    KUKA_ALLEGRO_STACK_ARM_WORKSPACE_UPPER,
)

from isaaclab_assets.robots import KUKA_ALLEGRO_CFG

_ARM_JOINT_NAMES = ["iiwa7_joint_(1|2|3|4|5|6|7)"]
_PINCH_JOINT_NAMES = ["index_joint_(0|1|2|3)", "thumb_joint_(0|1|2|3)"]
_ALL_HAND_TIP_BODY_NAMES = (
    "index_biotac_tip",
    "middle_biotac_tip",
    "ring_biotac_tip",
    "thumb_biotac_tip",
)
# The combined KUKA-Allegro USD authors dedicated BioTac tip bodies in addition
# to the terminal collision links used by contact sensors. DexSuite observes
# the same ``.*_tip`` bodies for hand geometry.
_PINCH_BODY_NAMES = ("index_biotac_tip", "thumb_biotac_tip")
_PINCH_CENTER_IN_PALM = (0.0570965, -0.0375159, 0.0498749)

_OPEN_PINCH_COMMAND = {
    "index_joint_0": KUKA_ALLEGRO_OPEN_PINCH_POSE[0],
    "index_joint_1": KUKA_ALLEGRO_OPEN_PINCH_POSE[1],
    "index_joint_2": KUKA_ALLEGRO_OPEN_PINCH_POSE[2],
    "index_joint_3": KUKA_ALLEGRO_OPEN_PINCH_POSE[3],
    "thumb_joint_0": KUKA_ALLEGRO_OPEN_PINCH_POSE[4],
    "thumb_joint_1": KUKA_ALLEGRO_OPEN_PINCH_POSE[5],
    "thumb_joint_2": KUKA_ALLEGRO_OPEN_PINCH_POSE[6],
    "thumb_joint_3": KUKA_ALLEGRO_OPEN_PINCH_POSE[7],
}


def _make_kuka_allegro_event_cfg() -> FrankaEventCfg:
    """Create the production reset sampler config with KUKA hand states."""
    cfg = FrankaEventCfg()
    cfg.reset_from_state_buffer.func = mdp.FullHandLargeCubeDiverseKukaAllegroStackResetStateTable
    cfg.reset_from_state_buffer.params["closed_hand_positions"] = KUKA_ALLEGRO_CLOSED_PINCH_POSE
    cfg.reset_from_state_buffer.params["open_hand_positions"] = KUKA_ALLEGRO_OPEN_PINCH_POSE
    return cfg


@configclass
class KukaAllegroCubeStackRLEnvCfg(FrankaCubeStackRLEnvCfg):
    """Stack 8 cm cubes with independent control of all 23 robot joints.

    The task reuses Franka's order-invariant stack objective, sparse rewards,
    reset curriculum, and robot-neutral observations. Its setup methods apply
    the KUKA-Allegro robot seam, full-hand observations, diverse reset bank,
    large-cube geometry, and final 7-arm plus 16-hand control contract in that
    order.
    """

    events: FrankaEventCfg = _make_kuka_allegro_event_cfg()

    def __post_init__(self) -> None:
        super().__post_init__()

        robot_semantic_tags = self.scene.robot.spawn.semantic_tags
        center_pregrasp = KUKA_ALLEGRO_STACK_ARM_POSES[4][1]
        robot_joint_positions = {
            **{f"iiwa7_joint_{joint_id + 1}": value for joint_id, value in enumerate(center_pregrasp)},
            **_OPEN_PINCH_COMMAND,
            **{
                f"{finger}_joint_{joint_id}": value
                for finger in ("middle", "ring")
                for joint_id, value in enumerate(KUKA_ALLEGRO_PARKED_FINGER_POSE)
            },
        }
        self.scene.robot = KUKA_ALLEGRO_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.0),
                # The authored arm reaches negative X. Rotate its fixed base
                # around world Z so every shared stack layout remains valid.
                rot=(0.0, 0.0, 1.0, 0.0),
                joint_pos=robot_joint_positions,
            ),
        )
        self.scene.robot.spawn.semantic_tags = robot_semantic_tags
        self.scene.robot.spawn.rigid_props.disable_gravity = False

        # Allegro self-collision and sixteen driven hand joints create many
        # more constraints than Panda's parallel jaws. Start from the proven
        # DexSuite KUKA capacities; preserve the stack task's per-substep
        # external contact refresh and non-speculative contact geometry.
        self.sim.physics = NewtonCfg(
            solver_cfg=MJWarpSolverCfg(
                solver="newton",
                integrator="implicitfast",
                njmax=300,
                nconmax=200,
                impratio=1.0,
                cone="pyramidal",
                update_data_interval=2,
                iterations=100,
                ls_iterations=15,
                use_mujoco_contacts=False,
                ccd_iterations=35,
            ),
            num_substeps=2,
            collision_decimation=1,
            use_cuda_graph=True,
            collision_cfg=NewtonCollisionPipelineCfg(
                broad_phase="explicit",
                reduce_contacts=True,
                rigid_contact_max=4_000_000,
            ),
            default_shape_cfg=NewtonShapeCfg(margin=0.0, gap=0.0),
        )

        self.actions.arm_action = mdp.WorkspaceBoundedRelativeJointPositionActionCfg(
            asset_name="robot",
            joint_names=_ARM_JOINT_NAMES,
            scale=0.20,
            max_delta=0.12,
            workspace_lower=KUKA_ALLEGRO_STACK_ARM_WORKSPACE_LOWER,
            workspace_upper=KUKA_ALLEGRO_STACK_ARM_WORKSPACE_UPPER,
            gravity_compensation=True,
        )

        # Reinstall the KUKA reset term after the Franka base class creates its
        # own concrete event object.
        self.events = _make_kuka_allegro_event_cfg()
        self.rewards = RewardsCfg()
        self.curriculum = CurriculumCfg()

        # SceneEntityCfg resolves names into IDs in place. Give every manager
        # term its own value so resolving one term cannot mutate another.
        def arm_entity_cfg() -> SceneEntityCfg:
            return SceneEntityCfg("robot", joint_names=_ARM_JOINT_NAMES)

        def pinch_entity_cfg() -> SceneEntityCfg:
            return SceneEntityCfg("robot", joint_names=_PINCH_JOINT_NAMES, preserve_order=True)

        self.rewards.joint_vel.params["asset_cfg"] = arm_entity_cfg()

        self.observations.policy.joint_pos.params["asset_cfg"] = arm_entity_cfg()
        self.observations.policy.joint_vel.params["asset_cfg"] = arm_entity_cfg()
        self.observations.policy.object.params = {
            "tool_body_name": "palm_link",
            "tool_offset": _PINCH_CENTER_IN_PALM,
        }
        self.observations.policy.gripper_pos = self.observations.policy.gripper_pos.replace(
            func=mdp.two_finger_gripper_posture,
            params={
                "asset_cfg": pinch_entity_cfg(),
                "open_joint_positions": KUKA_ALLEGRO_OPEN_PINCH_POSE,
                "closed_joint_positions": KUKA_ALLEGRO_CLOSED_PINCH_POSE,
                "finger_joint_counts": (4, 4),
            },
        )
        self.observations.policy.eef_velocity = self.observations.policy.eef_velocity.replace(
            func=mdp.tool_velocity,
            params={
                "tool_body_name": "palm_link",
                "tool_offset": _PINCH_CENTER_IN_PALM,
            },
        )
        self.observations.policy.eef_axes = self.observations.policy.eef_axes.replace(
            func=mdp.tool_axes,
            params={
                "tool_body_name": "palm_link",
                "tool_offset": _PINCH_CENTER_IN_PALM,
            },
        )

        # Keep the physical success and adaptive-reset evidence robot-neutral:
        # both use the same pinch center and posture projection as policy
        # observations. Franka continues through the legacy scalar-jaw path.
        self.gripper_joint_names = _PINCH_JOINT_NAMES
        gripper_params = {
            "open_gripper_joint_positions": KUKA_ALLEGRO_OPEN_PINCH_POSE,
            "closed_gripper_joint_positions": KUKA_ALLEGRO_CLOSED_PINCH_POSE,
            "gripper_finger_joint_counts": (4, 4),
            "maximum_gripper_closure": 0.2,
        }
        self.terminations.progress_context.params.update(
            {
                "gripper_cfg": pinch_entity_cfg(),
                **gripper_params,
            }
        )
        self.terminations.learning_progress_context.params.update(
            {
                "tool_body_name": "palm_link",
                "tool_offset": _PINCH_CENTER_IN_PALM,
                "gripper_cfg": pinch_entity_cfg(),
                **gripper_params,
                "minimum_gripper_closure": 0.8,
            }
        )
        self.terminations.nonfinite_robot_state.params["robot_cfg"] = arm_entity_cfg()

        self.scene.ee_frame = None
        self.viewer.eye = (1.5, 1.5, 1.1)
        self.viewer.lookat = (0.48, 0.0, 0.12)

        self._configure_extended_observations()
        self._configure_diverse_resets()
        self._configure_large_cube_geometry()
        self._configure_full_hand_control()

    def _configure_extended_observations(self) -> None:
        """Add explicit active-hand, cube-yaw, and pinch-geometry observations."""

        def pinch_entity_cfg() -> SceneEntityCfg:
            return SceneEntityCfg("robot", joint_names=_PINCH_JOINT_NAMES, preserve_order=True)

        self.observations.policy.pinch_joint_pos = ObsTerm(
            func=mdp.joint_pos,
            params={"asset_cfg": pinch_entity_cfg()},
        )
        self.observations.policy.pinch_joint_vel = ObsTerm(
            func=mdp.joint_vel,
            params={"asset_cfg": pinch_entity_cfg()},
        )
        self.observations.policy.cube_x_axes = ObsTerm(func=mdp.role_conditioned_cube_x_axes)
        self.observations.policy.pinch_tip_positions = ObsTerm(
            func=mdp.body_positions_relative_to_tool,
            params={
                "body_cfg": SceneEntityCfg("robot", body_names=_PINCH_BODY_NAMES, preserve_order=True),
                "tool_body_name": "palm_link",
                "tool_offset": _PINCH_CENTER_IN_PALM,
            },
        )

    def _configure_diverse_resets(self) -> None:
        """Add the wrist- and grasp-pair-diverse reset table."""

        self.events.reset_from_state_buffer.params.update(
            {
                # The finite bank already contains coherent arm/object
                # variation. Runtime joint noise would detach cached grasps.
                "arm_joint_noise_range": 0.0,
                "table_arm_joint_noise_range": 0.0,
                "table_cube_planar_translation_range": 0.0,
                "table_cube_rotation_range": 0.0,
            }
        )

        self.actions.arm_action.workspace_lower = KUKA_ALLEGRO_DIVERSE_ARM_WORKSPACE_LOWER
        self.actions.arm_action.workspace_upper = KUKA_ALLEGRO_DIVERSE_ARM_WORKSPACE_UPPER

        pair_closure_params = {
            "joint_names_by_pair": KUKA_ALLEGRO_GRASP_PAIR_JOINT_NAMES,
            "open_joint_positions_by_pair": KUKA_ALLEGRO_GRASP_PAIR_OPEN_POSES,
            "closed_joint_positions_by_pair": KUKA_ALLEGRO_GRASP_PAIR_CLOSED_POSES,
            "finger_joint_counts": (4, 4),
        }
        pair_goal_params = {
            "grasp_pair_joint_names": KUKA_ALLEGRO_GRASP_PAIR_JOINT_NAMES,
            "grasp_pair_open_joint_positions": KUKA_ALLEGRO_GRASP_PAIR_OPEN_POSES,
            "grasp_pair_closed_joint_positions": KUKA_ALLEGRO_GRASP_PAIR_CLOSED_POSES,
        }

        self.observations.policy.object.params["grasp_pair_tool_offsets"] = KUKA_ALLEGRO_GRASP_PAIR_TOOL_OFFSETS
        self.observations.policy.gripper_pos = self.observations.policy.gripper_pos.replace(
            func=mdp.grasp_pair_gripper_posture,
            params=pair_closure_params,
        )
        self.observations.policy.eef_velocity = self.observations.policy.eef_velocity.replace(
            func=mdp.grasp_pair_tool_velocity,
            params={
                "tool_body_name": "palm_link",
                "tool_offsets_by_pair": KUKA_ALLEGRO_GRASP_PAIR_TOOL_OFFSETS,
            },
        )
        self.observations.policy.pinch_joint_pos = self.observations.policy.pinch_joint_pos.replace(
            func=mdp.grasp_pair_joint_pos,
            params={"joint_names_by_pair": KUKA_ALLEGRO_GRASP_PAIR_JOINT_NAMES},
        )
        self.observations.policy.pinch_joint_vel = self.observations.policy.pinch_joint_vel.replace(
            func=mdp.grasp_pair_joint_vel,
            params={"joint_names_by_pair": KUKA_ALLEGRO_GRASP_PAIR_JOINT_NAMES},
        )
        self.observations.policy.pinch_tip_positions = self.observations.policy.pinch_tip_positions.replace(
            func=mdp.grasp_pair_tip_positions_relative_to_tool,
            params={
                "tip_body_names_by_pair": KUKA_ALLEGRO_GRASP_PAIR_BODY_NAMES,
                "tool_body_name": "palm_link",
                "tool_offsets_by_pair": KUKA_ALLEGRO_GRASP_PAIR_TOOL_OFFSETS,
            },
        )

        self.terminations.progress_context.params.update(pair_goal_params)
        self.terminations.learning_progress_context.params.update(
            {
                "grasp_pair_tool_offsets": KUKA_ALLEGRO_GRASP_PAIR_TOOL_OFFSETS,
                **pair_goal_params,
            }
        )
        self.curriculum.reset_sampling.params.update(
            {
                # Preserve a total active epsilon prior of 3.2768 across the
                # 49,152 non-table rows.
                "epsilon": 6.666666666666667e-5,
                "balance_recipes": True,
                "balance_reset_modes": True,
            }
        )

    def _configure_large_cube_geometry(self) -> None:
        """Adapt geometry, grasps, and reset states to lightweight 8 cm cubes."""

        cube_height = KUKA_ALLEGRO_LARGE_CUBE_EDGE_LENGTH
        # Match the reset bank to the pinned Seattle collision surface. Live
        # Newton settling measures a stable center height of 36.94 mm.
        resting_cube_height = KUKA_ALLEGRO_LARGE_CUBE_RESTING_HEIGHT

        pair_closure_params = {
            "joint_names_by_pair": KUKA_ALLEGRO_GRASP_PAIR_JOINT_NAMES,
            "open_joint_positions_by_pair": KUKA_ALLEGRO_LARGE_CUBE_GRASP_PAIR_OPEN_POSES,
            "closed_joint_positions_by_pair": KUKA_ALLEGRO_LARGE_CUBE_GRASP_PAIR_CLOSED_POSES,
            "finger_joint_counts": (4, 4),
        }
        pair_goal_params = {
            "grasp_pair_joint_names": KUKA_ALLEGRO_GRASP_PAIR_JOINT_NAMES,
            "grasp_pair_open_joint_positions": KUKA_ALLEGRO_LARGE_CUBE_GRASP_PAIR_OPEN_POSES,
            "grasp_pair_closed_joint_positions": KUKA_ALLEGRO_LARGE_CUBE_GRASP_PAIR_CLOSED_POSES,
        }
        geometry_params = {
            "cube_height": cube_height,
            "xy_threshold": 0.025,
            "height_threshold": 0.012,
        }

        self.observations.policy.object.params.update(
            {
                "grasp_pair_tool_offsets": KUKA_ALLEGRO_LARGE_CUBE_GRASP_PAIR_TOOL_OFFSETS,
                **geometry_params,
            }
        )
        self.observations.policy.gripper_pos.params.update(pair_closure_params)
        self.observations.policy.eef_velocity.params["tool_offsets_by_pair"] = (
            KUKA_ALLEGRO_LARGE_CUBE_GRASP_PAIR_TOOL_OFFSETS
        )
        self.observations.policy.pinch_tip_positions.params["tool_offsets_by_pair"] = (
            KUKA_ALLEGRO_LARGE_CUBE_GRASP_PAIR_TOOL_OFFSETS
        )

        self.terminations.progress_context.params.update(
            {
                **pair_goal_params,
                **geometry_params,
            }
        )
        self.terminations.learning_progress_context.params.update(
            {
                "grasp_pair_tool_offsets": KUKA_ALLEGRO_LARGE_CUBE_GRASP_PAIR_TOOL_OFFSETS,
                **pair_goal_params,
                **geometry_params,
            }
        )

        # Keep the original 50 g mass deliberately: the larger objects are
        # lightweight blocks that improve Allegro contact geometry without
        # making acquisition eight times heavier.
        for cube in (self.scene.cube_1, self.scene.cube_2, self.scene.cube_3):
            cube.spawn.size = (cube_height, cube_height, cube_height)
            cube.init_state.pos = (
                cube.init_state.pos[0],
                cube.init_state.pos[1],
                resting_cube_height,
            )

        self.viewer.lookat = (0.48, 0.0, 0.18)

    def _configure_full_hand_control(self) -> None:
        """Install independent control and proprioception for all 23 joints."""

        # Full-hand episodes terminate falling cubes well above the inherited
        # plane at z=-1.05, so it has no physical role. Omitting it also keeps
        # headless Newton training independent of Isaac Sim's externally
        # resolved ``default_environment.usd`` asset.
        self.scene.plane = None

        self.events.reset_from_state_buffer.params["table_arm_joint_noise_range"] = 0.04
        self.events.reset_from_state_buffer.params["table_target_potential"] = 1.05

        all_hand_cfg = SceneEntityCfg(
            "robot",
            joint_names=list(KUKA_ALLEGRO_ALL_HAND_JOINT_NAMES),
            preserve_order=True,
        )
        self.actions.gripper_action = mdp.ResetPreservingRelativeJointPositionActionCfg(
            asset_name="robot",
            joint_names=list(KUKA_ALLEGRO_ALL_HAND_JOINT_NAMES),
            preserve_order=True,
            scale=0.10,
            max_delta=0.10,
            joint_limit_margin=0.02,
            reset_preload_joint_names=KUKA_ALLEGRO_ALL_HAND_JOINT_NAMES,
            reset_preload_commands_by_pair=KUKA_ALLEGRO_FULL_HAND_GRASP_PAIR_PRELOAD_COMMANDS,
            reset_open_commands_by_pair=KUKA_ALLEGRO_FULL_HAND_GRASP_PAIR_OPEN_COMMANDS,
            preload_release_threshold=0.5,
            preload_release_steps=2,
        )

        # A complete three-cube stack spans two grasps, two transports, two
        # placements, and a final release. The adaptive epsilon sampler selects
        # the competence frontier, but a final-stack-only reward
        # provides no learning signal when the policy crosses the next reset
        # milestone without completing every remaining stage. Emit exactly
        # one small, non-terminating pulse when the episode-specific reset
        # target is first reached. The target is always at least 0.25 beyond
        # the sampled state, so the pulse is neither free at reset nor
        # repeatable through oscillation. The strict released-stack pulse
        # remains four times larger (+2 versus +0.5 after dt scaling).
        self.rewards.reset_progress = RewTerm(
            func=mdp.stack_success_pulse,
            params={"context_term_name": "learning_progress_context"},
            weight=25.0,
        )

        # The pair-synergy task needs stiff 20/2 gains because one scalar
        # command must establish a grasp immediately. Independent 16-joint
        # residual control instead caps each target update. The exact 3/0.1
        # and 6/0.2 gains could not retain the 8 cm cube for one second;
        # 20/0.5 retains the screened preload. Production policies use a
        # 0.10 rad action scale and per-step cap, with the actuator effort
        # limit remaining the physical backstop.
        hand_actuator = self.scene.robot.actuators["kuka_allegro_actuators"]
        hand_expression = "(index|middle|ring|thumb)_joint_(0|1|2|3)"
        hand_actuator.stiffness[hand_expression] = 20.0
        hand_actuator.damping[hand_expression] = 0.5

        # Use one global epsilon accumulator for the fully actuated policy.
        # Equal quotas over every recipe/pair/yaw/tilt stratum keep
        # unmastered modes artificially common and prevent the competence
        # frontier from concentrating on reachable transitions.
        self.curriculum.reset_sampling.params.update(
            {
                "balance_recipes": False,
                "balance_reset_modes": False,
                "global_sampling": True,
            }
        )

        # Replace pair-gathered proprioception with complete hand state. The
        # existing pair-conditioned closure projections remain as compact,
        # normalized grasp features.
        self.observations.policy.pinch_joint_pos = None
        self.observations.policy.pinch_joint_vel = None
        self.observations.policy.pinch_tip_positions = None
        self.observations.policy.hand_joint_pos = ObsTerm(
            func=mdp.joint_pos,
            params={"asset_cfg": all_hand_cfg},
        )
        self.observations.policy.hand_joint_vel = ObsTerm(
            func=mdp.joint_vel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=list(KUKA_ALLEGRO_ALL_HAND_JOINT_NAMES),
                    preserve_order=True,
                )
            },
        )
        self.observations.policy.hand_tip_positions = ObsTerm(
            func=mdp.body_positions_relative_to_tool,
            params={
                "body_cfg": SceneEntityCfg(
                    "robot",
                    body_names=_ALL_HAND_TIP_BODY_NAMES,
                    preserve_order=True,
                ),
                "tool_body_name": "palm_link",
                "tool_offset": (0.0, 0.0, 0.0),
            },
        )
        self.observations.policy.grasp_pair = ObsTerm(
            func=mdp.grasp_pair_one_hot,
            params={"num_pairs": len(KUKA_ALLEGRO_GRASP_PAIR_JOINT_NAMES)},
        )
        self.observations.policy.object.params["grasp_pair_tool_offsets"] = (
            KUKA_ALLEGRO_FULL_HAND_GRASP_PAIR_TOOL_OFFSETS
        )
        self.observations.policy.gripper_pos.params.update(
            {
                "open_joint_positions_by_pair": KUKA_ALLEGRO_FULL_HAND_GRASP_PAIR_OPEN_POSES,
                "closed_joint_positions_by_pair": KUKA_ALLEGRO_FULL_HAND_GRASP_PAIR_CONTACT_POSES,
            }
        )
        self.observations.policy.eef_velocity.params["tool_offsets_by_pair"] = (
            KUKA_ALLEGRO_FULL_HAND_GRASP_PAIR_TOOL_OFFSETS
        )
        self.terminations.learning_progress_context.params.update(
            {
                "grasp_pair_tool_offsets": KUKA_ALLEGRO_FULL_HAND_GRASP_PAIR_TOOL_OFFSETS,
                "grasp_pair_open_joint_positions": KUKA_ALLEGRO_FULL_HAND_GRASP_PAIR_OPEN_POSES,
                "grasp_pair_closed_joint_positions": KUKA_ALLEGRO_FULL_HAND_GRASP_PAIR_CONTACT_POSES,
            }
        )

        # A fully actuated hand cannot use a selected-pair posture as proof of
        # release: one open finger and one closed finger can satisfy that
        # projection while another finger still touches the stack. Require all
        # four BioTac tips to clear every oriented cube, and include angular
        # cube speed in the stability dwell.
        self.terminations.progress_context.func = mdp.StableFullHandOrderInvariantStackGoal
        self.terminations.progress_context.params = {
            "minimum_episode_steps": 3,
            "hold_steps": 5,
            "maximum_cube_linear_velocity": 0.10,
            "maximum_cube_angular_velocity": 1.0,
            "minimum_fingertip_cube_clearance": 0.010,
            "fingertip_cfg": SceneEntityCfg(
                "robot",
                body_names=_ALL_HAND_TIP_BODY_NAMES,
                preserve_order=True,
            ),
            "xy_threshold": 0.025,
            "height_threshold": 0.012,
            "cube_height": KUKA_ALLEGRO_LARGE_CUBE_EDGE_LENGTH,
        }
