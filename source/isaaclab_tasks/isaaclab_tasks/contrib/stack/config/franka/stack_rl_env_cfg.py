# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reinforcement-learning configuration for Franka cube stacking."""

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg, NewtonShapeCfg
from isaaclab_newton.sim.schemas import NewtonMaterialPropertiesCfg

import isaaclab.sim as sim_utils
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sim.schemas import CollisionBaseCfg, RigidBodyBaseCfg, UsdPhysicsRigidBodyCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.contrib.stack import mdp
from isaaclab_tasks.contrib.stack.constants import (
    FRANKA_STACK_ARM_WORKSPACE_LOWER,
    FRANKA_STACK_ARM_WORKSPACE_UPPER,
)
from isaaclab_tasks.contrib.stack.spawners import ColoredCuboidCfg

from . import stack_joint_pos_env_cfg
from .franka_robot_cfg import FRANKA_PANDA_DEXSUITE_CFG


@configclass
class EventCfg:
    """Reset from a shared cache of validated full-task stack states."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_from_state_buffer = EventTerm(
        func=mdp.StackResetStateTable,
        mode="reset",
        params={
            "closed_finger_position": 0.020,
            "placed_finger_position": 0.021,
            "open_finger_position": 0.04,
            "neighbor_count": 8,
            # Eighteen balanced workspace/source-order layouts each receive 64
            # independently scattered table starts. This yields 1,152 broad
            # deployment starts without making the table family dominate.
            "table_rows_per_layout": 64,
            # Optional per-task override for the curriculum milestone assigned
            # to TABLE rows. None preserves each reset-table class's default.
            "table_target_potential": None,
            # Cached intermediate states are authored hand-object contacts and
            # must remain exact. Deployment-like table starts receive broad
            # robot variation and coherent planar cube transforms instead.
            "arm_joint_noise_range": 0.0,
            "table_arm_joint_noise_range": 0.080,
            "table_cube_planar_translation_range": 0.015,
            "table_cube_rotation_range": 0.45,
            "fixed_row_id": None,
            "fixed_recipe": None,
            "fixed_role_permutation": None,
            # Every cached state trains the deployment objective. Cached
            # phases are starting-state data, never local terminal goals.
            "continuation_probability": 1.0,
            "fixed_continue_to_final": True,
            "force_full_goal": True,
        },
    )


@configclass
class RewardsCfg:
    """Sparse terminal objective plus safety regularizers."""

    success = RewTerm(
        func=mdp.stack_success_pulse,
        params={"context_term_name": "progress_context"},
        # Isaac Lab multiplies this weight by the 20 ms policy step, producing
        # one +2 terminal pulse.
        weight=100.0,
    )
    failure = RewTerm(
        func=mdp.irrecoverable_stack_failure,
        params={"success_termination_name": "success"},
        weight=-0.01,
    )
    action_l2 = RewTerm(func=mdp.action_term_l2, params={"action_name": "arm_action"}, weight=-1.0e-4)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1.0e-4)
    joint_vel = RewTerm(
        func=mdp.finite_joint_velocity_l2,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["panda_joint.*"]),
            "maximum_velocity": 3.0,
        },
        weight=-1.0e-4,
    )


@configclass
class CurriculumCfg:
    """Guaranteed table starts plus adaptive epsilon intermediate states."""

    reset_sampling = CurrTerm(
        func=mdp.StackResetTableCurriculum,
        params={
            "success_context_name": "learning_progress_context",
            "final_success_context_name": "progress_context",
            # Each physical reset row owns an exact Boolean rolling window. A
            # Beta kernel concentrates sampling near 50% success while epsilon
            # keeps every row alive.
            "monitored_history_len": 50,
            "target_success_rate": 0.50,
            "kappa": 1.0,
            # Preserve a total epsilon pseudocount mass of 3.2768. Scale it
            # over the 6,786-row table so additional spatial variants do not
            # silently increase the exploration prior.
            "epsilon": 4.83e-4,
            # Preserve a deployment-facing learning stream regardless of the
            # adaptive intermediate-state distribution. The remaining 65% is
            # sampled from non-table rows by the rolling-success kernel.
            "table_sampling_probability": 0.35,
            # Keep equal layout coverage by default. Experiments can opt into
            # one flat Beta-plus-epsilon distribution over active rows.
            "global_sampling": False,
        },
    )


@configclass
class FrankaCubeStackRLEnvCfg(stack_joint_pos_env_cfg.FrankaCubeStackEnvCfg):
    """Train a color-order-invariant three-cube stack from physical reset rows."""

    rewards: RewardsCfg = RewardsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self) -> None:
        super().__post_init__()

        robot_init_state = self.scene.robot.init_state
        robot_semantic_tags = self.scene.robot.spawn.semantic_tags
        self.scene.robot = FRANKA_PANDA_DEXSUITE_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=robot_init_state,
        )
        self.scene.robot.spawn.semantic_tags = robot_semantic_tags
        # Newton needs a fixed rigid root to discover the Seattle table's
        # separately instanced visual and collision subtrees. Keep this
        # compatibility override local to the two RL tasks; Kuka inherits it.
        self.scene.table.spawn.rigid_props = [UsdPhysicsRigidBodyCfg(kinematic_enabled=True)]
        # Keep physical world gravity enabled for the arm and cubes. The arm
        # action adds Newton's configuration-dependent g(q) as joint-effort
        # feedforward on top of the DexSuite impedance controller.
        self.sim.gravity = (0.0, 0.0, -9.81)
        self.scene.robot.spawn.rigid_props.disable_gravity = False

        # Use MJWarp only as the rigid-body constraint solver. Contact
        # generation is delegated to Newton's external collision pipeline.
        # The menagerie Franka plus three cubes reached 148 constraint rows in
        # the 4,096-environment reset distribution. Keep measured headroom so
        # contacts are never silently truncated while avoiding an oversized
        # contact buffer.
        self.sim.physics = NewtonCfg(
            solver_cfg=MJWarpSolverCfg(
                solver="newton",
                integrator="implicitfast",
                njmax=256,
                nconmax=100,
                impratio=10.0,
                cone="elliptic",
                use_mujoco_contacts=False,
            ),
            # Match the fast, stable Newton Cube Lift integration cadence:
            # two 5 ms solver steps per 10 ms physics tick. Ten 1 ms substeps
            # were stable but made collection roughly eight times slower.
            num_substeps=2,
            # Refresh external contacts for every 5 ms solver step. Reusing
            # one contact set for the full 10 ms tick is sufficient for Lift,
            # but lets a released three-cube tower drift or tip.
            collision_decimation=1,
            use_cuda_graph=True,
            collision_cfg=NewtonCollisionPipelineCfg(
                broad_phase="explicit",
                reduce_contacts=True,
            ),
            # Preserve the authored 4 cm cube and finger surfaces. A positive
            # gap creates speculative contacts before those surfaces touch;
            # fresh 5 ms collision queries make that unnecessary here.
            default_shape_cfg=NewtonShapeCfg(margin=0.0, gap=0.0),
        )
        self.scene.replicate_physics = True
        # Match Cube Lift's 50 Hz policy rate: two 100 Hz physics ticks for
        # every relative joint-position command.
        self.decimation = 2
        self.sim.render_interval = 2

        # Keep the standard Franka Lift joint-position interface, but express
        # each command as a residual around the measured reset pose. The reset
        # table contains valid grasps and near-placement states; a zero-mean
        # absolute target immediately destroys those states, whereas a zero
        # residual holds them. The larger scale restores enough joint-space
        # authority to acquire cubes from table starts.
        self.actions.arm_action = mdp.WorkspaceBoundedRelativeJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            scale=0.25,
            max_delta=0.15,
            workspace_lower=FRANKA_STACK_ARM_WORKSPACE_LOWER,
            workspace_upper=FRANKA_STACK_ARM_WORKSPACE_UPPER,
            gravity_compensation=True,
        )
        self.actions.gripper_action = mdp.ResetBufferedGripperActionCfg(
            asset_name="robot",
            joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": 0.04},
            close_command_expr={"panda_finger_.*": 0.0},
            # Give Newton 100 ms to establish reset-authored finger contacts.
            # Supported place endpoints are tagged as released, so this guard
            # cannot crush a cube that already rests on the stack.
            force_close_steps=5,
        )

        # The demonstration parent config installs a different reset event set.
        self.events = EventCfg()
        self.rewards = RewardsCfg()
        self.curriculum = CurriculumCfg()
        # Side-by-side table rows must have time for two picks, two placements,
        # and release. Near-goal rows still terminate early on success.
        self.episode_length_s = 20.0

        # Keep cube identity stable through an episode. The reset table maps
        # stack roles over all six physical cube permutations, providing order
        # invariance as data augmentation without dynamically re-sorting input
        # slots at the critical grasp-to-lift transition.
        arm_cfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"])
        self.observations.policy.cube_positions = None
        self.observations.policy.cube_orientations = None
        self.observations.policy.eef_pos = None
        self.observations.policy.eef_quat = None
        self.observations.policy.joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": arm_cfg})
        self.observations.policy.joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": arm_cfg})
        self.observations.policy.object = ObsTerm(func=mdp.role_conditioned_stack_obs)
        self.observations.policy.stack_state = None
        self.observations.policy.eef_velocity = ObsTerm(func=mdp.franka_ee_velocity)
        self.observations.policy.eef_axes = ObsTerm(func=mdp.franka_ee_axes)
        self.observations.policy.concatenate_terms = True
        self.observations.rgb_camera = None
        self.observations.subtask_terms = None
        self.scene.ee_frame = None

        # Keep the default viewer focused on the task workspace.
        self.viewer.eye = (1.4, 1.4, 0.9)
        self.viewer.lookat = (0.5, 0.0, 0.1)
        self.viewer.origin_type = "env"
        self.viewer.env_index = 0

        # Native cuboids avoid legacy block materials. Their standard geometry
        # owns both collision and rendering; a USD displayColor keeps the
        # colors available in both Kit and kitless Newton visualization.
        cube_colors = ((0.05, 0.15, 0.80), (0.80, 0.05, 0.05), (0.05, 0.65, 0.10))
        for cube, color in zip((self.scene.cube_1, self.scene.cube_2, self.scene.cube_3), cube_colors, strict=True):
            cube.spawn = ColoredCuboidCfg(
                size=(0.04, 0.04, 0.04),
                display_color=color,
                rigid_props=RigidBodyBaseCfg(disable_gravity=False),
                collision_props=CollisionBaseCfg(contact_offset=0.0, rest_offset=0.0),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
                physics_material=NewtonMaterialPropertiesCfg(
                    static_friction=1.0,
                    dynamic_friction=0.8,
                    restitution=0.0,
                    torsional_friction=0.002,
                    rolling_friction=0.0001,
                    # Reduce visible resting overlap in a three-cube tower.
                    # Newton's 2.5 kN/m fallback shortens the 12 cm tower by
                    # roughly 0.9 mm; these gains reduce that to about 0.25 mm.
                    contact_stiffness=1.0e4,
                    contact_damping=200.0,
                ),
            )

        self.terminations.progress_context = DoneTerm(
            func=mdp.StableOrderInvariantStackGoal,
            params={
                "minimum_episode_steps": 3,
                "hold_steps": 5,
                "maximum_cube_velocity": 0.10,
                "minimum_finger_release_position": 0.023,
            },
        )
        self.terminations.learning_progress_context = DoneTerm(
            func=mdp.StackResetLearningProgress,
            params={"minimum_episode_steps": 3},
        )
        self.terminations.success = DoneTerm(
            func=mdp.success_after_minimum_horizon,
            params={
                "context_term_name": "progress_context",
                # The context already requires five consecutive stable,
                # released-stack frames. Terminate on that physical hold
                # instead of leaving a solved scene alive for five seconds,
                # which invites post-success oscillation and reward cycling.
                "minimum_episode_length_s": 0.1,
            },
        )
        self.terminations.time_out = DoneTerm(
            func=mdp.time_out,
            time_out=True,
        )
        self.terminations.nonfinite_robot_state = DoneTerm(
            func=mdp.nonfinite_robot_state,
            params={"robot_cfg": arm_cfg},
        )
        self.terminations.nonfinite_cube_state = DoneTerm(func=mdp.nonfinite_cube_state)
        self.terminations.cube_workspace_invalid = DoneTerm(
            func=mdp.cube_out_of_workspace,
        )

    def play_mode(self) -> None:
        """Configure randomized table starts for policy evaluation."""
        super().play_mode()
        self.scene.env_spacing = 2.5
        self.episode_length_s = 30.0
        # Sample only from the randomized table partition. Avoid a brittle
        # numeric row ID so evaluation follows cache-size changes.
        self.events.reset_from_state_buffer.params["fixed_row_id"] = None
        self.events.reset_from_state_buffer.params["fixed_recipe"] = int(mdp.StackResetRecipe.TABLE)
        self.events.reset_from_state_buffer.params["force_full_goal"] = True
        self.curriculum = None
