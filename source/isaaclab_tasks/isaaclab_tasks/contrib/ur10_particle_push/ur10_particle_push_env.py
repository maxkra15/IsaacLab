# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Direct UR10 policy task for sweeping granular media into a side-mounted bin."""

from __future__ import annotations

import math
from collections.abc import Sequence

import newton
import torch
import warp as wp
from isaaclab_newton.cloner import newton_builder_clone_source, newton_builder_world_hook
from isaaclab_newton.ik import (
    NewtonIKJointLimitObjectiveCfg,
    NewtonIKPoseObjectiveCfg,
    NewtonIKSolver,
    NewtonIKSolverCfg,
)
from isaaclab_newton.physics import NewtonMPMManager

import isaaclab.sim as sim_utils
from isaaclab import cloner
from isaaclab.envs import DirectRLEnv
from isaaclab.envs.utils.spaces import spec_to_gym_space
from isaaclab.utils.math import quat_apply

from .reset_randomization import (
    PushResetPoseBank,
    build_particle_lattice_crop_mask,
    build_reset_paddle_targets,
    build_reset_pose_curriculum_levels,
    build_reset_pose_source_pile_indices,
    build_staged_particle_reset,
    sample_correlated_particle_translation,
    sample_particle_lattice_crop_resolution,
)
from .ur10_particle_push_env_cfg import UR10ParticlePushEnvCfg, configure_sparse_mpm_capacities


@wp.kernel(enable_backward=False)
def _mark_penetrating_reset_contacts(
    contact_count: wp.array(dtype=wp.int32),
    contact_max: int,
    contact_shape0: wp.array(dtype=wp.int32),
    contact_shape1: wp.array(dtype=wp.int32),
    contact_point0: wp.array(dtype=wp.vec3),
    contact_point1: wp.array(dtype=wp.vec3),
    contact_normal: wp.array(dtype=wp.vec3),
    contact_margin0: wp.array(dtype=wp.float32),
    contact_margin1: wp.array(dtype=wp.float32),
    shape_body: wp.array(dtype=wp.int32),
    shape_table: wp.array(dtype=wp.int32),
    body_world: wp.array(dtype=wp.int32),
    body_q: wp.array(dtype=wp.transform),
    body_is_robot: wp.array(dtype=wp.int32),
    body_tests_table: wp.array(dtype=wp.int32),
    penetration_tolerance: float,
    colliding_worlds: wp.array(dtype=wp.int32),
):
    """Mark reset worlds with penetrating robot self-contact or distal table contact."""
    contact_index = wp.tid()
    if contact_index >= contact_max or contact_index >= contact_count[0]:
        return
    shape0 = contact_shape0[contact_index]
    shape1 = contact_shape1[contact_index]
    body0 = shape_body[shape0]
    body1 = shape_body[shape1]
    table0 = shape_table[shape0] != 0
    table1 = shape_table[shape1] != 0

    robot_body = -1
    if table0 and body1 >= 0:
        robot_body = body1
    elif table1 and body0 >= 0:
        robot_body = body0
    elif not table0 and not table1 and body0 >= 0 and body1 >= 0:
        # The cloned source contains the complete workcell, including touching table/bin
        # members. Only robot-robot pairs are self-collision candidates; scene-scene and
        # robot-scene pairs are outside this dedicated self/table reset screen.
        if body_is_robot[body0] == 0 or body_is_robot[body1] == 0:
            return
        world0 = body_world[body0]
        world1 = body_world[body1]
        if world0 < 0 or world0 != world1:
            return
        point0_w = wp.transform_point(body_q[body0], contact_point0[contact_index])
        point1_w = wp.transform_point(body_q[body1], contact_point1[contact_index])
        separation = wp.dot(contact_normal[contact_index], point1_w - point0_w)
        separation = separation - contact_margin0[contact_index] - contact_margin1[contact_index]
        if separation < -penetration_tolerance:
            wp.atomic_max(colliding_worlds, world0, 1)
        return
    else:
        return

    if body_tests_table[robot_body] == 0:
        return
    world = body_world[robot_body]
    transform0 = wp.transform_identity()
    transform1 = wp.transform_identity()
    if body0 >= 0:
        transform0 = body_q[body0]
    if body1 >= 0:
        transform1 = body_q[body1]
    point0_w = wp.transform_point(transform0, contact_point0[contact_index])
    point1_w = wp.transform_point(transform1, contact_point1[contact_index])
    separation = wp.dot(contact_normal[contact_index], point1_w - point0_w)
    separation = separation - contact_margin0[contact_index] - contact_margin1[contact_index]
    if separation < -penetration_tolerance:
        wp.atomic_max(colliding_worlds, world, 1)


def update_success_streak(
    previous_streak: torch.Tensor,
    bin_fraction: torch.Tensor,
    spill_fraction: torch.Tensor,
    *,
    success_fraction: float | torch.Tensor,
    max_spill_fraction: float,
    dwell_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Update a persistent delivery predicate.

    The dwell requirement rejects transient fly-through occupancy. Global particle RMS speed is
    deliberately not part of success: motion in the much larger undelivered source pile must not
    veto a valid partial-curriculum delivery.
    """
    qualifies = (bin_fraction >= success_fraction) & (spill_fraction <= max_spill_fraction)
    next_streak = torch.where(
        qualifies,
        previous_streak + 1,
        torch.zeros_like(previous_streak),
    )
    return next_streak, next_streak >= dwell_steps


def update_curriculum_levels(
    level: torch.Tensor,
    success_streak: torch.Tensor,
    failure_streak: torch.Tensor,
    episode_success: torch.Tensor,
    *,
    max_level: int,
    successes_to_promote: int,
    failures_to_demote: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Advance or back off independent competence levels from completed episodes."""
    next_success_streak = torch.where(
        episode_success,
        success_streak + 1,
        torch.zeros_like(success_streak),
    )
    next_failure_streak = torch.where(
        episode_success,
        torch.zeros_like(failure_streak),
        failure_streak + 1,
    )
    promotion_ready = next_success_streak >= successes_to_promote
    demotion_ready = next_failure_streak >= failures_to_demote
    promoted = promotion_ready & (level < max_level)
    demoted = demotion_ready & (level > 0)
    next_level = (level + promoted.long() - demoted.long()).clamp(0, max_level)
    next_success_streak = torch.where(
        promotion_ready | demoted,
        torch.zeros_like(next_success_streak),
        next_success_streak,
    )
    next_failure_streak = torch.where(
        demotion_ready | promoted,
        torch.zeros_like(next_failure_streak),
        next_failure_streak,
    )
    return next_level, next_success_streak, next_failure_streak, promoted, demoted


def compute_masked_particle_mean(
    values: torch.Tensor,
    active_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Reduce a per-particle tensor without letting inactive reserve slots bias it."""
    if active_mask is None:
        return values.mean(dim=1)
    if active_mask.shape != values.shape[:2]:
        raise ValueError(f"active_mask must have shape {values.shape[:2]}, got {tuple(active_mask.shape)}.")
    weights = active_mask.to(dtype=values.dtype)
    while weights.ndim < values.ndim:
        weights = weights.unsqueeze(-1)
    denominator = active_mask.sum(dim=1).clamp_min(1).to(dtype=values.dtype)
    while denominator.ndim < values.ndim - 1:
        denominator = denominator.unsqueeze(-1)
    return (values * weights).sum(dim=1) / denominator


def compute_particle_metrics(
    particle_position_e: torch.Tensor,
    cfg: UR10ParticlePushEnvCfg,
    active_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute terminal delivery and irreversible-spill fractions from particle positions."""
    if active_mask is None:
        active_mask = torch.ones(particle_position_e.shape[:2], dtype=torch.bool, device=particle_position_e.device)
    elif active_mask.shape != particle_position_e.shape[:2]:
        raise ValueError(
            f"active_mask must have shape {particle_position_e.shape[:2]}, got {tuple(active_mask.shape)}."
        )
    active_count = active_mask.sum(dim=1).clamp_min(1)
    x, y, z = particle_position_e.unbind(dim=-1)
    delivered_below_rim = (
        (x >= cfg.bin_inner_x_bounds[0])
        & (x <= cfg.bin_inner_x_bounds[1])
        & (y >= cfg.bin_inner_y_bounds[0])
        & (y <= cfg.bin_inner_y_bounds[1])
        & (z >= cfg.bin_inner_z_bounds[0])
        & (z <= cfg.bin_inner_z_bounds[1])
    )
    bin_fraction = (delivered_below_rim & active_mask).sum(dim=1) / active_count

    in_physical_bin = (
        (x >= cfg.bin_inner_x_bounds[0])
        & (x <= cfg.bin_inner_x_bounds[1])
        & (y >= cfg.bin_inner_y_bounds[0])
        & (y <= cfg.bin_inner_y_bounds[1])
        & (z >= cfg.bin_physical_z_bounds[0])
        & (z <= cfg.bin_physical_z_bounds[1])
    )
    below_table = (z < -0.03) & ~in_physical_bin
    outside_lateral_workspace = (y.abs() > 0.455) | (x < -0.2461) | (x > cfg.bin_inner_x_bounds[1] + 0.05)
    spill_fraction = ((below_table | outside_lateral_workspace) & active_mask).sum(dim=1) / active_count

    return bin_fraction, spill_fraction


def compute_capped_bin_goal_progress(
    bin_fraction: torch.Tensor,
    success_fraction: float | torch.Tensor,
) -> torch.Tensor:
    """Return a bounded delivery potential normalized by the active curriculum target."""
    target = torch.as_tensor(success_fraction, dtype=bin_fraction.dtype, device=bin_fraction.device)
    return (bin_fraction / target).clamp(0.0, 1.0)


def build_bin_goal_mask(
    cfg: UR10ParticlePushEnvCfg,
    *,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Rasterize the calibrated bin region into the overhead heightmap frame."""
    height_cells, width_cells = cfg.heightmap_shape
    x_lo, x_hi = cfg.heightmap_x_bounds
    y_lo, y_hi = cfg.heightmap_y_bounds
    x = torch.linspace(
        x_lo + 0.5 * (x_hi - x_lo) / width_cells,
        x_hi - 0.5 * (x_hi - x_lo) / width_cells,
        width_cells,
        device=device,
    )
    y = torch.linspace(
        y_lo + 0.5 * (y_hi - y_lo) / height_cells,
        y_hi - 0.5 * (y_hi - y_lo) / height_cells,
        height_cells,
        device=device,
    )
    inside_x = (x >= cfg.bin_inner_x_bounds[0]) & (x <= cfg.bin_inner_x_bounds[1])
    inside_y = (y >= cfg.bin_inner_y_bounds[0]) & (y <= cfg.bin_inner_y_bounds[1])
    return (inside_y[:, None] & inside_x[None, :]).float()


def compute_transport_progress(
    particle_position_e: torch.Tensor,
    cfg: UR10ParticlePushEnvCfg,
    start_x: float | torch.Tensor | None = None,
    active_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return bounded source-centroid motion from the episode reset toward the bin."""
    if start_x is None:
        start_x = cfg.pile_nominal_center[0]
    if isinstance(start_x, (float, int)) and cfg.bin_inner_x_bounds[0] <= start_x:
        raise ValueError("The bin mouth must lie in front of the source pile.")
    start = torch.as_tensor(start_x, dtype=particle_position_e.dtype, device=particle_position_e.device)
    travel = (cfg.bin_inner_x_bounds[0] - start).clamp_min(torch.finfo(particle_position_e.dtype).eps)
    centroid_x = compute_masked_particle_mean(particle_position_e[..., 0], active_mask)
    return ((centroid_x - start) / travel).clamp(0.0, 1.0)


def compute_paddle_reach_potential(
    particle_position_e: torch.Tensor,
    paddle_position_e: torch.Tensor,
    paddle_normal_e: torch.Tensor,
    paddle_vertical_e: torch.Tensor,
    cfg: UR10ParticlePushEnvCfg,
    active_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Measure an upright, laterally aligned paddle reaching the pile's trailing face.

    The pile centroid minus its nominal half-length is a smooth trailing-face estimate. The target
    extends slightly into the pile so the potential spans first contact; it then saturates, leaving
    particle transport and delivery to provide the remaining learning signal.
    """
    nominal_half_length = 0.5 * (cfg.scene.media.spawn.upper[0] - cfg.scene.media.spawn.lower[0])
    particle_centroid = compute_masked_particle_mean(particle_position_e, active_mask)
    contact_target_x = particle_centroid[:, 0] - nominal_half_length + cfg.paddle_reach_contact_depth
    normal_x_alignment = paddle_normal_e[:, 0].abs().clamp(0.0, 1.0)
    vertical_z_alignment = paddle_vertical_e[:, 2].abs().clamp(0.0, 1.0)
    paddle_front_x = paddle_position_e[:, 0] + 0.5 * cfg.paddle_size[2] * normal_x_alignment
    longitudinal_gap = (contact_target_x - paddle_front_x).clamp_min(0.0)
    longitudinal_reach = (1.0 - longitudinal_gap / cfg.paddle_reach_distance).clamp(0.0, 1.0)

    pile_center_y = particle_centroid[:, 1]
    lateral_error = (paddle_position_e[:, 1] - pile_center_y).abs()
    lateral_alignment = (1.0 - lateral_error / cfg.paddle_lateral_alignment_distance).clamp(0.0, 1.0)
    vertical_error = (paddle_position_e[:, 2] - cfg.paddle_reset_center[2]).abs()
    vertical_alignment = (1.0 - vertical_error / cfg.paddle_vertical_alignment_distance).clamp(0.0, 1.0)
    orientation_alignment = normal_x_alignment * vertical_z_alignment
    return longitudinal_reach * lateral_alignment * vertical_alignment * orientation_alignment


def compute_transition_reward(
    bin_progress: torch.Tensor,
    transport_progress_delta: torch.Tensor,
    paddle_reach_potential: torch.Tensor,
    spill_fraction: torch.Tensor,
    action_magnitude: torch.Tensor,
    action_rate: torch.Tensor,
    success: torch.Tensor,
    failure: torch.Tensor,
    cfg: UR10ParticlePushEnvCfg,
    *,
    step_dt: float,
) -> torch.Tensor:
    """Combine signed potential progress with terminal events and safety costs."""
    if not math.isfinite(step_dt) or step_dt <= 0.0:
        raise ValueError("step_dt must be finite and positive.")
    return (
        cfg.reward_success * success.float()
        + cfg.reward_bin_progress * bin_progress
        + cfg.reward_transport_progress * transport_progress_delta
        + cfg.reward_paddle_reach_potential_per_second * step_dt * paddle_reach_potential
        + cfg.penalty_spill_fraction_per_second * step_dt * spill_fraction
        + cfg.penalty_action_magnitude_per_second * step_dt * action_magnitude
        + cfg.penalty_action_rate_per_second * step_dt * action_rate
        + cfg.penalty_failure * failure.float()
    )


class UR10ParticlePushEnv(DirectRLEnv):
    """Sweep an initially randomized granular pile into an open side bin."""

    cfg: UR10ParticlePushEnvCfg

    def __init__(self, cfg: UR10ParticlePushEnvCfg, render_mode: str | None = None, **kwargs):
        # Re-resolve capacities after any CLI/Hydra num-environment override.
        configure_sparse_mpm_capacities(cfg)
        self._particle_max_velocity = float(cfg.particle_max_velocity)
        with newton_builder_world_hook(self._configure_newton_builder):
            super().__init__(cfg, render_mode, **kwargs)

        self._joint_ids, joint_names = self._robot.find_joints(
            list(self.cfg.scene.robot.init_state.joint_pos),
            preserve_order=True,
        )
        if len(self._joint_ids) != self.cfg.action_space:
            raise RuntimeError(f"Expected six UR10 arm joints, resolved {joint_names}.")
        ee_ids, _ = self._robot.find_bodies(self.cfg.ee_body_name, preserve_order=True)
        if len(ee_ids) != 1:
            raise RuntimeError(f"Expected one {self.cfg.ee_body_name!r} body, found {len(ee_ids)}.")
        self._ee_body_id = ee_ids[0]

        self._action_scale = torch.tensor(
            self.cfg.relative_joint_position_action_scale,
            device=self.device,
            dtype=torch.float32,
        )
        self._joint_velocity_limits = self._robot.data.soft_joint_vel_limits.torch[:, self._joint_ids].clamp_min(1.0e-6)
        self._particle_workspace_lower = torch.tensor(
            self.cfg.particle_workspace_lower_bound,
            device=self.device,
        )
        self._particle_workspace_upper = torch.tensor(
            self.cfg.particle_workspace_upper_bound,
            device=self.device,
        )
        self._particle_longitudinal_offset_range = torch.tensor(
            self.cfg.reset_particle_longitudinal_offset_range,
            device=self.device,
        )
        self._particle_paddle_residual_half_range = torch.tensor(
            self.cfg.reset_particle_paddle_residual_half_range,
            device=self.device,
        )
        source_shape_profiles = torch.tensor(
            self.cfg.reset_source_shape_profiles,
            dtype=torch.float32,
            device=self.device,
        )
        self._reset_source_vertical_cell_count = source_shape_profiles[:, 0].long()
        self._reset_source_footprint_aspect_ratio = source_shape_profiles[:, 1]
        self._reset_source_shape_cycle_cursor = 0
        self._paddle_offset = torch.tensor(self.cfg.paddle_offset, device=self.device).expand(self.num_envs, -1)
        self._paddle_local_normal = torch.tensor((0.0, 0.0, 1.0), device=self.device).expand(self.num_envs, -1)
        self._paddle_local_vertical = torch.tensor((1.0, 0.0, 0.0), device=self.device).expand(self.num_envs, -1)
        self._bin_mouth = torch.tensor(
            (self.cfg.bin_inner_x_bounds[0], 0.0, 0.0),
            device=self.device,
        ).expand(self.num_envs, -1)
        self._paddle_position_center = torch.tensor((0.6, 0.0, 0.17), device=self.device)
        self._paddle_position_scale = torch.tensor((0.9, 0.55, 0.65), device=self.device)

        self._actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self._previous_actions = torch.zeros_like(self._actions)
        self._joint_targets = self._robot.data.default_joint_pos.torch[:, self._joint_ids].clone()

        self._particle_reset_template_e = (
            self._media.data.default_particle_state_w.torch[..., :3] - self.scene.env_origins[:, None, :]
        ).clone()
        self._particle_active_mask = torch.ones(
            (self.num_envs, self._media.particles_per_object),
            dtype=torch.bool,
            device=self.device,
        )
        self._particle_focused_source_mask = self._particle_active_mask.clone()
        self._bin_fraction = torch.zeros(self.num_envs, device=self.device)
        self._previous_bin_fraction = torch.zeros_like(self._bin_fraction)
        self._spill_fraction = torch.zeros_like(self._bin_fraction)
        self._success_streak = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._success_this_step = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._episode_success = torch.zeros_like(self._success_this_step)
        self._transport_progress = torch.zeros_like(self._bin_fraction)
        self._previous_transport_progress = torch.zeros_like(self._bin_fraction)
        self._paddle_reach_potential = torch.zeros_like(self._bin_fraction)
        self._rms_particle_speed = torch.zeros_like(self._bin_fraction)
        self._invalid_state = torch.zeros_like(self._success_this_step)
        self._invalid_action = torch.zeros_like(self._success_this_step)
        self._escaped_workspace = torch.zeros_like(self._success_this_step)
        self._excessive_spill = torch.zeros_like(self._success_this_step)
        self._episode_reward_components = torch.zeros((self.num_envs, 8), device=self.device)

        height, width = self.cfg.heightmap_shape
        history_length = self.cfg.heightmap_history_steps + 1
        self._heightmap_xy_offset = torch.zeros((self.num_envs, 2), device=self.device)
        self._heightmap_history = torch.zeros((self.num_envs, history_length, height, width), device=self.device)
        self._heightmap_history_pointer = -1
        self._heightmap_history_step = -1
        self._heightmap_history_reset = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self._heightmap_history_reset_pending = True
        self._heightmap_goal_mask = build_bin_goal_mask(self.cfg, device=self.device).expand(self.num_envs, -1, -1)
        self._success_dwell_steps = max(1, math.ceil(self.cfg.success_dwell_time_s / self.step_dt))

        self._curriculum_pile_center_x = torch.tensor(
            self.cfg.curriculum_pile_center_x,
            device=self.device,
            dtype=torch.float32,
        )
        self._curriculum_randomization_scale = torch.tensor(
            self.cfg.curriculum_randomization_scale,
            device=self.device,
            dtype=torch.float32,
        )
        self._curriculum_initial_bin_fraction = torch.tensor(
            self.cfg.curriculum_initial_bin_fraction,
            device=self.device,
            dtype=torch.float32,
        )
        self._curriculum_source_pile_count = torch.tensor(
            self.cfg.curriculum_source_pile_count,
            device=self.device,
            dtype=torch.long,
        )
        self._curriculum_source_lateral_offset = torch.tensor(
            self.cfg.curriculum_source_lateral_offset,
            device=self.device,
            dtype=torch.float32,
        )
        if self.cfg.curriculum_level_override is not None:
            initial_level = self.cfg.curriculum_level_override
        elif self.cfg.reset_curriculum_level_cycle is not None:
            initial_level = self.cfg.reset_curriculum_level_cycle[0]
        else:
            initial_level = 0
        self._curriculum_level = torch.full(
            (self.num_envs,),
            initial_level,
            device=self.device,
            dtype=torch.long,
        )
        self._curriculum_success_streak = torch.zeros_like(self._curriculum_level)
        self._curriculum_failure_streak = torch.zeros_like(self._curriculum_level)
        self._episode_success_fraction = torch.full(
            (self.num_envs,), self.cfg.success_fraction, device=self.device, dtype=torch.float32
        )
        self._episode_start_centroid_x = self._curriculum_pile_center_x[self._curriculum_level].clone()

        paddle_position_e, paddle_quaternion = build_reset_paddle_targets(self.cfg, device=self.device)
        self._reset_pose_bank = PushResetPoseBank(
            joint_position=self._solve_reset_joint_positions(paddle_position_e, paddle_quaternion),
            paddle_position_e=paddle_position_e,
            curriculum_level=build_reset_pose_curriculum_levels(self.cfg, device=self.device),
            source_pile_index=build_reset_pose_source_pile_indices(self.cfg, device=self.device),
        )
        level_count = len(self.cfg.curriculum_pile_center_x)
        self._reset_pose_cycle_rows = [torch.empty(0, dtype=torch.long, device=self.device) for _ in range(level_count)]
        self._reset_pose_cycle_cursors = [0] * level_count
        self._reset_curriculum_level_cycle = (
            torch.tensor(self.cfg.reset_curriculum_level_cycle, dtype=torch.long, device=self.device)
            if self.cfg.reset_curriculum_level_cycle is not None
            else None
        )
        self._reset_curriculum_cycle_cursor = 0
        self._reset_pose_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._reset_initialized = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _collision_free_reset_candidates(
        self,
        prototype_builder: newton.ModelBuilder,
        candidate_q: torch.Tensor,
    ) -> torch.Tensor:
        """Screen every IK endpoint against imported self-collision and the work surface."""
        if candidate_q.ndim != 3 or candidate_q.shape[2] != prototype_builder.joint_coord_count:
            raise ValueError(
                "Reset IK candidates must have shape (rows, candidates, prototype coordinates), "
                f"got {tuple(candidate_q.shape)}."
            )
        row_count, candidate_count, coordinate_count = candidate_q.shape
        shape_cfg = newton.ModelBuilder.ShapeConfig(
            density=0.0,
            margin=0.0,
            has_shape_collision=True,
            has_particle_collision=False,
            is_visible=False,
        )
        validation_builder = newton.ModelBuilder(up_axis=prototype_builder.up_axis)
        for row in range(row_count):
            validation_builder.begin_world(label=f"push_reset_{row}")
            validation_builder.add_builder(prototype_builder)
            validation_builder.add_shape_plane(
                -1,
                xform=wp.transform_identity(),
                width=0.0,
                length=0.0,
                cfg=shape_cfg,
                label="PushReset/WorkSurface",
            )
            validation_builder.end_world()
        validation_model = validation_builder.finalize(device=self.device)
        if validation_model.world_count != row_count:
            raise RuntimeError(f"Expected {row_count} reset-validation worlds, got {validation_model.world_count}.")
        validation_coordinate_count = validation_model.joint_coord_count // row_count
        if validation_coordinate_count != coordinate_count:
            raise RuntimeError(
                "Reset-validation coordinate count does not match the IK prototype: "
                f"{validation_coordinate_count} != {coordinate_count}."
            )

        body_names = [str(label).rsplit("/", 1)[-1] for label in validation_model.body_label]
        body_is_robot = torch.as_tensor(
            ["/Robot/" in str(label) for label in validation_model.body_label],
            device=self.device,
            dtype=torch.int32,
        )
        table_test_bodies = {
            "upper_arm_link",
            "forearm_link",
            "wrist_1_link",
            "wrist_2_link",
            "wrist_3_link",
            self.cfg.ee_body_name,
            "Paddle",
        }
        body_tests_table = torch.as_tensor(
            [name in table_test_bodies for name in body_names],
            device=self.device,
            dtype=torch.int32,
        )
        shape_table = torch.as_tensor(
            [str(label).endswith("PushReset/WorkSurface") for label in validation_model.shape_label],
            device=self.device,
            dtype=torch.int32,
        )
        if int(shape_table.sum()) != row_count:
            raise RuntimeError("Reset collision validation did not create one work surface per row.")

        pipeline = newton.CollisionPipeline(
            validation_model,
            broad_phase="explicit",
            include_static_kinematic_pairs=False,
            soft_contact_max=0,
            verify_buffers=True,
        )
        contacts = pipeline.contacts()
        state = validation_model.state()
        validation_q = wp.to_torch(validation_model.joint_q).reshape(row_count, coordinate_count)
        colliding_worlds = wp.zeros(row_count, dtype=wp.int32, device=validation_model.device)
        collision_free = torch.ones((row_count, candidate_count), dtype=torch.bool, device=self.device)
        for candidate_index in range(candidate_count):
            validation_q.copy_(candidate_q[:, candidate_index].contiguous())
            newton.eval_fk(
                validation_model,
                validation_model.joint_q,
                validation_model.joint_qd,
                state,
            )
            pipeline.collide(state, contacts)
            colliding_worlds.zero_()
            wp.launch(
                _mark_penetrating_reset_contacts,
                dim=contacts.rigid_contact_max,
                inputs=[
                    contacts.rigid_contact_count,
                    contacts.rigid_contact_max,
                    contacts.rigid_contact_shape0,
                    contacts.rigid_contact_shape1,
                    contacts.rigid_contact_point0,
                    contacts.rigid_contact_point1,
                    contacts.rigid_contact_normal,
                    contacts.rigid_contact_margin0,
                    contacts.rigid_contact_margin1,
                    validation_model.shape_body,
                    wp.from_torch(shape_table, dtype=wp.int32),
                    validation_model.body_world,
                    state.body_q,
                    wp.from_torch(body_is_robot, dtype=wp.int32),
                    wp.from_torch(body_tests_table, dtype=wp.int32),
                    1.0e-4,
                ],
                outputs=[colliding_worlds],
                device=validation_model.device,
            )
            collision_free[:, candidate_index] = wp.to_torch(colliding_worlds) == 0
        wp.synchronize_device(validation_model.device)
        generated_contact_count = int(wp.to_torch(contacts.rigid_contact_count)[0])
        if generated_contact_count > contacts.rigid_contact_max:
            raise RuntimeError(
                f"Reset validation generated {generated_contact_count} contacts for capacity "
                f"{contacts.rigid_contact_max}."
            )
        return collision_free

    def _solve_reset_joint_positions(
        self,
        paddle_position_e: torch.Tensor,
        paddle_quaternion: torch.Tensor,
    ) -> torch.Tensor:
        """Solve one small, canonical-branch, collision-screened reset pose bank."""
        pose_count = self.cfg.reset_pose_count
        if paddle_position_e.shape != (pose_count, 3):
            raise ValueError(
                f"Reset paddle positions have shape {tuple(paddle_position_e.shape)}; expected {(pose_count, 3)}."
            )
        if paddle_quaternion.shape != (pose_count, 4):
            raise ValueError(
                f"Reset paddle quaternions have shape {tuple(paddle_quaternion.shape)}; expected {(pose_count, 4)}."
            )
        plan = sim_utils.SimulationContext.instance().get_clone_plan()
        resolved = cloner.query.path_to_source(plan, self._robot.cfg.prim_path) if plan is not None else None
        if resolved is None:
            raise RuntimeError(f"Could not resolve clone-plan source for {self._robot.cfg.prim_path!r}.")
        source_path = resolved[0]
        prototype_origin = -self.scene.env_origins[0]
        prototype_xform = wp.transform(wp.vec3(*prototype_origin.tolist()), wp.quat_identity())

        source_builder = newton_builder_clone_source(source_path)
        prototype_builder = newton.ModelBuilder(up_axis=source_builder.up_axis)
        prototype_builder.add_builder(source_builder, xform=prototype_xform)
        model = prototype_builder.finalize(device=self.device)

        ee_matches = [
            body_id
            for body_id, label in enumerate(model.body_label)
            if str(label).rsplit("/", 1)[-1] == self.cfg.ee_body_name
        ]
        if len(ee_matches) != 1:
            raise RuntimeError(f"Expected one {self.cfg.ee_body_name!r} body in the IK prototype, found {ee_matches}.")
        ee_body_id = ee_matches[0]
        joint_labels = [str(label).rsplit("/", 1)[-1] for label in model.joint_label]
        joint_q_start = wp.to_torch(model.joint_q_start).to(device=self.device, dtype=torch.long)

        def coordinate_id(joint_name: str) -> int:
            matches = [joint_id for joint_id, label in enumerate(joint_labels) if label == joint_name]
            if len(matches) != 1:
                raise RuntimeError(f"Expected one {joint_name!r} joint in the IK prototype, found {matches}.")
            return int(joint_q_start[matches[0]].item())

        arm_coordinate_ids = torch.tensor(
            [coordinate_id(name) for name in self.cfg.scene.robot.init_state.joint_pos],
            device=self.device,
            dtype=torch.long,
        )
        home_arm_q = self._robot.data.default_joint_pos.torch[0, self._joint_ids]
        seed_q = wp.to_torch(model.joint_q).to(device=self.device, dtype=torch.float32).clone()
        seed_q[arm_coordinate_ids] = home_arm_q

        target_name = "reset_paddle"
        seed_count = self.cfg.reset_ik_seeds
        solver = NewtonIKSolver(
            NewtonIKSolverCfg(
                optimizer="lm",
                jacobian_mode="analytic",
                sampler="gauss",
                n_seeds=seed_count,
                noise_std=self.cfg.reset_ik_noise_std,
                rng_seed=self.cfg.reset_seed,
                iterations=self.cfg.reset_ik_iterations,
                lambda_initial=0.1,
            ),
            model=model,
            num_envs=pose_count,
            device=str(model.device),
            objectives=[
                NewtonIKPoseObjectiveCfg(
                    body_name=self.cfg.ee_body_name,
                    name=target_name,
                    body_offset_pos=self.cfg.paddle_offset,
                    position_weight=100.0,
                    rotation_weight=10.0,
                ),
                NewtonIKJointLimitObjectiveCfg(weight=1.0),
            ],
            link_resolver=lambda _body_name: ee_body_id,
        )
        pose_objective = solver.objectives_by_name[target_name]
        pose_objective.position_objective.set_target_positions(
            wp.from_torch(paddle_position_e.contiguous(), dtype=wp.vec3)
        )
        pose_objective.rotation_objective.set_target_rotations(
            wp.from_torch(paddle_quaternion.contiguous(), dtype=wp.vec4)
        )

        joint_limits = self._robot.data.soft_joint_pos_limits.torch[0, self._joint_ids]
        solver.solve(
            wp.from_torch(
                seed_q.expand(pose_count, -1).contiguous(),
                dtype=wp.float32,
            )
        )
        candidate_q = wp.to_torch(solver.joint_q).reshape(pose_count, seed_count, -1).clone()
        candidate_cost = wp.to_torch(solver.costs).reshape(pose_count, seed_count).clone()
        arm_q = candidate_q[..., arm_coordinate_ids]
        joint_margin = torch.minimum(
            arm_q - joint_limits[:, 0],
            joint_limits[:, 1] - arm_q,
        ).amin(dim=-1)
        valid = (
            torch.isfinite(arm_q).all(dim=-1)
            & torch.isfinite(candidate_cost)
            & (candidate_cost <= self.cfg.reset_ik_max_cost)
            & (joint_margin >= self.cfg.joint_limit_margin)
        )
        valid &= self._collision_free_reset_candidates(prototype_builder, candidate_q)
        missing = torch.nonzero(~valid.any(dim=1), as_tuple=False).flatten()
        if missing.numel() > 0:
            raise RuntimeError(f"Newton IK found no canonical collision-free reset pose for rows {missing.tolist()}.")

        # Anchor every target to the same nominal branch. Cost breaks ties between nearby seeds;
        # squared home distance prevents equivalent UR10 revolutions from entering the reset bank.
        home_distance = torch.square(arm_q - home_arm_q).sum(dim=-1)
        score = candidate_cost + 1.0e-4 * home_distance
        score = torch.where(valid, score, torch.full_like(score, torch.inf))
        selected = score.argmin(dim=1)
        row_ids = torch.arange(pose_count, device=self.device)
        joint_position = arm_q[row_ids, selected].contiguous()
        lower = joint_limits[:, 0] + self.cfg.joint_limit_margin
        upper = joint_limits[:, 1] - self.cfg.joint_limit_margin
        if bool(torch.any((joint_position < lower) | (joint_position > upper))):
            raise RuntimeError("Selected reset pose violates the configured joint-limit margin.")
        return joint_position

    def _sample_reset_pose_ids(self, curriculum_level: torch.Tensor) -> torch.Tensor:
        """Sample collision-screened paddle poses matching each episode's curriculum level."""
        count = curriculum_level.numel()
        level_count = len(self.cfg.curriculum_pile_center_x)
        rows_per_level = self._reset_pose_bank.row_count // level_count
        if self.cfg.reset_cycle:
            sampled = torch.empty_like(curriculum_level)
            for level_tensor in curriculum_level.unique(sorted=True):
                level = int(level_tensor.item())
                output_ids = torch.where(curriculum_level == level)[0]
                rows = torch.nonzero(
                    self._reset_pose_bank.curriculum_level == level,
                    as_tuple=False,
                ).flatten()
                chunks: list[torch.Tensor] = []
                remaining = output_ids.numel()
                while remaining > 0:
                    cursor = self._reset_pose_cycle_cursors[level]
                    cycle_rows = self._reset_pose_cycle_rows[level]
                    if cursor == cycle_rows.numel():
                        cycle_rows = rows[torch.randperm(rows.numel(), device=self.device)]
                        cursor = 0
                    available = cycle_rows.numel() - cursor
                    take = min(remaining, available)
                    stop = cursor + take
                    chunks.append(cycle_rows[cursor:stop])
                    self._reset_pose_cycle_rows[level] = cycle_rows
                    self._reset_pose_cycle_cursors[level] = stop
                    remaining -= take
                sampled[output_ids] = torch.cat(chunks) if chunks else rows[:0]
            return sampled
        local_rows = torch.randint(rows_per_level, (count,), device=self.device)
        return curriculum_level * rows_per_level + local_rows

    def _sample_reset_curriculum_levels(self, count: int) -> torch.Tensor | None:
        """Return the next deterministic playback levels, if level cycling is enabled."""
        if self._reset_curriculum_level_cycle is None:
            return None
        cycle_count = self._reset_curriculum_level_cycle.numel()
        indices = (
            torch.arange(count, dtype=torch.long, device=self.device) + self._reset_curriculum_cycle_cursor
        ).remainder(cycle_count)
        self._reset_curriculum_cycle_cursor = (self._reset_curriculum_cycle_cursor + count) % cycle_count
        return self._reset_curriculum_level_cycle[indices]

    def _configure_gym_env_spaces(self) -> None:
        """Declare the image group consumed by the mixed CNN/vector RSL-RL actor."""
        super()._configure_gym_env_spaces()
        self.single_observation_space["heightmap"] = spec_to_gym_space([3, *self.cfg.heightmap_shape])

    def _configure_newton_builder(self, builder, _env_id: int, _position, _quaternion) -> None:
        """Apply task-wide particle safeguards before the Newton model is finalized."""
        builder.particle_max_velocity = self._particle_max_velocity

    def _setup_scene(self) -> None:
        self._robot = self.scene["robot"]
        self._media = self.scene["media"]

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # Reset metrics describe one completed-outcome batch. Do not replay a previous batch on
        # every subsequent simulator step when no episode completed.
        self.extras.pop("log", None)
        self._previous_actions.copy_(self._actions)
        self._invalid_action = ~torch.isfinite(actions).all(dim=1)
        self._actions.copy_(torch.nan_to_num(actions, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0))
        current_position = self._robot.data.joint_pos.torch[:, self._joint_ids]
        limits = self._robot.data.soft_joint_pos_limits.torch[:, self._joint_ids]
        lower = limits[..., 0] + self.cfg.joint_limit_margin
        upper = limits[..., 1] - self.cfg.joint_limit_margin
        self._joint_targets = torch.maximum(
            torch.minimum(current_position + self._actions * self._action_scale, upper),
            lower,
        )

    def _apply_action(self) -> None:
        self._robot.set_joint_position_target_index(
            target=self._joint_targets,
            joint_ids=self._joint_ids,
        )

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        particle_position_e = self._particle_position_e()
        particle_velocity_e = self._media.data.particle_vel_w.torch
        joint_position = self._robot.data.joint_pos.torch[:, self._joint_ids]
        joint_velocity = self._robot.data.joint_vel.torch[:, self._joint_ids]
        ee_pose_w = self._robot.data.body_link_pose_w.torch[:, self._ee_body_id]
        ee_velocity_w = self._robot.data.body_link_vel_w.torch[:, self._ee_body_id]
        finite_particle_slots = torch.isfinite(particle_position_e).all(dim=2) & torch.isfinite(
            particle_velocity_e
        ).all(dim=2)
        finite_particles = (finite_particle_slots | ~self._particle_active_mask).all(dim=1)
        finite_robot = (
            torch.isfinite(joint_position).all(dim=1)
            & torch.isfinite(joint_velocity).all(dim=1)
            & torch.isfinite(ee_pose_w).all(dim=1)
            & torch.isfinite(ee_velocity_w).all(dim=1)
        )
        safe_position_e = torch.nan_to_num(particle_position_e, nan=0.0, posinf=10.0, neginf=-10.0)
        safe_velocity_e = torch.nan_to_num(particle_velocity_e, nan=0.0, posinf=20.0, neginf=-20.0)
        particle_speed = torch.linalg.vector_norm(safe_velocity_e, dim=-1)
        self._rms_particle_speed = torch.sqrt(
            compute_masked_particle_mean(torch.square(particle_speed), self._particle_active_mask)
        )

        self._bin_fraction, self._spill_fraction = compute_particle_metrics(
            safe_position_e,
            self.cfg,
            self._particle_active_mask,
        )
        self._excessive_spill = self._spill_fraction > self.cfg.failure_max_spill_fraction
        self._transport_progress = compute_transport_progress(
            safe_position_e,
            self.cfg,
            self._episode_start_centroid_x,
            self._particle_active_mask,
        )

        escaped_particle_fraction = compute_masked_particle_mean(
            ((safe_position_e < self._particle_workspace_lower) | (safe_position_e > self._particle_workspace_upper))
            .any(dim=2)
            .float(),
            self._particle_active_mask,
        )
        self._escaped_workspace = escaped_particle_fraction > self.cfg.max_escaped_particle_fraction
        safe_ee_pose_w = torch.nan_to_num(ee_pose_w, nan=0.0, posinf=10.0, neginf=-10.0)
        paddle_position_e = (
            safe_ee_pose_w[:, :3] + quat_apply(safe_ee_pose_w[:, 3:7], self._paddle_offset)
        ) - self.scene.env_origins
        paddle_normal_e = quat_apply(safe_ee_pose_w[:, 3:7], self._paddle_local_normal)
        paddle_vertical_e = quat_apply(safe_ee_pose_w[:, 3:7], self._paddle_local_vertical)
        self._paddle_reach_potential = compute_paddle_reach_potential(
            safe_position_e,
            paddle_position_e,
            paddle_normal_e,
            paddle_vertical_e,
            self.cfg,
            self._particle_focused_source_mask,
        )
        joint_limits = self._robot.data.soft_joint_pos_limits.torch[:, self._joint_ids]
        joint_position_in_bounds = (
            (joint_position >= joint_limits[..., 0] - self.cfg.state_bound_joint_position_margin)
            & (joint_position <= joint_limits[..., 1] + self.cfg.state_bound_joint_position_margin)
        ).all(dim=1)
        joint_velocity_in_bounds = (joint_velocity.abs() <= self.cfg.state_bound_max_joint_velocity).all(dim=1)
        ee_linear_velocity_in_bounds = (
            torch.linalg.vector_norm(ee_velocity_w[:, :3], dim=1) <= self.cfg.state_bound_max_ee_linear_velocity
        )
        ee_angular_velocity_in_bounds = (
            torch.linalg.vector_norm(ee_velocity_w[:, 3:], dim=1) <= self.cfg.state_bound_max_ee_angular_velocity
        )
        valid_quaternion = (torch.linalg.vector_norm(safe_ee_pose_w[:, 3:7], dim=1) - 1.0).abs() < 0.05
        bounded_robot = (
            joint_position_in_bounds
            & joint_velocity_in_bounds
            & ee_linear_velocity_in_bounds
            & ee_angular_velocity_in_bounds
        )
        self._invalid_state = (
            ~(finite_particles & finite_robot & valid_quaternion & bounded_robot) | self._invalid_action
        )
        self._success_streak, self._success_this_step = update_success_streak(
            self._success_streak,
            self._bin_fraction,
            self._spill_fraction,
            success_fraction=self._episode_success_fraction,
            max_spill_fraction=self.cfg.success_max_spill_fraction,
            dwell_steps=self._success_dwell_steps,
        )
        # Terminate successful episodes or states that violate the numerical-safety envelope.
        unsafe = self._invalid_state | self._escaped_workspace | self._excessive_spill
        self._success_streak = torch.where(unsafe, torch.zeros_like(self._success_streak), self._success_streak)
        self._success_this_step &= ~unsafe
        self._episode_success |= self._success_this_step

        terminated = self._success_this_step | unsafe
        time_out = (self.episode_length_buf >= self.max_episode_length) & ~terminated
        return terminated, time_out

    def _get_rewards(self) -> torch.Tensor:
        action_magnitude = torch.square(self._actions).sum(dim=1)
        action_rate = torch.square(self._actions - self._previous_actions).sum(dim=1)
        bin_progress = compute_capped_bin_goal_progress(
            self._bin_fraction,
            self._episode_success_fraction,
        ) - compute_capped_bin_goal_progress(
            self._previous_bin_fraction,
            self._episode_success_fraction,
        )
        transport_progress = self._transport_progress - self._previous_transport_progress
        failure = self._invalid_state | self._escaped_workspace | self._excessive_spill
        reward = compute_transition_reward(
            bin_progress,
            transport_progress,
            self._paddle_reach_potential,
            self._spill_fraction,
            action_magnitude,
            action_rate,
            self._success_this_step,
            failure,
            self.cfg,
            step_dt=self.step_dt,
        )
        components = torch.stack(
            (
                self.cfg.reward_success * self._success_this_step.float(),
                self.cfg.reward_bin_progress * bin_progress,
                self.cfg.reward_transport_progress * transport_progress,
                self.cfg.reward_paddle_reach_potential_per_second * self.step_dt * self._paddle_reach_potential,
                self.cfg.penalty_spill_fraction_per_second * self.step_dt * self._spill_fraction,
                self.cfg.penalty_failure * failure.float(),
                self.cfg.penalty_action_magnitude_per_second * self.step_dt * action_magnitude,
                self.cfg.penalty_action_rate_per_second * self.step_dt * action_rate,
            ),
            dim=1,
        )
        self._episode_reward_components += components
        self._previous_bin_fraction.copy_(self._bin_fraction)
        self._previous_transport_progress.copy_(self._transport_progress)
        return reward

    def _get_observations(self) -> dict[str, torch.Tensor]:
        particle_position_e = torch.nan_to_num(
            self._particle_position_e(),
            nan=0.0,
            posinf=10.0,
            neginf=-10.0,
        )
        heightmap = self._heightmap_observation(particle_position_e)

        joint_position = torch.nan_to_num(
            self._robot.data.joint_pos.torch[:, self._joint_ids],
            nan=0.0,
            posinf=10.0,
            neginf=-10.0,
        )
        joint_velocity = torch.nan_to_num(
            self._robot.data.joint_vel.torch[:, self._joint_ids],
            nan=0.0,
            posinf=20.0,
            neginf=-20.0,
        )
        limits = self._robot.data.soft_joint_pos_limits.torch[:, self._joint_ids]
        midpoint = 0.5 * (limits[..., 0] + limits[..., 1])
        half_range = (0.5 * (limits[..., 1] - limits[..., 0])).clamp_min(1.0e-6)
        normalized_joint_position = ((joint_position - midpoint) / half_range).clamp(-1.0, 1.0)
        normalized_joint_velocity = (joint_velocity / self._joint_velocity_limits).clamp(-1.0, 1.0)

        paddle_position_e, paddle_normal_e, paddle_vertical_e = self._paddle_pose_e()
        normalized_paddle_position = (
            (paddle_position_e - self._paddle_position_center) / self._paddle_position_scale
        ).clamp(-2.0, 2.0)
        normalized_paddle_to_bin = ((self._bin_mouth - paddle_position_e) / self._paddle_position_scale).clamp(
            -2.0, 2.0
        )

        policy = torch.cat(
            (
                normalized_joint_position,
                normalized_joint_velocity,
                self._actions,
                normalized_paddle_position,
                paddle_normal_e,
                paddle_vertical_e,
                normalized_paddle_to_bin,
                self._episode_success_fraction[:, None],
            ),
            dim=-1,
        )

        particle_velocity_e = torch.nan_to_num(
            self._media.data.particle_vel_w.torch,
            nan=0.0,
            posinf=20.0,
            neginf=-20.0,
        ).clamp(-self._particle_max_velocity, self._particle_max_velocity)
        # Clamp each point before reduction: a single large-but-finite escaped particle must not
        # dominate the privileged centroid and produce an unbounded critic target.
        bounded_particle_position_e = torch.maximum(
            torch.minimum(particle_position_e, self._particle_workspace_upper),
            self._particle_workspace_lower,
        )
        centroid_position = compute_masked_particle_mean(bounded_particle_position_e, self._particle_active_mask)
        centroid_velocity = compute_masked_particle_mean(particle_velocity_e, self._particle_active_mask)
        normalized_centroid = (centroid_position - torch.tensor((0.6, 0.0, 0.05), device=self.device)) / torch.tensor(
            (0.9, 0.55, 0.50), device=self.device
        )
        privileged = torch.cat(
            (
                self._bin_fraction[:, None],
                self._spill_fraction[:, None],
                normalized_centroid.clamp(-2.0, 2.0),
                (centroid_velocity / self._particle_max_velocity).clamp(-1.0, 1.0),
                (self._rms_particle_speed / self.cfg.success_max_rms_particle_speed).clamp(0.0, 4.0)[:, None],
                self._transport_progress[:, None],
                (self._success_streak.float() / self._success_dwell_steps).clamp(0.0, 1.0)[:, None],
            ),
            dim=-1,
        )
        policy = torch.nan_to_num(policy, nan=0.0, posinf=4.0, neginf=-4.0).clamp(-4.0, 4.0)
        heightmap = torch.nan_to_num(heightmap, nan=0.0, posinf=1.0, neginf=0.0)
        critic = torch.nan_to_num(privileged, nan=0.0, posinf=4.0, neginf=-4.0).clamp(-4.0, 4.0)
        return {"policy": policy, "heightmap": heightmap, "critic": critic}

    def _reset_idx(self, env_ids: Sequence[int]) -> None:
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if len(env_ids) == 0:
            return

        completed = self._reset_initialized[env_ids] & (self.episode_length_buf[env_ids] > 0)
        completed_ids = env_ids[completed]
        cycled_level = self._sample_reset_curriculum_levels(len(env_ids))
        if len(completed_ids) > 0:
            completed_level = self._curriculum_level[completed_ids].clone()
            if cycled_level is not None:
                next_level = cycled_level[completed]
                promoted = torch.zeros_like(completed_level, dtype=torch.bool)
                demoted = torch.zeros_like(completed_level, dtype=torch.bool)
            elif self.cfg.curriculum_level_override is None:
                (
                    next_level,
                    next_success_streak,
                    next_failure_streak,
                    promoted,
                    demoted,
                ) = update_curriculum_levels(
                    completed_level,
                    self._curriculum_success_streak[completed_ids],
                    self._curriculum_failure_streak[completed_ids],
                    self._episode_success[completed_ids],
                    max_level=len(self.cfg.curriculum_pile_center_x) - 1,
                    successes_to_promote=self.cfg.curriculum_successes_to_promote,
                    failures_to_demote=self.cfg.curriculum_failures_to_demote,
                )
                self._curriculum_level[completed_ids] = next_level
                self._curriculum_success_streak[completed_ids] = next_success_streak
                self._curriculum_failure_streak[completed_ids] = next_failure_streak
            else:
                next_level = torch.full_like(completed_level, self.cfg.curriculum_level_override)
                promoted = torch.zeros_like(completed_level, dtype=torch.bool)
                demoted = torch.zeros_like(completed_level, dtype=torch.bool)

            failure_causes = torch.stack(
                (
                    self._invalid_state[completed_ids],
                    self._escaped_workspace[completed_ids],
                    self._excessive_spill[completed_ids],
                ),
                dim=1,
            )
            failure = failure_causes.any(dim=1)
            success = self._episode_success[completed_ids]
            episode_components = self._episode_reward_components[completed_ids]
            episode_log = self.extras.setdefault("log", {})
            episode_log.update(
                {
                    # RSL-RL concatenates these one-value-per-episode vectors across the rollout.
                    # Returning batch means here would weight a one-episode reset batch the same as
                    # a large batch and produce the unstable rates seen in distributed training.
                    "Episode_Reward/success": episode_components[:, 0],
                    "Episode_Reward/bin_progress": episode_components[:, 1],
                    "Episode_Reward/transport_progress": episode_components[:, 2],
                    "Episode_Reward/paddle_reach": episode_components[:, 3],
                    "Episode_Reward/spill": episode_components[:, 4],
                    "Episode_Reward/failure": episode_components[:, 5],
                    "Episode_Reward/action_magnitude": episode_components[:, 6],
                    "Episode_Reward/action_rate": episode_components[:, 7],
                    "Episode_Reward/total": episode_components.sum(dim=1),
                    "Metrics/success_rate": success.float(),
                    "Metrics/final_bin_fraction": self._bin_fraction[completed_ids],
                    "Metrics/final_spill_fraction": self._spill_fraction[completed_ids],
                    "Curriculum/episode_level": completed_level.float(),
                    "Curriculum/next_level": next_level.float(),
                    "Curriculum/promoted": promoted.float(),
                    "Curriculum/demoted": demoted.float(),
                    "Episode_Termination/success": success.float(),
                    "Episode_Termination/failure": failure.float(),
                    "Episode_Termination/time_out": self.reset_time_outs[completed_ids].float(),
                    "Episode_Termination/invalid_state": failure_causes[:, 0].float(),
                    "Episode_Termination/escaped_workspace": failure_causes[:, 1].float(),
                    "Episode_Termination/excessive_spill": failure_causes[:, 2].float(),
                }
            )
        if cycled_level is not None:
            self._curriculum_level[env_ids] = cycled_level
            self._curriculum_success_streak[env_ids] = 0
            self._curriculum_failure_streak[env_ids] = 0
        elif self.cfg.curriculum_level_override is not None:
            self._curriculum_level[env_ids] = self.cfg.curriculum_level_override
            self._curriculum_success_streak[env_ids] = 0
            self._curriculum_failure_streak[env_ids] = 0
        super()._reset_idx(env_ids)

        episode_level = self._curriculum_level[env_ids]
        pose_ids = self._sample_reset_pose_ids(episode_level)
        default_root_pose = self._robot.data.default_root_pose.torch[env_ids].clone()
        default_root_pose[:, :3] += self.scene.env_origins[env_ids]
        default_root_velocity = self._robot.data.default_root_vel.torch[env_ids].clone()
        joint_position = self._reset_pose_bank.joint_position[pose_ids]
        joint_velocity = torch.zeros_like(joint_position)

        self._robot.write_root_pose_to_sim_index(root_pose=default_root_pose, env_ids=env_ids)
        self._robot.write_root_velocity_to_sim_index(root_velocity=default_root_velocity, env_ids=env_ids)
        self._robot.write_joint_position_to_sim_index(
            position=joint_position,
            joint_ids=self._joint_ids,
            env_ids=env_ids,
        )
        self._robot.write_joint_velocity_to_sim_index(
            velocity=joint_velocity,
            joint_ids=self._joint_ids,
            env_ids=env_ids,
        )
        self._robot.set_joint_position_target_index(
            target=joint_position,
            joint_ids=self._joint_ids,
            env_ids=env_ids,
        )
        self._joint_targets[env_ids] = joint_position
        # Consume FK invalidation before MPM snapshots the reset paddle transform.
        _ = self._robot.data.body_link_pose_w

        reset_count = len(env_ids)
        randomization_scale = self._curriculum_randomization_scale[episode_level]
        source_pile_index = self._reset_pose_bank.source_pile_index[pose_ids]
        source_pile_count = self._curriculum_source_pile_count[episode_level]
        source_lateral_offset = self._curriculum_source_lateral_offset[episode_level]
        nominal_paddle_xy = torch.stack(
            (
                self.cfg.paddle_reset_center[0]
                + self._curriculum_pile_center_x[episode_level]
                - self.cfg.pile_nominal_center[0],
                torch.where(source_pile_index == 0, -source_lateral_offset, source_lateral_offset),
            ),
            dim=1,
        )
        # Correlate the absolute paddle and source transforms so each reset keeps a short approach
        # to its selected pile. A smaller residual supplies particle-only deployment variation.
        particle_lateral_range = torch.where(
            source_pile_count == 2,
            self.cfg.reset_particle_split_max_lateral_offset,
            self.cfg.reset_particle_max_lateral_offset,
        )
        translation_lower_bound = (
            torch.stack(
                (
                    self._particle_longitudinal_offset_range[0].expand(reset_count),
                    -particle_lateral_range,
                ),
                dim=1,
            )
            * randomization_scale[:, None]
        )
        translation_upper_bound = (
            torch.stack(
                (
                    self._particle_longitudinal_offset_range[1].expand(reset_count),
                    particle_lateral_range,
                ),
                dim=1,
            )
            * randomization_scale[:, None]
        )
        translation_xy = sample_correlated_particle_translation(
            self._reset_pose_bank.paddle_position_e[pose_ids, :2],
            nominal_paddle_xy,
            translation_lower_bound,
            translation_upper_bound,
            self._particle_paddle_residual_half_range * randomization_scale[:, None],
        )
        yaw = (
            (2.0 * torch.rand(reset_count, device=self.device) - 1.0)
            * self.cfg.reset_particle_max_yaw
            * randomization_scale
        )
        pile_lattice_resolution = sample_particle_lattice_crop_resolution(
            self.cfg.reset_pile_lattice_min_resolution,
            self.cfg.reset_pile_lattice_max_resolution,
            randomization_scale,
        )
        particle_active_mask = build_particle_lattice_crop_mask(
            self._particle_reset_template_e[env_ids],
            pile_lattice_resolution,
            lattice_resolution=self.cfg.reset_pile_lattice_max_resolution,
        )
        shape_profile_count = self._reset_source_vertical_cell_count.numel()
        if self.cfg.reset_cycle:
            source_shape_profile = (
                torch.arange(reset_count, dtype=torch.long, device=self.device) + self._reset_source_shape_cycle_cursor
            ).remainder(shape_profile_count)
            self._reset_source_shape_cycle_cursor = (
                self._reset_source_shape_cycle_cursor + reset_count
            ) % shape_profile_count
        else:
            source_shape_profile = torch.randint(shape_profile_count, (reset_count,), device=self.device)
        particle_jitter = (
            2.0
            * torch.rand(
                (reset_count, self._media.particles_per_object, 3),
                device=self.device,
            )
            - 1.0
        ) * self.cfg.reset_particle_jitter
        particle_reset = build_staged_particle_reset(
            self._particle_reset_template_e[env_ids],
            particle_active_mask,
            self._curriculum_pile_center_x[episode_level],
            source_pile_count,
            source_lateral_offset,
            self._curriculum_initial_bin_fraction[episode_level],
            source_pile_index,
            translation_xy,
            yaw,
            particle_jitter,
            self.cfg,
            source_vertical_cell_count=self._reset_source_vertical_cell_count[source_shape_profile],
            source_footprint_aspect_ratio=self._reset_source_footprint_aspect_ratio[source_shape_profile],
        )
        particle_position_e = particle_reset.position_e
        particle_state = self._media.data.default_particle_state_w.torch[env_ids].clone()
        particle_state[..., :3] = particle_position_e + self.scene.env_origins[env_ids, None, :]
        particle_state[..., 3:] = 0.0
        self._media.write_particle_state_to_sim_index(particle_state, env_ids=env_ids)
        self._media.write_particle_active_mask_to_sim_index(particle_reset.active_mask, env_ids=env_ids)

        # Clear constitutive/contact/collider history for exactly the reset worlds.
        # Newton reset masks include one trailing slot for global (world -1) entities.
        world_mask = torch.zeros(self.num_envs + 1, dtype=torch.bool, device=self.device)
        world_mask[env_ids] = True
        NewtonMPMManager.reset_solver_state(
            world_mask=wp.from_torch(world_mask, dtype=wp.bool),
            flags=newton.StateFlags.BODY | newton.StateFlags.PARTICLE,
        )

        bin_fraction, spill_fraction = compute_particle_metrics(
            particle_position_e, self.cfg, particle_reset.active_mask
        )
        start_centroid_x = compute_masked_particle_mean(particle_position_e[..., 0], particle_reset.active_mask)
        transport_progress = compute_transport_progress(
            particle_position_e,
            self.cfg,
            start_centroid_x,
            particle_reset.active_mask,
        )
        paddle_position_e, paddle_normal_e, paddle_vertical_e = self._paddle_pose_e()
        paddle_reach_potential = compute_paddle_reach_potential(
            particle_position_e,
            paddle_position_e[env_ids],
            paddle_normal_e[env_ids],
            paddle_vertical_e[env_ids],
            self.cfg,
            particle_reset.focused_source_mask,
        )
        self._particle_active_mask[env_ids] = particle_reset.active_mask
        self._particle_focused_source_mask[env_ids] = particle_reset.focused_source_mask
        self._bin_fraction[env_ids] = bin_fraction
        self._previous_bin_fraction[env_ids] = bin_fraction
        self._spill_fraction[env_ids] = spill_fraction
        self._episode_start_centroid_x[env_ids] = start_centroid_x
        self._transport_progress[env_ids] = transport_progress
        self._previous_transport_progress[env_ids] = transport_progress
        self._paddle_reach_potential[env_ids] = paddle_reach_potential

        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        self._episode_reward_components[env_ids] = 0.0
        self._success_streak[env_ids] = 0
        self._success_this_step[env_ids] = False
        self._episode_success[env_ids] = False
        self._rms_particle_speed[env_ids] = 0.0
        self._invalid_state[env_ids] = False
        self._invalid_action[env_ids] = False
        self._escaped_workspace[env_ids] = False
        self._excessive_spill[env_ids] = False
        if self.cfg.heightmap_xy_noise_std > 0.0:
            offset = torch.randn((reset_count, 2), device=self.device) * self.cfg.heightmap_xy_noise_std
            self._heightmap_xy_offset[env_ids] = offset.clamp(
                -3.0 * self.cfg.heightmap_xy_noise_std,
                3.0 * self.cfg.heightmap_xy_noise_std,
            )
        else:
            self._heightmap_xy_offset[env_ids] = 0.0
        self._heightmap_history_reset[env_ids] = True
        self._heightmap_history_reset_pending = True
        self._reset_pose_ids[env_ids] = pose_ids
        self._reset_initialized[env_ids] = True

    def _particle_position_e(self) -> torch.Tensor:
        return self._media.data.particle_pos_w.torch - self.scene.env_origins[:, None, :]

    def _paddle_pose_e(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ee_pose_w = self._robot.data.body_link_pose_w.torch[:, self._ee_body_id]
        paddle_position_w = ee_pose_w[:, :3] + quat_apply(ee_pose_w[:, 3:7], self._paddle_offset)
        paddle_position_e = paddle_position_w - self.scene.env_origins
        paddle_normal_e = quat_apply(ee_pose_w[:, 3:7], self._paddle_local_normal)
        paddle_vertical_e = quat_apply(ee_pose_w[:, 3:7], self._paddle_local_vertical)
        return paddle_position_e, paddle_normal_e, paddle_vertical_e

    def _heightmap_observation(self, particle_position_e: torch.Tensor) -> torch.Tensor:
        """Return current/delayed particle surfaces and the calibrated bin mask."""
        new_step = self._heightmap_history_step != self.common_step_counter
        if new_step or self._heightmap_history_reset_pending:
            surface = self._segmented_overhead_heightmap(particle_position_e)
            if new_step:
                self._heightmap_history_pointer = (self._heightmap_history_pointer + 1) % self._heightmap_history.shape[
                    1
                ]
                self._heightmap_history[:, self._heightmap_history_pointer].copy_(surface)
                self._heightmap_history_step = self.common_step_counter
            if self._heightmap_history_reset_pending:
                reset_mask = self._heightmap_history_reset
                self._heightmap_history[reset_mask] = surface[reset_mask, None]
                reset_mask.zero_()
                self._heightmap_history_reset_pending = False

        current = self._heightmap_history[:, self._heightmap_history_pointer]
        delayed_index = (self._heightmap_history_pointer + 1) % self._heightmap_history.shape[1]
        delayed = self._heightmap_history[:, delayed_index]
        return torch.stack((current, delayed, self._heightmap_goal_mask), dim=1)

    def _segmented_overhead_heightmap(self, particle_position_e: torch.Tensor) -> torch.Tensor:
        height_cells, width_cells = self.cfg.heightmap_shape
        cell_count = height_cells * width_cells
        x_lo, x_hi = self.cfg.heightmap_x_bounds
        y_lo, y_hi = self.cfg.heightmap_y_bounds
        x, y, z = particle_position_e.unbind(dim=-1)
        if self.cfg.heightmap_xy_noise_std > 0.0:
            x = x + self._heightmap_xy_offset[:, None, 0]
            y = y + self._heightmap_xy_offset[:, None, 1]

        ix = torch.floor((x - x_lo) * width_cells / (x_hi - x_lo)).long()
        iy = torch.floor((y - y_lo) * height_cells / (y_hi - y_lo)).long()
        valid = (
            (ix >= 0)
            & (ix < width_cells)
            & (iy >= 0)
            & (iy < height_cells)
            & torch.isfinite(z)
            & self._particle_active_mask
        )
        flat_index = (iy.clamp(0, height_cells - 1) * width_cells + ix.clamp(0, width_cells - 1)).long()

        particle_surface_z = z + float(self.cfg.scene.media.spawn.radius)
        normalized_height = ((particle_surface_z - self.cfg.heightmap_z_min) / self.cfg.heightmap_z_range).clamp(
            0.0, 1.0
        )
        height = torch.zeros((self.num_envs, cell_count), device=self.device)
        height.scatter_reduce_(
            dim=1,
            index=flat_index,
            src=torch.where(valid, normalized_height, torch.zeros_like(normalized_height)),
            reduce="amax",
            include_self=True,
        )
        occupancy = (height > 0.0).to(dtype=height.dtype)

        if self.cfg.heightmap_depth_noise_std > 0.0:
            height = (
                height
                + occupancy
                * torch.randn_like(height)
                * (self.cfg.heightmap_depth_noise_std / self.cfg.heightmap_z_range)
            ).clamp(0.0, 1.0)
        if self.cfg.heightmap_dropout_probability > 0.0:
            keep = torch.rand_like(occupancy) >= self.cfg.heightmap_dropout_probability
            height *= keep

        return height.reshape(self.num_envs, height_cells, width_cells)
