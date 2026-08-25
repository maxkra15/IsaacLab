# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for immutable Franka RJ45 pick-and-insert validation thresholds."""

import importlib
import json
import math
import sys
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from isaaclab_tasks.contrib.franka_rj45_insertion.pick_insert_env_cfg import (
    PICK_INSERT_CLOSED_FINGER_POSITION,
    PICK_INSERT_GRASP_PROXY_FRICTION,
    PICK_INSERT_SUCCESS_MAX_PLUG_SPEED,
    FrankaRJ45PickInsertEnvCfg,
    pick_insert_reset_dataset_task_contract,
)
from isaaclab_tasks.contrib.franka_rj45_insertion.pick_insert_reset_dataset_io import (
    PICK_INSERT_GOAL_MAX_ARM_JOINT_SPEED_RAD_S,
    PICK_INSERT_GOAL_MAX_AUTHORED_PLUG_ANGLE_RAD,
    PICK_INSERT_GOAL_MAX_AUTHORED_SEAT_ERROR_M,
    PICK_INSERT_GOAL_MAX_CABLE_SPEED_M_S,
    PICK_INSERT_GOAL_MAX_FINGER_JOINT_SPEED_M_S,
    PICK_INSERT_GOAL_MAX_PLUG_RELATIVE_LATCH_ANGLE_RAD,
    PICK_INSERT_GOAL_MAX_SOCKET_DRIFT_M,
    PICK_INSERT_GOAL_MAX_TASK_BODY_DRIFT_M,
    PICK_INSERT_RECOVERY_CARTESIAN_C2_POLICY,
)

_SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SCRIPT_DIR))
try:
    validator = importlib.import_module("validate_franka_rj45_pick_insert_resets")
    fast_validator = importlib.import_module("validate_franka_rj45_pick_insert_fast_resets")
    generator = importlib.import_module("generate_franka_rj45_pick_insert_reset_dataset")
    ValidationCfg = validator.ValidationCfg
finally:
    sys.path.remove(str(_SCRIPT_DIR))


_TEST_RECOVERY_CARTESIAN_SPEED_COMPONENTS = (
    "plug_linear_speed",
    "plug_angular_speed",
    "arm_joint_speed",
    "finger_joint_speed",
)
_TEST_RECOVERY_DIAGNOSTIC_EVIDENCE_NAMES = (
    "motion_policy",
    "pick_insert_phase",
    "overtravel_distance",
    "used_canonical_goal_arm_target",
    "compensation_iterations",
    "goal_error",
    "goal_error_history",
    "plug_translation_error_history",
    "correction_norm_history",
    "start_preload_bias_by_joint_rad",
    "goal_preload_bias_by_joint_rad",
    "preload_bias_difference_by_joint_rad",
    "maximum_preload_bias_difference_rad",
    "maximum_observed_raw_ik_joint_step_rad",
    "maximum_commanded_joint_step_before_densification_rad",
    "maximum_commanded_joint_step_after_densification_rad",
    "command_densification_required_subknot_count",
    "command_densification_executed_subknot_count",
    "start_target_anchor_error_rad",
    "canonical_endpoint_anchor_error_rad",
    "maximum_segment_boundary_command_jump_rad",
    "cartesian_route_waypoint_count_before_densification",
    "cartesian_route_waypoint_count_after_densification",
    "cartesian_route_waypoint_count",
    "cartesian_motion_sample_count",
    "cartesian_motion_maximum_cable_linear_speed",
    "cartesian_motion_maximum_plug_linear_speed",
    "cartesian_motion_maximum_plug_angular_speed",
    "cartesian_motion_maximum_arm_joint_speed",
    "cartesian_motion_maximum_finger_joint_speed",
    *(
        f"cartesian_motion_first_{component}_failure_{field}"
        for component in _TEST_RECOVERY_CARTESIAN_SPEED_COMPONENTS
        for field in ("mask", "step", "segment", "knot", "time_s")
    ),
    "lane_failure_masks",
)


def _fake_recovery_diagnostic_metrics(num_envs: int, *, phase: int = 2) -> dict[str, object]:
    lane_values = torch.arange(num_envs, dtype=torch.float32)
    metrics: dict[str, object] = {
        "motion_policy": PICK_INSERT_RECOVERY_CARTESIAN_C2_POLICY["motion_policy"],
        "pick_insert_phase": torch.full((num_envs,), phase, dtype=torch.int64),
        "overtravel_distance": torch.zeros(num_envs),
        "used_canonical_goal_arm_target": torch.ones(num_envs, dtype=torch.bool),
        "compensation_iterations": torch.ones(num_envs, dtype=torch.int64),
        "goal_error": lane_values + 0.001,
        "goal_error_history": torch.stack((lane_values + 0.002, lane_values + 0.001), dim=1),
        "plug_translation_error_history": torch.zeros((num_envs, 2, 3)),
        "correction_norm_history": torch.full((num_envs, 1), 0.0005),
        "start_preload_bias_by_joint_rad": torch.full((num_envs, 7), 0.01),
        "goal_preload_bias_by_joint_rad": torch.full((num_envs, 7), 0.02),
        "preload_bias_difference_by_joint_rad": torch.full((num_envs, 7), 0.01),
        "maximum_preload_bias_difference_rad": lane_values + 0.03,
        "maximum_observed_raw_ik_joint_step_rad": lane_values + 0.04,
        "maximum_commanded_joint_step_before_densification_rad": lane_values + 0.05,
        "maximum_commanded_joint_step_after_densification_rad": lane_values + 0.006,
        "command_densification_required_subknot_count": torch.arange(num_envs, dtype=torch.int64) + 2,
        "command_densification_executed_subknot_count": torch.arange(num_envs, dtype=torch.int64) + 1,
        "start_target_anchor_error_rad": lane_values + 0.007,
        "canonical_endpoint_anchor_error_rad": lane_values + 0.008,
        "maximum_segment_boundary_command_jump_rad": lane_values + 0.009,
        "cartesian_route_waypoint_count_before_densification": torch.full((num_envs,), 23, dtype=torch.int64),
        "cartesian_route_waypoint_count_after_densification": torch.full((num_envs,), 31, dtype=torch.int64),
        "cartesian_route_waypoint_count": torch.full((num_envs,), 31, dtype=torch.int64),
        "cartesian_motion_sample_count": torch.full((num_envs,), 47, dtype=torch.int64),
        "cartesian_motion_maximum_cable_linear_speed": lane_values + 0.10,
        "cartesian_motion_maximum_plug_linear_speed": lane_values + 0.11,
        "cartesian_motion_maximum_plug_angular_speed": lane_values + 0.12,
        "cartesian_motion_maximum_arm_joint_speed": lane_values + 0.13,
        "cartesian_motion_maximum_finger_joint_speed": lane_values + 0.14,
        "lane_failure_masks": {
            "recovery-cartesian-motion-plug-linear-speed": torch.arange(num_envs) % 2 == 1,
        },
    }
    for component_index, component in enumerate(_TEST_RECOVERY_CARTESIAN_SPEED_COMPONENTS):
        prefix = f"cartesian_motion_first_{component}_failure"
        metrics[f"{prefix}_mask"] = torch.arange(num_envs) % 2 == component_index % 2
        metrics[f"{prefix}_step"] = torch.full((num_envs,), 10 + component_index, dtype=torch.int64)
        metrics[f"{prefix}_segment"] = torch.full((num_envs,), 20 + component_index, dtype=torch.int64)
        metrics[f"{prefix}_knot"] = torch.full((num_envs,), 30 + component_index, dtype=torch.int64)
        metrics[f"{prefix}_time_s"] = lane_values + 0.5 + component_index
    return metrics


class _LaneMotionFakeEnv:
    num_envs = 2
    device = "cpu"
    _arm_joint_ids = torch.arange(7)

    def __init__(self, *, tcp_target_after_call: int | None = None, tcp_target: torch.Tensor | None = None):
        limits = torch.tensor([[[-10.0, 10.0]] * 7])
        self._robot = SimpleNamespace(data=SimpleNamespace(soft_joint_pos_limits=SimpleNamespace(torch=limits)))
        self.arm_q = torch.zeros((2, 7))
        self.current_arm_target = torch.zeros((2, 7))
        self.current_finger_target = torch.zeros((2, 2))
        self.target_writes: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.physics_commands: list[tuple[torch.Tensor, torch.Tensor]] = []
        self._tcp_calls = 0
        self._tcp_target_after_call = tcp_target_after_call
        self._tcp_target = torch.zeros((2, 3)) if tcp_target is None else tcp_target.clone()

    def read_robot_state(self):
        return self.arm_q.clone(), torch.zeros_like(self.arm_q), torch.zeros((2, 2)), torch.zeros((2, 2))

    def set_robot_targets(self, arm_target, finger_target):
        self.current_arm_target = arm_target.clone()
        self.current_finger_target = finger_target.clone()
        self.target_writes.append((arm_target.clone(), finger_target.clone()))

    def advance(self, _duration_s, update=None, *, post_step=None):
        if update is not None:
            update(0, 1, 1.0)
        self.physics_commands.append((self.current_arm_target.clone(), self.current_finger_target.clone()))
        if post_step is not None:
            post_step(0, 1, 1.0)
        return 1

    def tcp_pose_e(self):
        self._tcp_calls += 1
        pose = torch.zeros((2, 7))
        pose[:, 6] = 1.0
        if self._tcp_target_after_call is not None and self._tcp_calls >= self._tcp_target_after_call:
            pose[:, :3] = self._tcp_target
        return pose


class _InvalidThenValidIK:
    def __init__(self):
        self.calls = 0
        self.positions: list[torch.Tensor] = []

    def solve(self, position, *_args, **_kwargs):
        self.positions.append(position.clone())
        first = self.calls == 0
        self.calls += 1
        arm_q = torch.tensor(
            (
                (9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0),
                (0.1 * self.calls,) * 7,
            )
        )
        return SimpleNamespace(
            arm_q=arm_q,
            valid=torch.tensor((not first, True)),
            tcp_position=torch.zeros((2, 3)),
            position_residual=torch.zeros(2),
            rotation_residual=torch.zeros(2),
        )


def test_validation_cfg_uses_canonical_goal_thresholds():
    cfg = ValidationCfg()
    generator_cfg = generator.GeneratorCfg()
    env_cfg = FrankaRJ45PickInsertEnvCfg()

    assert cfg.finger_closed_target == PICK_INSERT_CLOSED_FINGER_POSITION
    assert generator_cfg.maximum_goal_body_drift_m == PICK_INSERT_GOAL_MAX_TASK_BODY_DRIFT_M
    assert env_cfg.grasp_proxy_friction == PICK_INSERT_GRASP_PROXY_FRICTION
    assert env_cfg.success_max_plug_speed == PICK_INSERT_SUCCESS_MAX_PLUG_SPEED
    assert cfg.maximum_goal_socket_drift_m == PICK_INSERT_GOAL_MAX_SOCKET_DRIFT_M
    assert cfg.maximum_goal_body_drift_m == PICK_INSERT_GOAL_MAX_TASK_BODY_DRIFT_M
    assert cfg.maximum_goal_cable_speed_m_s == PICK_INSERT_GOAL_MAX_CABLE_SPEED_M_S
    assert cfg.maximum_goal_arm_joint_speed_rad_s == PICK_INSERT_GOAL_MAX_ARM_JOINT_SPEED_RAD_S
    assert cfg.maximum_goal_finger_joint_speed_m_s == PICK_INSERT_GOAL_MAX_FINGER_JOINT_SPEED_M_S
    assert cfg.maximum_goal_authored_seat_error_m == PICK_INSERT_GOAL_MAX_AUTHORED_SEAT_ERROR_M
    assert cfg.maximum_goal_authored_plug_angle_rad == PICK_INSERT_GOAL_MAX_AUTHORED_PLUG_ANGLE_RAD
    assert cfg.maximum_goal_plug_relative_latch_angle_rad == PICK_INSERT_GOAL_MAX_PLUG_RELATIVE_LATCH_ANGLE_RAD
    assert (cfg.ik_sampler, cfg.ik_seed_count, cfg.ik_iterations, cfg.ik_noise_std) == ("none", 1, 160, 0.0)


@pytest.mark.parametrize(
    ("rows_per_phase", "minimum_unique"),
    ((96, 90), (3_334, 3_000), (137, 124), (1, 1)),
)
def test_generation_environment_task_contract_matches_requested_reset_bank_shape(rows_per_phase, minimum_unique):
    env_cfg = FrankaRJ45PickInsertEnvCfg()

    generator._configure_generation_reset_dataset_shape(env_cfg, rows_per_phase)
    env_cfg.validate_config()
    contract = pick_insert_reset_dataset_task_contract(env_cfg)

    assert contract["pick_insert"]["reset_dataset_rows_per_phase"] == rows_per_phase
    diversity = contract["pick_insert"]["full_pick_diversity"]
    assert (
        diversity["minimum_unique_socket_rows"],
        diversity["minimum_unique_plug_rows"],
        diversity["minimum_unique_arm_rows"],
    ) == (minimum_unique,) * 3


def test_canonical_goal_task_projection_ignores_only_reset_bank_cardinality():
    legacy_cfg = FrankaRJ45PickInsertEnvCfg()
    generator._configure_generation_reset_dataset_shape(legacy_cfg, 96)
    reference_cfg = FrankaRJ45PickInsertEnvCfg()
    generator._configure_generation_reset_dataset_shape(reference_cfg, 3_334)
    legacy = generator._canonical_goal_task_contract_projection(pick_insert_reset_dataset_task_contract(legacy_cfg))
    reference = generator._canonical_goal_task_contract_projection(
        pick_insert_reset_dataset_task_contract(reference_cfg)
    )

    assert generator.reset_dataset_digest(legacy) == generator.reset_dataset_digest(reference)
    changed = deepcopy(reference)
    changed["pick_insert"]["reach_reward_scale_m"] *= 2.0
    assert generator.reset_dataset_digest(legacy) != generator.reset_dataset_digest(changed)


def test_phase_4_pregrasp_orientation_sampler_contract_has_exact_ranges():
    cfg = generator.GeneratorCfg()
    contract = generator._phase_4_pregrasp_orientation_sampling_contract(cfg)
    runtime_contract = pick_insert_reset_dataset_task_contract(FrankaRJ45PickInsertEnvCfg())

    assert contract == runtime_contract["pick_insert"]["phase_4_pregrasp_orientation_sampling"]
    assert contract["sampler_version"] == 1
    assert contract["phase"] == 4
    assert contract["phase_name"] == "pregrasp"
    assert contract["starts_grasped"] is False
    assert contract["clearance_height_m"] == pytest.approx(0.045)
    assert contract["frame"] == "canonical-grasp-tool-local"
    assert contract["top_down_tilt_distribution"] == "uniform-solid-angle-cone"
    assert contract["top_down_tilt_range_rad"] == pytest.approx((0.0, math.radians(25.0)))
    assert contract["closing_axis_twist_distribution"] == "uniform"
    assert contract["closing_axis_twist_range_rad"] == pytest.approx((-math.radians(60.0), math.radians(60.0)))
    assert contract["starts_grasped_phases_use_canonical_orientation"] == (0, 1, 2, 3)
    assert contract["full_pick_phase_5_orientation_sampling"] == "unchanged-away-pose"
    assert "phase_4_pregrasp_orientation_sampling" not in generator._canonical_goal_generation_contract(cfg)


def test_phase_0_reverse_curriculum_sampler_contract_and_reference_profile_are_exact():
    contract = generator.pick_insert_phase_0_reverse_curriculum_sampling_contract()
    cfg = generator.GeneratorCfg(generation_mode="fast-ik", rows_per_phase=3_334, batch_size=256)
    profile = generator._fast_reset_bank_profile_contract(cfg)

    assert contract["sampler_version"] == 1
    assert contract["frame"] == "goal-plug-local"
    assert contract["axial_offset_ranges_m"] == (
        (0.0010, 0.0016),
        (0.0016, 0.0035),
        (0.0035, 0.0120),
    )
    assert contract["band_weights"] == (0.35, 0.35, 0.30)
    assert contract["geometric_success_at_reset"] is False
    assert profile == {
        "contract_version": 1,
        "profile": "balanced-20004-v1",
        "reference_profile": True,
        "rows_per_phase": 3_334,
        "phase_counts": (3_334,) * 6,
        "total_rows": 20_004,
        "batch_size": 256,
        "maximum_batches_per_phase": 96,
        "simulation_steps_per_row": 0,
    }
    assert cfg.max_batches_per_phase >= math.ceil(cfg.rows_per_phase / cfg.batch_size)


def test_phase_0_reverse_curriculum_sampler_is_deterministic_bounded_and_band_weighted():
    sample_count = 20_004
    first_rng = torch.Generator().manual_seed(2026)
    second_rng = torch.Generator().manual_seed(2026)

    first_shortfalls, first_bands = generator.sample_phase_0_reverse_curriculum_axial_shortfalls(
        sample_count,
        device="cpu",
        rng=first_rng,
    )
    second_shortfalls, second_bands = generator.sample_phase_0_reverse_curriculum_axial_shortfalls(
        sample_count,
        device="cpu",
        rng=second_rng,
    )

    torch.testing.assert_close(first_shortfalls, second_shortfalls, rtol=0.0, atol=0.0)
    assert torch.equal(first_bands, second_bands)
    assert float(first_shortfalls.min()) >= 0.0010
    assert float(first_shortfalls.max()) <= 0.0120
    assert float(first_shortfalls.min()) > FrankaRJ45PickInsertEnvCfg().success_axial_tolerance
    counts = torch.bincount(first_bands, minlength=3).float() / sample_count
    torch.testing.assert_close(counts, torch.tensor((0.35, 0.35, 0.30)), rtol=0.0, atol=0.01)
    for band_index, (lower, upper) in enumerate(generator.PICK_INSERT_PHASE_0_REVERSE_CURRICULUM_AXIAL_RANGES_M):
        values = first_shortfalls[first_bands == band_index]
        assert len(values) > 0
        assert float(values.min()) >= lower
        assert float(values.max()) <= upper


def test_phase_0_reverse_curriculum_sampler_is_batch_partition_invariant_and_consumes_one_uniform_per_row():
    whole_rng = torch.Generator().manual_seed(4815)
    chunked_rng = torch.Generator().manual_seed(4815)
    reference_rng = torch.Generator().manual_seed(4815)

    whole_shortfalls, whole_bands = generator.sample_phase_0_reverse_curriculum_axial_shortfalls(
        256,
        device="cpu",
        rng=whole_rng,
    )
    chunks = [
        generator.sample_phase_0_reverse_curriculum_axial_shortfalls(64, device="cpu", rng=chunked_rng)
        for _ in range(4)
    ]
    torch.rand((256,), generator=reference_rng)

    torch.testing.assert_close(torch.cat([chunk[0] for chunk in chunks]), whole_shortfalls, rtol=0.0, atol=0.0)
    assert torch.equal(torch.cat([chunk[1] for chunk in chunks]), whole_bands)
    assert torch.equal(chunked_rng.get_state(), whole_rng.get_state())
    assert torch.equal(whole_rng.get_state(), reference_rng.get_state())


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("phase_4_pregrasp_maximum_top_down_tilt_error_rad", 0.0, "top-down tilt"),
        ("phase_4_pregrasp_maximum_top_down_tilt_error_rad", math.radians(25.01), "top-down tilt"),
        ("phase_4_pregrasp_maximum_closing_axis_twist_error_rad", 0.0, "closing-axis twist"),
        ("phase_4_pregrasp_maximum_closing_axis_twist_error_rad", math.radians(60.01), "closing-axis twist"),
    ),
)
def test_phase_4_pregrasp_orientation_sampler_rejects_out_of_contract_ranges(field, value, message):
    with pytest.raises(ValueError, match=message):
        generator.GeneratorCfg(**{field: value})


def test_phase_4_pregrasp_orientation_sampler_is_deterministic_bounded_and_area_uniform():
    sample_count = 8192
    first_rng = torch.Generator().manual_seed(4815)
    second_rng = torch.Generator().manual_seed(4815)

    first = generator.sample_phase_4_pregrasp_orientation_errors(
        sample_count,
        device="cpu",
        rng=first_rng,
    )
    second = generator.sample_phase_4_pregrasp_orientation_errors(
        sample_count,
        device="cpu",
        rng=second_rng,
    )

    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        torch.linalg.vector_norm(first, dim=-1),
        torch.ones(sample_count),
        rtol=1.0e-6,
        atol=1.0e-6,
    )
    canonical_tool_z = torch.tensor((0.0, 0.0, 1.0)).expand(sample_count, -1)
    sampled_tool_z = generator.math_utils.quat_apply(first, canonical_tool_z)
    tilt_cosine = sampled_tool_z[:, 2].clamp(-1.0, 1.0)
    tilt_error = torch.acos(tilt_cosine)
    # For error = tilt * twist with a tilt axis in tool XY, 2*atan2(z, w)
    # recovers the signed tool-Z closing-axis twist independently of tilt.
    twist_error = 2.0 * torch.atan2(first[:, 2], first[:, 3])

    assert float(tilt_error.min()) >= 0.0
    assert float(tilt_error.max()) <= math.radians(25.0) + 1.0e-6
    assert float(twist_error.min()) >= -math.radians(60.0) - 1.0e-6
    assert float(twist_error.max()) <= math.radians(60.0) + 1.0e-6
    assert float(tilt_error.max()) > math.radians(24.0)
    assert float(twist_error.min()) < -math.radians(58.0)
    assert float(twist_error.max()) > math.radians(58.0)
    expected_mean_cosine = 0.5 * (1.0 + math.cos(math.radians(25.0)))
    assert float(tilt_cosine.mean()) == pytest.approx(expected_mean_cosine, abs=1.0e-3)


def test_phase_4_pregrasp_orientation_sampler_maps_rows_independently_of_batch_partition():
    whole_rng = torch.Generator().manual_seed(2026)
    chunked_rng = torch.Generator().manual_seed(2026)

    whole = generator.sample_phase_4_pregrasp_orientation_errors(96, device="cpu", rng=whole_rng)
    chunked = torch.cat(
        tuple(generator.sample_phase_4_pregrasp_orientation_errors(24, device="cpu", rng=chunked_rng) for _ in range(4))
    )

    torch.testing.assert_close(chunked, whole, rtol=0.0, atol=0.0)
    assert torch.equal(chunked_rng.get_state(), whole_rng.get_state())


def test_only_phase_4_consumes_the_pregrasp_orientation_rng_stream():
    owner = object.__new__(generator.PickInsertResetDatasetGenerator)
    owner.env = SimpleNamespace(num_envs=8)
    owner.cfg = generator.GeneratorCfg(batch_size=8)
    owner.device = torch.device("cpu")
    owner.random = torch.Generator().manual_seed(91)
    initial_state = owner.random.get_state().clone()

    assert all(owner._sample_phase_tcp_orientation_error(phase) is None for phase in (0, 1, 2, 3, 5))
    assert torch.equal(owner.random.get_state(), initial_state)
    sampled = owner._sample_phase_tcp_orientation_error(4)

    assert sampled is not None
    assert sampled.shape == (8, 4)
    assert not torch.equal(owner.random.get_state(), initial_state)


def test_desired_tcp_pose_applies_pregrasp_error_after_the_canonical_grasp():
    owner = object.__new__(generator.PickInsertResetDatasetGenerator)
    owner.env = SimpleNamespace(cfg=SimpleNamespace(plug_grasp_offset=(0.0, 0.0, 0.0)))
    owner.device = torch.device("cpu")
    owner.local_grasp_orientation = torch.tensor(((math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0),))
    plug_pose = torch.tensor(((0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0),))
    orientation_error = generator.math_utils.quat_from_angle_axis(
        torch.tensor((math.radians(30.0),)),
        torch.tensor(((0.0, 0.0, 1.0),)),
    )

    position, orientation = owner._desired_tcp_pose(
        plug_pose,
        orientation_error_xyzw=orientation_error,
    )
    expected = generator.math_utils.quat_mul(owner.local_grasp_orientation, orientation_error)

    torch.testing.assert_close(position, plug_pose[:, :3])
    torch.testing.assert_close(orientation, expected)


def test_starts_grasped_pickup_construction_rejects_pregrasp_orientation_errors():
    owner = object.__new__(generator.PickInsertResetDatasetGenerator)

    with pytest.raises(ValueError, match="Starts-grasped.*canonical grasp orientation"):
        owner._construct_pickup(
            torch.zeros((1, 7)),
            torch.zeros((1, 7)),
            acquire=True,
            pregrasp_orientation_error_xyzw=torch.tensor(((0.0, 0.0, 0.0, 1.0),)),
        )


def test_generator_contract_selects_the_shared_pick_insert_recovery_policy():
    contract = generator._canonical_goal_generation_contract(generator.GeneratorCfg())

    assert contract["oracle_entry_replay"] == {
        "contract_version": 1,
        "restore_source": "stored-candidate-with-vbd-pose-history",
        "controller_semantics": "persistent-absolute",
        "duration_s": generator.PICK_INSERT_RESET_REPLAY_DURATION_S,
        "post_step_samples": generator.PICK_INSERT_RESET_REPLAY_POST_STEP_SAMPLES,
        "post_replay_arm_target": "runtime-persistent-arm-target",
        "ungrasped_acquisition_move_attempt_count": 5,
        "ungrasped_acquisition_move_settle_s": 0.30,
        "phase_modes": {
            "0": "verify-existing-physical-grasp-after-replay",
            "1": "verify-existing-physical-grasp-after-replay",
            "2": "verify-existing-physical-grasp-after-replay",
            "3": "verify-existing-physical-grasp-after-replay",
            "4": "guarded-full-physical-acquisition-after-replay",
            "5": "guarded-full-physical-acquisition-after-replay",
        },
        "gates": (
            "finite",
            "contact-buffers-empty-after-restore",
            "collision",
            "phase-contact-state",
            "construction-drives-disabled",
            "joint-and-target-limits",
            "absolute-target-unclamped-and-stable",
            "target-tracking",
            "robot-and-cable-speed",
            "task-body-drift",
            "vbd-pose-history-applied-exactly-once",
        ),
    }
    assert contract["scripted_recovery"] == PICK_INSERT_RECOVERY_CARTESIAN_C2_POLICY
    assert (
        contract["scripted_recovery"]["contract_version"]
        == PICK_INSERT_RECOVERY_CARTESIAN_C2_POLICY["contract_version"]
    )
    assert contract["scripted_recovery"]["motion_policy"] == PICK_INSERT_RECOVERY_CARTESIAN_C2_POLICY["motion_policy"]
    assert contract["scripted_recovery"]["compensation"] == PICK_INSERT_RECOVERY_CARTESIAN_C2_POLICY["compensation"]
    assert contract["scripted_recovery"]["phase_route_modes"] == {
        "0": "insertion-corridor",
        "1": "insertion-corridor",
        "2": "clearance-via-preinsert",
        "3": "clearance-via-preinsert",
        "4": "clearance-via-preinsert",
        "5": "clearance-via-preinsert",
    }


def test_recovery_failure_checks_are_reported_without_changing_acceptance():
    base_checks = {
        "first": torch.tensor((True, True, False)),
        "second": torch.tensor((True, False, True)),
    }
    lane_failures = {
        "recovery-dwell-goal:arm-joint-speed": torch.tensor((False, False, True)),
        "recovery-cartesian-motion-plug-linear-speed": torch.tensor((False, True, False)),
    }

    valid, reported = generator._validity_and_recovery_failure_checks(base_checks, lane_failures)

    assert valid.tolist() == [True, False, False]
    assert tuple(reported) == (
        "first",
        "second",
        "oracle_recovery_lane_recovery-cartesian-motion-plug-linear-speed",
        "oracle_recovery_lane_recovery-dwell-goal:arm-joint-speed",
    )
    assert reported["oracle_recovery_lane_recovery-cartesian-motion-plug-linear-speed"].tolist() == [
        True,
        False,
        True,
    ]
    assert reported["oracle_recovery_lane_recovery-dwell-goal:arm-joint-speed"].tolist() == [
        True,
        True,
        False,
    ]
    owner = object.__new__(generator.PickInsertResetDatasetGenerator)
    owner.rejection_counts = {0: dict.fromkeys((*reported, "accepted"), 0)}
    owner._record_rejections(0, reported, valid)
    assert owner.rejection_counts[0]["oracle_recovery_lane_recovery-cartesian-motion-plug-linear-speed"] == 1
    assert owner.rejection_counts[0]["oracle_recovery_lane_recovery-dwell-goal:arm-joint-speed"] == 1
    assert owner.rejection_counts[0]["accepted"] == 1


def test_scripted_recovery_diagnostic_evidence_routes_the_exact_shared_v2_fields():
    metrics = _fake_recovery_diagnostic_metrics(3, phase=1)

    evidence = generator._scripted_recovery_diagnostic_evidence(metrics)

    assert tuple(evidence) == _TEST_RECOVERY_DIAGNOSTIC_EVIDENCE_NAMES
    assert all(evidence[name] is metrics[name] for name in _TEST_RECOVERY_DIAGNOSTIC_EVIDENCE_NAMES)


def test_validator_constructs_one_exact_sampler_free_ik_owner(monkeypatch):
    calls = []

    class FakeIK:
        def __init__(self, env, **kwargs):
            calls.append((env, kwargs))

    env = object()
    monkeypatch.setattr(validator, "FrankaResetIK", FakeIK)

    owner = validator._new_validator_ik(env, ValidationCfg(), prior_solve_calls=17)

    assert len(calls) == 1
    assert calls[0] == (
        env,
        {"seed": 2027, "seeds": 1, "iterations": 160, "noise_std": 0.0, "sampler": "none"},
    )
    assert owner.solve_calls == 17


@pytest.mark.parametrize(
    ("field", "invalid"),
    (("ik_sampler", "gauss"), ("ik_seed_count", 2), ("ik_iterations", 159), ("ik_noise_std", 0.5)),
)
def test_validation_cfg_rejects_any_ik_policy_change(field, invalid):
    with pytest.raises(ValueError, match="IK policy is immutable"):
        ValidationCfg(**{field: invalid})


def _checkpoint_progress() -> dict:
    progress = {
        "status": "rows",
        "created_utc": "2026-08-17T00:00:00+00:00",
        "goal_replay": {"passed": True},
        "completed_batches": [{"ordinal": 0, "phase": 0, "phase_batch_index": 1, "row_ids": [0]}],
        "rows": [{"row_id": 0, "phase": 0}],
        "ik_solve_call_count": 3,
        "torch_rng_state": validator._torch_rng_state_json(),
        "report": None,
    }
    progress["counters"] = validator._validation_progress_counters(
        progress["rows"],
        progress["completed_batches"],
    )
    return progress


def test_validator_checkpoint_round_trip_is_atomic_and_digest_bound(tmp_path):
    path = tmp_path / "validation.checkpoint.json"
    metadata = {"artifact_content_sha256": "a" * 64, "tuple_normalizes": (1, 2)}

    validator._write_validation_checkpoint(metadata, _checkpoint_progress(), path)
    loaded = validator._load_validation_checkpoint(path)

    assert loaded["metadata"]["tuple_normalizes"] == [1, 2]
    assert loaded["progress"]["status"] == "rows"
    assert not list(tmp_path.glob(".validation.checkpoint.json.*.tmp"))

    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["progress"]["ik_solve_call_count"] += 1
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="content digest"):
        validator._load_validation_checkpoint(path)


def test_validator_checkpoint_cli_is_full_only_and_refuses_aliases(tmp_path):
    artifact = tmp_path / "reset_dataset.pt"
    stable = tmp_path / "reset_validation.json"
    checkpoint = tmp_path / "checkpoint.json"

    resolved, resuming = validator._validate_checkpoint_invocation(
        checkpoint=checkpoint,
        resume=None,
        keep_checkpoint=False,
        input_path=artifact,
        stable_output=stable,
        cfg=ValidationCfg(),
    )
    assert resolved == checkpoint.resolve()
    assert resuming is False

    with pytest.raises(ValueError, match="only for a full-dataset replay"):
        validator._validate_checkpoint_invocation(
            checkpoint=checkpoint,
            resume=None,
            keep_checkpoint=False,
            input_path=artifact,
            stable_output=stable,
            cfg=ValidationCfg(quick=True),
        )
    with pytest.raises(ValueError, match="cannot alias"):
        validator._validate_checkpoint_invocation(
            checkpoint=stable,
            resume=None,
            keep_checkpoint=False,
            input_path=artifact,
            stable_output=stable,
            cfg=ValidationCfg(),
        )


def test_validator_resolves_every_output_alias_before_simulation_and_preserves_input(tmp_path):
    artifact = tmp_path / "reset_dataset.pt"
    original = b"immutable-reset-artifact"
    artifact.write_bytes(original)
    stable_alias = tmp_path / "stable.json"
    stable_alias.symlink_to(artifact)

    with pytest.raises(ValueError, match="cannot alias the reset artifact"):
        validator._resolve_validation_output_paths(
            input_path=artifact,
            output_dir=tmp_path / "reports",
            stable_output=stable_alias,
        )

    with pytest.raises(ValueError, match="must be a directory distinct"):
        validator._resolve_validation_output_paths(
            input_path=artifact,
            output_dir=artifact,
            stable_output=tmp_path / "stable-other.json",
        )

    output_file = tmp_path / "not-a-directory"
    output_file.write_text("preserve", encoding="utf-8")
    with pytest.raises(ValueError, match="must be a directory distinct"):
        validator._resolve_validation_output_paths(
            input_path=artifact,
            output_dir=output_file,
            stable_output=tmp_path / "stable-other.json",
        )
    assert artifact.read_bytes() == original
    assert output_file.read_text(encoding="utf-8") == "preserve"


def test_timestamped_report_symlink_alias_is_rejected_before_simulation(tmp_path):
    artifact = tmp_path / "reset_dataset.pt"
    artifact.write_bytes(b"dataset")
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    created_at = datetime(2026, 8, 17, 12, 0, tzinfo=UTC)
    candidate = report_dir / "reset_validation_20260817T120000Z_aaaaaaaaaaaa_full.json"
    candidate.symlink_to(artifact)

    with pytest.raises(ValueError, match="cannot alias a protected"):
        validator._timestamped_validation_report_path(
            report_dir,
            artifact_content_sha256="a" * 64,
            quick=False,
            protected_paths=(artifact,),
            created_at=created_at,
        )
    assert artifact.read_bytes() == b"dataset"


@pytest.mark.parametrize("field", ("created_utc", "goal_replay", "rows", "ik_solve_call_count"))
def test_completed_checkpoint_report_must_match_progress(field: str, tmp_path):
    progress = _checkpoint_progress()
    progress["status"] = "stable-published"
    report = {
        "created_utc": progress["created_utc"],
        "goal_replay": deepcopy(progress["goal_replay"]),
        "rows": deepcopy(progress["rows"]),
        "ik_solve_call_count": progress["ik_solve_call_count"],
    }
    invalid = {
        "created_utc": "2026-08-17T00:00:01+00:00",
        "goal_replay": {"passed": False},
        "rows": [],
        "ik_solve_call_count": 4,
    }[field]
    report[field] = invalid
    report["content_sha256"] = validator.reset_validation_report_content_digest(report)
    progress["report"] = report
    path = tmp_path / "stable.checkpoint.json"
    validator._write_validation_checkpoint({}, progress, path)

    with pytest.raises(ValueError, match=rf"report {field} does not match checkpoint progress"):
        validator._load_validation_checkpoint(path)


def test_stable_published_resume_republishes_identically_then_cleans_checkpoint(monkeypatch, tmp_path):
    env_cfg = FrankaRJ45PickInsertEnvCfg()
    report = {"immutable": [1, 2, 3]}
    checkpoint = {
        "metadata": {"task_contract": validator.pick_insert_reset_dataset_task_contract(env_cfg)},
        "progress": {"status": "stable-published", "report": report},
    }
    checkpoint_path = tmp_path / "validation.checkpoint.json"
    checkpoint_path.write_text("checkpoint", encoding="utf-8")
    stable_output = tmp_path / "reset_validation.json"
    published: list[dict] = []

    def publish(observed_report, _payload, _cfg, output, **_kwargs):
        published.append(deepcopy(observed_report))
        output.write_text(json.dumps(observed_report, sort_keys=True), encoding="utf-8")
        return output

    monkeypatch.setattr(validator, "write_stable_validation_report", publish)
    first = validator._republish_stable_published_checkpoint(
        checkpoint,
        checkpoint_path=checkpoint_path,
        keep_checkpoint=True,
        payload={},
        env_cfg=env_cfg,
        stable_output=stable_output,
        expected_source_sha256={},
        expected_asset_closure={},
        protected_paths=(),
    )
    first_bytes = first.read_bytes()
    second = validator._republish_stable_published_checkpoint(
        checkpoint,
        checkpoint_path=checkpoint_path,
        keep_checkpoint=False,
        payload={},
        env_cfg=env_cfg,
        stable_output=stable_output,
        expected_source_sha256={},
        expected_asset_closure={},
        protected_paths=(),
    )

    assert published == [report, report]
    assert second.read_bytes() == first_bytes
    assert not checkpoint_path.exists()


def test_validator_resume_accepts_only_an_exact_whole_batch_prefix():
    phases = torch.tensor([1, 0, 1, 0, 2, 2], dtype=torch.int64)
    selected = torch.arange(len(phases), dtype=torch.int64)
    plan = validator._validation_batch_plan(phases, selected, batch_size=2)
    completed = deepcopy(plan[:2])
    rows = [{"row_id": row_id} for batch in completed for row_id in batch["row_ids"]]

    validator._validate_completed_batch_prefix(completed, rows, plan)
    assert [batch["row_ids"] for batch in plan[:3]] == [[1, 3], [0, 2], [4, 5]]

    completed[1]["row_ids"].reverse()
    with pytest.raises(ValueError, match="batch prefix"):
        validator._validate_completed_batch_prefix(completed, rows, plan)

    with pytest.raises(ValueError, match="exact completed-batch prefix"):
        validator._validate_completed_batch_prefix(plan[:1], [{"row_id": 3}], plan)


def _parsed_generator_args(
    output: Path,
    *,
    generation_mode: str = "physical-oracle",
    rows_per_phase: int = 96,
    batch_size: int = 24,
    quick: bool = False,
    diagnostic_goal_only: bool = False,
    diagnostic_pickup_only: bool = False,
    diagnostic_phase0_transport_only: bool = False,
    diagnostic_recovery_phase: int | None = None,
    canonical_goal_certificate_output: Path | None = None,
    canonical_goal_certificate_input: Path | None = None,
    checkpoint: Path | None = None,
    resume_from: Path | None = None,
    keep_checkpoint: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        output=output,
        generation_mode=generation_mode,
        rows_per_phase=rows_per_phase,
        batch_size=batch_size,
        quick=quick,
        diagnostic_goal_only=diagnostic_goal_only,
        diagnostic_pickup_only=diagnostic_pickup_only,
        diagnostic_phase0_transport_only=diagnostic_phase0_transport_only,
        diagnostic_recovery_phase=diagnostic_recovery_phase,
        diagnostic_reset_abcd=False,
        diagnostic_reset_e=False,
        diagnostic_p_relax_reseat=False,
        diagnostic_zero_finger_close_target=False,
        diagnostic_forward_grasp_offset=False,
        diagnostic_effective_grasp_friction_three=False,
        canonical_goal_certificate_output=canonical_goal_certificate_output,
        canonical_goal_certificate_input=canonical_goal_certificate_input,
        checkpoint=checkpoint,
        resume_from=resume_from,
        keep_checkpoint=keep_checkpoint,
        validate=False,
    )


def test_legacy_physical_generation_requires_explicit_noncanonical_output(tmp_path):
    args = _parsed_generator_args(generator.DEFAULT_DATASET_PATH)

    with pytest.raises(ValueError, match="requires --canonical-goal-certificate-input"):
        generator._validate_parsed_artifact_policy(args)

    args.canonical_goal_certificate_input = tmp_path / "goal-certificate.pt"

    with pytest.raises(ValueError, match="cannot overwrite the canonical 20,004-row reset bank"):
        generator._validate_parsed_artifact_policy(args)

    args.output = tmp_path / "legacy-physical-reset-bank.pt"

    assert generator._validate_parsed_artifact_policy(args) is False


@pytest.mark.parametrize("batch_size", (1, 4))
def test_canonical_goal_certifier_policy_allows_only_one_or_four_envs(tmp_path, batch_size):
    args = _parsed_generator_args(
        generator.DEFAULT_DATASET_PATH,
        batch_size=batch_size,
        canonical_goal_certificate_output=tmp_path / "goal-certificate.pt",
    )

    assert generator._validate_parsed_artifact_policy(args) is True


def test_canonical_goal_certificate_output_cannot_overwrite_canonical_dataset():
    args = _parsed_generator_args(
        generator.DEFAULT_DATASET_PATH,
        batch_size=4,
        canonical_goal_certificate_output=generator.DEFAULT_DATASET_PATH,
    )

    with pytest.raises(ValueError, match="cannot overwrite the canonical reset-dataset path"):
        generator._validate_parsed_artifact_policy(args)


def test_canonical_goal_certificate_input_cannot_alias_dataset_output(tmp_path):
    output = tmp_path / "reset-dataset.pt"
    args = _parsed_generator_args(
        output,
        canonical_goal_certificate_input=tmp_path / "nested" / ".." / output.name,
    )

    with pytest.raises(ValueError, match="cannot also be the reset-dataset output path"):
        generator._validate_parsed_artifact_policy(args)


def test_canonical_goal_certificate_modes_are_mutually_exclusive(tmp_path):
    args = _parsed_generator_args(
        generator.DEFAULT_DATASET_PATH,
        batch_size=4,
        canonical_goal_certificate_output=tmp_path / "goal-certificate.pt",
        canonical_goal_certificate_input=tmp_path / "other-goal-certificate.pt",
    )

    with pytest.raises(ValueError, match="mutually exclusive"):
        generator._validate_parsed_artifact_policy(args)


def test_pickup_only_diagnostic_disables_artifact_saving_for_requested_batch_width(tmp_path):
    args = _parsed_generator_args(
        tmp_path / "must-not-be-written.pt",
        batch_size=7,
        diagnostic_pickup_only=True,
    )

    assert generator._validate_parsed_artifact_policy(args) is True
    assert args.batch_size == 7


def test_phase0_transport_diagnostic_disables_artifact_saving_for_requested_batch_width(tmp_path):
    args = _parsed_generator_args(
        tmp_path / "must-not-be-written.pt",
        batch_size=24,
        diagnostic_phase0_transport_only=True,
        canonical_goal_certificate_input=tmp_path / "goal-certificate.pt",
    )

    assert generator._validate_parsed_artifact_policy(args) is True
    assert args.batch_size == 24


def test_phase0_transport_diagnostic_requires_canonical_goal_certificate(tmp_path):
    args = _parsed_generator_args(
        tmp_path / "must-not-be-written.pt",
        batch_size=24,
        diagnostic_phase0_transport_only=True,
    )

    with pytest.raises(ValueError, match="requires --canonical-goal-certificate-input"):
        generator._validate_parsed_artifact_policy(args)


@pytest.mark.parametrize("phase", (1, 2, 4, 5))
def test_recovery_diagnostic_requires_exact_certificate_backed_production_shape(tmp_path, phase):
    certificate = tmp_path / "goal-certificate.pt"
    args = _parsed_generator_args(
        tmp_path / "must-not-be-written.pt",
        diagnostic_recovery_phase=phase,
        canonical_goal_certificate_input=certificate,
    )

    assert generator._validate_parsed_artifact_policy(args) is True

    args.rows_per_phase = 95
    with pytest.raises(ValueError, match="requires exact production shape"):
        generator._validate_parsed_artifact_policy(args)


@pytest.mark.parametrize("phase", (0, 3, 6, True, 1.0))
def test_recovery_diagnostic_policy_rejects_unsupported_phase_values(tmp_path, phase):
    args = _parsed_generator_args(
        tmp_path / "must-not-be-written.pt",
        diagnostic_recovery_phase=phase,
        canonical_goal_certificate_input=tmp_path / "goal-certificate.pt",
    )

    with pytest.raises(ValueError, match="must be 1, 2, 4, or 5"):
        generator._validate_parsed_artifact_policy(args)


def test_recovery_diagnostic_requires_canonical_goal_certificate(tmp_path):
    args = _parsed_generator_args(
        tmp_path / "must-not-be-written.pt",
        diagnostic_recovery_phase=1,
    )

    with pytest.raises(ValueError, match="requires --canonical-goal-certificate-input"):
        generator._validate_parsed_artifact_policy(args)


@pytest.mark.parametrize("conflicting_field", ("quick", "validate"))
def test_recovery_diagnostic_rejects_save_or_validation_controls(tmp_path, conflicting_field):
    args = _parsed_generator_args(
        tmp_path / "must-not-be-written.pt",
        diagnostic_recovery_phase=2,
        canonical_goal_certificate_input=tmp_path / "goal-certificate.pt",
    )
    setattr(args, conflicting_field, True)

    with pytest.raises(ValueError, match="writes no artifact"):
        generator._validate_parsed_artifact_policy(args)


@pytest.mark.parametrize("phase", (1, 2, 4, 5))
def test_recovery_diagnostic_cli_parses_phase_and_routes_as_no_save(monkeypatch, tmp_path, phase):
    certificate = tmp_path / "goal-certificate.pt"
    observed: dict[str, object] = {}
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(generator.__file__),
            "--diagnostic-recovery-phase",
            str(phase),
            "--canonical-goal-certificate-input",
            str(certificate),
        ],
    )
    monkeypatch.setattr(generator, "_canonical_goal_source_digests", lambda: {"source": "digest"})

    def execute(args, **kwargs):
        observed["phase"] = args.diagnostic_recovery_phase
        observed.update(kwargs)

    monkeypatch.setattr(generator, "_execute_parsed_invocation", execute)

    generator.main()

    assert observed["phase"] == phase
    assert observed["reset_dataset_saving_disabled"] is True
    assert observed["certificate_input_source_snapshot"] == {"source": "digest"}
    assert observed["checkpoint_path"] is None
    assert observed["resuming_checkpoint"] is False


def test_phase0_transport_diagnostic_is_mutually_exclusive_with_pickup_only(tmp_path):
    args = _parsed_generator_args(
        tmp_path / "must-not-be-written.pt",
        diagnostic_pickup_only=True,
        diagnostic_phase0_transport_only=True,
    )

    with pytest.raises(ValueError, match="mutually exclusive"):
        generator._validate_parsed_artifact_policy(args)


def test_recovery_diagnostic_is_mutually_exclusive_with_existing_batch_diagnostics(tmp_path):
    args = _parsed_generator_args(
        tmp_path / "must-not-be-written.pt",
        diagnostic_phase0_transport_only=True,
        diagnostic_recovery_phase=1,
        canonical_goal_certificate_input=tmp_path / "goal-certificate.pt",
    )

    with pytest.raises(ValueError, match="mutually exclusive"):
        generator._validate_parsed_artifact_policy(args)


@pytest.mark.parametrize("conflicting_field", ("quick", "validate"))
def test_phase0_transport_diagnostic_rejects_row_or_artifact_controls(tmp_path, conflicting_field):
    args = _parsed_generator_args(
        tmp_path / "must-not-be-written.pt",
        diagnostic_phase0_transport_only=True,
        canonical_goal_certificate_input=tmp_path / "goal-certificate.pt",
    )
    setattr(args, conflicting_field, True)

    with pytest.raises(ValueError, match="writes no artifact"):
        generator._validate_parsed_artifact_policy(args)


@pytest.mark.parametrize(
    "diagnostic_field",
    ("diagnostic_pickup_only", "diagnostic_phase0_transport_only", "diagnostic_recovery_phase"),
)
@pytest.mark.parametrize("checkpoint_field", ("checkpoint", "resume_from", "keep_checkpoint"))
def test_batch_diagnostics_reject_generation_checkpoint_controls(tmp_path, diagnostic_field, checkpoint_field):
    args = _parsed_generator_args(
        tmp_path / "must-not-be-written.pt",
        canonical_goal_certificate_input=(
            tmp_path / "goal-certificate.pt"
            if diagnostic_field in ("diagnostic_phase0_transport_only", "diagnostic_recovery_phase")
            else None
        ),
    )
    setattr(args, diagnostic_field, 1 if diagnostic_field == "diagnostic_recovery_phase" else True)
    if checkpoint_field == "keep_checkpoint":
        setattr(args, checkpoint_field, True)
    else:
        checkpoint_path = tmp_path / f"{checkpoint_field}.json"
        if checkpoint_field == "resume_from":
            checkpoint_path.touch()
        setattr(args, checkpoint_field, checkpoint_path)

    with pytest.raises(ValueError, match="writes no artifact"):
        generator._validate_parsed_artifact_policy(args)


@pytest.mark.parametrize(
    "conflicting_field",
    (
        "diagnostic_goal_only",
        "diagnostic_reset_abcd",
        "diagnostic_reset_e",
        "diagnostic_p_relax_reseat",
        "diagnostic_zero_finger_close_target",
        "diagnostic_forward_grasp_offset",
        "diagnostic_effective_grasp_friction_three",
        "canonical_goal_certificate_output",
        "canonical_goal_certificate_input",
    ),
)
def test_pickup_only_diagnostic_rejects_goal_and_certificate_modes(tmp_path, conflicting_field):
    args = _parsed_generator_args(
        tmp_path / "must-not-be-written.pt",
        diagnostic_pickup_only=True,
    )
    setattr(args, conflicting_field, tmp_path / "goal-certificate.pt" if "certificate" in conflicting_field else True)

    with pytest.raises(ValueError, match="cannot be combined with goal or certificate modes"):
        generator._validate_parsed_artifact_policy(args)


@pytest.mark.parametrize("conflicting_field", ("quick", "validate"))
def test_pickup_only_diagnostic_rejects_row_or_artifact_controls(tmp_path, conflicting_field):
    args = _parsed_generator_args(
        tmp_path / "must-not-be-written.pt",
        diagnostic_pickup_only=True,
    )
    setattr(args, conflicting_field, True)

    with pytest.raises(ValueError, match="writes no artifact"):
        generator._validate_parsed_artifact_policy(args)


@pytest.mark.parametrize("batch_size", (2, 24))
def test_canonical_goal_certifier_policy_rejects_other_batch_widths(tmp_path, batch_size):
    args = _parsed_generator_args(
        generator.DEFAULT_DATASET_PATH,
        batch_size=batch_size,
        canonical_goal_certificate_output=tmp_path / "goal-certificate.pt",
    )

    with pytest.raises(ValueError, match="batch-size 1 or --batch-size 4"):
        generator._validate_parsed_artifact_policy(args)


@pytest.mark.parametrize(
    "overrides",
    (
        {"quick": True},
        {"rows_per_phase": 95},
        {"batch_size": 4},
    ),
)
def test_physical_canonical_goal_certificate_input_requires_exact_batch24_generation(tmp_path, overrides):
    args = _parsed_generator_args(
        generator.DEFAULT_DATASET_PATH,
        canonical_goal_certificate_input=tmp_path / "goal-certificate.pt",
        **overrides,
    )

    with pytest.raises(ValueError, match="exact non-quick production generation"):
        generator._validate_parsed_artifact_policy(args)


def test_reference_fast_bank_policy_allows_canonical_20004_row_output(tmp_path):
    args = _parsed_generator_args(
        generator.DEFAULT_DATASET_PATH,
        generation_mode="fast-ik",
        rows_per_phase=3_334,
        batch_size=256,
        canonical_goal_certificate_input=tmp_path / "goal-certificate.pt",
    )

    assert generator._validate_parsed_artifact_policy(args) is False


def test_fast_bank_policy_allows_custom_noncanonical_shape(tmp_path):
    args = _parsed_generator_args(
        tmp_path / "custom-fast-reset-bank.pt",
        generation_mode="fast-ik",
        rows_per_phase=137,
        batch_size=32,
        canonical_goal_certificate_input=tmp_path / "goal-certificate.pt",
    )

    assert generator._validate_parsed_artifact_policy(args) is False


def test_fast_bank_policy_reserves_canonical_output_for_reference_shape(tmp_path):
    args = _parsed_generator_args(
        generator.DEFAULT_DATASET_PATH,
        generation_mode="fast-ik",
        rows_per_phase=3_333,
        batch_size=256,
        canonical_goal_certificate_input=tmp_path / "goal-certificate.pt",
    )

    with pytest.raises(ValueError, match="Canonical fast-IK.*3334 rows per phase"):
        generator._validate_parsed_artifact_policy(args)


def test_generation_checkpoint_policy_is_exact_certificate_input_only(tmp_path):
    certificate = tmp_path / "goal.pt"
    certificate.touch()
    checkpoint = tmp_path / "rows.json"
    args = _parsed_generator_args(
        tmp_path / "legacy-physical-reset-bank.pt",
        canonical_goal_certificate_input=certificate,
        checkpoint=checkpoint,
    )

    assert generator._validate_parsed_artifact_policy(args) is False
    assert generator._validate_generation_checkpoint_invocation(args) == (checkpoint.resolve(), False)

    args.quick = True
    with pytest.raises(ValueError, match="exact non-quick production generation"):
        generator._validate_parsed_artifact_policy(args)


def test_generation_checkpoint_policy_rejects_missing_mode_and_bad_paths(tmp_path):
    output = tmp_path / "reset.pt"
    certificate = tmp_path / "goal.pt"
    certificate.touch()

    with pytest.raises(ValueError, match="requires --canonical-goal-certificate-input"):
        generator._validate_generation_checkpoint_invocation(
            _parsed_generator_args(output, checkpoint=tmp_path / "rows.json")
        )
    with pytest.raises(ValueError, match="existing checkpoint file"):
        generator._validate_generation_checkpoint_invocation(
            _parsed_generator_args(
                output,
                canonical_goal_certificate_input=certificate,
                resume_from=tmp_path / "missing.json",
            )
        )
    with pytest.raises(ValueError, match=".json suffix"):
        generator._validate_generation_checkpoint_invocation(
            _parsed_generator_args(
                output,
                canonical_goal_certificate_input=certificate,
                checkpoint=tmp_path / "rows.pt",
            )
        )
    existing = tmp_path / "existing.json"
    existing.touch()
    with pytest.raises(ValueError, match="refuses to overwrite"):
        generator._validate_generation_checkpoint_invocation(
            _parsed_generator_args(
                output,
                canonical_goal_certificate_input=certificate,
                checkpoint=existing,
            )
        )
    with pytest.raises(ValueError, match="cannot alias the reset-dataset output"):
        generator._validate_generation_checkpoint_invocation(
            _parsed_generator_args(
                tmp_path / "reset.json",
                canonical_goal_certificate_input=certificate,
                checkpoint=tmp_path / "reset.json",
            )
        )


def test_generation_checkpoint_policy_rejects_ambiguous_lifecycle_flags(tmp_path):
    certificate = tmp_path / "goal.pt"
    certificate.touch()
    resume = tmp_path / "resume.json"
    resume.touch()
    args = _parsed_generator_args(
        tmp_path / "reset.pt",
        canonical_goal_certificate_input=certificate,
        checkpoint=tmp_path / "new.json",
        resume_from=resume,
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        generator._validate_generation_checkpoint_invocation(args)

    args = _parsed_generator_args(tmp_path / "reset.pt", keep_checkpoint=True)
    with pytest.raises(ValueError, match="requires --checkpoint or --resume-from"):
        generator._validate_generation_checkpoint_invocation(args)


def _certificate_goal_state(task_body_count: int = 3) -> dict[str, torch.Tensor]:
    result = {
        "arm_joint_position": torch.zeros(7),
        "arm_joint_target": torch.zeros(7),
        "arm_joint_velocity": torch.zeros(7),
        "finger_joint_position": torch.zeros(2),
        "finger_joint_velocity": torch.zeros(2),
        "finger_joint_target": torch.zeros(2),
        "task_body_pose": torch.zeros((task_body_count, 7)),
        "task_body_previous_pose": torch.zeros((task_body_count, 7)),
        "task_body_coupling_previous_pose": torch.zeros((task_body_count, 7)),
        "task_body_velocity": torch.zeros((task_body_count, 6)),
    }
    for name in ("task_body_pose", "task_body_previous_pose", "task_body_coupling_previous_pose"):
        result[name][:, 6] = 1.0
    return result


def _certificate_physical_contract() -> dict[str, float]:
    return {
        "finger_closed_target_m": 0.0,
        "live_finger_close_position_m": 0.0,
        "configured_grasp_proxy_raw_friction": 4.5,
        "live_grasp_proxy_raw_friction": 4.5,
        "effective_finger_proxy_friction": 3.0,
        "success_max_plug_speed": PICK_INSERT_SUCCESS_MAX_PLUG_SPEED,
    }


def _certificate_production_evidence(certifier_env_count: int = 4) -> dict[str, object]:
    survivors = [True] * certifier_env_count
    return {
        "passed": True,
        "classification": "production-canonical-relax-reseat-cold-proof",
        "diagnostic_cli": False,
        "physical_contract": {
            "finger_closed_target_m": 0.0,
            "finger_raw_friction": 2.0,
            "grasp_proxy_raw_friction": 4.5,
            "effective_finger_proxy_friction": 3.0,
        },
        "construction_surviving_mask": survivors,
        "continuous_relaxation": {"passed": True},
        "authored_reseat": {"count": 1},
        "reseat_trailing_equilibrium": {"passed": True},
        "cold_proofs": {
            "same_original_capture_restored_both_times": True,
            "endpoint_promotion_count": 0,
            "cold_30s": {"passed": True, "stage": "canonical-cold-30s"},
            "cold_60s": {"passed": True, "stage": "canonical-cold-60s"},
        },
        "final_surviving_mask": survivors,
        "final_surviving_lane_ids": list(range(certifier_env_count)),
        "selected_original_lane": 0,
    }


def _certificate_contracts(task_body_count: int = 3) -> dict[str, object]:
    return {
        "task_body_count": task_body_count,
        "expected_task_contract": {"task_body_count": task_body_count, "contract_version": 6},
        "expected_physical_contract": _certificate_physical_contract(),
        "expected_generation_contract": generator._canonical_goal_generation_contract(generator.GeneratorCfg()),
        "expected_versions": {"newton": "test", "warp": "test", "isaaclab": "test", "torch": "test"},
        "expected_source_sha256": {"generator": "a" * 64, "physics": "b" * 64},
    }


def _canonical_goal_certificate(task_body_count: int = 3) -> tuple[dict[str, object], dict[str, object]]:
    contracts = _certificate_contracts(task_body_count)
    certificate = generator._build_canonical_goal_certificate(
        goal_state=_certificate_goal_state(task_body_count),
        production_evidence=_certificate_production_evidence(),
        row_rng_state=torch.Generator().manual_seed(2026).get_state(),
        certifier_env_count=4,
        task_body_count=task_body_count,
        task_contract=contracts["expected_task_contract"],
        physical_contract=contracts["expected_physical_contract"],
        generation_contract=contracts["expected_generation_contract"],
        versions=contracts["expected_versions"],
        source_sha256=contracts["expected_source_sha256"],
    )
    return certificate, contracts


def _checkpoint_owner(contracts, certificate):
    owner = object.__new__(generator.PickInsertResetDatasetGenerator)
    owner.cfg = generator.GeneratorCfg()
    owner.device = torch.device("cpu")
    owner.random = torch.Generator()
    owner.random.set_state(certificate["row_rng_state"])
    owner._ik_solve_call_count = 0
    owner.attempt_counts = [0] * 6
    owner.rejection_counts = {phase: {} for phase in range(6)}
    owner.accepted_oracle_metrics = {phase: [] for phase in range(6)}
    owner._canonical_goal_certificate_validation_kwargs = lambda: deepcopy(contracts)
    return owner


def _checkpoint_row_chunk(phase: int, *, row_count: int = 1, task_body_count: int = 3):
    specs = generator._generation_checkpoint_row_specs(task_body_count)
    chunk = {name: torch.zeros((row_count, *shape), dtype=dtype) for name, (dtype, shape) in specs.items()}
    chunk["phase"].fill_(phase)
    chunk["starts_grasped"].fill_(generator._PHASE_STARTS_GRASPED[phase])
    for name in (
        "task_body_pose",
        "task_body_previous_pose",
        "task_body_coupling_previous_pose",
        "goal_task_body_pose",
    ):
        chunk[name][..., 6] = 1.0
    return chunk


def test_generation_checkpoint_stop_resume_restores_rng_rows_counters_and_logical_ik(tmp_path):
    certificate, contracts = _canonical_goal_certificate()
    checkpoint_path = tmp_path / "generation.json"
    first = _checkpoint_owner(contracts, certificate)
    checkpoint = generator._GenerationCheckpoint.open(
        first,
        certificate,
        path=checkpoint_path,
        resuming=False,
    )
    checkpoint.restore_generator(first, certificate)
    torch.rand(5, generator=first.random)
    first._ik_solve_call_count = 7
    first.attempt_counts[0] = 24
    first.rejection_counts[0] = {
        "accepted": 1,
        "finite": 0,
        "oracle_recovery_lane_recovery-cartesian-motion-plug-linear-speed": 2,
    }
    first.accepted_oracle_metrics[0] = [{"goal_error": 0.01}]
    checkpoint.commit_batch(first, 0, 0, _checkpoint_row_chunk(0))
    expected_next = float(torch.rand((), generator=first.random))

    resumed_owner = _checkpoint_owner(contracts, certificate)
    resumed = generator._GenerationCheckpoint.open(
        resumed_owner,
        certificate,
        path=checkpoint_path,
        resuming=True,
    )
    resumed.restore_generator(resumed_owner, certificate)

    assert resumed.status == "generating"
    assert resumed.next_batch_index(0) == 1
    assert resumed_owner._ik_solve_call_count == 0
    assert resumed.document["progress"]["logical_ik_solve_call_count"] == 7
    assert resumed_owner.attempt_counts == [24, 0, 0, 0, 0, 0]
    assert resumed_owner.rejection_counts[0] == {
        "accepted": 1,
        "finite": 0,
        "oracle_recovery_lane_recovery-cartesian-motion-plug-linear-speed": 2,
    }
    assert len(resumed.phase_chunks(0, device=torch.device("cpu"))) == 1
    assert float(torch.rand((), generator=resumed_owner.random)) == expected_next


def test_generation_checkpoint_uses_custom_fast_bank_row_stride_and_batch_width(tmp_path):
    certificate, contracts = _canonical_goal_certificate()
    owner = _checkpoint_owner(contracts, certificate)
    owner.cfg = generator.GeneratorCfg(generation_mode="fast-ik", rows_per_phase=5, batch_size=3)
    checkpoint = generator._GenerationCheckpoint.open(
        owner,
        certificate,
        path=tmp_path / "custom-fast-bank.json",
        resuming=False,
    )

    owner.attempt_counts[0] = 3
    owner.accepted_oracle_metrics[0] = [{"accepted": True}] * 3
    checkpoint.commit_batch(owner, 0, 0, _checkpoint_row_chunk(0, row_count=3))
    owner.attempt_counts[0] = 6
    owner.accepted_oracle_metrics[0] = [{"accepted": True}] * 5
    checkpoint.commit_batch(owner, 0, 1, _checkpoint_row_chunk(0, row_count=2))
    owner.attempt_counts[1] = 3
    owner.accepted_oracle_metrics[1] = [{"accepted": True}]
    checkpoint.commit_batch(owner, 1, 0, _checkpoint_row_chunk(1))

    assert checkpoint.document["metadata"]["artifact_contract"]["rows_per_phase"] == 5
    assert checkpoint.document["metadata"]["artifact_contract"]["batch_size"] == 3
    assert [record["row_ids"] for record in checkpoint.document["progress"]["completed_batches"]] == [
        [0, 1, 2],
        [3, 4],
        [5],
    ]
    reloaded = generator._load_generation_checkpoint(
        checkpoint.path,
        expected_metadata=checkpoint.expected_metadata,
    )
    assert [chunk["row_ids"] for chunk in reloaded.accepted_chunks] == [[0, 1, 2], [3, 4], [5]]


def test_generation_checkpoint_rejects_digest_corruption_and_recomputed_unknown_fields(tmp_path):
    certificate, contracts = _canonical_goal_certificate()
    owner = _checkpoint_owner(contracts, certificate)
    metadata = generator._generation_checkpoint_metadata(owner, certificate)
    document = generator._initial_generation_checkpoint_document(metadata=metadata, certificate=certificate)

    corrupted = deepcopy(document)
    corrupted["progress"]["row_rng_state"]["data"][0] ^= 1
    with pytest.raises(ValueError, match="content digest"):
        generator._validate_generation_checkpoint(corrupted, expected_metadata=metadata)

    unknown = deepcopy(document)
    unknown["progress"]["partial_batch"] = True
    unknown["content_sha256"] = generator._generation_checkpoint_content_digest(unknown)
    with pytest.raises(ValueError, match="unexpected or missing progress fields"):
        generator._validate_generation_checkpoint(unknown, expected_metadata=metadata)

    path = tmp_path / "checkpoint.json"
    generator._write_generation_checkpoint_atomic(document, path, expected_metadata=metadata)
    assert not list(tmp_path.glob(".checkpoint.json.*.tmp"))


def test_generation_checkpoint_final_permutation_is_drawn_once_then_reused():
    checkpoint = object.__new__(generator._GenerationCheckpoint)
    checkpoint.document = {"progress": {"status": "rows-complete", "final_artifact": None}}
    owner = SimpleNamespace(device=torch.device("cpu"), random=torch.Generator().manual_seed(81))

    first = checkpoint.final_permutation(owner, 32)
    rng_after_first = owner.random.get_state().clone()
    checkpoint.document["progress"]["status"] = "artifact-ready"
    checkpoint.document["progress"]["final_artifact"] = {
        "permutation": first.tolist(),
        "content_sha256": "a" * 64,
        "row_count": 32,
    }
    second = checkpoint.final_permutation(owner, 32)

    assert torch.equal(first, second)
    assert torch.equal(owner.random.get_state(), rng_after_first)


def test_canonical_goal_certificate_binds_current_fresh_sampler_free_row_ik_stream():
    certificate, contracts = _canonical_goal_certificate()
    row_ik_stream = contracts["expected_generation_contract"]["row_ik_stream"]

    assert row_ik_stream == {
        "owner": "PickInsertResetDatasetGenerator.ik",
        "owner_count": 1,
        "seed": 2026,
        "sampler": "none",
        "stochastic_sampler": False,
        "seed_count": 1,
        "noise_std": 0.0,
        "iterations": 160,
        "serialized_cursor": False,
        "fresh_row_stream": True,
        "required_initial_state": "fresh-sampler-free-single-owner-before-any-solve",
        "goal_derivation_process_separate": True,
    }
    assert certificate["metadata"]["rng_contract"]["row_ik_stream"] == row_ik_stream


def test_canonical_goal_certificate_binds_direct_staging_pickup_sequence():
    certificate, contracts = _canonical_goal_certificate()
    sequence = contracts["expected_generation_contract"]["pickup_construction_sequence"]

    assert sequence == {
        "construction_sequence_version": 2,
        "construction_robot_staging": "kinematic-open-clearance-before-loose-task-placement",
        "coherent_task_placement": "task-only-rigid-write-preserves-staged-robot-state-and-targets",
        "drive_free_local_alignment": "fresh-live-clearance-with-2mm-translation-and-2deg-rotation-gates",
        "physical_close": "prepositioned-2mm-step-descent-bilateral-close-and-post-contact-settle",
        "local_descent_maximum_translation_step_m": 0.002,
        "local_descent_waypoint_count": 23,
        "live_clearance_maximum_translation_error_m": 0.002,
        "live_clearance_maximum_rotation_error_rad": math.radians(2.0),
        "full_route_reserved_for_ungrasped_oracle": True,
    }
    assert certificate["metadata"]["generation_contract"]["pickup_construction_sequence"] == sequence


def test_canonical_goal_certificate_binds_segment_continuous_grasped_transport_schedule():
    certificate, contracts = _canonical_goal_certificate()
    schedule = contracts["expected_generation_contract"]["grasped_transport_schedule"]

    assert schedule == {
        "schedule_version": 5,
        "planning": "precomputed-sequential-ik-knots",
        "execution": "c2-endpoint-time-law-piecewise-linear-joint-cruise",
        "time_law": "c2-endpoint-ramp-with-constant-speed-cruise",
        "time_law_endpoint_continuity": "C2",
        "joint_path_interpolation": "piecewise-linear-through-precomputed-ik-knots",
        "joint_path_internal_knot_continuity": "C0-with-bounded-target-velocity-jumps",
        "internal_knot_settles": 0,
        "segment_end_settle": True,
        "scope": "all-shared-scripted-grasped-carry-including-phase-realization-and-canonical-reseat",
        "transient_cable_speed_policy": "sample-every-step-observation-only-during-scripted-carry",
        "transient_cable_speed_is_rejection_gate": False,
        "stored_final_reset_replay_cable_speed_limit_m_s": 0.04,
        "final_cable_speed_is_rejection_gate": True,
        "cold_reset_replay_cable_speed_is_rejection_gate": True,
        "canonical_reseat_followup_goal_cable_speed_limit_m_s": 0.01,
        "transient_motion_speed_gates": {
            "plug_linear_m_s": 0.04,
            "plug_angular_rad_s": 0.35,
            "arm_joint_rad_s": 0.5,
            "finger_joint_m_s": 0.05,
        },
        "c2_ramp_fraction": 0.10,
        "maximum_normalized_progress_rate": 1.0 / 0.9,
        "segment_duration_per_knot_s": 0.20,
        "maximum_translation_step_m": 0.002,
        "maximum_rotation_step_rad": math.radians(2.0),
        "maximum_raw_ik_joint_step_rad": 0.02,
        "derived_maximum_internal_target_velocity_jump_rad_s": 2.0 * 0.02 / (0.9 * 0.20),
        "maximum_waypoints": 430,
        "endpoint_policies": {
            "canonical-and-default": {
                "policy": "strict",
                "position_tolerance_m": 0.002,
                "terminal_correction_enabled": True,
            },
            "reset-row-phases-0-through-3": {
                "policy": "reset-row",
                "position_tolerance_m": 0.006,
                "terminal_correction_enabled": False,
                "acceptance": "settled-endpoint",
            },
        },
        "terminal_correction": {
            "scope": "strict-endpoint-policy-only",
            "trigger": "plug-position-or-tcp-position-outside-final-tolerance",
            "priority": "plug-position-before-tcp-position",
            "translation_vector": "selected-goal-position-minus-selected-live-position",
            "maximum_translation_step_m": 0.001,
            "rotation_correction": "none",
            "maximum_iterations": 6,
            "position_tolerance_m": 0.002,
            "progress_gate": (
                "selected-metric-reaches-tolerance-or-improves-beyond-epsilon-and-"
                "already-in-tolerance-secondary-remains-within-tolerance-plus-epsilon"
            ),
            "progress_epsilon_m": 1.0e-6,
            "ik_raw-step-and-joint-limit-gates_unchanged": True,
            "final_strict_tcp-and-plug-position-and-orientation-gates_unchanged": True,
            "final_reset_row_orientation-and-physical-gates_unchanged": True,
        },
    }
    assert certificate["metadata"]["generation_contract"]["grasped_transport_schedule"] == schedule


def test_grasped_transport_c2_schedule_is_symmetric_monotone_and_flat_at_endpoints():
    progress = [index / 1000 for index in range(1001)]
    path = [generator._grasped_transport_c2_progress(value) for value in progress]

    assert path[0] == 0.0
    assert path[-1] == 1.0
    assert all(path[index] <= path[index + 1] for index in range(len(path) - 1))
    assert all(
        left == pytest.approx(1.0 - right, abs=1.0e-12) for left, right in zip(path, reversed(path), strict=True)
    )
    epsilon = 1.0e-5
    assert generator._grasped_transport_c2_progress(epsilon) / epsilon < 1.0e-6
    assert (1.0 - generator._grasped_transport_c2_progress(1.0 - epsilon)) / epsilon < 1.0e-6


def test_grasped_transport_knot_interpolation_passes_every_knot_without_overshoot():
    knots = torch.tensor(
        (
            ((0.0, 0.0), (10.0, 10.0)),
            ((1.0, 2.0), (11.0, 12.0)),
            ((2.0, 4.0), (12.0, 14.0)),
            ((3.0, 6.0), (13.0, 16.0)),
        )
    )

    for index in range(len(knots)):
        observed = generator._interpolate_grasped_transport_knots(knots, index / (len(knots) - 1))
        assert torch.equal(observed, knots[index])

    before = generator._interpolate_grasped_transport_knots(knots, 1.0 / 3.0 - 1.0e-6)
    after = generator._interpolate_grasped_transport_knots(knots, 1.0 / 3.0 + 1.0e-6)
    assert torch.linalg.vector_norm(after - before).item() < 1.0e-4
    samples = torch.stack([generator._interpolate_grasped_transport_knots(knots, index / 100) for index in range(101)])
    assert bool((samples >= knots.amin(dim=0)).all())
    assert bool((samples <= knots.amax(dim=0)).all())


def test_grasped_transport_reports_the_deliberately_noncollinear_knot_velocity_jump():
    knots = torch.tensor(
        (
            ((0.00, 0.00), (0.00, 0.00)),
            ((0.02, 0.01), (0.01, 0.00)),
            ((0.00, 0.02), (0.02, 0.00)),
        )
    )

    jump = generator._grasped_transport_internal_target_velocity_jump(
        knots,
        duration_per_knot_s=0.20,
    )

    assert jump.tolist() == pytest.approx((2.0 * 0.02 / (0.9 * 0.20), 0.0))


@pytest.mark.parametrize(
    ("segment_waypoint_counts", "legacy_steps", "scheduled_steps"),
    (
        ((51, 89, 89, 51, 38), 2226, 1916),
        ((51, 85, 85, 51, 38), 2170, 1868),
        ((51, 96, 96, 52, 38), 2331, 2007),
    ),
)
def test_grasped_transport_route_budget_removes_only_internal_knot_settles(
    segment_waypoint_counts,
    legacy_steps,
    scheduled_steps,
):
    budget = generator._grasped_transport_route_control_budget(
        segment_waypoint_counts,
        duration_per_knot_s=0.20,
        segment_end_settle_s=1.0 / 30.0,
        advance_dt=1.0 / 30.0,
    )

    waypoint_count = sum(segment_waypoint_counts)
    assert budget["waypoint_count"] == waypoint_count
    assert budget["legacy_route_motion_control_step_count"] == 6 * waypoint_count
    assert budget["legacy_internal_knot_settle_control_step_count"] == waypoint_count
    assert budget["legacy_route_control_step_count"] == legacy_steps
    assert budget["scheduled_segment_end_settle_control_step_count"] == len(segment_waypoint_counts)
    assert budget["scheduled_route_control_step_count"] == scheduled_steps
    assert budget["scheduled_route_control_step_reduction"] == legacy_steps - scheduled_steps


def test_grasped_transport_target_selection_never_reactivates_a_failed_lane():
    held = torch.tensor(((0.1, 0.2), (1.0, 2.0)))
    active = torch.tensor((False, True))

    first = generator._retain_active_grasped_transport_target(
        torch.tensor(((9.0, 9.0), (3.0, 4.0))),
        held,
        active,
    )
    second = generator._retain_active_grasped_transport_target(
        torch.tensor(((8.0, 8.0), (5.0, 6.0))),
        first,
        active,
    )

    assert torch.equal(first[0], held[0])
    assert torch.equal(second[0], held[0])
    assert torch.equal(second[1], torch.tensor((5.0, 6.0)))


def test_grasped_transport_endpoint_policies_keep_canonical_strict_and_rows_settled():
    cfg = generator.GeneratorCfg()

    assert generator._resolve_grasped_transport_endpoint_policy(
        cfg,
        generator._GRASPED_TRANSPORT_STRICT_ENDPOINT_POLICY,
    ) == (0.002, True)
    assert generator._resolve_grasped_transport_endpoint_policy(
        cfg,
        generator._GRASPED_TRANSPORT_RESET_ROW_ENDPOINT_POLICY,
    ) == (0.006, False)
    with pytest.raises(ValueError, match="Unknown grasped-transport endpoint policy"):
        generator._resolve_grasped_transport_endpoint_policy(cfg, "ambiguous")


def test_grasped_transport_endpoint_position_gates_are_inclusive_and_context_specific():
    tcp_error = torch.tensor((0.002, 0.002001, 0.006, 0.006001, float("nan")))
    plug_error = torch.tensor((0.002, 0.001, 0.006, 0.001, 0.001))

    strict = generator._grasped_transport_endpoint_position_mask(
        tcp_error,
        plug_error,
        position_tolerance_m=0.002,
    )
    reset_row = generator._grasped_transport_endpoint_position_mask(
        tcp_error,
        plug_error,
        position_tolerance_m=0.006,
    )

    assert strict.tolist() == [True, False, False, False, False]
    assert reset_row.tolist() == [True, True, True, False, False]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("tcp_compensation_tolerance_m", 0.002001, "exact 2 mm"),
        ("grasped_transport_row_endpoint_position_tolerance_m", 0.006001, "exact 6 mm"),
    ),
)
def test_grasped_transport_endpoint_tolerances_are_immutable(field, value, message):
    with pytest.raises(ValueError, match=message):
        generator.GeneratorCfg(**{field: value})


def test_grasped_transport_terminal_translation_is_bounded_and_plug_priority():
    step, correction_needed, plug_priority = generator._grasped_transport_terminal_translation_step(
        current_tcp_position=torch.tensor(((0.0, 0.0, 0.0),) * 4),
        target_tcp_position=torch.tensor(
            (
                (0.0, 0.003, 0.0),
                (0.0, 0.003, 0.0),
                (0.0, 0.001, 0.0),
                (0.0, 0.002, 0.0),
            )
        ),
        current_plug_position=torch.tensor(((0.0, 0.0, 0.0),) * 4),
        target_plug_position=torch.tensor(
            (
                (0.004, 0.0, 0.0),
                (0.001, 0.0, 0.0),
                (0.001, 0.0, 0.0),
                (0.002, 0.0, 0.0),
            )
        ),
        position_tolerance_m=0.002,
        maximum_step_m=0.001,
    )

    assert correction_needed.tolist() == [True, True, False, False]
    assert plug_priority.tolist() == [True, False, False, False]
    torch.testing.assert_close(
        step,
        torch.tensor(
            (
                (0.001, 0.0, 0.0),
                (0.0, 0.001, 0.0),
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
            )
        ),
    )


def test_grasped_transport_terminal_progress_is_selected_and_tolerance_aware():
    progress = generator._grasped_transport_terminal_progress_mask(
        tcp_error_before_m=torch.tensor((0.0030, 0.0015, 0.0030, 0.0030, 0.0030, 0.0015)),
        tcp_error_after_m=torch.tensor((0.0035, 0.0021, 0.0025, 0.0025, 0.0032, 0.0019)),
        plug_error_before_m=torch.tensor((0.0040, 0.0040, 0.0015, 0.0015, 0.0020005, 0.0020005)),
        plug_error_after_m=torch.tensor((0.0030, 0.0030, 0.0019, 0.0021, 0.0020001, 0.0020)),
        correction_mask=torch.ones(6, dtype=torch.bool),
        plug_priority=torch.tensor((True, True, False, False, True, True)),
        position_tolerance_m=0.002,
        progress_epsilon_m=1.0e-6,
    )

    assert progress.tolist() == [True, False, True, False, False, True]


def test_grasped_transport_observes_transient_cable_speed_without_weakening_motion_speed_gates():
    cable_within_reset_limit, motion_speeds_bounded = generator._grasped_transport_transient_speed_masks(
        torch.tensor((0.10, 0.01, 0.01, 0.01, 0.01, 0.04)),
        torch.tensor((0.01, 0.041, 0.01, 0.01, 0.01, 0.04)),
        torch.tensor((0.01, 0.01, 0.351, 0.01, 0.01, 0.35)),
        torch.tensor((0.01, 0.01, 0.01, 0.501, 0.01, 0.5)),
        torch.tensor((0.01, 0.01, 0.01, 0.01, 0.051, 0.05)),
        maximum_reset_cable_speed_m_s=0.04,
        maximum_transport_plug_linear_speed_m_s=0.04,
        maximum_transport_plug_angular_speed_rad_s=0.35,
        maximum_transport_arm_joint_speed_rad_s=0.5,
        maximum_transport_finger_joint_speed_m_s=0.05,
    )

    assert cable_within_reset_limit.tolist() == [False, True, True, True, True, True]
    assert motion_speeds_bounded.tolist() == [True, False, False, False, False, True]


def test_generator_constructs_one_sampler_free_row_ik_owner(monkeypatch):
    constructed: list[dict[str, object]] = []

    class FakeIK:
        def __init__(self, _env, **kwargs):
            constructed.append(kwargs)

    task_q = torch.zeros((2, 5, 7))
    task_q[..., 6] = 1.0
    env = SimpleNamespace(
        num_envs=2,
        device="cpu",
        cfg=SimpleNamespace(
            actions=SimpleNamespace(
                gripper_action=SimpleNamespace(neutral_position=0.04, default_position=0.04),
            ),
            plug_grasp_orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
            task_translation=(0.0, 0.0, 0.0),
            task_rotation_xyzw=(0.0, 0.0, 0.0, 1.0),
        ),
        rj45_runtime=SimpleNamespace(
            layout=SimpleNamespace(
                body_count=5,
                socket_body_index=0,
                plug_body_index=1,
                latch_body_index=2,
                cable_body_slice=slice(3, 5),
            )
        ),
        restore_default_task=lambda: None,
        read_task_state=lambda: (task_q.clone(), torch.zeros((2, 5, 6))),
    )
    monkeypatch.setattr(generator, "FrankaResetIK", FakeIK)
    monkeypatch.setattr(generator, "pick_insert_tool_physical_contract", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(generator, "configured_arm_home", lambda _env: torch.zeros(7))

    owner = generator.PickInsertResetDatasetGenerator(env, generator.GeneratorCfg(batch_size=2))

    assert len(constructed) == 1
    assert constructed[0] == {
        "seed": 2026,
        "seeds": 1,
        "iterations": 160,
        "noise_std": 0.0,
        "sampler": "none",
    }
    assert owner._ik_solve_call_count == 0


def test_pickup_only_runner_routes_without_goal_derivation_or_artifact_output(capsys):
    class FakeGenerator:
        pickup_calls = 0
        goal_calls = 0

        def run_diagnostic_pickup_once(self):
            self.pickup_calls += 1
            return {"mode": "diagnostic-pickup-only", "attempted_batch_count": 1}

        def derive_goal(self):
            self.goal_calls += 1
            raise AssertionError("Pickup-only routing must not derive a canonical goal.")

    owner = FakeGenerator()

    evidence = generator._run_save_disabled_diagnostic(
        owner,
        SimpleNamespace(diagnostic_pickup_only=True),
    )

    assert evidence["attempted_batch_count"] == 1
    assert owner.pickup_calls == 1
    assert owner.goal_calls == 0
    output = capsys.readouterr().out
    assert output.startswith("[PICK-INSERT PICKUP-ONLY COMPLETE] {")
    assert '"attempted_batch_count": 1' in output


def test_phase0_transport_runner_routes_without_goal_derivation_or_artifact_output(capsys):
    class FakeGenerator:
        transport_calls = 0
        pickup_calls = 0
        goal_calls = 0

        def __init__(self):
            self._ik_solve_call_count = 0
            self.random = torch.Generator().manual_seed(999)

        def run_diagnostic_phase0_transport_once(self, canonical_goal):
            self.transport_calls += 1
            assert set(canonical_goal) == set(generator.RESET_DATASET_GOAL_STATE_NAMES)
            assert torch.equal(self.random.get_state(), certified_rng_state)
            return {"mode": "diagnostic-phase0-transport-only", "attempted_batch_count": 1}

        def run_diagnostic_pickup_once(self):
            self.pickup_calls += 1
            raise AssertionError("Phase-0 transport routing must not enter pickup-only mode.")

        def derive_goal(self):
            self.goal_calls += 1
            raise AssertionError("Phase-0 transport routing must not derive a canonical goal.")

    owner = FakeGenerator()
    certified_rng = torch.Generator().manual_seed(91)
    torch.rand(3, generator=certified_rng)
    certified_rng_state = certified_rng.get_state().clone()
    certificate = {
        "goal_state": {name: torch.tensor((1.0,)) for name in generator.RESET_DATASET_GOAL_STATE_NAMES},
        "row_rng_state": certified_rng_state,
    }

    evidence = generator._run_save_disabled_diagnostic(
        owner,
        SimpleNamespace(diagnostic_pickup_only=False, diagnostic_phase0_transport_only=True),
        certificate,
    )

    assert evidence["attempted_batch_count"] == 1
    assert owner.transport_calls == 1
    assert owner.pickup_calls == 0
    assert owner.goal_calls == 0
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("phase", (2, 4, 5))
def test_recovery_runner_restores_certified_rng_and_routes_exact_phase_without_goal_derivation(capsys, phase):
    class FakeGenerator:
        def __init__(self):
            self._ik_solve_call_count = 0
            self.random = torch.Generator().manual_seed(999)
            self.calls: list[tuple[int, dict[str, torch.Tensor]]] = []

        def run_diagnostic_recovery_phase_once(self, phase, canonical_goal):
            assert torch.equal(self.random.get_state(), certified_rng_state)
            self.calls.append((phase, canonical_goal))
            return {"mode": "diagnostic-recovery-phase", "phase": phase, "attempted_batch_count": 1}

        def derive_goal(self):
            raise AssertionError("Recovery diagnostic routing must not derive a canonical goal.")

    owner = FakeGenerator()
    certified_rng = torch.Generator().manual_seed(91)
    torch.rand(3, generator=certified_rng)
    certified_rng_state = certified_rng.get_state().clone()
    certificate = {
        "goal_state": {name: torch.tensor((1.0,)) for name in generator.RESET_DATASET_GOAL_STATE_NAMES},
        "row_rng_state": certified_rng_state,
    }

    evidence = generator._run_save_disabled_diagnostic(
        owner,
        SimpleNamespace(
            diagnostic_pickup_only=False,
            diagnostic_phase0_transport_only=False,
            diagnostic_recovery_phase=phase,
        ),
        certificate,
    )

    assert evidence == {"mode": "diagnostic-recovery-phase", "phase": phase, "attempted_batch_count": 1}
    assert len(owner.calls) == 1
    assert owner.calls[0][0] == phase
    assert set(owner.calls[0][1]) == set(generator.RESET_DATASET_GOAL_STATE_NAMES)
    assert capsys.readouterr().out == ""


def test_phase0_transport_runner_requires_fresh_row_ik_before_restoring_rng():
    certified_rng_state = torch.Generator().manual_seed(91).get_state()
    owner = SimpleNamespace(
        _ik_solve_call_count=1,
        random=torch.Generator().manual_seed(999),
        run_diagnostic_phase0_transport_once=lambda _goal: pytest.fail("diagnostic must not run"),
    )
    initial_rng_state = owner.random.get_state().clone()
    certificate = {
        "goal_state": {name: torch.tensor((1.0,)) for name in generator.RESET_DATASET_GOAL_STATE_NAMES},
        "row_rng_state": certified_rng_state,
    }

    with pytest.raises(RuntimeError, match="fresh, unadvanced row IK stream"):
        generator._run_save_disabled_diagnostic(
            owner,
            SimpleNamespace(diagnostic_pickup_only=False, diagnostic_phase0_transport_only=True),
            certificate,
        )

    assert torch.equal(owner.random.get_state(), initial_rng_state)


def test_phase0_transport_completion_is_published_only_after_live_revalidation(monkeypatch, capsys):
    events: list[str] = []
    owner = SimpleNamespace(
        _canonical_goal_certificate_validation_kwargs=lambda: events.append("snapshot") or {"live": True}
    )
    monkeypatch.setattr(
        generator,
        "_require_unchanged_canonical_goal_validation_snapshot",
        lambda *_args, **_kwargs: events.append("unchanged"),
    )
    monkeypatch.setattr(
        generator,
        "_validate_canonical_goal_certificate",
        lambda *_args, **_kwargs: events.append("validated"),
    )

    generator._finalize_certificate_backed_phase0_diagnostic(
        owner,
        {"certificate": True},
        {"before": True},
        {"attempted_batch_count": 1},
    )

    assert events == ["snapshot", "unchanged", "validated"]
    assert capsys.readouterr().out.startswith("[PICK-INSERT PHASE0-TRANSPORT-ONLY COMPLETE] {")


def test_phase0_transport_completion_is_silent_when_live_revalidation_fails(monkeypatch, capsys):
    owner = SimpleNamespace(_canonical_goal_certificate_validation_kwargs=lambda: {"live": True})
    monkeypatch.setattr(
        generator,
        "_require_unchanged_canonical_goal_validation_snapshot",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("source drift")),
    )

    with pytest.raises(RuntimeError, match="source drift"):
        generator._finalize_certificate_backed_phase0_diagnostic(
            owner,
            {"certificate": True},
            {"before": True},
            {"attempted_batch_count": 1},
        )

    assert capsys.readouterr().out == ""


def test_recovery_completion_is_published_only_after_live_revalidation(monkeypatch, capsys):
    events: list[str] = []
    owner = SimpleNamespace(
        _canonical_goal_certificate_validation_kwargs=lambda: events.append("snapshot") or {"live": True}
    )
    monkeypatch.setattr(
        generator,
        "_require_unchanged_canonical_goal_validation_snapshot",
        lambda *_args, **kwargs: events.append(kwargs["operation"]),
    )
    monkeypatch.setattr(
        generator,
        "_validate_canonical_goal_certificate",
        lambda *_args, **_kwargs: events.append("validated"),
    )

    evidence = {
        "phase": 2,
        "attempted_batch_count": 1,
        "scripted_recovery_evidence": {
            "maximum_commanded_joint_step_after_densification_rad": [0.019, 0.018],
            "cartesian_motion_first_plug_linear_speed_failure_step": [-1, 42],
        },
    }
    generator._finalize_certificate_backed_recovery_diagnostic(
        owner,
        {"certificate": True},
        {"before": True},
        evidence,
    )

    assert events == ["snapshot", "certificate-backed phase-2 recovery diagnostic", "validated"]
    output = capsys.readouterr().out
    prefix = "[PICK-INSERT RECOVERY-PHASE2 COMPLETE] "
    assert output.startswith(prefix)
    assert json.loads(output.removeprefix(prefix)) == evidence


def test_diagnostic_pickup_once_samples_and_constructs_exactly_one_batch():
    class FakeEnv:
        num_envs = 3
        device = "cpu"
        advance_dt = 0.05

        def __init__(self):
            self.advance_durations: list[float] = []

        def advance(self, duration_s, update=None, *, post_step=None):
            self.advance_durations.append(duration_s)
            return 4

    env = FakeEnv()
    owner = object.__new__(generator.PickInsertResetDatasetGenerator)
    owner.env = env
    owner.device = torch.device("cpu")
    owner.cfg = SimpleNamespace(seed=91)
    owner._ik_solve_call_count = 5
    sample_calls = 0
    construct_calls = 0
    socket_pose = torch.zeros((3, 7))
    pickup_pose = torch.ones((3, 7))

    def sample_scene():
        nonlocal sample_calls
        sample_calls += 1
        return socket_pose, pickup_pose

    def construct_pickup(
        observed_socket_pose,
        observed_pickup_pose,
        *,
        acquire,
        active_mask,
        diagnostic_evidence,
    ):
        nonlocal construct_calls
        construct_calls += 1
        assert observed_socket_pose is socket_pose
        assert observed_pickup_pose is pickup_pose
        assert acquire is True
        assert active_mask.tolist() == [True, True, True]
        env.advance(0.1)
        env.advance(0.2)
        owner._ik_solve_call_count += 3
        diagnostic_evidence.update(
            {
                "initial_active_mask": [True, True, True],
                "survival_mask": [True, False, True],
                "lane_results": [],
                "construction_robot_staging": {
                    "advance_call_count": 1,
                    "control_step_count": 2,
                    "ik_solve_call_count": 1,
                },
                "drive_free_local_alignment": {
                    "local_acquisition_counts": {
                        "call_count": 1,
                        "advance_call_count": 7,
                        "control_step_count": 31,
                        "ik_solve_call_count": 24,
                        "descent_waypoint_count": 23,
                    }
                },
                "placement": {},
                "acquisition": {},
            }
        )
        return torch.zeros((3, 7)), torch.zeros((3, 2)), torch.tensor((True, False, True))

    owner._sample_scene = sample_scene
    owner._construct_pickup = construct_pickup

    evidence = owner.run_diagnostic_pickup_once()

    assert sample_calls == 1
    assert construct_calls == 1
    assert env.advance_durations == [0.1, 0.2]
    assert "advance" not in env.__dict__
    assert evidence["attempted_batch_count"] == 1
    assert evidence["sampled_scene_count"] == 1
    assert evidence["construct_pickup_call_count"] == 1
    assert evidence["ik_solve_call_count_before"] == 5
    assert evidence["ik_solve_call_count_after"] == 8
    assert evidence["ik_solve_call_delta"] == 3
    assert evidence["advance_call_count"] == 2
    assert evidence["control_step_count"] == 8
    assert evidence["construction_staging_advance_call_count"] == 1
    assert evidence["construction_staging_control_step_count"] == 2
    assert evidence["construction_staging_ik_solve_call_count"] == 1
    assert evidence["local_acquisition_call_count"] == 1
    assert evidence["local_acquisition_advance_call_count"] == 7
    assert evidence["local_acquisition_control_step_count"] == 31
    assert evidence["local_acquisition_ik_solve_call_count"] == 24
    assert evidence["local_descent_waypoint_count"] == 23
    assert evidence["simulated_time_s"] == pytest.approx(0.4)
    assert evidence["survivor_count"] == 2
    assert evidence["yield_fraction"] == pytest.approx(2.0 / 3.0)
    assert evidence["survival_mask"] == [True, False, True]


def test_diagnostic_phase0_transport_once_reuses_pickup_and_phase_realization_paths():
    class FakeEnv:
        num_envs = 3
        device = "cpu"
        advance_dt = 0.05

        def __init__(self):
            self.advance_durations: list[float] = []
            self.task_q = torch.zeros((3, 5, 7))
            self.task_q[..., 6] = 1.0
            self.task_qd = torch.zeros((3, 5, 6))

        def restore_default_task(self):
            return None

        def read_task_state(self):
            return self.task_q.clone(), self.task_qd.clone()

        def advance(self, duration_s, update=None, *, post_step=None):
            self.advance_durations.append(duration_s)
            return int(math.ceil(duration_s / self.advance_dt))

    env = FakeEnv()
    owner = object.__new__(generator.PickInsertResetDatasetGenerator)
    owner.env = env
    owner.device = torch.device("cpu")
    owner.cfg = generator.GeneratorCfg(batch_size=3, seed=91)
    owner.socket_index = 0
    owner._ik_solve_call_count = 5
    sample_calls = 0
    row_goal_calls = 0
    construct_calls = 0
    realize_calls = 0
    socket_pose = torch.tensor(
        (
            (0.10, 0.20, 0.30, 0.0, 0.0, 0.0, 1.0),
            (0.20, 0.30, 0.40, 0.0, 0.0, 0.0, 1.0),
            (0.30, 0.40, 0.50, 0.0, 0.0, 0.0, 1.0),
        )
    )
    pickup_pose = socket_pose.clone()

    def sample_scene():
        nonlocal sample_calls
        sample_calls += 1
        return socket_pose, pickup_pose

    canonical_goal = {"task_body_pose": torch.full((5, 7), 7.0)}

    def row_goal(observed_canonical_goal, observed_socket_pose):
        nonlocal row_goal_calls
        row_goal_calls += 1
        assert observed_canonical_goal is not canonical_goal
        torch.testing.assert_close(observed_canonical_goal["task_body_pose"], canonical_goal["task_body_pose"])
        assert observed_socket_pose is socket_pose
        owner._ik_solve_call_count += 1
        goal_q = env.task_q.clone()
        goal_q[:, 0] = socket_pose
        return goal_q, env.task_qd.clone(), torch.zeros((3, 7)), torch.ones(3, dtype=torch.bool)

    def construct_pickup(
        observed_socket_pose,
        observed_pickup_pose,
        *,
        acquire,
        active_mask,
        diagnostic_evidence,
    ):
        nonlocal construct_calls
        construct_calls += 1
        assert observed_socket_pose is socket_pose
        assert observed_pickup_pose is pickup_pose
        assert acquire is True
        assert active_mask.tolist() == [True, True, True]
        env.advance(0.1)
        owner._ik_solve_call_count += 3
        diagnostic_evidence.update({"survival_mask": [True, True, False]})
        return torch.zeros((3, 7)), torch.zeros((3, 2)), torch.tensor((True, True, False))

    def realize_phase(
        phase,
        observed_pickup_pose,
        goal_q,
        pickup_arm_target,
        *,
        pickup_finger_target,
        active_mask,
    ):
        nonlocal realize_calls
        realize_calls += 1
        assert phase == 0
        assert observed_pickup_pose is pickup_pose
        torch.testing.assert_close(goal_q[:, 0], socket_pose)
        assert pickup_arm_target.shape == (3, 7)
        assert pickup_finger_target.shape == (3, 2)
        assert active_mask.tolist() == [True, True, False]
        env.advance(0.2)
        owner._ik_solve_call_count += 10
        owner.last_grasped_motion_evidence = {
            "transport_schedule": "c2-endpoint-time-law-piecewise-linear-joint-cruise",
            "maximum_internal_target_velocity_jump_rad_s": torch.tensor((0.1, 0.2, 0.0)),
        }
        return pickup_arm_target, pickup_finger_target, torch.tensor((True, False, False))

    owner._sample_scene = sample_scene
    owner._row_goal = row_goal
    owner._construct_pickup = construct_pickup
    owner._realize_phase = realize_phase

    evidence = owner.run_diagnostic_phase0_transport_once(canonical_goal)

    assert sample_calls == 1
    assert row_goal_calls == 1
    assert construct_calls == 1
    assert realize_calls == 1
    assert env.advance_durations == [0.1, 0.2]
    assert "advance" not in env.__dict__
    assert evidence["attempted_batch_count"] == 1
    assert evidence["construct_pickup_call_count"] == 1
    assert evidence["realize_phase0_call_count"] == 1
    assert evidence["ik_solve_call_delta"] == 14
    assert evidence["goal_ik_solve_call_count"] == 1
    assert evidence["pickup_ik_solve_call_count"] == 3
    assert evidence["phase0_transport_ik_solve_call_count"] == 10
    assert evidence["advance_call_count"] == 2
    assert evidence["control_step_count"] == 6
    assert evidence["pickup_control_step_count"] == 2
    assert evidence["phase0_transport_control_step_count"] == 4
    assert evidence["pickup_survivor_count"] == 2
    assert evidence["phase0_survivor_count"] == 1
    assert evidence["phase0_yield_given_pickup"] == pytest.approx(0.5)
    assert evidence["goal_is_canonical_certificate"] is True
    assert evidence["goal_ik_valid_mask"] == [True, True, True]
    assert evidence["grasped_transport_evidence"]["maximum_internal_target_velocity_jump_rad_s"] == pytest.approx(
        [0.1, 0.2, 0.0]
    )


def test_diagnostic_recovery_phase_once_stops_at_one_complete_production_batch():
    owner = object.__new__(generator.PickInsertResetDatasetGenerator)
    owner.env = SimpleNamespace(num_envs=24)
    owner.device = torch.device("cpu")
    owner.cfg = SimpleNamespace(seed=91, quick=False, rows_per_phase=96)
    owner._ik_solve_call_count = 0
    owner.attempt_counts = [0] * 6
    owner.rejection_counts = {phase: {} for phase in range(6)}
    observed: list[tuple[int, dict[str, torch.Tensor]]] = []
    canonical_goal = {"task_body_pose": torch.zeros((3, 7))}
    batch_evidence = generator._scripted_recovery_diagnostic_evidence(
        _fake_recovery_diagnostic_metrics(owner.env.num_envs, phase=2)
    )

    def generate_phase(
        phase,
        observed_goal,
        *,
        diagnostic_batch_evidence_callback,
        completed_batch_callback,
    ):
        observed.append((phase, observed_goal))
        owner._ik_solve_call_count += 17
        owner.attempt_counts[phase] += owner.env.num_envs
        owner.rejection_counts[phase] = {
            "oracle_full_recovery": 20,
            "oracle_recovery_lane_recovery-cartesian-motion-arm-joint-speed": 19,
            "accepted": 4,
        }
        diagnostic_batch_evidence_callback(phase, 0, batch_evidence)
        completed_batch_callback(
            phase,
            0,
            {"phase": torch.full((4,), phase, dtype=torch.int64)},
        )
        raise AssertionError("The diagnostic callback must stop the production loop.")

    owner._generate_phase = generate_phase

    evidence = owner.run_diagnostic_recovery_phase_once(2, canonical_goal)

    assert len(observed) == 1
    assert observed[0][0] == 2
    assert observed[0][1] is not canonical_goal
    assert observed[0][1]["task_body_pose"] is canonical_goal["task_body_pose"]
    assert evidence["phase"] == 2
    assert evidence["attempted_batch_count"] == 1
    assert evidence["production_phase_batch_index"] == 0
    assert {
        evidence[name]
        for name in (
            "sampled_scene_count",
            "row_goal_call_count",
            "construct_pickup_call_count",
            "realize_phase_call_count",
            "cold_replay_call_count",
            "phase_semantics_call_count",
            "oracle_call_count",
        )
    } == {1}
    assert evidence["attempt_count_delta"] == 24
    assert evidence["accepted_row_count"] == 4
    assert evidence["yield_fraction"] == pytest.approx(1.0 / 6.0)
    assert evidence["ik_solve_call_delta"] == 17
    assert evidence["rejection_counts"] == owner.rejection_counts[2]
    assert evidence["production_path"][-1] == "scripted-recovery-oracle"
    assert evidence["scripted_recovery"] == PICK_INSERT_RECOVERY_CARTESIAN_C2_POLICY
    assert evidence["scripted_recovery_evidence"] == generator._plain_certificate_value(batch_evidence)
    assert (
        evidence["scripted_recovery_evidence"]["motion_policy"]
        == PICK_INSERT_RECOVERY_CARTESIAN_C2_POLICY["motion_policy"]
    )
    assert evidence["scripted_recovery_evidence"]["pick_insert_phase"] == [2] * 24
    assert evidence["scripted_recovery_evidence"]["cartesian_motion_first_arm_joint_speed_failure_step"] == [12] * 24


def test_diagnostic_pickup_lane_results_include_placement_and_acquisition_failures():
    results = generator._diagnostic_pickup_lane_results(
        active_mask=torch.tensor((True, True, True)),
        survival_mask=torch.tensor((True, False, False)),
        placement_gate_masks={"socket-drift-bounded": torch.tensor((True, False, True))},
        acquisition_failure_masks={"open-near-ik-continuity": torch.tensor((False, False, True))},
    )

    assert results == [
        {"lane": 0, "active": True, "survived": True, "failure_reasons": []},
        {
            "lane": 1,
            "active": True,
            "survived": False,
            "failure_reasons": ["placement/socket-drift-bounded"],
        },
        {
            "lane": 2,
            "active": True,
            "survived": False,
            "failure_reasons": ["acquisition/open-near-ik-continuity"],
        },
    ]


def test_construction_robot_staging_freezes_failed_lane_and_preserves_open_target(monkeypatch):
    class FakeEnv:
        num_envs = 2
        device = "cpu"
        _arm_joint_ids = torch.arange(7)
        _finger_joint_ids = torch.arange(7, 9)

        def __init__(self):
            self.cfg = SimpleNamespace(
                plug_grasp_offset=(0.0, 0.0, 0.0),
                actions=SimpleNamespace(
                    arm_action=SimpleNamespace(tracking_error_limits=(0.1,) * 7),
                ),
            )
            self.arm_q = torch.tensor(((-0.1,) * 7, (0.08,) * 7))
            self.arm_qd = torch.zeros_like(self.arm_q)
            self.finger_q = torch.full((2, 2), 0.04)
            self.finger_qd = torch.zeros_like(self.finger_q)
            self.arm_target = torch.full((2, 7), 0.1)
            self.finger_target = self.finger_q.clone()
            joint_target = torch.cat((self.arm_target, self.finger_target), dim=-1)
            self._robot = SimpleNamespace(data=SimpleNamespace(joint_pos_target=SimpleNamespace(torch=joint_target)))
            self.robot_writes: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
            self.target_writes: list[tuple[torch.Tensor, torch.Tensor]] = []
            self.flush_count = 0
            self.clearance_pose = torch.zeros((2, 7))
            self.clearance_pose[:, 2] = 0.145
            self.clearance_pose[:, 3] = 1.0

        def write_robot_state(
            self,
            arm_q,
            finger_q,
            *,
            arm_target,
            arm_qd,
            finger_qd,
            finger_target,
        ):
            self.arm_q = arm_q.clone()
            self.arm_qd = arm_qd.clone()
            self.finger_q = finger_q.clone()
            self.finger_qd = finger_qd.clone()
            self.arm_target = arm_target.clone()
            self.finger_target = finger_target.clone()
            self._robot.data.joint_pos_target.torch = torch.cat(
                (self.arm_target, self.finger_target),
                dim=-1,
            )
            self.robot_writes.append((arm_q.clone(), arm_target.clone(), arm_qd.clone(), finger_q.clone()))

        def flush_reset_history(self):
            self.flush_count += 1

        def set_robot_targets(self, arm_target, finger_target):
            self.arm_target = arm_target.clone()
            self.finger_target = finger_target.clone()
            self._robot.data.joint_pos_target.torch = torch.cat(
                (self.arm_target, self.finger_target),
                dim=-1,
            )
            self.target_writes.append((arm_target.clone(), finger_target.clone()))

        def advance(self, _duration_s, update=None, *, post_step=None):
            if update is not None:
                update(0, 1, 1.0)
            if post_step is not None:
                post_step(0, 1, 1.0)
            return 1

        def read_task_state(self):
            task_q = torch.zeros((2, 2, 7))
            task_q[..., 3] = 1.0
            return task_q, torch.zeros((2, 2, 6))

        def read_robot_state(self):
            return self.arm_q, self.arm_qd, self.finger_q, self.finger_qd

        def tcp_pose_e(self):
            return self.clearance_pose.clone()

    class FakeIK:
        def __init__(self):
            self.orientations: list[torch.Tensor] = []

        def solve(self, _position, orientation, *_args, **_kwargs):
            self.orientations.append(orientation.clone())
            return SimpleNamespace(
                arm_q=torch.tensor(((0.3,) * 7, (0.2,) * 7)),
                valid=torch.tensor((True, True)),
            )

    env = FakeEnv()
    owner = object.__new__(generator.PickInsertResetDatasetGenerator)
    owner.env = env
    owner.ik = FakeIK()
    owner.cfg = generator.GeneratorCfg(batch_size=2)
    owner.device = torch.device("cpu")
    owner.home_arm_q = torch.tensor(((0.1,) * 7,) * 2)
    owner.open_finger_q = torch.full((2, 2), 0.04)
    owner.local_grasp_orientation = torch.tensor(((1.0, 0.0, 0.0, 0.0),) * 2)
    owner._ik_solve_call_count = 0
    owner._drive_enabled = lambda: torch.ones(2, dtype=torch.bool)
    owner._orientation_hold_enabled = lambda: torch.ones(2, dtype=torch.bool)
    monkeypatch.setattr(generator, "joint_limit_mask", lambda *_args, **_kwargs: torch.ones(2, dtype=torch.bool))
    monkeypatch.setattr(
        generator,
        "task_state_is_finite_and_normalized",
        lambda *_args, **_kwargs: torch.ones(2, dtype=torch.bool),
    )
    monkeypatch.setattr(
        generator,
        "collision_metrics",
        lambda *_args, **_kwargs: SimpleNamespace(
            valid=torch.ones(2, dtype=torch.bool),
            left_grasp_contact_count=torch.zeros(2, dtype=torch.long),
            right_grasp_contact_count=torch.zeros(2, dtype=torch.long),
            contact_overflow=False,
            invalid_contact_pairs=(),
        ),
    )
    pickup_pose = torch.zeros((2, 7))
    pickup_pose[:, 2] = 0.1
    pickup_pose[:, 3] = 1.0
    orientation_error = generator.math_utils.quat_from_angle_axis(
        torch.tensor((math.radians(20.0),) * 2),
        torch.tensor(((0.0, 0.0, 1.0),) * 2),
    )
    expected_position, expected_orientation = owner._desired_tcp_pose(
        pickup_pose,
        orientation_error_xyzw=orientation_error,
    )
    expected_position[:, 2] += owner.cfg.grasp_open_clearance_m
    env.clearance_pose = torch.cat((expected_position, expected_orientation), dim=-1)

    staged_target, valid, evidence = owner._stage_robot_for_pickup(
        pickup_pose,
        active_mask=torch.ones(2, dtype=torch.bool),
        orientation_error_xyzw=orientation_error,
    )

    assert valid.tolist() == [False, True]
    assert torch.equal(staged_target[0], owner.home_arm_q[0])
    assert torch.allclose(staged_target[1], torch.full((7,), 0.22))
    raw_q, biased_target, arm_qd, written_finger_q = env.robot_writes[0]
    assert torch.equal(raw_q[0], owner.home_arm_q[0])
    assert torch.allclose(raw_q[1], torch.full((7,), 0.2))
    assert torch.equal(biased_target, staged_target)
    assert torch.equal(arm_qd, torch.zeros_like(arm_qd))
    assert torch.equal(written_finger_q, owner.open_finger_q)
    assert evidence["stage_entry_bias_within_tracking_limits"].tolist() == [False, True]
    assert torch.allclose(evidence["stage_entry_equilibrium_bias"][1], torch.full((7,), 0.02))
    assert torch.equal(evidence["raw_staged_arm_q"], raw_q)
    assert torch.equal(evidence["biased_staged_arm_target"], biased_target)
    assert env.flush_count == 1
    assert owner._ik_solve_call_count == 1
    assert evidence["ik_solve_call_count"] == 1
    assert evidence["control_step_count"] == 1
    torch.testing.assert_close(owner.ik.orientations[0], expected_orientation)
    assert all(torch.equal(finger, owner.open_finger_q) for _, finger in env.target_writes)


def test_coherent_task_write_preserves_staged_robot_state_and_targets():
    class FakeEnv:
        _arm_joint_ids = torch.arange(7)
        _finger_joint_ids = torch.arange(7, 9)

        def __init__(self):
            self.arm_q = torch.arange(14, dtype=torch.float32).reshape(2, 7)
            self.arm_qd = torch.zeros_like(self.arm_q)
            self.finger_q = torch.full((2, 2), 0.04)
            self.finger_qd = torch.zeros_like(self.finger_q)
            targets = torch.cat((self.arm_q + 0.5, self.finger_q), dim=-1)
            self._robot = SimpleNamespace(data=SimpleNamespace(joint_pos_target=SimpleNamespace(torch=targets)))
            self.task_writes = 0

        def read_robot_state(self):
            return tuple(value.clone() for value in (self.arm_q, self.arm_qd, self.finger_q, self.finger_qd))

        def write_task_state(self, _task_q, _task_qd):
            self.task_writes += 1

    env = FakeEnv()
    owner = object.__new__(generator.PickInsertResetDatasetGenerator)
    owner.env = env
    staged_arm_target = env._robot.data.joint_pos_target.torch[:, :7].clone()
    staged_finger_target = env._robot.data.joint_pos_target.torch[:, 7:9].clone()

    evidence = owner._write_task_state_preserving_staged_robot(
        torch.zeros((2, 3, 7)),
        torch.zeros((2, 3, 6)),
        staged_arm_target=staged_arm_target,
        staged_finger_target=staged_finger_target,
    )

    assert env.task_writes == 1
    assert evidence["unchanged"].tolist() == [True, True]
    assert all(not bool(value.any()) for name, value in evidence.items() if name != "unchanged")


def test_local_pickup_descent_uses_twenty_three_two_millimeter_waypoints():
    cfg = generator.GeneratorCfg()

    assert generator._LOCAL_PICKUP_DESCENT_STEP_M == 0.002
    assert math.ceil(cfg.grasp_open_clearance_m / generator._LOCAL_PICKUP_DESCENT_STEP_M) == 23


def test_grasped_row_phases_route_only_through_the_six_millimeter_settled_endpoint_policy():
    class FakeEnv:
        num_envs = 2
        device = "cpu"

        def __init__(self):
            self.task_q = torch.zeros((2, 1, 7))
            self.task_q[..., 6] = 1.0

        def read_task_state(self):
            return self.task_q.clone(), torch.zeros((2, 1, 6))

        def advance(self, _duration_s, update=None, *, post_step=None):
            return 1

    env = FakeEnv()
    owner = object.__new__(generator.PickInsertResetDatasetGenerator)
    owner.env = env
    owner.cfg = generator.GeneratorCfg(batch_size=2)
    owner.device = torch.device("cpu")
    owner.plug_index = 0
    owner._assert_drive_disabled = lambda _context: None
    policies: list[str] = []

    def record_move(_plug_target, arm_seed, **kwargs):
        policies.append(kwargs["endpoint_policy"])
        return arm_seed, kwargs["lane_hold"].active_mask

    owner._move_grasped_plug = record_move
    arm_target = torch.zeros((2, 7))
    finger_target = torch.zeros((2, 2))
    lane_hold = SimpleNamespace(
        active_mask=torch.ones(2, dtype=torch.bool),
        last_sent_arm_target=arm_target,
        last_sent_finger_target=finger_target,
    )
    pickup_pose = env.task_q[:, 0].clone()
    goal_q = env.task_q.clone()

    for phase in range(4):
        _, _, valid = owner._realize_phase_per_lane(
            phase,
            pickup_pose,
            goal_q,
            arm_target,
            lane_hold=lane_hold,
        )
        assert valid.tolist() == [True, True]

    assert policies == [generator._GRASPED_TRANSPORT_RESET_ROW_ENDPOINT_POLICY] * 4


def test_phase_four_preserves_constructed_open_clearance_without_broad_motion():
    env = _LaneMotionFakeEnv()
    owner = object.__new__(generator.PickInsertResetDatasetGenerator)
    owner.env = env
    owner.cfg = generator.GeneratorCfg(batch_size=2)
    owner.device = torch.device("cpu")
    owner.plug_index = 0
    owner.open_finger_q = torch.full((2, 2), 0.04)
    owner._open_gripper = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("phase 4 must not run a physical open route")
    )
    owner._move_tcp = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("phase 4 must not run redundant broad IK motion")
    )
    owner._assert_drive_disabled = lambda _context: None
    staged_arm_target = torch.tensor(((0.2,) * 7, (0.4,) * 7))
    pickup_pose = torch.zeros((2, 7))
    goal_q = torch.zeros((2, 1, 7))

    with generator._PerLaneTargetHold(
        env,
        torch.ones(2, dtype=torch.bool),
        staged_arm_target,
        owner.open_finger_q,
    ) as lane_hold:
        arm_target, finger_target, valid = owner._realize_phase_per_lane(
            4,
            pickup_pose,
            goal_q,
            staged_arm_target,
            lane_hold=lane_hold,
        )

    assert valid.tolist() == [True, True]
    assert torch.equal(arm_target, staged_arm_target)
    assert torch.equal(finger_target, owner.open_finger_q)
    assert len(env.physics_commands) == 1
    assert torch.equal(env.physics_commands[0][0], staged_arm_target)
    assert torch.equal(env.physics_commands[0][1], owner.open_finger_q)


def test_phase_five_keeps_physical_move_away_from_constructed_clearance(monkeypatch):
    env = _LaneMotionFakeEnv()
    env.cfg = SimpleNamespace(arm_reset_joint_noise=0.0)
    owner = object.__new__(generator.PickInsertResetDatasetGenerator)
    owner.env = env
    owner.cfg = generator.GeneratorCfg(batch_size=2)
    owner.device = torch.device("cpu")
    owner.plug_index = 0
    owner.open_finger_q = torch.full((2, 2), 0.04)
    owner.home_arm_q = torch.tensor(((0.7,) * 7,) * 2)
    owner.random = torch.Generator().manual_seed(7)
    owner._open_gripper = lambda *_args, **_kwargs: None
    owner._assert_drive_disabled = lambda _context: None
    interpolations: list[tuple[torch.Tensor, torch.Tensor]] = []

    def record_interpolation(observed_env, start, end, finger, _duration):
        interpolations.append((start.clone(), end.clone()))
        observed_env.set_robot_targets(end, finger)

    monkeypatch.setattr(generator, "joint_limit_mask", lambda *_args, **_kwargs: torch.ones(2, dtype=torch.bool))
    monkeypatch.setattr(generator, "interpolate_arm_motion", record_interpolation)
    staged_arm_target = torch.tensor(((0.2,) * 7, (0.4,) * 7))

    with generator._PerLaneTargetHold(
        env,
        torch.ones(2, dtype=torch.bool),
        staged_arm_target,
        owner.open_finger_q,
    ) as lane_hold:
        arm_target, finger_target, valid = owner._realize_phase_per_lane(
            5,
            torch.zeros((2, 7)),
            torch.zeros((2, 1, 7)),
            staged_arm_target,
            lane_hold=lane_hold,
        )

    assert valid.tolist() == [True, True]
    assert len(interpolations) == 1
    assert torch.equal(interpolations[0][1], owner.home_arm_q)
    assert torch.equal(arm_target, owner.home_arm_q)
    assert torch.equal(finger_target, owner.open_finger_q)
    assert len(env.physics_commands) == 2


def test_ungrasped_oracle_acquisition_retains_full_physical_route():
    owner = object.__new__(generator.PickInsertResetDatasetGenerator)
    owner.open_finger_q = torch.full((2, 2), 0.04)
    full_route_calls = []

    def full_route(arm_seed, *, duration_s, active_mask, move_attempt_count, move_settle_s):
        full_route_calls.append((arm_seed.clone(), duration_s, active_mask.clone(), move_attempt_count, move_settle_s))
        return arm_seed + 1.0, active_mask.clone(), {"last_finger_target": owner.open_finger_q.clone()}

    owner._acquire_current_plug = full_route
    owner._acquire_prepositioned_current_plug = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("ungrasped oracle must not use the construction-only local path")
    )
    arm_seed = torch.zeros((2, 7))
    active_mask = torch.tensor((True, False))

    arm_target, acquired, finger_target = owner._acquire_grasp(arm_seed, active_mask=active_mask)

    assert len(full_route_calls) == 1
    assert full_route_calls[0][1] == 2.5
    assert torch.equal(full_route_calls[0][2], active_mask)
    assert full_route_calls[0][3:] == (5, 0.30)
    assert torch.equal(arm_target, arm_seed + 1.0)
    assert torch.equal(acquired, active_mask)
    assert torch.equal(finger_target, owner.open_finger_q)


def test_starts_grasped_oracle_entry_restores_and_executes_full_reset_replay(monkeypatch):
    class FakeEnv:
        num_envs = 2

        def read_task_state(self):
            task_q = torch.zeros((2, 3, 7))
            task_q[..., 6] = 1.0
            return task_q, torch.zeros((2, 3, 6))

        def read_robot_state(self):
            return torch.zeros((2, 7)), torch.zeros((2, 7)), torch.zeros((2, 2)), torch.zeros((2, 2))

    owner = object.__new__(generator.PickInsertResetDatasetGenerator)
    owner.env = FakeEnv()
    owner.device = torch.device("cpu")
    owner.cfg = generator.GeneratorCfg(batch_size=2)
    restored: list[dict[str, torch.Tensor]] = []
    asserted_contexts: list[str] = []
    owner._restore_state = lambda state: restored.append(state) or {"body_order_exact": True}
    owner._vbd_pose_history_applied_mask = lambda _evidence: torch.ones(2, dtype=torch.bool)
    owner._assert_drive_disabled = asserted_contexts.append
    observed: dict[str, object] = {}
    reset_contact_count = 0

    def replay_hold(
        observed_env,
        duration_s,
        arm_target,
        finger_target,
        *,
        clamp_evidence,
        replay_evidence,
        starts_grasped,
    ):
        observed.update(
            {
                "env": observed_env,
                "duration_s": duration_s,
                "arm_target": arm_target.clone(),
                "finger_target": finger_target.clone(),
                "starts_grasped": starts_grasped,
            }
        )
        clamp_evidence.update(
            {
                "maximum_arm_target_clamp_delta": torch.zeros(2),
                "any_arm_target_clamped": torch.zeros(2, dtype=torch.bool),
                "maximum_arm_target_drift": torch.zeros(2),
            }
        )
        true = torch.ones(2, dtype=torch.bool)
        zero = torch.zeros(2)
        replay_evidence.update(
            {
                "stored_state_finite": true.clone(),
                "stored_task_state_finite_and_normalized": true.clone(),
                "stored_drive_disabled": true.clone(),
                "all_post_step_state_finite": true.clone(),
                "all_post_step_task_state_finite_and_normalized": true.clone(),
                "all_post_step_collision_free": true.clone(),
                "maximum_invalid_contact_count": torch.zeros(2, dtype=torch.long),
                "any_contact_overflow": False,
                "all_post_step_drive_disabled": true.clone(),
                "all_post_step_expected_contact_state": true.clone(),
                "all_post_step_bilateral_grasp": true.clone(),
                "all_post_step_proxy_bilateral_contact": true.clone(),
                "minimum_left_proxy_contact_count": torch.ones(2, dtype=torch.long),
                "minimum_right_proxy_contact_count": torch.ones(2, dtype=torch.long),
                "all_post_step_zero_proxy_contacts": true.clone(),
                "maximum_left_proxy_contact_count": torch.zeros(2, dtype=torch.long),
                "maximum_right_proxy_contact_count": torch.zeros(2, dtype=torch.long),
                "stored_arm_target_tracking_bounded": true.clone(),
                "all_post_step_arm_target_tracking_bounded": true.clone(),
                "stored_maximum_arm_joint_speed_rad_s": zero.clone(),
                "maximum_post_step_arm_joint_speed_rad_s": zero.clone(),
                "final_arm_joint_speed_rad_s": zero.clone(),
                "stored_maximum_finger_joint_speed_m_s": zero.clone(),
                "maximum_post_step_finger_joint_speed_m_s": zero.clone(),
                "final_finger_joint_speed_m_s": zero.clone(),
                "stored_maximum_cable_speed_m_s": zero.clone(),
                "maximum_post_step_cable_speed_m_s": zero.clone(),
                "final_cable_speed_m_s": zero.clone(),
                "maximum_body_excursion_m": zero.clone(),
                "maximum_plug_excursion_m": zero.clone(),
                "maximum_socket_excursion_m": zero.clone(),
                "post_step_samples": generator.PICK_INSERT_RESET_REPLAY_POST_STEP_SAMPLES,
            }
        )
        return generator.PICK_INSERT_RESET_REPLAY_POST_STEP_SAMPLES

    monkeypatch.setattr(generator, "advance_reset_absolute_target_hold", replay_hold)
    monkeypatch.setattr(generator, "_contact_count", lambda: reset_contact_count)
    monkeypatch.setattr(
        generator,
        "runtime_persistent_arm_target",
        lambda _env, target: (target + 0.25, torch.zeros(2)),
    )
    monkeypatch.setattr(
        generator,
        "task_state_is_finite_and_normalized",
        lambda *_args: torch.ones(2, dtype=torch.bool),
    )
    monkeypatch.setattr(generator, "joint_limit_mask", lambda *_args, **_kwargs: torch.ones(2, dtype=torch.bool))
    state = {
        "arm_joint_position": torch.zeros((2, 7)),
        "arm_joint_target": torch.full((2, 7), 0.5),
        "finger_joint_target": torch.zeros((2, 2)),
    }

    settled_target, valid = owner._replay_oracle_entry(
        state,
        starts_grasped=True,
        active_mask=torch.tensor((True, False)),
    )

    assert restored == [state]
    assert observed["env"] is owner.env
    assert observed["duration_s"] == generator.PICK_INSERT_RESET_REPLAY_DURATION_S
    assert observed["starts_grasped"] is True
    assert torch.equal(observed["arm_target"], state["arm_joint_target"])
    assert torch.equal(observed["finger_target"], state["finger_joint_target"])
    assert torch.equal(settled_target, state["arm_joint_target"] + 0.25)
    assert valid.tolist() == [True, False]
    assert asserted_contexts == ["generator oracle entry replay"]

    reset_contact_count = 1
    _, dirty_contact_valid = owner._replay_oracle_entry(
        state,
        starts_grasped=True,
        active_mask=torch.ones(2, dtype=torch.bool),
    )

    assert dirty_contact_valid.tolist() == [False, False]
    assert asserted_contexts == ["generator oracle entry replay"] * 2


def test_ungrasped_generator_oracle_uses_post_replay_target_and_survival_mask(monkeypatch):
    class FakeLaneHold:
        def __init__(self, _env, active_mask, arm_target, finger_target):
            self.active_mask = active_mask.clone()
            self.last_sent_arm_target = arm_target.clone()
            self.last_sent_finger_target = finger_target.clone()
            self.reason_masks: dict[str, torch.Tensor] = {}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def deactivate(self, mask, *, reason):
            failed = self.active_mask & mask
            self.active_mask &= ~failed
            self.reason_masks[reason] = failed.clone()

    owner = object.__new__(generator.PickInsertResetDatasetGenerator)
    owner.env = SimpleNamespace(num_envs=2)
    owner.device = torch.device("cpu")
    owner.cfg = generator.GeneratorCfg(batch_size=2)
    owner.local_grasp_orientation = torch.tensor((0.0, 0.0, 0.0, 1.0))
    owner.closed_finger_target = torch.zeros((2, 2))
    owner.plug_index = 0
    owner.latch_index = 1
    owner._counted_ik = object()
    owner._drive_enabled = lambda: torch.zeros(2, dtype=torch.bool)
    owner._orientation_hold_enabled = lambda: torch.zeros(2, dtype=torch.bool)
    observed: dict[str, object] = {}
    post_replay_target = torch.full((2, 7), 0.25)
    post_replay_mask = torch.tensor((True, False))

    def replay_entry(observed_state, *, starts_grasped, active_mask):
        observed["replay_state"] = observed_state
        observed["replay_starts_grasped"] = starts_grasped
        observed["replay_active_mask"] = active_mask.clone()
        return post_replay_target.clone(), post_replay_mask.clone()

    def acquire_grasp(arm_seed, *, active_mask):
        observed["acquisition_arm_seed"] = arm_seed.clone()
        observed["acquisition_active_mask"] = active_mask.clone()
        return arm_seed + 0.5, active_mask.clone(), owner.closed_finger_target

    owner._replay_oracle_entry = replay_entry
    owner._acquire_grasp = acquire_grasp

    def fake_recovery(_env, _ik, _goal_q, _orientation, _finger_target, **kwargs):
        observed.update(kwargs)
        return torch.tensor((True, False)), {
            **_fake_recovery_diagnostic_metrics(2, phase=4),
            "exact_success_dwell_satisfied": torch.ones(2, dtype=torch.bool),
            "exact_success_all_samples_collision_free": torch.ones(2, dtype=torch.bool),
            "exact_success_all_samples_bilateral_grasp": torch.ones(2, dtype=torch.bool),
            "exact_success_all_samples_finite": torch.ones(2, dtype=torch.bool),
            "exact_success_maximum_body_excursion": torch.zeros(2),
            "exact_success_maximum_cable_linear_speed": torch.zeros(2),
            "goal_error": torch.tensor((0.0, 1.0)),
        }

    monkeypatch.setattr(generator, "_PerLaneTargetHold", FakeLaneHold)
    monkeypatch.setattr(generator, "scripted_recovery", fake_recovery)
    monkeypatch.setattr(
        generator,
        "collision_metrics",
        lambda _env: SimpleNamespace(contact_overflow=False, valid=torch.ones(2, dtype=torch.bool)),
    )
    state = {"arm_joint_target": torch.zeros((2, 7))}

    valid, evidence = owner._oracle(
        state,
        torch.zeros((2, 3, 7)),
        torch.zeros((2, 7)),
        phase=4,
        starts_grasped=False,
    )

    assert valid.tolist() == [True, False]
    assert observed["replay_state"] is state
    assert observed["replay_starts_grasped"] is False
    assert torch.equal(observed["replay_active_mask"], torch.ones(2, dtype=torch.bool))
    assert torch.equal(observed["acquisition_arm_seed"], post_replay_target)
    assert torch.equal(observed["acquisition_active_mask"], post_replay_mask)
    assert observed["motion_policy"] == PICK_INSERT_RECOVERY_CARTESIAN_C2_POLICY["motion_policy"]
    assert observed["pick_insert_phase"] == 4
    assert tuple(evidence["recovery_diagnostic_evidence"]) == _TEST_RECOVERY_DIAGNOSTIC_EVIDENCE_NAMES
    assert evidence["recovery_diagnostic_evidence"]["pick_insert_phase"].tolist() == [4, 4]
    assert evidence["recovery_diagnostic_evidence"][
        "cartesian_motion_first_plug_linear_speed_failure_step"
    ].tolist() == [10, 10]
    assert evidence["recovery_lane_failure_masks"]["generator-oracle-recovery-final-validation"].tolist() == [
        False,
        False,
    ]


def test_canonical_goal_certificate_round_trip_is_cpu_safe_exact_and_path_free(tmp_path):
    certificate, contracts = _canonical_goal_certificate()
    output = tmp_path / "goal-certificate.pt"
    generator.save_torch_atomic(certificate, output)

    loaded = generator._load_canonical_goal_certificate(output, **contracts)
    embedded = generator._canonical_goal_certificate_embedding(loaded)

    assert loaded["content_sha256"] == generator.reset_dataset_content_digest(loaded)
    assert tuple(loaded["goal_state"]) == generator.RESET_DATASET_GOAL_STATE_NAMES
    assert len(loaded["goal_state"]) == 10
    assert all(value.device.type == "cpu" for value in loaded["goal_state"].values())
    assert embedded["content_sha256"] == loaded["content_sha256"]
    assert embedded["metadata"]["production_evidence"] == _certificate_production_evidence()
    assert set(embedded["goal_state"]) == set(generator.RESET_DATASET_GOAL_STATE_NAMES)
    assert str(output) not in repr(embedded)


def test_canonical_goal_certificate_load_is_weights_only_and_cpu_mapped(monkeypatch, tmp_path):
    certificate, contracts = _canonical_goal_certificate()
    observed: dict[str, object] = {}

    def fake_load(path, *, map_location, weights_only):
        observed.update(path=path, map_location=map_location, weights_only=weights_only)
        return certificate

    monkeypatch.setattr(generator.torch, "load", fake_load)
    requested = tmp_path / "goal-certificate.pt"

    generator._load_canonical_goal_certificate(requested, **contracts)

    assert observed == {
        "path": requested.resolve(),
        "map_location": "cpu",
        "weights_only": True,
    }


def test_canonical_goal_certificate_rejects_tampering_and_extra_goal_fields():
    certificate, contracts = _canonical_goal_certificate()
    tampered = deepcopy(certificate)
    tampered["goal_state"]["arm_joint_position"][0] = 1.0

    with pytest.raises(ValueError, match="content digest"):
        generator._validate_canonical_goal_certificate(tampered, **contracts)

    extra = deepcopy(certificate)
    extra["goal_state"]["extra"] = torch.zeros(1)
    extra["content_sha256"] = generator.reset_dataset_content_digest(extra)
    with pytest.raises(ValueError, match="exactly the ten runtime fields"):
        generator._validate_canonical_goal_certificate(extra, **contracts)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda certificate: certificate["metadata"]["production_evidence"].update(diagnostic_cli=True), "diagnostic"),
        (
            lambda certificate: certificate["metadata"]["production_evidence"]["cold_proofs"].update(
                endpoint_promotion_count=1
            ),
            "cannot promote",
        ),
        (
            lambda certificate: certificate["metadata"]["production_evidence"]["cold_proofs"]["cold_60s"].update(
                passed=False
            ),
            "canonical-cold-60s",
        ),
    ),
)
def test_canonical_goal_certificate_rejects_nonproduction_evidence(mutation, message):
    certificate, contracts = _canonical_goal_certificate()
    mutation(certificate)
    certificate["content_sha256"] = generator.reset_dataset_content_digest(certificate)

    with pytest.raises(ValueError, match=message):
        generator._validate_canonical_goal_certificate(certificate, **contracts)


def test_canonical_goal_certificate_rejects_contract_and_rng_state_changes():
    certificate, contracts = _canonical_goal_certificate()
    changed_contracts = deepcopy(contracts)
    changed_contracts["expected_task_contract"]["contract_version"] = 7

    with pytest.raises(ValueError, match="task contract"):
        generator._validate_canonical_goal_certificate(certificate, **changed_contracts)

    legacy_sequence = deepcopy(certificate)
    legacy_sequence["metadata"]["generation_contract"]["pickup_construction_sequence"][
        "construction_sequence_version"
    ] = 1
    legacy_sequence["content_sha256"] = generator.reset_dataset_content_digest(legacy_sequence)
    with pytest.raises(ValueError, match="generation contract"):
        generator._validate_canonical_goal_certificate(legacy_sequence, **contracts)

    changed_rng = deepcopy(certificate)
    changed_rng["row_rng_state"][0] ^= 1
    changed_rng["content_sha256"] = generator.reset_dataset_content_digest(changed_rng)
    with pytest.raises(ValueError, match="RNG contract"):
        generator._validate_canonical_goal_certificate(changed_rng, **contracts)


def test_canonical_goal_certificate_source_digest_rejects_a_later_source_edit(monkeypatch, tmp_path):
    source = tmp_path / "generator.py"
    source.write_bytes(b"before")
    monkeypatch.setattr(generator, "_REPO_ROOT", tmp_path)
    monkeypatch.setattr(generator, "_CANONICAL_GOAL_SOURCE_ROOTS", ())
    monkeypatch.setattr(generator, "_CANONICAL_GOAL_SOURCE_FILES", ("generator.py",))
    before = generator._canonical_goal_source_digests()

    certificate, contracts = _canonical_goal_certificate()
    certificate["metadata"]["source_sha256"] = before
    certificate["content_sha256"] = generator.reset_dataset_content_digest(certificate)
    source.write_bytes(b"after")
    after = generator._canonical_goal_source_digests()
    contracts["expected_source_sha256"] = after

    assert before != after
    with pytest.raises(ValueError, match="source-digest contract"):
        generator._validate_canonical_goal_certificate(certificate, **contracts)


def test_canonical_goal_source_manifest_recurses_all_python_roots_in_sorted_order(monkeypatch, tmp_path):
    roots = (
        "source/isaaclab/isaaclab",
        "source/isaaclab_newton/isaaclab_newton",
        "source/isaaclab_contrib/isaaclab_contrib",
        "source/isaaclab_assets/isaaclab_assets",
        "source/isaaclab_tasks/isaaclab_tasks",
    )
    expected_contents: dict[str, bytes] = {}
    for index, relative_root in enumerate(roots):
        relative_name = f"{relative_root}/package_{index}.py"
        source = tmp_path / relative_name
        source.parent.mkdir(parents=True, exist_ok=True)
        contents = f"ROOT = {index}\n".encode()
        source.write_bytes(contents)
        expected_contents[relative_name] = contents
    nested_stub = tmp_path / roots[-1] / "nested" / "contract.pyi"
    nested_stub.parent.mkdir(parents=True)
    nested_stub.write_bytes(b"VALUE: int\n")
    expected_contents[nested_stub.relative_to(tmp_path).as_posix()] = b"VALUE: int\n"
    (nested_stub.parent / "ignored.pyc").write_bytes(b"ignored")
    (nested_stub.parent / "ignored.txt").write_bytes(b"ignored")
    for relative_name, contents in {
        "scripts/tools/generate_franka_rj45_pick_insert_reset_dataset.py": b"generator",
        "scripts/tools/_franka_rj45_reset_tools.py": b"helper",
        "uv.lock": b"lock",
    }.items():
        source = tmp_path / relative_name
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(contents)
        expected_contents[relative_name] = contents
    monkeypatch.setattr(generator, "_REPO_ROOT", tmp_path)
    monkeypatch.setattr(generator, "_CANONICAL_GOAL_SOURCE_ROOTS", roots)
    monkeypatch.setattr(
        generator,
        "_CANONICAL_GOAL_SOURCE_FILES",
        (
            "scripts/tools/generate_franka_rj45_pick_insert_reset_dataset.py",
            "scripts/tools/_franka_rj45_reset_tools.py",
            "uv.lock",
        ),
    )

    manifest = generator._canonical_goal_source_digests()

    assert list(manifest) == sorted(expected_contents)
    assert manifest == {
        relative_name: generator.hashlib.sha256(expected_contents[relative_name]).hexdigest()
        for relative_name in sorted(expected_contents)
    }


def test_canonical_goal_certifier_rejects_any_source_edit_during_proof(monkeypatch, tmp_path):
    source_root = tmp_path / "source/package"
    source_root.mkdir(parents=True)
    member = source_root / "nested.pyi"
    member.write_bytes(b"VALUE: int\n")
    generator_source = tmp_path / "generator.py"
    generator_source.write_bytes(b"generator")
    monkeypatch.setattr(generator, "_REPO_ROOT", tmp_path)
    monkeypatch.setattr(generator, "_CANONICAL_GOAL_SOURCE_ROOTS", ("source/package",))
    monkeypatch.setattr(generator, "_CANONICAL_GOAL_SOURCE_FILES", ("generator.py",))
    contracts = _certificate_contracts()
    owner = object.__new__(generator.PickInsertResetDatasetGenerator)
    owner.env = SimpleNamespace(num_envs=4)
    owner._ik_solve_call_count = 0
    owner.random = torch.Generator().manual_seed(2026)

    def validation_snapshot():
        snapshot = deepcopy(contracts)
        snapshot["expected_source_sha256"] = generator._canonical_goal_source_digests()
        return snapshot

    def derive_goal():
        member.write_bytes(b"VALUE: str\n")
        return _certificate_goal_state(), _certificate_production_evidence()

    owner._canonical_goal_certificate_validation_kwargs = validation_snapshot
    owner.derive_goal = derive_goal

    with pytest.raises(RuntimeError, match="snapshot changed during canonical-goal certification"):
        owner.derive_goal_certificate()


def test_certificate_input_generation_rejects_any_source_edit_before_atomic_save(monkeypatch, tmp_path):
    source_root = tmp_path / "source/package"
    source_root.mkdir(parents=True)
    member = source_root / "nested.py"
    member.write_bytes(b"VALUE = 1\n")
    generator_source = tmp_path / "generator.py"
    generator_source.write_bytes(b"generator")
    monkeypatch.setattr(generator, "_REPO_ROOT", tmp_path)
    monkeypatch.setattr(generator, "_CANONICAL_GOAL_SOURCE_ROOTS", ("source/package",))
    monkeypatch.setattr(generator, "_CANONICAL_GOAL_SOURCE_FILES", ("generator.py",))
    contracts = _certificate_contracts()
    before = generator._canonical_goal_source_digests()
    contracts["expected_source_sha256"] = before
    certificate = generator._build_canonical_goal_certificate(
        goal_state=_certificate_goal_state(),
        production_evidence=_certificate_production_evidence(),
        row_rng_state=torch.Generator().manual_seed(2026).get_state(),
        certifier_env_count=4,
        task_body_count=contracts["task_body_count"],
        task_contract=contracts["expected_task_contract"],
        physical_contract=contracts["expected_physical_contract"],
        generation_contract=contracts["expected_generation_contract"],
        versions=contracts["expected_versions"],
        source_sha256=before,
    )
    certificate_input = tmp_path / "certificate.pt"
    generator.save_torch_atomic(certificate, certificate_input)
    owner = SimpleNamespace()

    def validation_snapshot():
        snapshot = deepcopy(contracts)
        snapshot["expected_source_sha256"] = generator._canonical_goal_source_digests()
        return snapshot

    def generate(_certificate):
        member.write_bytes(b"VALUE = 2\n")
        return {"payload": torch.zeros(1)}

    owner._canonical_goal_certificate_validation_kwargs = validation_snapshot
    owner.generate = generate
    save_calls: list[Path] = []
    monkeypatch.setattr(generator, "save_torch_atomic", lambda _payload, output: save_calls.append(output))
    output = tmp_path / "reset-dataset.pt"

    with pytest.raises(RuntimeError, match="snapshot changed during certificate-input reset-row generation"):
        generator._generate_and_save_reset_dataset_artifact(
            owner,
            output=output,
            certificate_input=certificate_input,
            pre_environment_source_sha256=before,
        )

    assert save_calls == []
    assert not output.exists()


def test_certificate_generation_restores_row_rng_and_embeds_state_not_path(monkeypatch):
    certificate, contracts = _canonical_goal_certificate()
    owner = object.__new__(generator.PickInsertResetDatasetGenerator)
    owner.cfg = generator.GeneratorCfg()
    owner.env = SimpleNamespace(num_envs=24, cfg=object())
    owner.device = torch.device("cpu")
    owner.random = torch.Generator().manual_seed(999)
    owner._ik_solve_call_count = 0
    owner.layout = SimpleNamespace(body_count=3)
    owner.socket_index = 0
    owner.reference_socket_body_pose = torch.zeros(7)
    owner.attempt_counts = [0] * 6
    owner.rejection_counts = {phase: {} for phase in range(6)}
    owner.accepted_oracle_metrics = {phase: [] for phase in range(6)}
    owner._canonical_goal_certificate_validation_kwargs = lambda: contracts
    observed_draws: list[float] = []

    def fake_phase(phase, _canonical_goal):
        observed_draws.append(float(torch.rand((), generator=owner.random)))
        result = {name: torch.zeros(1) for name in generator.RESET_DATASET_STATE_NAMES}
        result["phase"] = torch.tensor((phase,), dtype=torch.int64)
        result["goal_task_body_pose"] = torch.zeros((1, 3, 7))
        return result

    owner._generate_phase = fake_phase
    monkeypatch.setattr(
        generator,
        "pick_insert_reset_dataset_task_contract",
        lambda _cfg: contracts["expected_task_contract"],
    )
    monkeypatch.setattr(generator, "reset_dataset_validate_runtime", lambda *_args, **_kwargs: None)
    expected_rng = torch.Generator()
    expected_rng.set_state(certificate["row_rng_state"])
    expected_draws = [float(torch.rand((), generator=expected_rng)) for _ in range(6)]

    payload = owner.generate(certificate)

    assert observed_draws == expected_draws
    assert payload["metadata"]["canonical_goal_certificate"]["content_sha256"] == certificate["content_sha256"]
    assert set(payload["metadata"]["canonical_goal_certificate"]["goal_state"]) == set(
        generator.RESET_DATASET_GOAL_STATE_NAMES
    )
    assert all(
        torch.equal(payload["goal_state"][name], certificate["goal_state"][name])
        for name in generator.RESET_DATASET_GOAL_STATE_NAMES
    )
    assert {
        name: payload["metadata"]["initial_state_policy"][name]
        for name in generator._pickup_construction_sequence_contract()
    } == generator._pickup_construction_sequence_contract()
    assert payload["metadata"]["initial_state_policy"]["scripted_recovery"] == (
        PICK_INSERT_RECOVERY_CARTESIAN_C2_POLICY
    )
    assert payload["metadata"]["initial_state_policy"]["phase_4_pregrasp_orientation_sampling"] == (
        generator._phase_4_pregrasp_orientation_sampling_contract(owner.cfg)
    )
    assert "certificate_input" not in repr(payload["metadata"])


def test_certificate_generation_rejects_an_advanced_row_ik_stream():
    certificate, contracts = _canonical_goal_certificate()
    owner = object.__new__(generator.PickInsertResetDatasetGenerator)
    owner.cfg = generator.GeneratorCfg()
    owner.env = SimpleNamespace(num_envs=24, cfg=object())
    owner._ik_solve_call_count = 1
    owner._canonical_goal_certificate_validation_kwargs = lambda: contracts

    with pytest.raises(RuntimeError, match="freshly constructed, unadvanced row IK stream"):
        owner.generate(certificate)


@pytest.mark.parametrize(
    "overrides",
    (
        {"quick": True},
        {"rows_per_phase": 95},
        {"batch_size": 23},
    ),
)
def test_canonical_output_policy_rejects_quick_or_custom_shapes(overrides):
    args = _parsed_generator_args(generator.DEFAULT_DATASET_PATH, **overrides)

    with pytest.raises(ValueError, match="explicit noncanonical --output"):
        generator._validate_parsed_artifact_policy(args)


@pytest.mark.parametrize(
    "overrides",
    (
        {"quick": True},
        {"rows_per_phase": 95},
        {"batch_size": 23},
    ),
)
def test_noncanonical_output_policy_allows_quick_or_custom_shapes(tmp_path, overrides):
    args = _parsed_generator_args(tmp_path / "scratch-reset-dataset.pt", **overrides)

    assert generator._validate_parsed_artifact_policy(args) is False


@pytest.mark.parametrize(
    "diagnostic_flag",
    (
        "diagnostic_goal_only",
        "diagnostic_reset_abcd",
        "diagnostic_reset_e",
        "diagnostic_p_relax_reseat",
    ),
)
def test_diagnostic_output_policy_is_no_save_regardless_of_output_shape(diagnostic_flag):
    args = _parsed_generator_args(
        generator.DEFAULT_DATASET_PATH,
        rows_per_phase=1,
        batch_size=1,
        quick=True,
    )
    setattr(args, diagnostic_flag, True)

    assert generator._validate_parsed_artifact_policy(args) is True


def test_generator_compatibility_diagnostics_cannot_change_the_production_close_or_proxy_contract():
    cfg = generator.GeneratorCfg(
        diagnostic_p_relax_reseat=True,
        diagnostic_zero_finger_close_target=True,
        diagnostic_effective_grasp_friction_three=True,
    )

    assert cfg.finger_closed_target == PICK_INSERT_CLOSED_FINGER_POSITION
    with pytest.raises(ValueError, match="immutable production pick-insert close target"):
        generator.GeneratorCfg(
            diagnostic_p_relax_reseat=True,
            diagnostic_zero_finger_close_target=True,
            finger_closed_target=0.001,
        )


def test_validation_cfg_rejects_changed_production_close_target():
    with pytest.raises(ValueError, match="immutable production closed-finger target"):
        ValidationCfg(finger_closed_target=0.001)


def _canonical_phase_rows(rows_per_phase: int = 3_334) -> torch.Tensor:
    return torch.repeat_interleave(
        torch.arange(6, dtype=torch.int64),
        rows_per_phase,
    )


def _mutated_phase_rows(phase: int, delta: int) -> torch.Tensor:
    phases = _canonical_phase_rows()
    if delta < 0:
        removed = int(torch.where(phases == phase)[0][0])
        return torch.cat((phases[:removed], phases[removed + 1 :]))
    return torch.cat((phases, torch.tensor((phase,), dtype=torch.int64)))


def test_full_validation_phase_count_gate_accepts_only_the_canonical_bank():
    cfg = ValidationCfg()
    env_cfg = FrankaRJ45PickInsertEnvCfg()

    counts = validator._validate_invocation_phase_counts(
        _canonical_phase_rows(),
        cfg,
        expected_rows_per_phase=env_cfg.reset_dataset_rows_per_phase,
    )

    assert counts == (3_334,) * 6


@pytest.mark.parametrize("phase", range(6))
@pytest.mark.parametrize("delta", (-1, 1), ids=("deficit", "surplus"))
def test_full_validation_and_stable_publish_reject_each_phase_count_mutation(
    tmp_path,
    phase: int,
    delta: int,
):
    phases = _mutated_phase_rows(phase, delta)
    cfg = ValidationCfg()
    env_cfg = FrankaRJ45PickInsertEnvCfg()

    with pytest.raises(ValueError, match="phase counts do not match the configured production size"):
        validator._validate_invocation_phase_counts(
            phases,
            cfg,
            expected_rows_per_phase=env_cfg.reset_dataset_rows_per_phase,
        )

    payload = {"content_sha256": "0" * 64, "states": {"phase": phases}}
    stable_output = tmp_path / "reset_validation.json"
    with pytest.raises(ValueError, match="phase counts do not match the configured production size"):
        validator.write_stable_validation_report(
            {},
            payload,
            env_cfg,
            stable_output,
            expected_validation_policy=validator.FRANKA_RJ45_PICK_INSERT_RESET_VALIDATION_POLICY,
            expected_source_sha256={},
            expected_asset_closure={},
        )
    assert not stable_output.exists()


def test_quick_validation_preserves_noncanonical_diagnostic_phase_counts():
    env_cfg = FrankaRJ45PickInsertEnvCfg()

    counts = validator._validate_invocation_phase_counts(
        torch.arange(6, dtype=torch.int64),
        ValidationCfg(quick=True),
        expected_rows_per_phase=env_cfg.reset_dataset_rows_per_phase,
    )

    assert counts is None


def test_generator_cfg_rejects_changed_goal_body_drift_limit():
    with pytest.raises(ValueError, match="immutable canonical-goal replay limit"):
        generator.GeneratorCfg(maximum_goal_body_drift_m=PICK_INSERT_GOAL_MAX_TASK_BODY_DRIFT_M * 2.0)


@pytest.mark.parametrize(
    ("field", "invalid"),
    (
        ("maximum_goal_socket_drift_m", PICK_INSERT_GOAL_MAX_SOCKET_DRIFT_M * 2.0),
        ("maximum_goal_body_drift_m", PICK_INSERT_GOAL_MAX_TASK_BODY_DRIFT_M * 2.0),
        ("maximum_goal_cable_speed_m_s", PICK_INSERT_GOAL_MAX_CABLE_SPEED_M_S * 2.0),
        ("maximum_goal_arm_joint_speed_rad_s", PICK_INSERT_GOAL_MAX_ARM_JOINT_SPEED_RAD_S * 2.0),
        ("maximum_goal_finger_joint_speed_m_s", PICK_INSERT_GOAL_MAX_FINGER_JOINT_SPEED_M_S * 2.0),
        ("maximum_goal_authored_seat_error_m", PICK_INSERT_GOAL_MAX_AUTHORED_SEAT_ERROR_M * 2.0),
        ("maximum_goal_authored_plug_angle_rad", PICK_INSERT_GOAL_MAX_AUTHORED_PLUG_ANGLE_RAD * 2.0),
        (
            "maximum_goal_plug_relative_latch_angle_rad",
            PICK_INSERT_GOAL_MAX_PLUG_RELATIVE_LATCH_ANGLE_RAD * 2.0,
        ),
    ),
)
def test_validation_cfg_rejects_changed_goal_threshold(field: str, invalid: float):
    with pytest.raises(ValueError, match="Canonical goal replay thresholds are immutable"):
        ValidationCfg(**{field: invalid})


def test_fixed_bias_waypoint_moves_valid_lanes_when_a_peer_ik_lane_fails(monkeypatch):
    class FakeEnv:
        num_envs = 2
        device = "cpu"

        def __init__(self):
            self.targets: list[tuple[torch.Tensor, torch.Tensor]] = []
            self.advances: list[float] = []
            self.arm_q = torch.zeros((2, 7))

        def set_robot_targets(self, arm_target, finger_target):
            self.targets.append((arm_target.clone(), finger_target.clone()))

        def advance(self, duration_s, update=None, *, post_step=None):
            self.advances.append(duration_s)
            return 1

        def read_robot_state(self):
            return self.arm_q, torch.zeros_like(self.arm_q), torch.zeros((2, 2)), torch.zeros((2, 2))

        def tcp_pose_e(self):
            pose = torch.zeros((2, 7))
            pose[:, 6] = 1.0
            return pose

    class FakeIK:
        def solve(self, *_args, **_kwargs):
            return SimpleNamespace(
                arm_q=torch.tensor(
                    (
                        (4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0),
                        (0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1),
                    )
                ),
                valid=torch.tensor((False, True)),
            )

    env = FakeEnv()
    interpolations: list[tuple[torch.Tensor, torch.Tensor]] = []
    monkeypatch.setattr(validator, "joint_limit_mask", lambda *_args, **_kwargs: torch.ones(2, dtype=torch.bool))
    monkeypatch.setattr(
        validator,
        "interpolate_arm_motion",
        lambda _env, start, end, _finger, _duration: interpolations.append((start.clone(), end.clone())),
    )
    raw_seed = torch.zeros((2, 7))
    finger_target = torch.full((2, 2), 0.04)

    with validator._PerLaneTargetHold(env, torch.ones(2, dtype=torch.bool), raw_seed, finger_target) as lane_hold:
        safe_raw, safe_target, valid = validator._move_tcp_with_fixed_bias(
            env,
            FakeIK(),
            torch.zeros((2, 3)),
            torch.tensor(((0.0, 0.0, 0.0, 1.0),) * 2),
            finger_target,
            raw_arm_seed=raw_seed,
            arm_target_bias=torch.zeros_like(raw_seed),
            cfg=ValidationCfg(),
            maximum_raw_joint_step_rad=0.5,
            lane_hold=lane_hold,
            failure_reason="test-fixed-bias",
        )

    assert valid.tolist() == [False, True]
    assert torch.equal(safe_raw[0], raw_seed[0])
    assert torch.allclose(safe_raw[1], torch.full((7,), 0.1))
    assert torch.equal(safe_target, safe_raw)
    assert len(interpolations) == 1
    assert torch.equal(interpolations[0][1], safe_target)
    assert torch.equal(env.targets[-1][0], safe_target)
    assert len(env.advances) == 1


def test_generator_fixed_bias_waypoint_moves_valid_lanes_when_a_peer_ik_lane_fails(monkeypatch):
    class FakeEnv:
        num_envs = 2
        device = "cpu"
        _arm_joint_ids = torch.arange(7)

        def __init__(self):
            limits = torch.tensor([[[-10.0, 10.0]] * 7])
            self._robot = SimpleNamespace(data=SimpleNamespace(soft_joint_pos_limits=SimpleNamespace(torch=limits)))
            self.targets: list[tuple[torch.Tensor, torch.Tensor]] = []
            self.arm_q = torch.zeros((2, 7))

        def set_robot_targets(self, arm_target, finger_target):
            self.targets.append((arm_target.clone(), finger_target.clone()))

        def advance(self, _duration_s, update=None, *, post_step=None):
            return 1

        def read_robot_state(self):
            return self.arm_q, torch.zeros_like(self.arm_q), torch.zeros((2, 2)), torch.zeros((2, 2))

        def tcp_pose_e(self):
            pose = torch.zeros((2, 7))
            pose[:, 6] = 1.0
            return pose

    class FakeIK:
        def solve(self, *_args, **_kwargs):
            return SimpleNamespace(
                arm_q=torch.tensor(
                    (
                        (4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0),
                        (0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1),
                    )
                ),
                valid=torch.tensor((False, True)),
                position_residual=torch.zeros(2),
                rotation_residual=torch.zeros(2),
            )

    env = FakeEnv()
    owner = object.__new__(generator.PickInsertResetDatasetGenerator)
    owner.env = env
    owner.ik = FakeIK()
    owner.cfg = generator.GeneratorCfg()
    owner.device = torch.device("cpu")
    interpolations: list[tuple[torch.Tensor, torch.Tensor]] = []
    monkeypatch.setattr(generator, "joint_limit_mask", lambda *_args, **_kwargs: torch.ones(2, dtype=torch.bool))
    monkeypatch.setattr(
        generator,
        "interpolate_arm_motion",
        lambda _env, start, end, _finger, _duration: interpolations.append((start.clone(), end.clone())),
    )
    raw_seed = torch.zeros((2, 7))
    finger_target = torch.full((2, 2), 0.04)
    diagnostics = []

    with generator._PerLaneTargetHold(env, torch.ones(2, dtype=torch.bool), raw_seed, finger_target) as lane_hold:
        safe_raw, safe_target, valid = owner._move_tcp_with_fixed_arm_bias(
            torch.zeros((2, 3)),
            torch.tensor(((0.0, 0.0, 0.0, 1.0),) * 2),
            finger_target,
            raw_arm_seed=raw_seed,
            arm_target_bias=torch.zeros_like(raw_seed),
            duration_s=0.5,
            diagnostic_label="mixed-validity",
            diagnostics=diagnostics,
            compensation_iterations=0,
            require_tracking_tolerance=False,
            settle_s=0.0,
            maximum_raw_ik_joint_step_rad=0.5,
            lane_hold=lane_hold,
            failure_reason="test-fixed-bias",
        )

    assert valid.tolist() == [False, True]
    assert torch.equal(safe_raw[0], raw_seed[0])
    assert torch.allclose(safe_raw[1], torch.full((7,), 0.1))
    assert torch.equal(safe_target, safe_raw)
    assert len(interpolations) == 1
    assert torch.equal(env.targets[-1][0], safe_target)
    assert diagnostics[0]["motion_skipped_before_physics"] is False


def test_generator_successful_waypoints_do_not_serialize_cpu_diagnostics(monkeypatch):
    class AlwaysValidIK:
        def solve(self, position, *_args, **_kwargs):
            return SimpleNamespace(
                arm_q=torch.zeros((len(position), 7)),
                valid=torch.ones(len(position), dtype=torch.bool),
                tcp_position=position.clone(),
                position_residual=torch.zeros(len(position)),
                rotation_residual=torch.zeros(len(position)),
            )

    env = _LaneMotionFakeEnv()
    owner = object.__new__(generator.PickInsertResetDatasetGenerator)
    owner.env = env
    owner.ik = AlwaysValidIK()
    owner.cfg = generator.GeneratorCfg()
    owner.device = torch.device("cpu")
    monkeypatch.setattr(generator, "joint_limit_mask", lambda *_args, **_kwargs: torch.ones(2, dtype=torch.bool))
    monkeypatch.setattr(generator, "interpolate_arm_motion", lambda *_args, **_kwargs: None)
    arm_seed = torch.zeros((2, 7))
    finger_target = torch.full((2, 2), 0.04)
    orientation = torch.tensor(((0.0, 0.0, 0.0, 1.0),) * 2)
    iterative_diagnostics: list[dict[str, object]] = []

    with generator._PerLaneTargetHold(env, torch.ones(2, dtype=torch.bool), arm_seed, finger_target) as lane_hold:
        _, iterative_valid = owner._move_tcp(
            torch.zeros((2, 3)),
            orientation,
            finger_target,
            arm_seed=arm_seed,
            diagnostic_label="successful-iterative-waypoint",
            diagnostics=iterative_diagnostics,
            lane_hold=lane_hold,
            failure_reason="test-successful-iterative-waypoint",
        )

    fixed_bias_diagnostics: list[dict[str, object]] = []
    with generator._PerLaneTargetHold(env, torch.ones(2, dtype=torch.bool), arm_seed, finger_target) as lane_hold:
        _, _, fixed_bias_valid = owner._move_tcp_with_fixed_arm_bias(
            torch.zeros((2, 3)),
            orientation,
            finger_target,
            raw_arm_seed=arm_seed,
            arm_target_bias=torch.zeros_like(arm_seed),
            duration_s=0.5,
            diagnostic_label="successful-fixed-bias-waypoint",
            diagnostics=fixed_bias_diagnostics,
            compensation_iterations=0,
            settle_s=0.0,
            lane_hold=lane_hold,
            failure_reason="test-successful-fixed-bias-waypoint",
        )

    assert iterative_valid.all()
    assert fixed_bias_valid.all()
    assert iterative_diagnostics == []
    assert fixed_bias_diagnostics == []


def test_generator_iterative_tcp_freezes_a_failed_lane_before_later_valid_ik(monkeypatch):
    target = torch.full((2, 3), 0.01)
    env = _LaneMotionFakeEnv(tcp_target_after_call=4, tcp_target=target)
    ik = _InvalidThenValidIK()
    owner = object.__new__(generator.PickInsertResetDatasetGenerator)
    owner.env = env
    owner.ik = ik
    owner.cfg = generator.GeneratorCfg(tcp_compensation_iterations=1)
    owner.device = torch.device("cpu")
    monkeypatch.setattr(generator, "joint_limit_mask", lambda *_args, **_kwargs: torch.ones(2, dtype=torch.bool))
    monkeypatch.setattr(generator, "interpolate_arm_motion", lambda *_args, **_kwargs: None)
    arm_seed = torch.zeros((2, 7))
    finger_target = torch.full((2, 2), 0.04)

    with generator._PerLaneTargetHold(env, torch.ones(2, dtype=torch.bool), arm_seed, finger_target) as lane_hold:
        arm_target, valid = owner._move_tcp(
            target,
            torch.tensor(((0.0, 0.0, 0.0, 1.0),) * 2),
            finger_target,
            arm_seed=arm_seed,
            lane_hold=lane_hold,
            failure_reason="test-iterative-ik",
        )
        reason_masks = lane_hold.reason_masks

    assert ik.calls == 2
    assert valid.tolist() == [False, True]
    assert torch.equal(reason_masks["test-iterative-ik"], torch.tensor((True, False)))
    assert torch.equal(arm_target[0], torch.zeros(7))
    assert torch.isfinite(ik.positions[1]).all()
    assert torch.equal(ik.positions[1][0], torch.zeros(3))
    assert all(torch.equal(command[0][0], torch.zeros(7)) for command in env.physics_commands)
    assert all(torch.equal(target_write[0][0], torch.zeros(7)) for target_write in env.target_writes)


def test_generator_iterative_tcp_honors_oracle_attempt_and_settle_overrides(monkeypatch):
    class AlwaysValidIK:
        def __init__(self):
            self.calls = 0

        def solve(self, position, *_args, **_kwargs):
            self.calls += 1
            return SimpleNamespace(
                arm_q=torch.zeros((len(position), 7)),
                valid=torch.ones(len(position), dtype=torch.bool),
                tcp_position=position.clone(),
                position_residual=torch.zeros(len(position)),
                rotation_residual=torch.zeros(len(position)),
            )

    env = _LaneMotionFakeEnv()
    advance_durations: list[float] = []
    original_advance = env.advance

    def record_advance(duration_s, update=None, *, post_step=None):
        advance_durations.append(duration_s)
        return original_advance(duration_s, update, post_step=post_step)

    env.advance = record_advance
    owner = object.__new__(generator.PickInsertResetDatasetGenerator)
    owner.env = env
    owner.ik = AlwaysValidIK()
    owner.cfg = generator.GeneratorCfg(tcp_compensation_iterations=9)
    owner.device = torch.device("cpu")
    monkeypatch.setattr(generator, "joint_limit_mask", lambda *_args, **_kwargs: torch.ones(2, dtype=torch.bool))
    monkeypatch.setattr(generator, "interpolate_arm_motion", lambda *_args, **_kwargs: None)
    arm_seed = torch.zeros((2, 7))
    finger_target = torch.full((2, 2), 0.04)

    with generator._PerLaneTargetHold(env, torch.ones(2, dtype=torch.bool), arm_seed, finger_target) as lane_hold:
        _, valid = owner._move_tcp(
            torch.full((2, 3), 0.01),
            torch.tensor(((0.0, 0.0, 0.0, 1.0),) * 2),
            finger_target,
            arm_seed=arm_seed,
            attempt_count=5,
            settle_s=0.30,
            lane_hold=lane_hold,
            failure_reason="test-oracle-iterative-controls",
        )

    assert owner.ik.calls == 5
    assert advance_durations == [0.30] * 5
    assert valid.tolist() == [False, False]


def test_validator_iterative_tcp_freezes_a_failed_lane_before_later_valid_ik(monkeypatch):
    target = torch.full((2, 3), 0.01)
    env = _LaneMotionFakeEnv(tcp_target_after_call=4, tcp_target=target)
    ik = _InvalidThenValidIK()
    monkeypatch.setattr(validator, "joint_limit_mask", lambda *_args, **_kwargs: torch.ones(2, dtype=torch.bool))
    monkeypatch.setattr(validator, "interpolate_arm_motion", lambda *_args, **_kwargs: None)
    arm_seed = torch.zeros((2, 7))
    finger_target = torch.full((2, 2), 0.04)

    with validator._PerLaneTargetHold(env, torch.ones(2, dtype=torch.bool), arm_seed, finger_target) as lane_hold:
        arm_target, valid = validator._move_tcp_for_acquisition(
            env,
            ik,
            target,
            torch.tensor(((0.0, 0.0, 0.0, 1.0),) * 2),
            arm_seed,
            finger_target,
            ValidationCfg(),
            lane_hold=lane_hold,
            failure_reason="test-iterative-ik",
        )
        reason_masks = lane_hold.reason_masks

    assert ik.calls == 2
    assert valid.tolist() == [False, True]
    assert torch.equal(reason_masks["test-iterative-ik"], torch.tensor((True, False)))
    assert torch.equal(arm_target[0], torch.zeros(7))
    assert torch.isfinite(ik.positions[1]).all()
    assert torch.equal(ik.positions[1][0], torch.zeros(3))
    assert all(torch.equal(command[0][0], torch.zeros(7)) for command in env.physics_commands)
    assert all(torch.equal(target_write[0][0], torch.zeros(7)) for target_write in env.target_writes)


def test_generator_fixed_bias_compensation_never_reactivates_a_failed_lane(monkeypatch):
    env = _LaneMotionFakeEnv()
    ik = _InvalidThenValidIK()
    owner = object.__new__(generator.PickInsertResetDatasetGenerator)
    owner.env = env
    owner.ik = ik
    owner.cfg = generator.GeneratorCfg()
    owner.device = torch.device("cpu")
    monkeypatch.setattr(generator, "joint_limit_mask", lambda *_args, **_kwargs: torch.ones(2, dtype=torch.bool))
    monkeypatch.setattr(generator, "interpolate_arm_motion", lambda *_args, **_kwargs: None)
    arm_seed = torch.zeros((2, 7))
    finger_target = torch.full((2, 2), 0.04)

    with generator._PerLaneTargetHold(env, torch.ones(2, dtype=torch.bool), arm_seed, finger_target) as lane_hold:
        raw_target, arm_target, valid = owner._move_tcp_with_fixed_arm_bias(
            torch.full((2, 3), 0.01),
            torch.tensor(((0.0, 0.0, 0.0, 1.0),) * 2),
            finger_target,
            raw_arm_seed=arm_seed,
            arm_target_bias=torch.zeros_like(arm_seed),
            duration_s=0.5,
            diagnostic_label="mixed-validity-compensation",
            diagnostics=[],
            compensation_iterations=1,
            require_tracking_tolerance=False,
            settle_s=0.0,
            maximum_raw_ik_joint_step_rad=10.0,
            lane_hold=lane_hold,
            failure_reason="test-fixed-bias-compensation",
        )

    assert ik.calls == 2
    assert valid.tolist() == [False, True]
    assert torch.equal(raw_target[0], torch.zeros(7))
    assert torch.equal(arm_target[0], torch.zeros(7))
    assert torch.isfinite(ik.positions[1]).all()
    assert torch.equal(ik.positions[1][0], torch.zeros(3))
    assert all(torch.equal(target_write[0][0], torch.zeros(7)) for target_write in env.target_writes)


def test_diagnostic_rolling_dwell_requires_a_fresh_uninterrupted_window_after_a_miss():
    dwell = generator._DiagnosticRollingDwell(required_duration_s=2.0)

    assert not dwell.observe(0.0, 0, True)
    assert not dwell.observe(0.75, 1, True)
    assert not dwell.observe(1.0, 2, False)
    assert not dwell.observe(1.25, 3, True)
    assert not dwell.observe(3.0, 4, True)
    assert dwell.observe(3.25, 5, True)

    assert dwell.streak_start_time_s == 1.25
    assert dwell.streak_start_sample_index == 3
    assert dwell.maximum_streak_duration_s == 2.0
    assert dwell.miss_sample_count == 1
    assert dwell.miss_episode_count == 1
    assert dwell.first_miss_time_s == 1.0


@pytest.mark.parametrize(
    ("field", "failure_value", "expected_gate"),
    (
        ("axial_error", 0.801, "exact_axial_error"),
        ("signed_axial_error", 0.201, "exact_axial_overtravel"),
        ("radial_error", 0.751, "exact_radial_error"),
        ("plug_angle_error", 0.051, "exact_plug_angle_error"),
        ("latch_angle_error", 0.061, "exact_latch_angle_error"),
    ),
)
def test_diagnostic_rolling_exact_geometry_failures_are_named_hard_gates(
    field: str,
    failure_value: float,
    expected_gate: str,
):
    success_cfg = SimpleNamespace(
        success_axial_tolerance=0.8,
        success_axial_overtravel_tolerance=0.2,
        success_radial_tolerance=0.75,
        success_plug_angle_tolerance=0.05,
        success_latch_angle_tolerance=0.06,
    )
    exact = SimpleNamespace(
        axial_error=torch.tensor((0.0,)),
        signed_axial_error=torch.tensor((0.0,)),
        radial_error=torch.tensor((0.0,)),
        plug_angle_error=torch.tensor((0.0,)),
        latch_angle_error=torch.tensor((0.0,)),
        plug_spatial_speed=torch.tensor((float("inf"),)),
    )
    setattr(exact, field, torch.tensor((failure_value,)))

    gate_masks = generator._diagnostic_exact_geometry_gate_masks(exact, success_cfg)

    assert tuple(name for name, valid in gate_masks.items() if not bool(valid.all())) == (expected_gate,)
    assert "plug_spatial_speed" not in gate_masks


def test_diagnostic_failure_statistics_include_a_terminal_hard_failure():
    samples = [
        {"time_s": 0.0, "qualifies": True},
        {"time_s": 0.5, "qualifies": False},
    ]

    assert generator._diagnostic_failure_statistics(samples, "qualifies") == {
        "miss_sample_count": 1,
        "miss_episode_count": 1,
        "recurrent_miss_episode_count": 0,
        "first_miss_time_s": 0.5,
    }


def test_diagnostic_rolling_summary_keeps_early_transient_and_split_plug_speed_evidence():
    def sample(
        time_s: float,
        *,
        plug_linear: float,
        plug_angular: float,
        body_x: float,
        qualifies: bool,
    ) -> dict[str, object]:
        plug_spatial = math.sqrt(plug_linear**2 + plug_angular**2)
        body_positions = torch.tensor(((0.0, 0.0, 0.0), (body_x, 0.0, 0.0)))
        return {
            "time_s": time_s,
            "hard_valid": True,
            "exact_success": qualifies,
            "speed_limits_satisfied": qualifies,
            "plug_spatial_speed_satisfied": qualifies,
            "cable_speed_satisfied": True,
            "arm_speed_satisfied": True,
            "finger_speed_satisfied": True,
            "qualifies": qualifies,
            "plug_linear_speed_m_s": plug_linear,
            "plug_angular_speed_rad_s": plug_angular,
            "plug_spatial_speed": plug_spatial,
            "cable_speed_m_s": 0.002,
            "arm_joint_speed_rad_s": 0.003,
            "finger_joint_speed_m_s": 0.0004,
            "authored_seat_error_m": 0.0003,
            "authored_plug_tilt_rad": 0.01,
            "plug_relative_latch_angle_rad": 0.02,
            "arm_target_tracking_error_rad": 0.004,
            "arm_target_drift_rad": 0.0,
            "arm_target_clamp_delta_rad": 0.0,
            "body_positions": body_positions,
            "plug_position": torch.tensor((body_x, 0.0, 0.0)),
            "plug_orientation": torch.tensor((0.0, 0.0, 0.0, 1.0)),
        }

    samples = [
        sample(0.0, plug_linear=0.001, plug_angular=0.002, body_x=0.0, qualifies=True),
        sample(0.5, plug_linear=0.001, plug_angular=0.012, body_x=0.001, qualifies=False),
        sample(1.0, plug_linear=0.002, plug_angular=0.003, body_x=0.002, qualifies=True),
    ]

    summary = generator._summarize_diagnostic_rolling_window(samples)

    assert summary["maximum_plug_linear_speed_m_s"] == 0.002
    assert summary["maximum_plug_angular_speed_rad_s"] == 0.012
    assert summary["maximum_plug_angular_speed_rad_s_time_s"] == 0.5
    assert summary["final_plug_angular_speed_rad_s"] == 0.003
    assert summary["maximum_body_excursion_m"] == pytest.approx(0.002)
    assert summary["maximum_plug_translation_excursion_m"] == pytest.approx(0.002)
    assert summary["qualification_failures"] == {
        "miss_sample_count": 1,
        "miss_episode_count": 1,
        "recurrent_miss_episode_count": 0,
        "first_miss_time_s": 0.5,
    }


def _canonical_capture_state(lane_count: int, body_count: int = 3) -> dict[str, torch.Tensor]:
    """Return a complete lane-distinguishable canonical capture for pure tests."""
    shapes = {
        "arm_joint_position": (lane_count, 7),
        "arm_joint_target": (lane_count, 7),
        "arm_joint_velocity": (lane_count, 7),
        "finger_joint_position": (lane_count, 2),
        "finger_joint_velocity": (lane_count, 2),
        "finger_joint_target": (lane_count, 2),
        "task_body_pose": (lane_count, body_count, 7),
        "task_body_previous_pose": (lane_count, body_count, 7),
        "task_body_coupling_previous_pose": (lane_count, body_count, 7),
        "task_body_velocity": (lane_count, body_count, 6),
    }
    state = {}
    for field_index, (name, shape) in enumerate(shapes.items()):
        row_size = math.prod(shape[1:])
        row = torch.arange(row_size, dtype=torch.float32).reshape(shape[1:])
        state[name] = torch.stack([row + lane * 1000.0 + field_index * 100.0 for lane in range(lane_count)])
    return state


def test_per_lane_rolling_dwell_resets_only_speed_misses_in_the_affected_lane():
    dwell = generator._PerLaneRollingDwell(required_duration_s=2.0, lane_count=3, device="cpu")
    active = torch.ones(3, dtype=torch.bool)

    assert dwell.observe(0.0, 0, torch.tensor((True, True, True)), active).tolist() == [False, False, False]
    assert dwell.observe(1.0, 1, torch.tensor((False, True, False)), active).tolist() == [False, False, False]
    assert dwell.observe(2.0, 2, torch.tensor((True, True, True)), active).tolist() == [False, True, False]
    assert dwell.observe(3.0, 3, torch.tensor((True, True, True)), active).tolist() == [False, True, False]
    assert dwell.observe(4.0, 4, torch.tensor((True, True, True)), active).tolist() == [True, True, True]

    assert dwell.miss_sample_count.tolist() == [1, 0, 1]
    assert dwell.miss_episode_count.tolist() == [1, 0, 1]
    first_miss = dwell.first_miss_time_s.tolist()
    assert first_miss[0] == 1.0
    assert math.isnan(first_miss[1])
    assert first_miss[2] == 1.0
    assert dwell.streak_start_sample_index.tolist() == [2, 0, 2]


def test_canonical_quarantine_copies_every_state_history_robot_and_target_field():
    state = _canonical_capture_state(4)
    original = {name: value.clone() for name, value in state.items()}
    active = torch.tensor((False, True, False, True))

    quarantined, donor = generator._quarantine_inactive_state(state, active)

    assert donor == 1
    assert tuple(quarantined) == tuple(state)
    assert set(quarantined) == {
        "arm_joint_position",
        "arm_joint_target",
        "arm_joint_velocity",
        "finger_joint_position",
        "finger_joint_velocity",
        "finger_joint_target",
        "task_body_pose",
        "task_body_previous_pose",
        "task_body_coupling_previous_pose",
        "task_body_velocity",
    }
    for name, value in quarantined.items():
        assert torch.equal(value[active], original[name][active])
        assert torch.equal(value[~active], original[name][donor].expand_as(value[~active]))
        assert torch.equal(state[name], original[name])


def test_canonical_quarantine_and_selection_reject_zero_survivors_and_select_lowest_original_lane():
    state = _canonical_capture_state(4)
    survivors = torch.tensor((False, False, True, True))

    selected, selected_lane = generator._select_lowest_surviving_lane(state, survivors)

    assert selected_lane == 2
    for name, value in selected.items():
        assert value.device.type == "cpu"
        assert value.dtype == torch.float32
        assert value.is_contiguous()
        assert torch.equal(value, state[name][2])
    with pytest.raises(RuntimeError, match="no surviving lane"):
        generator._quarantine_inactive_state(state, torch.zeros(4, dtype=torch.bool))


def test_canonical_cold_sequence_restores_the_same_capture_without_endpoint_promotion(monkeypatch):
    owner = object.__new__(generator.PickInsertResetDatasetGenerator)
    owner.cfg = generator.GeneratorCfg(batch_size=3)
    capture = _canonical_capture_state(3)
    calls = []

    def fake_cold_proof(
        _self,
        candidate,
        active_mask,
        _authored_target,
        _authored_orientation,
        *,
        duration_s,
        use_authored_exact_reference,
        stage_name,
    ):
        calls.append(
            {
                "identity": id(candidate),
                "active": active_mask.clone(),
                "duration_s": duration_s,
                "authored": use_authored_exact_reference,
                "stage": stage_name,
            }
        )
        survivors = (
            torch.tensor((False, True, True)) if use_authored_exact_reference else torch.tensor((False, False, True))
        )
        return survivors, {"stage": stage_name}

    monkeypatch.setattr(generator.PickInsertResetDatasetGenerator, "_run_canonical_goal_cold_proof", fake_cold_proof)
    survivors, evidence = owner._run_canonical_goal_cold_sequence(
        capture,
        torch.ones(3, dtype=torch.bool),
        torch.zeros((3, 3)),
        torch.tensor(((0.0, 0.0, 0.0, 1.0),) * 3),
    )

    assert survivors.tolist() == [False, False, True]
    assert [call["identity"] for call in calls] == [id(capture), id(capture)]
    assert [call["active"].tolist() for call in calls] == [[True, True, True], [False, True, True]]
    assert [call["authored"] for call in calls] == [True, False]
    assert [call["duration_s"] for call in calls] == [30.0, 60.0]
    assert evidence["same_original_capture_restored_both_times"] is True
    assert evidence["endpoint_promotion_count"] == 0


def test_production_canonical_sequence_continues_after_lane_failures_and_reseats_once(monkeypatch):
    class FakeEnv:
        num_envs = 3
        device = "cpu"
        advance_dt = 1.0
        grasp_proxy_friction = PICK_INSERT_GRASP_PROXY_FRICTION

        def __init__(self):
            self.env_origins = torch.zeros((3, 3))
            self.cfg = SimpleNamespace(
                actions=SimpleNamespace(
                    arm_action=SimpleNamespace(tracking_error_limits=(1.0,) * 7),
                )
            )
            self.arm_q = torch.zeros((3, 7))
            self.task_q = torch.zeros((3, 2, 7))
            self.task_q[..., 6] = 1.0
            self.arm_target = torch.zeros((3, 7))
            self.finger_target = torch.zeros((3, 2))
            self.physics_commands = []

        def read_robot_state(self):
            return self.arm_q.clone(), torch.zeros_like(self.arm_q), torch.zeros((3, 2)), torch.zeros((3, 2))

        def read_task_state(self):
            return self.task_q.clone(), torch.zeros((3, 2, 6))

        def set_robot_targets(self, arm_target, finger_target):
            self.arm_target = arm_target.clone()
            self.finger_target = finger_target.clone()

        def advance(self, duration_s, update=None, *, post_step=None):
            steps = max(1, round(duration_s))
            for step in range(steps):
                progress = (step + 1) / steps
                if update is not None:
                    update(step, steps, progress)
                self.physics_commands.append((self.arm_target.clone(), self.finger_target.clone()))
                if post_step is not None:
                    post_step(step, steps, progress)
            return steps

    env = FakeEnv()
    owner = object.__new__(generator.PickInsertResetDatasetGenerator)
    owner.env = env
    owner.cfg = generator.GeneratorCfg(batch_size=3)
    owner.device = torch.device("cpu")
    owner.plug_index = 0
    owner.closed_finger_target = torch.zeros((3, 2))
    owner.last_grasped_motion_evidence = {}
    capture = _canonical_capture_state(3, body_count=2)
    acquisition_finger_target = torch.tensor(((0.04, 0.04), (0.0, 0.0), (0.0, 0.0)))
    stages = []
    reseat_calls = 0

    monkeypatch.setattr(
        generator.PickInsertResetDatasetGenerator,
        "_assert_drive_disabled",
        lambda _self, _context: None,
    )

    def fake_continuous(self, *_args, lane_hold):
        stages.append("continuous")
        assert lane_hold.active_mask.tolist() == [False, True, True]
        self.env.advance(
            3.0,
            lambda *_: self.env.set_robot_targets(torch.ones((3, 7)), self.closed_finger_target),
        )
        return {"surviving_mask": lane_hold.active_mask.tolist()}

    def fake_reseat(self, *_args, lane_hold, **_kwargs):
        nonlocal reseat_calls
        reseat_calls += 1
        stages.append("reseat")
        assert lane_hold.active_mask.tolist() == [False, True, True]
        self.env.advance(
            2.0,
            lambda *_: self.env.set_robot_targets(2.0 * torch.ones((3, 7)), self.closed_finger_target),
        )
        lane_hold.deactivate(torch.tensor((False, True, False)), reason="fake-reseat-hard-failure")
        self.last_grasped_motion_evidence = {"surviving_mask": lane_hold.active_mask.clone()}
        return lane_hold.last_sent_arm_target, lane_hold.active_mask

    def fake_rolling(self, *_args, lane_hold, **_kwargs):
        stages.append("rolling")
        assert lane_hold.active_mask.tolist() == [False, False, True]
        self.env.advance(
            4.0,
            lambda *_: self.env.set_robot_targets(3.0 * torch.ones((3, 7)), self.closed_finger_target),
        )
        return lane_hold.active_mask, {"surviving_mask": lane_hold.active_mask.tolist()}

    def fake_cold(_self, candidate, active_mask, *_args):
        stages.append("cold")
        assert candidate is capture
        assert active_mask.tolist() == [False, False, True]
        return active_mask, {"same_original_capture_restored_both_times": True}

    monkeypatch.setattr(
        generator.PickInsertResetDatasetGenerator,
        "_run_continuous_goal_relaxation",
        fake_continuous,
    )
    monkeypatch.setattr(generator.PickInsertResetDatasetGenerator, "_move_grasped_plug", fake_reseat)
    monkeypatch.setattr(
        generator.PickInsertResetDatasetGenerator,
        "_run_diagnostic_reseat_rolling_equilibrium",
        fake_rolling,
    )
    monkeypatch.setattr(generator.PickInsertResetDatasetGenerator, "_capture_state", lambda _self, *_args: capture)
    monkeypatch.setattr(
        generator.PickInsertResetDatasetGenerator,
        "_run_canonical_goal_cold_sequence",
        fake_cold,
    )

    canonical, evidence = owner._run_production_canonical_goal_sequence(
        torch.zeros((3, 7)),
        acquisition_finger_target,
        torch.tensor((False, True, True)),
        torch.zeros((3, 3)),
        torch.tensor(((0.0, 0.0, 0.0, 1.0),) * 3),
    )

    assert stages == ["continuous", "reseat", "rolling", "cold"]
    assert reseat_calls == 1
    assert evidence["authored_reseat"]["count"] == 1
    assert evidence["authored_reseat"]["intermediate_fractions"] == (0.25, 0.5, 0.75)
    assert evidence["selected_original_lane"] == 2
    assert torch.equal(canonical["arm_joint_position"], capture["arm_joint_position"][2])
    assert len(env.physics_commands) == 9
    assert all(torch.equal(finger_target[0], acquisition_finger_target[0]) for _, finger_target in env.physics_commands)


def test_canonical_goal_stage_keeps_contact_overflow_batch_fatal(monkeypatch):
    class FakeEnv:
        num_envs = 2
        device = "cpu"

        def read_task_state(self):
            task_q = torch.zeros((2, 2, 7))
            task_q[..., 6] = 1.0
            return task_q, torch.zeros((2, 2, 6))

        def read_robot_state(self):
            return torch.zeros((2, 7)), torch.zeros((2, 7)), torch.zeros((2, 2)), torch.zeros((2, 2))

    owner = object.__new__(generator.PickInsertResetDatasetGenerator)
    owner.env = FakeEnv()
    owner.device = torch.device("cpu")
    owner.plug_index = 0
    owner.latch_index = 1
    owner.closed_finger_target = torch.zeros((2, 2))
    monkeypatch.setattr(
        generator.PickInsertResetDatasetGenerator,
        "_assert_drive_disabled",
        lambda _self, _context: None,
    )
    monkeypatch.setattr(
        generator,
        "collision_metrics",
        lambda *_args, **_kwargs: SimpleNamespace(contact_overflow=True),
    )
    task_q, _ = owner.env.read_task_state()

    with pytest.raises(RuntimeError, match="Global contact-buffer overflow"):
        owner._sample_canonical_goal_stage(
            goal_task_q=task_q,
            immutable_arm_target=torch.zeros((2, 7)),
            baseline_q=task_q,
            authored_seat_target_e=torch.zeros((2, 3)),
            authored_plug_orientation=torch.tensor(((0.0, 0.0, 0.0, 1.0),) * 2),
            tracking_limits=torch.ones(7),
            context="test canonical stage",
        )


def test_generation_checkpoint_lock_rejects_aliases_and_is_exclusive_without_mutation(tmp_path):
    checkpoint = tmp_path / "generation.json"
    lock_path = tmp_path / "generation.json.lock"
    protected_artifact = tmp_path / "goal.pt"
    protected_bytes = b"immutable-goal-certificate"
    protected_artifact.write_bytes(protected_bytes)

    lock_path.symlink_to(protected_artifact)
    with pytest.raises(RuntimeError, match="non-symlink regular file"):
        with generator._GenerationCheckpointLock(checkpoint, protected_paths=(protected_artifact,)):
            pass
    assert protected_artifact.read_bytes() == protected_bytes
    lock_path.unlink()

    lock_path.hardlink_to(protected_artifact)
    with pytest.raises(RuntimeError, match="must not be hard-linked"):
        with generator._GenerationCheckpointLock(checkpoint, protected_paths=(protected_artifact,)):
            pass
    assert protected_artifact.read_bytes() == protected_bytes
    lock_path.unlink()

    lock_path.mkdir()
    with pytest.raises(RuntimeError, match="non-symlink regular file"):
        with generator._GenerationCheckpointLock(checkpoint, protected_paths=(protected_artifact,)):
            pass
    lock_path.rmdir()

    stale_lock_bytes = b"stale-lock-metadata-must-not-be-truncated"
    lock_path.write_bytes(stale_lock_bytes)
    with generator._GenerationCheckpointLock(checkpoint, protected_paths=(protected_artifact,)):
        assert lock_path.read_bytes() == stale_lock_bytes
        with pytest.raises(RuntimeError, match="already locked"):
            with generator._GenerationCheckpointLock(checkpoint, protected_paths=(protected_artifact,)):
                pass
        assert lock_path.read_bytes() == stale_lock_bytes
    assert lock_path.read_bytes() == stale_lock_bytes
    assert protected_artifact.read_bytes() == protected_bytes


def test_generator_recovery_ik_adapter_counts_every_delegated_solve():
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    result = object()

    class FakeIK:
        def solve(self, *args, **kwargs):
            calls.append((args, kwargs))
            return result

    owner = object.__new__(generator.PickInsertResetDatasetGenerator)
    owner.ik = FakeIK()
    owner._ik_solve_call_count = 0
    owner._counted_ik = SimpleNamespace(solve=owner._solve_ik)

    assert owner._counted_ik.solve("first", arm_seed="seed-a") is result
    assert owner._counted_ik.solve("second", arm_seed="seed-b") is result
    assert owner._ik_solve_call_count == 2
    assert calls == [(("first",), {"arm_seed": "seed-a"}), (("second",), {"arm_seed": "seed-b"})]


def test_generation_checkpoint_interrupted_resume_exactly_matches_fresh_576_row_stream(monkeypatch, tmp_path):
    certificate, contracts = _canonical_goal_certificate()
    monkeypatch.setattr(generator, "franka_rj45_asset_contract", lambda: {"schema_version": 5})
    monkeypatch.setattr(
        generator,
        "pick_insert_reset_dataset_task_contract",
        lambda _cfg: deepcopy(contracts["expected_task_contract"]),
    )
    monkeypatch.setattr(generator, "package_versions", lambda: deepcopy(contracts["expected_versions"]))
    monkeypatch.setattr(generator, "reset_dataset_validate_runtime", lambda *_args, **_kwargs: None)

    def make_owner():
        owner = _checkpoint_owner(contracts, certificate)
        owner.env = SimpleNamespace(num_envs=24, cfg=object())
        owner.layout = SimpleNamespace(body_count=3)
        owner.socket_index = 0
        owner.reference_socket_body_pose = torch.tensor((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0))
        return owner

    def install_synthetic_phase_stream(owner, *, crash_after=None):
        def synthetic_phase(
            phase,
            _canonical_goal,
            *,
            initial_accepted=None,
            start_batch_index=0,
            completed_batch_callback=None,
        ):
            accepted = [] if initial_accepted is None else list(initial_accepted)
            accepted_count = sum(int(chunk["phase"].shape[0]) for chunk in accepted)
            for batch_index in range(start_batch_index, 4):
                if accepted_count >= 96:
                    break
                row_count = min(24, 96 - accepted_count)
                draws = torch.rand(24, generator=owner.random)
                chunk = _checkpoint_row_chunk(phase, row_count=row_count)
                chunk["arm_joint_position"][:, 0] = draws[:row_count]
                chunk["arm_joint_target"][:, 0] = phase + batch_index / 10.0
                owner._ik_solve_call_count += 3
                owner.attempt_counts[phase] += 24
                phase_rejections = owner.rejection_counts[phase]
                phase_rejections["accepted"] = phase_rejections.get("accepted", 0) + row_count
                owner.accepted_oracle_metrics[phase].extend(
                    {
                        "goal_error": float(value),
                        "synthetic_batch_index": batch_index,
                    }
                    for value in draws[:row_count]
                )
                accepted.append(chunk)
                accepted_count += row_count
                if completed_batch_callback is not None:
                    completed_batch_callback(phase, batch_index, chunk)
                if crash_after == (phase, batch_index):
                    raise RuntimeError("injected post-commit interruption")
            return {
                name: torch.cat([chunk[name] for chunk in accepted], dim=0)
                for name in generator.RESET_DATASET_STATE_NAMES
            }

        owner._generate_phase = synthetic_phase

    fresh_owner = make_owner()
    install_synthetic_phase_stream(fresh_owner)
    fresh_payload = fresh_owner.generate(certificate)
    assert fresh_owner._ik_solve_call_count == 72

    checkpoint_path = tmp_path / "generation.json"
    interrupted_owner = make_owner()
    interrupted_checkpoint = generator._GenerationCheckpoint.open(
        interrupted_owner,
        certificate,
        path=checkpoint_path,
        resuming=False,
    )
    install_synthetic_phase_stream(interrupted_owner, crash_after=(2, 1))
    with pytest.raises(RuntimeError, match="post-commit interruption"):
        interrupted_owner.generate(certificate, generation_checkpoint=interrupted_checkpoint)

    resumed_owner = make_owner()
    resumed_checkpoint = generator._GenerationCheckpoint.open(
        resumed_owner,
        certificate,
        path=checkpoint_path,
        resuming=True,
    )
    install_synthetic_phase_stream(resumed_owner)
    resumed_payload = resumed_owner.generate(certificate, generation_checkpoint=resumed_checkpoint)

    assert resumed_checkpoint.status == "artifact-ready"
    assert resumed_checkpoint.document["progress"]["logical_ik_solve_call_count"] == 72
    assert (
        resumed_checkpoint.document["progress"]["final_artifact"]["content_sha256"] == fresh_payload["content_sha256"]
    )
    assert resumed_payload["content_sha256"] == fresh_payload["content_sha256"]
    assert resumed_payload["metadata"] == fresh_payload["metadata"]
    assert all(
        torch.equal(resumed_payload["states"][name], fresh_payload["states"][name])
        for name in generator.RESET_DATASET_STATE_NAMES
    )
    assert all(
        torch.equal(resumed_payload["goal_state"][name], fresh_payload["goal_state"][name])
        for name in generator.RESET_DATASET_GOAL_STATE_NAMES
    )


def test_fast_phase_0_task_state_is_preseat_banded_and_never_geometrically_successful():
    num_envs = 2_048
    env_cfg = FrankaRJ45PickInsertEnvCfg()
    owner = object.__new__(generator.PickInsertResetDatasetGenerator)
    owner.env = SimpleNamespace(
        num_envs=num_envs,
        cfg=env_cfg,
        rj45_runtime=SimpleNamespace(layout=SimpleNamespace(body_count=3, plug_body_index=0, latch_body_index=1)),
        advance=lambda *_args, **_kwargs: pytest.fail("physics advanced"),
    )
    owner.cfg = generator.GeneratorCfg(generation_mode="fast-ik", batch_size=num_envs)
    owner.device = torch.device("cpu")
    owner.random = torch.Generator().manual_seed(91)
    owner.layout = SimpleNamespace(body_count=3)
    owner.plug_index = 0
    owner.latch_index = 1
    owner.socket_index = 2
    goal_q = torch.zeros((num_envs, 3, 7))
    goal_q[..., 6] = 1.0

    task_q, task_qd = owner._fast_task_state(0, torch.zeros((num_envs, 7)), goal_q)
    local_error = generator.math_utils.quat_apply_inverse(
        goal_q[:, owner.plug_index, 3:7],
        task_q[:, owner.plug_index, :3] - goal_q[:, owner.plug_index, :3],
    )
    shortfall = -local_error[:, 1]
    band_indices = generator.phase_0_reverse_curriculum_band_indices(shortfall)
    success = generator.exact_success_from_state(
        owner.env,
        task_q,
        task_qd,
        goal_q,
        plug_body_index=owner.plug_index,
        latch_body_index=owner.latch_index,
    ).mask

    assert bool((local_error[:, 1] < 0.0).all())
    assert bool((band_indices >= 0).all())
    assert set(band_indices.tolist()) == {0, 1, 2}
    assert not bool(success.any())
    assert not bool((task_qd != 0.0).any())


def test_fast_phase_4_ik_uses_the_bounded_pregrasp_orientation_sample():
    owner = object.__new__(generator.PickInsertResetDatasetGenerator)
    owner.env = SimpleNamespace(
        num_envs=4,
        cfg=SimpleNamespace(plug_grasp_offset=(0.0, 0.0, 0.0)),
    )
    owner.cfg = generator.GeneratorCfg(batch_size=4)
    owner.device = torch.device("cpu")
    owner.random = torch.Generator().manual_seed(37)
    owner.plug_index = 0
    owner.local_grasp_orientation = torch.tensor(((math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0),) * 4)
    owner.closed_finger_target = torch.zeros((4, 2))
    owner.open_finger_q = torch.full((4, 2), 0.04)
    owner.home_arm_q = torch.zeros((4, 7))
    observed: dict[str, torch.Tensor] = {}

    def solve(position, orientation, finger_target, *, arm_seed):
        observed.update(
            position=position.clone(),
            orientation=orientation.clone(),
            finger_target=finger_target.clone(),
            arm_seed=arm_seed.clone(),
        )
        return SimpleNamespace(arm_q=arm_seed.clone(), valid=torch.ones(4, dtype=torch.bool))

    owner._solve_ik = solve
    task_q = torch.zeros((4, 1, 7))
    task_q[..., 6] = 1.0
    expected_rng = torch.Generator().manual_seed(37)
    expected_error = generator.sample_phase_4_pregrasp_orientation_errors(
        4,
        device="cpu",
        rng=expected_rng,
    )
    expected_orientation = generator.math_utils.quat_mul(owner.local_grasp_orientation, expected_error)

    _, finger_target = owner._fast_row_ik(4, task_q, torch.ones((4, 7)))

    torch.testing.assert_close(observed["position"][:, 2], torch.full((4,), 0.045))
    torch.testing.assert_close(observed["orientation"], expected_orientation)
    assert torch.equal(finger_target, owner.open_finger_q)
    assert torch.equal(observed["finger_target"], owner.open_finger_q)
    assert torch.equal(observed["arm_seed"], owner.home_arm_q)


def test_fast_ik_generator_mode_is_explicit_and_legacy_remains_default():
    assert generator.GeneratorCfg().generation_mode == generator._GENERATION_MODE_PHYSICAL_ORACLE
    assert (
        generator.GeneratorCfg(generation_mode=generator._GENERATION_MODE_FAST_IK).generation_mode
        == generator._GENERATION_MODE_FAST_IK
    )
    with pytest.raises(ValueError, match="generation_mode must be one of"):
        generator.GeneratorCfg(generation_mode="implicit-shortcut")


def test_fast_collision_authoring_replaces_invalid_lanes_without_admitting_them():
    owner = object.__new__(generator.PickInsertResetDatasetGenerator)
    owner.home_arm_q = torch.full((2, 7), 0.25)
    goal_q = torch.zeros((2, 3, 7))
    goal_q[..., 6] = 1.0
    goal_q[:, :, 0] = 0.5
    task_q = goal_q.clone()
    task_q[0, :, 0] = 0.1
    task_q[1, 0, 0] = torch.nan
    task_qd = torch.ones((2, 3, 6))
    arm_target = torch.zeros((2, 7))
    arm_target[1, 0] = torch.nan

    collision_q, collision_qd, collision_arm = owner._fast_collision_authoring_state(
        task_q=task_q,
        task_qd=task_qd,
        goal_q=goal_q,
        arm_target=arm_target,
        ik_valid=torch.tensor((True, False)),
    )

    torch.testing.assert_close(collision_q[0], task_q[0], rtol=0.0, atol=0.0)
    torch.testing.assert_close(collision_qd[0], task_qd[0], rtol=0.0, atol=0.0)
    torch.testing.assert_close(collision_arm[0], arm_target[0], rtol=0.0, atol=0.0)
    torch.testing.assert_close(collision_q[1], goal_q[1], rtol=0.0, atol=0.0)
    assert not bool((collision_qd[1] != 0.0).any())
    torch.testing.assert_close(collision_arm[1], owner.home_arm_q[1], rtol=0.0, atol=0.0)
    assert bool(torch.isfinite(collision_q).all())
    assert bool(torch.isfinite(collision_arm).all())


@pytest.mark.parametrize(
    ("starts_grasped", "grasp_contact_count", "grasp_aligned", "expected"),
    (
        (True, 2, True, True),
        (True, 2, False, False),
        (False, 0, False, True),
        (False, 1, False, False),
    ),
)
def test_fast_collide_only_checks_enforce_grasped_and_open_contact_semantics(
    monkeypatch,
    starts_grasped,
    grasp_contact_count,
    grasp_aligned,
    expected,
):
    owner = object.__new__(generator.PickInsertResetDatasetGenerator)
    calls: list[str] = []
    owner.device = torch.device("cpu")
    owner.env = SimpleNamespace(
        num_envs=1,
        write_task_state=lambda *_args, **_kwargs: calls.append("task"),
        write_robot_state=lambda *_args, **_kwargs: calls.append("robot"),
    )
    monkeypatch.setattr(
        generator,
        "collide_only_metrics",
        lambda *_args, **_kwargs: SimpleNamespace(
            valid=torch.ones(1, dtype=torch.bool),
            invalid_contact_count=torch.zeros(1, dtype=torch.long),
            grasp_contact_count=torch.full((1,), grasp_contact_count, dtype=torch.long),
            left_grasp_contact_count=torch.full((1,), int(grasp_contact_count > 0), dtype=torch.long),
            right_grasp_contact_count=torch.full((1,), int(grasp_contact_count > 1), dtype=torch.long),
            contact_overflow=False,
        ),
    )
    monkeypatch.setattr(
        generator,
        "grasp_metrics",
        lambda *_args, **_kwargs: SimpleNamespace(valid=torch.tensor((grasp_aligned,))),
    )

    valid, evidence = owner._fast_collide_only_checks(
        task_q=torch.zeros((1, 3, 7)),
        task_qd=torch.zeros((1, 3, 6)),
        arm_target=torch.zeros((1, 7)),
        finger_position=torch.zeros((1, 2)),
        finger_target=torch.zeros((1, 2)),
        starts_grasped=starts_grasped,
    )

    assert calls == ["task", "robot"]
    assert bool(valid[0]) is expected
    assert int(evidence["collide_only_grasp_contact_count"][0]) == grasp_contact_count
    assert bool(evidence["collide_only_contact_overflow"][0]) is False


def test_fast_phase_generation_never_enters_dynamics_replay_or_recovery(monkeypatch):
    owner = object.__new__(generator.PickInsertResetDatasetGenerator)
    authored: list[str] = []
    owner.env = SimpleNamespace(
        num_envs=2,
        advance=lambda *_args, **_kwargs: pytest.fail("physics advanced"),
        write_task_state=lambda *_args, **_kwargs: authored.append("task"),
        write_robot_state=lambda *_args, **_kwargs: authored.append("robot"),
    )
    owner.device = torch.device("cpu")
    owner.home_arm_q = torch.zeros((2, 7))
    owner.cfg = SimpleNamespace(rows_per_phase=2, max_batches_per_phase=1)
    owner.attempt_counts = [0] * 6
    owner.rejection_counts = {phase: {} for phase in range(6)}
    owner.accepted_oracle_metrics = {phase: [] for phase in range(6)}
    task_q = torch.zeros((2, 3, 7))
    task_q[..., 6] = 1.0
    task_qd = torch.zeros((2, 3, 6))
    arm_q = torch.zeros((2, 7))
    finger_q = torch.zeros((2, 2))
    goal_q = task_q.clone()
    owner._sample_scene = lambda: (torch.zeros((2, 7)), torch.zeros((2, 7)))
    owner._row_goal = lambda *_args: (goal_q, task_qd, arm_q, torch.ones(2, dtype=torch.bool))
    owner._fast_task_state = lambda *_args: (task_q, task_qd)
    owner._fast_row_ik = lambda *_args: (
        SimpleNamespace(
            arm_q=arm_q,
            tcp_position=torch.zeros((2, 3)),
            position_residual=torch.zeros(2),
            rotation_residual=torch.zeros(2),
            valid=torch.ones(2, dtype=torch.bool),
        ),
        finger_q,
    )
    owner._last_goal_ik_result = SimpleNamespace(
        position_residual=torch.zeros(2),
        rotation_residual=torch.zeros(2),
    )
    owner._fast_static_checks = lambda *_args: (
        {
            name: torch.ones(2, dtype=torch.bool)
            for name in generator.FRANKA_RJ45_PICK_INSERT_FAST_RESET_POLICY["checks"]
        },
        {
            "initial_goal_error": torch.full((2,), 0.03),
            "initial_tcp_grasp_distance": torch.full((2,), 0.04),
            "minimum_workspace_margin_m": torch.full((2,), 0.1),
            "minimum_cable_support_clearance_m": torch.zeros(2),
            "minimum_nonadjacent_cable_separation_m": torch.full((2,), 0.01),
            "minimum_cable_socket_center_distance_m": torch.full((2,), 0.1),
            "phase_0_signed_axial_error_m": torch.full((2,), -0.0012),
            "phase_0_axial_shortfall_m": torch.full((2,), 0.0012),
            "phase_0_reverse_curriculum_band_index": torch.zeros(2, dtype=torch.int64),
            "initial_runtime_geometric_success": torch.zeros(2, dtype=torch.bool),
        },
    )
    owner._record_rejections = lambda phase, _checks, valid: owner.rejection_counts[phase].update(
        accepted=int(valid.sum())
    )
    owner._cold_replay = lambda *_args, **_kwargs: pytest.fail("cold replay entered")
    owner._oracle = lambda *_args, **_kwargs: pytest.fail("recovery oracle entered")
    monkeypatch.setattr(
        generator,
        "collide_only_metrics",
        lambda *_args, **_kwargs: SimpleNamespace(
            valid=torch.ones(2, dtype=torch.bool),
            invalid_contact_count=torch.zeros(2, dtype=torch.long),
            grasp_contact_count=torch.full((2,), 2, dtype=torch.long),
            left_grasp_contact_count=torch.ones(2, dtype=torch.long),
            right_grasp_contact_count=torch.ones(2, dtype=torch.long),
            contact_overflow=False,
        ),
    )
    monkeypatch.setattr(
        generator,
        "grasp_metrics",
        lambda *_args, **_kwargs: SimpleNamespace(valid=torch.ones(2, dtype=torch.bool)),
    )

    rows = owner._generate_phase_fast(
        0,
        {"task_body_pose": task_q[0], "finger_joint_position": torch.full((1, 2), 0.007)},
    )

    assert rows["phase"].tolist() == [0, 0]
    assert authored == ["task", "robot"]
    assert owner.attempt_counts[0] == 2
    assert len(owner.accepted_oracle_metrics[0]) == 2
    assert owner.accepted_oracle_metrics[0][0]["checks"] == {
        name: True for name in generator.FRANKA_RJ45_PICK_INSERT_FAST_RESET_POLICY["checks"]
    }
    assert owner.accepted_oracle_metrics[0][0]["collide_only_invalid_contact_count"] == 0
    assert owner.accepted_oracle_metrics[0][0]["collide_only_grasp_contact_count"] == 2
    assert owner.accepted_oracle_metrics[0][0]["collide_only_left_grasp_contact_count"] == 1
    assert owner.accepted_oracle_metrics[0][0]["collide_only_right_grasp_contact_count"] == 1
    assert owner.accepted_oracle_metrics[0][0]["collide_only_contact_overflow"] is False


def test_fast_generation_embeds_distinct_policy_and_metrics(monkeypatch):
    certificate, contracts = _canonical_goal_certificate()
    owner = object.__new__(generator.PickInsertResetDatasetGenerator)
    owner.cfg = generator.GeneratorCfg(generation_mode=generator._GENERATION_MODE_FAST_IK)
    owner.env = SimpleNamespace(num_envs=24, cfg=object())
    owner.device = torch.device("cpu")
    owner.random = torch.Generator().manual_seed(999)
    owner._ik_solve_call_count = 0
    owner.layout = SimpleNamespace(body_count=3)
    owner.socket_index = 0
    owner.reference_socket_body_pose = torch.zeros(7)
    owner.attempt_counts = [0] * 6
    owner.rejection_counts = {phase: {} for phase in range(6)}
    owner.accepted_oracle_metrics = {
        phase: [
            {
                "checks": dict(generator.FRANKA_RJ45_PICK_INSERT_FAST_RESET_POLICY["checks"]),
                **(
                    {
                        "phase_0_reverse_curriculum_band": "immediate",
                        "phase_0_axial_shortfall_m": 0.0012,
                        "initial_runtime_geometric_success": False,
                    }
                    if phase == 0
                    else {}
                ),
            }
        ]
        for phase in range(6)
    }
    owner.validate_goal_certificate = lambda value: value
    calls: list[int] = []

    def fast_phase(phase, _canonical_goal):
        calls.append(phase)
        result = {name: torch.zeros(1) for name in generator.RESET_DATASET_STATE_NAMES}
        result["phase"] = torch.tensor((phase,), dtype=torch.int64)
        result["goal_task_body_pose"] = torch.zeros((1, 3, 7))
        return result

    owner._generate_phase_fast = fast_phase
    owner._generate_phase = lambda *_args, **_kwargs: pytest.fail("legacy phase generator entered")
    monkeypatch.setattr(
        generator,
        "pick_insert_reset_dataset_task_contract",
        lambda _cfg: contracts["expected_task_contract"],
    )
    monkeypatch.setattr(generator, "reset_dataset_validate_runtime", lambda *_args, **_kwargs: None)

    payload = owner.generate(certificate)

    assert calls == list(range(6))
    assert payload["metadata"]["initial_state_policy"]["fast_reset_policy"] == (
        generator.FRANKA_RJ45_PICK_INSERT_FAST_RESET_POLICY
    )
    assert "accepted_fast_reset_metrics" in payload["metadata"]
    assert "accepted_oracle_metrics" not in payload["metadata"]
    assert payload["metadata"]["initial_state_policy"]["fast_reset_policy"]["simulation_steps"] == 0
    assert payload["metadata"]["initial_state_policy"]["phase_0_reverse_curriculum_sampling"] == (
        generator.pick_insert_phase_0_reverse_curriculum_sampling_contract()
    )
    assert payload["metadata"]["initial_state_policy"]["phase_0_reverse_curriculum_evidence"] == {
        "accepted_row_count": 1,
        "accepted_band_counts": {"immediate": 1, "quick": 0, "boundary": 0},
        "accepted_band_fractions": {"immediate": 1.0, "quick": 0.0, "boundary": 0.0},
        "maximum_absolute_band_fraction_error": 0.65,
        "allowed_absolute_band_fraction_error": 1.0,
        "band_proportions_within_tolerance": True,
        "minimum_axial_shortfall_m": 0.0012,
        "maximum_axial_shortfall_m": 0.0012,
        "initial_runtime_geometric_success_count": 0,
        "all_rows_preseat_and_outside_geometric_success": True,
        "simulation_steps": 0,
    }
    bound_records = payload["metadata"]["accepted_fast_reset_metrics"]
    final_row_ids = sorted(record["final_row_id"] for records in bound_records.values() for record in records)
    assert final_row_ids == list(range(6))
    for records in bound_records.values():
        for record in records:
            assert record["state_sha256"] == generator.pick_insert_reset_dataset_row_digest(
                payload["states"], record["final_row_id"]
            )


def _fast_validator_metadata_fixture(*, rows_per_phase: int = 7, batch_size: int = 4):
    phases = torch.arange(6, dtype=torch.int64).repeat_interleave(rows_per_phase)
    states = {
        name: torch.arange(len(phases), dtype=torch.float32)
        for name in fast_validator.PICK_INSERT_FAST_RESET_ROW_BINDING_CONTRACT["state_names"]
    }
    states["phase"] = phases
    sampler = fast_validator.pick_insert_phase_0_reverse_curriculum_sampling_contract()
    band_names = tuple(sampler["band_names"])
    band_ranges = tuple(sampler["axial_offset_ranges_m"])
    accepted: dict[str, list[dict[str, object]]] = {}
    phase_0_shortfalls: list[float] = []
    phase_0_band_counts = {name: 0 for name in band_names}
    for phase in range(6):
        records: list[dict[str, object]] = []
        for row_index in range(rows_per_phase):
            starts_grasped = phase <= 3
            record: dict[str, object] = {
                "final_row_id": phase * rows_per_phase + row_index,
                "checks": deepcopy(fast_validator.FRANKA_RJ45_PICK_INSERT_FAST_RESET_POLICY["checks"]),
                "collide_only_invalid_contact_count": 0,
                "collide_only_grasp_contact_count": 2 if starts_grasped else 0,
                "collide_only_left_grasp_contact_count": 1 if starts_grasped else 0,
                "collide_only_right_grasp_contact_count": 1 if starts_grasped else 0,
                "collide_only_contact_overflow": False,
            }
            record["state_sha256"] = fast_validator.pick_insert_reset_dataset_row_digest(
                states, int(record["final_row_id"])
            )
            if phase == 0:
                band_index = row_index % len(band_names)
                band_name = band_names[band_index]
                lower, upper = band_ranges[band_index]
                shortfall = 0.5 * (lower + upper)
                record.update(
                    {
                        "phase_0_reverse_curriculum_band": band_name,
                        "phase_0_axial_shortfall_m": shortfall,
                        "initial_runtime_geometric_success": False,
                    }
                )
                phase_0_shortfalls.append(shortfall)
                phase_0_band_counts[band_name] += 1
            records.append(record)
        accepted[str(phase)] = records
    reference_profile = rows_per_phase == 3_334 and batch_size == 256
    metadata = {
        "phase_counts": [rows_per_phase] * 6,
        "accepted_fast_reset_metrics": accepted,
        "initial_state_policy": {
            "generation_mode": "fast-ik",
            "fast_reset_policy": deepcopy(fast_validator.FRANKA_RJ45_PICK_INSERT_FAST_RESET_POLICY),
            "accepted_row_binding": deepcopy(fast_validator.PICK_INSERT_FAST_RESET_ROW_BINDING_CONTRACT),
            "phase_0_accepted_band_proportions": deepcopy(
                fast_validator.PICK_INSERT_FAST_RESET_PHASE_0_BAND_ACCEPTANCE_CONTRACT
            ),
            "reset_bank_profile": {
                "contract_version": 1,
                "profile": "balanced-20004-v1" if reference_profile else "custom-balanced-fast-ik",
                "reference_profile": reference_profile,
                "rows_per_phase": rows_per_phase,
                "phase_counts": [rows_per_phase] * 6,
                "total_rows": rows_per_phase * 6,
                "batch_size": batch_size,
                "maximum_batches_per_phase": 96 if reference_profile else math.ceil(rows_per_phase / batch_size),
                "simulation_steps_per_row": 0,
            },
            "phase_0_reverse_curriculum_sampling": deepcopy(sampler),
            "phase_0_reverse_curriculum_evidence": {
                "accepted_row_count": rows_per_phase,
                "accepted_band_counts": phase_0_band_counts,
                "accepted_band_fractions": {
                    name: count / rows_per_phase for name, count in phase_0_band_counts.items()
                },
                "maximum_absolute_band_fraction_error": max(
                    abs(phase_0_band_counts[name] / rows_per_phase - weight)
                    for name, weight in zip(band_names, sampler["band_weights"], strict=True)
                ),
                "allowed_absolute_band_fraction_error": (
                    fast_validator.pick_insert_fast_reset_phase_0_band_fraction_tolerance(rows_per_phase)
                ),
                "band_proportions_within_tolerance": True,
                "minimum_axial_shortfall_m": min(phase_0_shortfalls),
                "maximum_axial_shortfall_m": max(phase_0_shortfalls),
                "initial_runtime_geometric_success_count": 0,
                "all_rows_preseat_and_outside_geometric_success": True,
                "simulation_steps": 0,
            },
        },
    }
    return metadata, states


def test_fast_validator_derives_nonlegacy_balanced_shape_from_artifact_contract():
    env_cfg = FrankaRJ45PickInsertEnvCfg()
    env_cfg.reset_dataset_rows_per_phase = 7
    env_cfg.reset_dataset_min_unique_full_pick_rows = 6
    payload = {
        "metadata": {"task_contract": pick_insert_reset_dataset_task_contract(env_cfg)},
        "states": {"phase": torch.arange(6, dtype=torch.int64).repeat_interleave(7)},
    }

    resolved = fast_validator._artifact_bound_env_cfg(payload)

    assert resolved.reset_dataset_rows_per_phase == 7
    assert resolved.reset_dataset_min_unique_full_pick_rows == 6
    assert pick_insert_reset_dataset_task_contract(resolved) == payload["metadata"]["task_contract"]


def test_fast_validator_rejects_artifact_cardinality_that_disagrees_with_state_rows():
    env_cfg = FrankaRJ45PickInsertEnvCfg()
    env_cfg.reset_dataset_rows_per_phase = 7
    env_cfg.reset_dataset_min_unique_full_pick_rows = 6
    payload = {
        "metadata": {"task_contract": pick_insert_reset_dataset_task_contract(env_cfg)},
        "states": {"phase": torch.arange(6, dtype=torch.int64).repeat_interleave(6)},
    }

    with pytest.raises(ValueError, match="phase counts do not match"):
        fast_validator._artifact_bound_env_cfg(payload)


def test_fast_validator_accepts_complete_policy_v2_collide_only_evidence_at_custom_scale():
    metadata, states = _fast_validator_metadata_fixture(rows_per_phase=7, batch_size=4)

    evidence, evidence_sha256 = fast_validator._fast_metadata_evidence(metadata, states)

    assert len(evidence) == len(states["phase"]) == 42
    assert evidence[0] is not evidence[len(states["phase"]) - 1]
    assert all(evidence[0].values())
    assert evidence_sha256 == fast_validator.reset_dataset_digest(metadata["accepted_fast_reset_metrics"])


def test_fast_validator_accepts_the_balanced_20004_row_reference_profile():
    metadata, states = _fast_validator_metadata_fixture(rows_per_phase=3_334, batch_size=256)

    evidence, _ = fast_validator._fast_metadata_evidence(metadata, states)

    assert len(evidence) == 20_004
    assert metadata["initial_state_policy"]["reset_bank_profile"]["reference_profile"] is True
    assert metadata["initial_state_policy"]["reset_bank_profile"]["profile"] == "balanced-20004-v1"


def test_fast_validator_canonical_promotion_requires_reference_profile_and_live_contract():
    reference_metadata, reference_states = _fast_validator_metadata_fixture(rows_per_phase=3_334, batch_size=256)
    reference_metadata["task_contract"] = pick_insert_reset_dataset_task_contract(FrankaRJ45PickInsertEnvCfg())

    fast_validator._require_reference_promotion({"metadata": reference_metadata, "states": reference_states})

    custom_metadata, custom_states = _fast_validator_metadata_fixture(rows_per_phase=7, batch_size=4)
    custom_cfg = FrankaRJ45PickInsertEnvCfg()
    custom_cfg.reset_dataset_rows_per_phase = 7
    custom_cfg.reset_dataset_min_unique_full_pick_rows = 7
    custom_metadata["task_contract"] = pick_insert_reset_dataset_task_contract(custom_cfg)
    with pytest.raises(ValueError, match="exact 20,004-row reference fast profile"):
        fast_validator._require_reference_promotion({"metadata": custom_metadata, "states": custom_states})

    reference_metadata["initial_state_policy"]["generation_mode"] = "physical-oracle"
    with pytest.raises(ValueError, match="exact reference profile and default live task contract"):
        fast_validator._require_reference_promotion({"metadata": reference_metadata, "states": reference_states})


@pytest.mark.parametrize("mutation", ("duplicate-row-id", "stale-state-digest", "wrong-record-digest"))
def test_fast_validator_rejects_unbound_final_row_evidence(mutation):
    metadata, states = _fast_validator_metadata_fixture()
    record = metadata["accepted_fast_reset_metrics"]["0"][0]
    if mutation == "duplicate-row-id":
        record["final_row_id"] = metadata["accepted_fast_reset_metrics"]["0"][1]["final_row_id"]
    elif mutation == "stale-state-digest":
        states["difficulty"][record["final_row_id"]] += 1.0
    else:
        record["state_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="not bound to its exact final artifact row"):
        fast_validator._fast_metadata_evidence(metadata, states)


@pytest.mark.parametrize(
    ("phase", "field", "value", "message"),
    (
        (0, "collide_only_right_grasp_contact_count", 0, "lacks bilateral"),
        (4, "collide_only_grasp_contact_count", 1, "open-gripper row has Newton proxy contacts"),
        (5, "collide_only_invalid_contact_count", 0.0, "non-negative plain integers"),
        (2, "collide_only_contact_overflow", True, "invalid or overflowing"),
    ),
)
def test_fast_validator_rejects_inconsistent_collide_only_evidence(phase, field, value, message):
    metadata, states = _fast_validator_metadata_fixture()
    metadata["accepted_fast_reset_metrics"][str(phase)][0][field] = value

    with pytest.raises(ValueError, match=message):
        fast_validator._fast_metadata_evidence(metadata, states)


def test_fast_validator_rejects_terminal_or_out_of_band_phase_0_metadata():
    metadata, states = _fast_validator_metadata_fixture()
    metadata["accepted_fast_reset_metrics"]["0"][0]["initial_runtime_geometric_success"] = True

    with pytest.raises(ValueError, match="already a geometric success"):
        fast_validator._fast_metadata_evidence(metadata, states)

    metadata, states = _fast_validator_metadata_fixture()
    metadata["accepted_fast_reset_metrics"]["0"][0]["phase_0_axial_shortfall_m"] = 0.02
    with pytest.raises(ValueError, match="outside its recorded band"):
        fast_validator._fast_metadata_evidence(metadata, states)


def test_fast_validator_rejects_grossly_skewed_accepted_phase_0_bands_with_matching_summary():
    metadata, states = _fast_validator_metadata_fixture(rows_per_phase=100, batch_size=20)
    records = metadata["accepted_fast_reset_metrics"]["0"]
    for record in records:
        record["phase_0_reverse_curriculum_band"] = "immediate"
        record["phase_0_axial_shortfall_m"] = 0.0012
    evidence = metadata["initial_state_policy"]["phase_0_reverse_curriculum_evidence"]
    evidence.update(
        {
            "accepted_band_counts": {"immediate": 100, "quick": 0, "boundary": 0},
            "accepted_band_fractions": {"immediate": 1.0, "quick": 0.0, "boundary": 0.0},
            "maximum_absolute_band_fraction_error": 0.65,
            "allowed_absolute_band_fraction_error": 0.05,
            "minimum_axial_shortfall_m": 0.0012,
            "maximum_axial_shortfall_m": 0.0012,
        }
    )

    with pytest.raises(ValueError, match="band proportions exceed"):
        fast_validator._fast_metadata_evidence(metadata, states)


def test_fast_validator_rejects_an_impossible_reset_bank_profile_capacity():
    metadata, states = _fast_validator_metadata_fixture(rows_per_phase=7, batch_size=2)
    metadata["initial_state_policy"]["reset_bank_profile"]["maximum_batches_per_phase"] = 3

    with pytest.raises(ValueError, match="reset_bank_profile is inconsistent"):
        fast_validator._fast_metadata_evidence(metadata, states)


def test_fast_validator_phase_0_geometry_accepts_only_banded_nonterminal_preseat_rows():
    env_cfg = FrankaRJ45PickInsertEnvCfg()
    task_contract = pick_insert_reset_dataset_task_contract(env_cfg)
    layout = task_contract["rj45_physics"]["task_layout"]
    plug_index = int(layout["plug_body_index"])
    body_count = int(task_contract["task_body_count"])
    shortfalls = torch.tensor((0.0010, 0.0016, 0.0035, 0.0120, 0.0007, 0.0121, -0.0012))
    task_pose = torch.zeros((len(shortfalls), body_count, 7))
    goal_pose = torch.zeros_like(task_pose)
    task_pose[..., 6] = 1.0
    goal_pose[..., 6] = 1.0
    task_pose[:, plug_index, 1] = -shortfalls
    states = {
        "task_body_pose": task_pose,
        "task_body_velocity": torch.zeros((len(shortfalls), body_count, 6)),
        "goal_task_body_pose": goal_pose,
    }

    valid = fast_validator._phase_0_reverse_curriculum_semantics(states, task_contract, env_cfg)

    assert valid.tolist() == [True, True, True, True, False, False, False]
