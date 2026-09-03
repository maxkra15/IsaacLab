# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pure geometry, configuration, and policy-contract tests for the dual-rack RJ45 task."""

from __future__ import annotations

from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

from isaaclab.managers import ObservationTermCfg
from isaaclab.utils import math as math_utils

from isaaclab_tasks.contrib.franka_rj45_insertion import mdp
from isaaclab_tasks.contrib.franka_rj45_insertion.config import franka as _franka_registration  # noqa: F401
from isaaclab_tasks.contrib.franka_rj45_insertion.config.franka.agents.dual_rack_rsl_rl_ppo_cfg import (
    FrankaRJ45DualRackInsertPPORunnerCfg,
)
from isaaclab_tasks.contrib.franka_rj45_insertion.config.franka.agents.gb300_rsl_rl_ppo_cfg import (
    FrankaRJ45Gb300InsertPPORunnerCfg,
)
from isaaclab_tasks.contrib.franka_rj45_insertion.dual_rack_env_cfg import (
    DUAL_RACK_ROBOT_PROXY_BODY_PATTERNS,
    FrankaRJ45DualRackInsertEnvCfg,
    dual_rack_reset_dataset_task_contract,
)
from isaaclab_tasks.contrib.franka_rj45_insertion.dual_rack_workcell import (
    DUAL_RACK_ANCHORED_SOCKET_POSITION_E,
    DUAL_RACK_CABLE_TABLE_CENTERLINE_HEIGHT_M,
    DUAL_RACK_SWITCH_MAX_SOCKET_LOCAL,
    DUAL_RACK_SWITCH_MIN_SOCKET_LOCAL,
    DUAL_RACK_TARGET_SOCKET_POSITION_E,
    DUAL_RACK_WORKCELL_CFG,
    dual_rack_cable_body_poses_torch,
    dual_rack_cable_workcell_intersection_mask_torch,
    dual_rack_cable_workcell_intersections_numpy,
    dual_rack_workcell_contract,
    route_dual_rack_cable_points_numpy,
    route_dual_rack_cable_points_torch,
)
from isaaclab_tasks.contrib.franka_rj45_insertion.dual_rack_workcell_presentation import (
    dual_rack_t_slot_marker_state,
)
from isaaclab_tasks.contrib.franka_rj45_insertion.gb300_asset import (
    GB300_SIMREADY_EXTERNAL_USD_SHA256,
    GB300_SIMREADY_EXTERNAL_USD_SIZE,
    GB300_SIMREADY_LICENSE,
    GB300_SIMREADY_REVISION,
    gb300_asset_contract,
)
from isaaclab_tasks.contrib.franka_rj45_insertion.gb300_env_cfg import (
    GB300_RJ45_ENTRY_BODY_PATTERNS,
    FrankaRJ45Gb300InsertEnvCfg,
    gb300_reset_dataset_task_contract,
)
from isaaclab_tasks.contrib.franka_rj45_insertion.gb300_presentation import (
    gb300_marker_state,
    gb300_presentation_contract,
)
from isaaclab_tasks.contrib.franka_rj45_insertion.gb300_workcell import (
    GB300_ANCHORED_SOCKET_POSITION_E,
    GB300_DEFAULT_TARGET_TASK_TRANSLATION,
    GB300_SIMREADY_COMPOSED_BOUNDS_MAX_M,
    GB300_SIMREADY_COMPOSED_BOUNDS_MIN_M,
    GB300_SIMREADY_ROOT_ROTATION_XYZ_DEG,
    GB300_SIMREADY_ROOT_TRANSLATION_M,
    GB300_STUDIO_FLOOR_HEIGHT_M,
    GB300_TARGET_SOCKET_POSITIONS_E,
    GB300_TARGET_TASK_TRANSLATIONS,
    GB300_WORKCELL_CFG,
    gb300_workcell_contract,
)
from isaaclab_tasks.contrib.franka_rj45_insertion.pick_insert_env_cfg import (
    PICK_INSERT_RJ45_ENTRY_BODY_PATTERNS,
    FrankaRJ45PickInsertEnvCfg,
)
from isaaclab_tasks.contrib.franka_rj45_insertion.rj45_env_cfg import RJ45_GRASP_PROXY_BODY_PATTERNS


def _synthetic_route_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a feasible four-anchor prefix and 45-segment cable profile."""
    prefix = np.asarray(
        [[0.43, -0.11 + 0.01 * index, 0.014] for index in range(5)],
        dtype=np.float64,
    )
    endpoint = np.asarray((0.58, 0.045, 0.0744), dtype=np.float64)
    lengths = np.full(45, 0.01, dtype=np.float64)
    return prefix, endpoint, lengths


def _synthetic_anchored_suffix(endpoint: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """Return four front-facing strain-relief spans in cable-route order."""
    outward = np.stack([endpoint + np.asarray((0.0, -float(lengths[-index:].sum()), 0.0)) for index in range(1, 5)])
    return np.concatenate((outward[::-1], endpoint[None]), axis=0)


def test_dual_rack_workcell_has_two_open_racks_and_one_t_slot_frame() -> None:
    """Pin cheap collision geometry independently from detailed presentation meshes."""
    boxes = DUAL_RACK_WORKCELL_CFG.boxes
    collidable = tuple(box for box in boxes if box.collidable)
    visible = tuple(box for box in boxes if box.visible)
    profile = tuple(box for box in boxes if box.visible and not box.collidable)
    assert len(boxes) == 130
    assert len(collidable) == 34
    assert len(visible) == 96
    assert len(profile) == 96
    assert len(tuple(box for box in collidable if box.name.startswith("Frame/"))) == 16
    assert all(not box.visible for box in collidable)
    assert all(box.name.startswith("TSlotVisual/") for box in visible)
    assert len(tuple(box for box in collidable if box.name.startswith("Racks/Target/"))) == 9
    assert len(tuple(box for box in collidable if box.name.startswith("Racks/Anchored/"))) == 9
    assert all(all(size > 0.0 for size in box.size_m) for box in boxes)

    # Neither rack collision shell is allowed to place a cuboid through its
    # 36 x 30 mm active connector corridor on the front plane.
    for rack_name, socket in (
        ("Target", DUAL_RACK_TARGET_SOCKET_POSITION_E),
        ("Anchored", DUAL_RACK_ANCHORED_SOCKET_POSITION_E),
    ):
        front = tuple(box for box in collidable if box.name.startswith(f"Racks/{rack_name}/Front"))
        assert len(front) == 4
        front_by_name = {box.name.rsplit("/", 1)[-1]: box for box in front}
        lower_box = front_by_name["FrontBottom"]
        upper_box = front_by_name["FrontTop"]
        left_box = front_by_name["FrontLeft"]
        right_box = front_by_name["FrontRight"]
        assert lower_box.center_m[2] + 0.5 * lower_box.size_m[2] == pytest.approx(socket[2] - 0.015)
        assert upper_box.center_m[2] - 0.5 * upper_box.size_m[2] == pytest.approx(socket[2] + 0.015)
        assert left_box.center_m[0] + 0.5 * left_box.size_m[0] == pytest.approx(socket[0] - 0.018)
        assert right_box.center_m[0] - 0.5 * right_box.size_m[0] == pytest.approx(socket[0] + 0.018)
        for box in front:
            lower = np.asarray(box.center_m) - 0.5 * np.asarray(box.size_m)
            upper = np.asarray(box.center_m) + 0.5 * np.asarray(box.size_m)
            inside_x = lower[0] < socket[0] < upper[0]
            inside_z = lower[2] < socket[2] < upper[2]
            assert not (inside_x and inside_z), box.name

    # Two front/rear T-slot supports touch the lower face of each switch;
    # neither chassis is visually floating.
    for rack_name, socket in (
        ("Target", DUAL_RACK_TARGET_SOCKET_POSITION_E),
        ("Anchored", DUAL_RACK_ANCHORED_SOCKET_POSITION_E),
    ):
        supports = tuple(box for box in collidable if f"RackSupports/{rack_name}" in box.name)
        assert len(supports) == 2
        expected_top = socket[2] + dual_rack_workcell_contract()["switch_socket_local_bounds_m"]["minimum"][2]
        assert all(box.center_m[2] + 0.5 * box.size_m[2] == pytest.approx(expected_top) for box in supports)

    # The outer frame shares one exact grid with the supports.  Side posts
    # touch the rack bounds, front/rear profile envelopes match its depth, and
    # only the top retains 10 mm service clearance.  The lower cable-facing
    # edge is deliberately open.
    by_name = {box.name: box for box in collidable}
    assert "Frame/Base/XFront" not in by_name
    rack_min_x = DUAL_RACK_TARGET_SOCKET_POSITION_E[0] + DUAL_RACK_SWITCH_MIN_SOCKET_LOCAL[0]
    rack_max_x = DUAL_RACK_TARGET_SOCKET_POSITION_E[0] + DUAL_RACK_SWITCH_MAX_SOCKET_LOCAL[0]
    rack_min_y = DUAL_RACK_TARGET_SOCKET_POSITION_E[1] + DUAL_RACK_SWITCH_MIN_SOCKET_LOCAL[1]
    rack_max_y = DUAL_RACK_TARGET_SOCKET_POSITION_E[1] + DUAL_RACK_SWITCH_MAX_SOCKET_LOCAL[1]
    upper_rack_top = (
        max(
            DUAL_RACK_TARGET_SOCKET_POSITION_E[2],
            DUAL_RACK_ANCHORED_SOCKET_POSITION_E[2],
        )
        + DUAL_RACK_SWITCH_MAX_SOCKET_LOCAL[2]
    )
    left = by_name["Frame/Posts/LeftFront"]
    right = by_name["Frame/Posts/RightFront"]
    front = by_name["Frame/Posts/LeftFront"]
    rear = by_name["Frame/Posts/LeftRear"]
    top_left = by_name["Frame/Top/XFrontLeft"]
    top_right = by_name["Frame/Top/XFrontRight"]
    assert rack_min_x - (left.center_m[0] + 0.5 * left.size_m[0]) == pytest.approx(0.0)
    assert (right.center_m[0] - 0.5 * right.size_m[0]) - rack_max_x == pytest.approx(0.0)
    assert front.center_m[1] - 0.5 * front.size_m[1] == pytest.approx(rack_min_y)
    assert rear.center_m[1] + 0.5 * rear.size_m[1] == pytest.approx(rack_max_y)
    assert (top_left.center_m[2] - 0.5 * top_left.size_m[2]) - upper_rack_top == pytest.approx(0.010)
    assert top_right.center_m[2] == pytest.approx(top_left.center_m[2])
    assert top_left.center_m[0] + 0.5 * top_left.size_m[0] == pytest.approx(
        DUAL_RACK_TARGET_SOCKET_POSITION_E[0] - 0.105
    )
    assert top_right.center_m[0] - 0.5 * top_right.size_m[0] == pytest.approx(
        DUAL_RACK_TARGET_SOCKET_POSITION_E[0] + 0.105
    )
    for rack_name in ("Target", "Anchored"):
        front_support = by_name[f"Frame/RackSupports/{rack_name}Front"]
        rear_support = by_name[f"Frame/RackSupports/{rack_name}Rear"]
        assert front_support.center_m[1] == pytest.approx(front.center_m[1])
        assert rear_support.center_m[1] == pytest.approx(rear.center_m[1])


def test_dual_rack_cable_route_is_exact_batched_and_table_clear() -> None:
    """The reset generator and authored default must share one exact rest-length route."""
    prefix, endpoint, lengths = _synthetic_route_inputs()
    suffix = _synthetic_anchored_suffix(endpoint, lengths)
    numpy_route = route_dual_rack_cable_points_numpy(
        prefix,
        endpoint,
        lengths,
        fixed_suffix_points=suffix,
    )
    torch_route = route_dual_rack_cable_points_torch(
        torch.from_numpy(prefix)[None].repeat(3, 1, 1),
        torch.from_numpy(endpoint)[None].repeat(3, 1),
        torch.from_numpy(lengths),
        fixed_suffix_points=torch.from_numpy(suffix)[None].repeat(3, 1, 1),
    )
    assert numpy_route.shape == (46, 3)
    assert tuple(torch_route.shape) == (3, 46, 3)
    assert np.array_equal(numpy_route[:5], prefix)
    assert np.array_equal(numpy_route[-1], endpoint)
    assert np.array_equal(numpy_route[-5:], suffix)
    assert torch.equal(torch_route[:, :5], torch.from_numpy(prefix)[None].repeat(3, 1, 1))
    assert torch.equal(torch_route[:, -1], torch.from_numpy(endpoint)[None].repeat(3, 1))
    assert torch.equal(torch_route[:, -5:], torch.from_numpy(suffix)[None].repeat(3, 1, 1))
    assert np.linalg.norm(np.diff(numpy_route, axis=0), axis=-1) == pytest.approx(lengths, abs=5.0e-8)
    assert torch.linalg.vector_norm(torch.diff(torch_route, dim=1), dim=-1) == pytest.approx(
        torch.from_numpy(lengths)[None].repeat(3, 1), abs=5.0e-8
    )
    assert torch_route == pytest.approx(torch.from_numpy(numpy_route)[None].repeat(3, 1, 1), abs=2.0e-8)
    assert float(numpy_route[:, 2].min()) >= DUAL_RACK_CABLE_TABLE_CENTERLINE_HEIGHT_M - 1.0e-9


def test_dual_rack_cable_fixture_intersection_query_detects_hidden_contacts() -> None:
    """The generator can reject a route crossing the hidden box approximation."""
    safe = np.asarray(((0.50, -0.12, 0.34), (0.62, -0.12, 0.34)))
    collidable = tuple(box for box in DUAL_RACK_WORKCELL_CFG.boxes if box.collidable)
    front_y = next(box.center_m[1] for box in collidable if box.name == "Frame/Posts/LeftFront")
    rear_y = next(box.center_m[1] for box in collidable if box.name == "Frame/Base/XRear")
    open_front = np.asarray(((0.40, front_y, 0.0125), (0.70, front_y, 0.0125)))
    crossing = np.asarray(((0.40, rear_y, 0.0125), (0.70, rear_y, 0.0125)))
    assert dual_rack_cable_workcell_intersections_numpy(safe, cable_radius_m=0.0025) == ()
    assert dual_rack_cable_workcell_intersections_numpy(open_front, cable_radius_m=0.0025) == ()
    assert "Frame/Base/XRear" in dual_rack_cable_workcell_intersections_numpy(
        crossing,
        cable_radius_m=0.0025,
    )
    batched = torch.from_numpy(np.stack((safe, crossing)))
    assert dual_rack_cable_workcell_intersection_mask_torch(
        batched,
        cable_radius_m=0.0025,
    ).tolist() == [False, True]


def test_dual_rack_t_slot_marker_uses_profile_pieces_not_collision_boxes() -> None:
    """NewtonGL gets the same open extrusion presentation that Kit authors."""
    origins = torch.tensor(((0.0, 0.0, 0.0), (2.0, -1.0, 0.5)))
    translations, orientations, scales, indices, environment_ids = dual_rack_t_slot_marker_state(origins)
    visible = tuple(box for box in DUAL_RACK_WORKCELL_CFG.boxes if box.visible)
    assert len(visible) == 96
    assert tuple(translations.shape) == (192, 3)
    assert tuple(orientations.shape) == (192, 4)
    assert tuple(scales.shape) == (192, 3)
    assert torch.equal(indices, torch.zeros(192, dtype=torch.int32))
    assert environment_ids.tolist() == [0] * 96 + [1] * 96
    assert translations[0] == pytest.approx(torch.tensor(visible[0].center_m))
    assert translations[96] == pytest.approx(torch.tensor(visible[0].center_m) + origins[1])
    assert orientations[:, 3] == pytest.approx(torch.ones(192))
    assert torch.all(scales > 0.0)


def test_dual_rack_cable_body_poses_preserve_prefix_and_align_remaining_capsules() -> None:
    prefix, endpoint, lengths = _synthetic_route_inputs()
    suffix = _synthetic_anchored_suffix(endpoint, lengths)
    route = route_dual_rack_cable_points_torch(
        torch.from_numpy(prefix)[None],
        torch.from_numpy(endpoint)[None],
        torch.from_numpy(lengths),
        fixed_suffix_points=torch.from_numpy(suffix)[None],
    )
    free_orientation = math_utils.quat_from_euler_xyz(
        torch.tensor([0.2], dtype=torch.float64),
        torch.tensor([-0.1], dtype=torch.float64),
        torch.tensor([0.35], dtype=torch.float64),
    )
    prefix_rotations = torch.tensor(
        [
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 2**-0.5, 0.0, 2**-0.5],
            [2**-0.5, 0.0, 0.0, 2**-0.5],
            [0.0, 0.0, 2**-0.5, 2**-0.5],
        ],
        dtype=torch.float64,
    )
    poses = dual_rack_cable_body_poses_torch(
        route,
        free_plug_orientation_xyzw=free_orientation,
        prefix_rotations_xyzw=prefix_rotations,
    )
    assert tuple(poses.shape) == (1, 45, 7)
    assert poses[0, :, :3] == pytest.approx(0.5 * (route[0, 1:] + route[0, :-1]), abs=1.0e-12)
    expected_prefix = math_utils.quat_mul(free_orientation.repeat(4, 1), prefix_rotations)
    assert poses[0, :4, 3:7] == pytest.approx(expected_prefix, abs=1.0e-12)
    local_z = torch.tensor((0.0, 0.0, 1.0), dtype=torch.float64).repeat(41, 1)
    aligned = math_utils.quat_apply(poses[0, 4:, 3:7], local_z)
    expected = torch.diff(route[0], dim=0)[4:]
    expected /= torch.linalg.vector_norm(expected, dim=-1, keepdim=True)
    assert aligned == pytest.approx(expected, abs=1.0e-8)


def test_dual_rack_config_contract_and_registration_are_role_explicit() -> None:
    cfg = FrankaRJ45DualRackInsertEnvCfg()
    cfg.validate_config()
    baseline = FrankaRJ45PickInsertEnvCfg()
    assert tuple(cfg.proxy_body_patterns) == DUAL_RACK_ROBOT_PROXY_BODY_PATTERNS
    assert tuple(baseline.proxy_body_patterns) == RJ45_GRASP_PROXY_BODY_PATTERNS
    assert tuple(cfg.rj45_entry_body_patterns) == PICK_INSERT_RJ45_ENTRY_BODY_PATTERNS
    assert cfg.reset_dataset_path.endswith("franka_rj45_dual_rack_insert/reset_dataset.pt")
    contract = dual_rack_reset_dataset_task_contract(cfg)
    assert contract["task_variant"] == "franka-rj45-dual-rack-insert"
    assert contract["dual_rack"]["physical_socket_count"] == 2
    assert contract["dual_rack"]["physical_plug_count"] == 2
    assert contract["dual_rack"]["workcell"] == dual_rack_workcell_contract()
    assert contract["coupler"]["proxy"]["body_patterns"] == DUAL_RACK_ROBOT_PROXY_BODY_PATTERNS

    spec = gym.spec("IsaacContrib-Franka-RJ45-Dual-Rack-Insert")
    assert spec.entry_point.endswith("dual_rack_env:FrankaRJ45DualRackInsertEnv")
    assert "dual_rack_env_cfg:FrankaRJ45DualRackInsertEnvCfg" in spec.kwargs["env_cfg_entry_point"]


def test_dual_rack_policy_interface_adds_only_anchored_end_state() -> None:
    cfg = FrankaRJ45DualRackInsertEnvCfg()
    policy_terms = tuple(
        name for name, value in vars(cfg.observations.policy).items() if isinstance(value, ObservationTermCfg)
    )
    assert policy_terms[-3:] == ("anchored_socket_pose", "anchored_plug_pose", "anchored_cable_endpoint_error")
    inherited_width = 135
    anchored_width = 7 + 7 + 3
    assert inherited_width + anchored_width == 152
    assert inherited_width + anchored_width + 3 == 155
    assert cfg.observations.policy.anchored_socket_pose.func is mdp.anchored_socket_pose_obs
    assert cfg.observations.policy.anchored_plug_pose.func is mdp.anchored_plug_pose_obs
    assert cfg.observations.policy.anchored_cable_endpoint_error.func is mdp.anchored_cable_endpoint_error_obs
    assert tuple(vars(cfg.actions)) == ("arm_action", "gripper_action")
    runner = FrankaRJ45DualRackInsertPPORunnerCfg()
    assert runner.experiment_name == "franka_rj45_dual_rack_insert"
    assert runner.obs_groups == {"actor": ["policy"], "critic": ["policy", "privileged"]}


def test_dual_rack_play_mode_replays_frozen_dataset_rows() -> None:
    cfg = FrankaRJ45DualRackInsertEnvCfg()
    contract = dual_rack_reset_dataset_task_contract(cfg)
    cfg.play_mode()
    cfg.validate_config()
    assert cfg.scene.num_envs == 1
    assert cfg.reset_source == "dataset"
    assert cfg.reset_dataset_sampling_mode == "uniform"
    assert cfg.curriculum_freeze is True
    assert cfg.curriculum is not None
    assert dual_rack_reset_dataset_task_contract(cfg) == contract


def test_dual_rack_anchored_observations_and_termination_are_finite_and_local() -> None:
    socket = torch.tensor([[0.58, 0.07, 0.075, 0.0, 0.0, 0.0, 1.0]])
    plug = torch.tensor([[0.58, 0.07, 0.075, 0.0, 0.0, 2**-0.5, 2**-0.5]])
    endpoint = torch.tensor([[0.58, 0.046, 0.074]])
    target = torch.tensor([[0.58, 0.045, 0.074]])
    env = SimpleNamespace(
        cfg=SimpleNamespace(anchored_cable_endpoint_tolerance_m=0.003),
        anchored_socket_pose_e=lambda: socket,
        anchored_plug_pose_e=lambda: plug,
        anchored_cable_endpoint_position_e=lambda: endpoint,
        anchored_cable_target_position_e=lambda: target,
    )
    assert mdp.anchored_socket_pose_obs(env) == pytest.approx(socket)
    normalized_plug = mdp.anchored_plug_pose_obs(env)
    assert torch.linalg.vector_norm(normalized_plug[:, 3:7], dim=-1) == pytest.approx(torch.ones(1))
    assert mdp.anchored_cable_endpoint_error_obs(env) == pytest.approx(torch.tensor([[0.001, 0.0, 0.0]]), abs=3.0e-9)
    assert mdp.anchored_cable_disconnected(env).tolist() == [False]
    endpoint[:, 0] += 0.004
    assert mdp.anchored_cable_disconnected(env).tolist() == [True]


def test_gb300_workcell_registers_native_sn2201_jacks_without_synthetic_visuals() -> None:
    """The SN2201 CAD supplies every port visual while hidden RJ45 geometry owns contact."""
    expected_targets = (
        (0.4329552, 0.3413757, 0.14),
        (0.4749552, 0.3413757, 0.14),
        (0.5231752, 0.3413757, 0.14),
        (0.5651752, 0.3413757, 0.14),
        (0.4329552, 0.3413757, 0.15412),
        (0.4749552, 0.3413757, 0.15412),
        (0.5231752, 0.3413757, 0.15412),
        (0.5651752, 0.3413757, 0.15412),
    )
    assert len(GB300_TARGET_SOCKET_POSITIONS_E) == 8
    assert len(set(GB300_TARGET_SOCKET_POSITIONS_E)) == 8
    assert np.asarray(GB300_TARGET_SOCKET_POSITIONS_E) == pytest.approx(np.asarray(expected_targets), abs=1.0e-9)
    assert pytest.approx((0.3769552, 0.3413757, 0.14), abs=1.0e-9) == GB300_ANCHORED_SOCKET_POSITION_E
    assert len(GB300_TARGET_TASK_TRANSLATIONS) == 8
    assert GB300_WORKCELL_CFG.presentation_kind == "gb300"
    collidable = tuple(box for box in GB300_WORKCELL_CFG.boxes if box.collidable)
    visible = tuple(box for box in GB300_WORKCELL_CFG.boxes if box.visible)
    assert len(collidable) == 6
    assert len(visible) == 0
    assert all(not box.visible for box in collidable)
    assert not any("CandidatePort" in box.name or "AnchoredPort" in box.name for box in GB300_WORKCELL_CFG.boxes)

    contract = gb300_workcell_contract()
    assert contract["target_socket_candidates_e_m"] == GB300_TARGET_SOCKET_POSITIONS_E
    assert contract["active_collision"] == "one-resettable-hidden-exact-rj45-sdf-at-selected-native-jack"
    assert contract["inactive_ports"] == "native-simready-sn2201-cad-only-no-sdf"
    assert contract["anchored_socket_position_e_m"] == GB300_ANCHORED_SOCKET_POSITION_E
    registration = contract["native_cad_port_registration"]
    assert registration["source_mesh"].endswith("/tn__0000_NV_MSN2201TOR_08132024_Ze0_Merged")
    assert registration["authored_port_count"] == 48
    assert registration["authored_layout"] == "two-rows-of-24-1gbase-t-rj45-jacks"
    assert registration["registration"].endswith("no-added-port-visual")
    presentation_transform = contract["presentation_transform"]
    assert presentation_transform["composition"] == (
        "placement-parent-plus-payload-child-preserves-simready-root-xform"
    )
    assert presentation_transform["simready_root_translation_m"] == GB300_SIMREADY_ROOT_TRANSLATION_M
    assert presentation_transform["simready_root_rotation_xyz_deg"] == GB300_SIMREADY_ROOT_ROTATION_XYZ_DEG
    assert presentation_transform["simready_composed_bounds_min_m"] == GB300_SIMREADY_COMPOSED_BOUNDS_MIN_M
    assert presentation_transform["simready_composed_bounds_max_m"] == GB300_SIMREADY_COMPOSED_BOUNDS_MAX_M
    assert presentation_transform["studio_floor_height_m"] == GB300_STUDIO_FLOOR_HEIGHT_M
    assert contract["simready_asset"] == gb300_asset_contract()


def test_gb300_simready_provenance_is_pinned_but_not_a_physics_dependency() -> None:
    contract = gb300_asset_contract()
    assert contract == {
        "repository": "nvidia/simready-dsx",
        "revision": GB300_SIMREADY_REVISION,
        "relative_path": "GB300/simready_usd/payloads/external.usd",
        "file_sha256": GB300_SIMREADY_EXTERNAL_USD_SHA256,
        "file_size_bytes": GB300_SIMREADY_EXTERNAL_USD_SIZE,
        "license": GB300_SIMREADY_LICENSE,
        "role": "render-only-no-physics-or-reset-state-dependency",
    }


def test_gb300_config_contract_registration_and_policy_abi() -> None:
    cfg = FrankaRJ45Gb300InsertEnvCfg()
    cfg.validate_config()
    assert tuple(cfg.task_translation) == GB300_DEFAULT_TARGET_TASK_TRANSLATION
    assert tuple(cfg.task_rotation_xyzw) == (0.0, 0.0, 0.0, 1.0)
    assert tuple(cfg.scene.robot.init_state.pos) == (0.0, 0.0, 0.0)
    assert tuple(cfg.scene.robot.init_state.rot) == (0.0, 0.0, 0.0, 1.0)
    assert tuple(cfg.rj45_entry_body_patterns) == GB300_RJ45_ENTRY_BODY_PATTERNS
    assert cfg.scene.table.spawn is None
    assert cfg.scene.table_contact_surface.spawn is None
    assert tuple(cfg.scene.ground.init_state.pos) == (0.0, 0.0, GB300_STUDIO_FLOOR_HEIGHT_M)
    assert tuple(cfg.scene.ground.spawn.color) == (0.96, 0.97, 0.99)
    assert tuple(cfg.socket_position_lower) == tuple(min(values) for values in zip(*GB300_TARGET_TASK_TRANSLATIONS))
    assert tuple(cfg.socket_position_upper) == tuple(max(values) for values in zip(*GB300_TARGET_TASK_TRANSLATIONS))
    assert cfg.reset_dataset_path.endswith("franka_rj45_gb300_insert/reset_dataset.pt")
    contract = gb300_reset_dataset_task_contract(cfg)
    assert contract["task_variant"] == "franka-rj45-gb300-insert"
    assert contract["gb300"]["physical_plug_count"] == 2
    assert contract["gb300"]["active_physical_socket_count_per_world"] == 2
    assert contract["gb300"]["inactive_candidate_ports"] == "native-simready-sn2201-cad-only-no-collision"
    assert contract["static_scene"]["table_spawn"] is None
    assert contract["static_scene"]["table_contact_spawn"] is None
    assert contract["pick_insert"]["full_pick_diversity"]["minimum_unique_socket_rows"] == 8
    assert contract["pick_insert"]["full_pick_diversity"]["minimum_unique_plug_rows"] == 3000

    spec = gym.spec("IsaacContrib-Franka-RJ45-GB300-Insert")
    assert spec.entry_point.endswith("gb300_env:FrankaRJ45Gb300InsertEnv")
    assert "gb300_env_cfg:FrankaRJ45Gb300InsertEnvCfg" in spec.kwargs["env_cfg_entry_point"]
    runner = FrankaRJ45Gb300InsertPPORunnerCfg()
    assert runner.experiment_name == "franka_rj45_gb300_insert"
    assert runner.obs_groups == {"actor": ["policy"], "critic": ["policy", "privileged"]}

    cfg.task_rotation_xyzw = (0.0, 0.0, 1.0, 0.0)
    with pytest.raises(ValueError, match="identity task rotation"):
        cfg.validate_config()


def test_gb300_studio_has_eight_consistently_front_facing_racks() -> None:
    presentation = gb300_presentation_contract()
    assert presentation["rack_count"] == 8
    assert presentation["rack_spacing_x_m"] == pytest.approx(0.6569000133217895)
    translations = presentation["rack_translations_e_m"]
    assert len(translations) == 8
    assert translations[0] == pytest.approx((0.58, 0.6833, -1.89475))
    assert [position[0] for position in translations] == pytest.approx(
        [0.58 + 0.6569000133217895 * index for index in range(8)]
    )
    assert len({position[1:] for position in translations}) == 1
    rotations = presentation["rack_rotations_xyzw"]
    assert np.asarray(rotations) == pytest.approx(np.asarray(((0.0, 0.0, -(2**-0.5), 2**-0.5),) * 8), abs=1.0e-12)
    assert presentation["asset_composition"] == ("placement-parent-plus-payload-child-preserves-simready-root-xform")
    assert presentation["table"] == "absent"
    assert presentation["franka_pedestal"]["center_e_m"] == pytest.approx((0.0, 0.0, 0.5 * GB300_STUDIO_FLOOR_HEIGHT_M))
    assert presentation["franka_pedestal"]["size_m"] == pytest.approx((0.38, 0.38, -GB300_STUDIO_FLOOR_HEIGHT_M))
    assert presentation["cables"]["visible_total"] == 1
    assert presentation["cables"]["physical_task_cables"] == 1
    assert presentation["cables"]["render_only_drops"] == 0
    assert presentation["cables"]["free_visible_plug_ends"] == 1
    assert presentation["floor"] == {
        "color": (0.965, 0.975, 0.985),
        "roughness": 0.10,
        "material": "UsdPreviewSurface",
    }
    assert presentation["backwall"]["size_m"] == (6.40, 0.08, 3.20)
    assert presentation["lighting"]["style"] == "neutral-key-cool-fill-and-rim"
    assert presentation["physics_effect"] == "none-render-only-racks-and-franka-pedestal"


def test_gb300_newton_gl_fallback_is_environment_major_and_shape_only() -> None:
    origins = torch.tensor(((0.0, 0.0, 0.0), (1.0, -2.0, 0.5)))
    translations, orientations, scales, indices, environment_ids = gb300_marker_state(origins)
    boxes = GB300_WORKCELL_CFG.boxes
    assert tuple(translations.shape) == (2 * len(boxes), 3)
    assert tuple(orientations.shape) == (2 * len(boxes), 4)
    assert tuple(scales.shape) == (2 * len(boxes), 3)
    assert tuple(indices.shape) == (2 * len(boxes),)
    assert environment_ids.tolist() == [0] * len(boxes) + [1] * len(boxes)
    assert translations[0] == pytest.approx(torch.tensor(boxes[0].center_m))
    assert translations[len(boxes)] == pytest.approx(torch.tensor(boxes[0].center_m) + origins[1])
    assert orientations[:, 3] == pytest.approx(torch.ones(2 * len(boxes)))
    assert torch.all(scales > 0.0)
