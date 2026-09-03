# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration and retargeting tests for the standalone Sharpa cable task."""

from __future__ import annotations

import math
from xml.etree import ElementTree

import gymnasium as gym
import pytest
import torch
import yaml
from isaaclab_newton.envs.mdp.actions.newton_ik_actions_cfg import NewtonInverseKinematicsActionCfg
from isaaclab_newton.ik.newton_ik_objectives_cfg import NewtonIKJointPostureObjectiveCfg, NewtonIKPoseObjectiveCfg
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, VBDSolverCfg
from scipy.spatial.transform import Rotation

import isaaclab.envs.mdp as mdp

from isaaclab_contrib.coupling import CouplerProxyCfg

from isaaclab_tasks.contrib.rizon_sharpa_cable import RIZON_SHARPA_CABLE_TASK_ID
from isaaclab_tasks.contrib.rizon_sharpa_cable.actions import update_absolute_pose_with_dropout
from isaaclab_tasks.contrib.rizon_sharpa_cable.cable import connector_render_parts, socket_render_part
from isaaclab_tasks.contrib.rizon_sharpa_cable.cable_cfg import (
    CABLE_CONNECTOR_RIGID_SPAN_M,
    CABLE_FLEX_SEGMENT_LENGTH_M,
    CABLE_INITIAL_LATERAL_OFFSET_M,
    CABLE_INITIAL_VERTICAL_SPAN_M,
    CABLE_LENGTH_M,
    CABLE_SEGMENT_COUNT,
    hanging_cable_positions,
)
from isaaclab_tasks.contrib.rizon_sharpa_cable.env_cfg import (
    CABLE_ANCHOR_POSITION_E,
    CABLE_FREE_END_POSITION_E,
    CAMERA_EYE_E,
    CAMERA_LOOKAT_E,
    INSERTION_SOCKET_POSITION_E,
    INSERTION_SOCKET_ROTATION_XYZW,
    ROBOT_BASE_POSITION_E,
    STAND_CENTER_E,
    STAND_SIZE_M,
    XR_ANCHOR_POSITION_E,
    XR_ANCHOR_ROTATION_XYZW,
    RizonSharpaCableEnvCfg,
    rizon_sharpa_cable_contract,
)
from isaaclab_tasks.contrib.rizon_sharpa_cable.robot_asset import (
    RIZON_SHARPA_END_EFFECTOR_BODY_NAME,
    RIZON_SHARPA_NATIVE_PALM_BODY_NAME,
    RIZON_SHARPA_RIGHT_HAND_JOINT_NAMES,
    RIZON_SHARPA_RIGHT_HAND_LIMITS_RAD,
)
from isaaclab_tasks.contrib.rizon_sharpa_cable.sharpa_hand_retargeting import (
    ISAAC_TELEOP_REVISION,
    SHARPA_DEXPILOT_CONFIG_SHA256,
    SHARPA_DEXPILOT_URDF_SHA256,
    SHARPA_HANDTRACKING_TO_BASELINK,
    SHARPA_OPENXR_TO_CANONICAL_PALM_RPY_DEG,
    SHARPA_URDF_REVISION,
    sharpa_dexpilot_config_path,
    sharpa_dexpilot_urdf_path,
    sharpa_hand_retargeting_contract,
)
from isaaclab_tasks.contrib.rizon_sharpa_cable.showroom import (
    GB300_SHOWROOM_RACK_COUNT,
    GB300_SHOWROOM_RACK_SPACING_X_M,
    GB300_SHOWROOM_RACK_TRANSLATIONS_E,
    rizon_sharpa_showroom_contract,
)
from isaaclab_tasks.contrib.rizon_sharpa_cable.teleop import (
    SHARPA_THUMB_RETARGETING_GAINS,
    apply_sharpa_thumb_retargeting_gain,
)


def test_task_is_registered_as_a_standalone_environment() -> None:
    """The clean scene must not remain a mode of the Franka package."""
    spec = gym.spec(RIZON_SHARPA_CABLE_TASK_ID)
    assert spec.entry_point == "isaaclab_tasks.contrib.rizon_sharpa_cable.env:RizonSharpaCableEnv"
    assert spec.kwargs["env_cfg_entry_point"].startswith("isaaclab_tasks.contrib.rizon_sharpa_cable.")
    with pytest.raises(gym.error.Error):
        gym.spec("IsaacContrib-Rizon-Sharpa-GB300-XR-Teleop")


def test_scene_contains_one_half_meter_top_anchored_cable() -> None:
    """Pin the physical task topology and cable discretization."""
    cfg = RizonSharpaCableEnvCfg()
    positions = torch.tensor(hanging_cable_positions(), dtype=torch.float64)
    assert positions.shape == (CABLE_SEGMENT_COUNT + 1, 3)
    span_lengths = torch.linalg.vector_norm(torch.diff(positions, dim=0), dim=-1)
    assert span_lengths[0] == pytest.approx(CABLE_CONNECTOR_RIGID_SPAN_M)
    assert span_lengths[1:] == pytest.approx(
        torch.full((CABLE_SEGMENT_COUNT - 1,), CABLE_FLEX_SEGMENT_LENGTH_M, dtype=torch.float64)
    )
    assert span_lengths.sum() == pytest.approx(CABLE_LENGTH_M)
    assert positions[-1, 0] - positions[0, 0] == pytest.approx(CABLE_INITIAL_LATERAL_OFFSET_M)
    assert positions[-1, 2] - positions[0, 2] == pytest.approx(CABLE_INITIAL_VERTICAL_SPAN_M)
    assert torch.linalg.vector_norm(positions[-1] - positions[0]) == pytest.approx(CABLE_LENGTH_M)
    assert cfg.scene.cable.spawn.positions == hanging_cable_positions()
    assert tuple(cfg.scene.cable.init_state.pos) == CABLE_FREE_END_POSITION_E
    assert cfg.scene.cable.spawn.insertion_target_position_e == INSERTION_SOCKET_POSITION_E
    assert cfg.scene.cable.spawn.insertion_target_rotation_xyzw == INSERTION_SOCKET_ROTATION_XYZW
    anchor_delta = torch.tensor(CABLE_ANCHOR_POSITION_E) - torch.tensor(CABLE_FREE_END_POSITION_E)
    assert torch.linalg.vector_norm(anchor_delta) == pytest.approx(CABLE_LENGTH_M)

    contract = rizon_sharpa_cable_contract()["scene"]
    assert contract["cable_count"] == 1
    assert contract["connector_rigid_span_m"] == pytest.approx(CABLE_CONNECTOR_RIGID_SPAN_M)
    assert contract["flex_segment_length_m"] == pytest.approx(CABLE_FLEX_SEGMENT_LENGTH_M)
    assert contract["floating_socket_position_e_m"] == INSERTION_SOCKET_POSITION_E
    assert contract["floating_socket_rotation_xyzw"] == INSERTION_SOCKET_ROTATION_XYZW
    assert contract["props"] == (
        "glossy-white-ground",
        "single-pedestal",
        "small-cable-anchor",
        "white-backwall",
    )
    assert contract["gb300"]["rack_count"] == 8
    assert contract["gb300"]["physics_effect"] == "none-render-only-racks-wall-and-lights"
    assert contract["franka"] == "absent"
    assert CABLE_FREE_END_POSITION_E[2] == pytest.approx(1.60)
    assert CABLE_ANCHOR_POSITION_E[2] == pytest.approx(1.60 + CABLE_INITIAL_VERTICAL_SPAN_M)
    assert INSERTION_SOCKET_POSITION_E[2] == pytest.approx(1.60)
    assert CAMERA_EYE_E[2] == pytest.approx(2.30)
    assert CAMERA_LOOKAT_E[2] == pytest.approx(1.50)
    assert ROBOT_BASE_POSITION_E[2] == pytest.approx(0.50)
    assert STAND_CENTER_E[2] == pytest.approx(0.25)
    assert STAND_SIZE_M[2] == pytest.approx(0.50)
    assert ROBOT_BASE_POSITION_E[2] == pytest.approx(STAND_CENTER_E[2] + 0.5 * STAND_SIZE_M[2])


def test_connector_uses_canonical_embedded_strain_relief_and_socket_is_exact_mesh() -> None:
    """The connector and strain relief are one body, with the first flex joint behind them."""
    plug, latch = connector_render_parts(CABLE_CONNECTOR_RIGID_SPAN_M)
    del latch
    plug_points = torch.tensor(plug.points, dtype=torch.float64)
    attachment_z = -0.5 * CABLE_CONNECTOR_RIGID_SPAN_M
    first_flexible_joint_z = 0.5 * CABLE_CONNECTOR_RIGID_SPAN_M
    # The canonical source centerline begins inside the housing.  Expressed in
    # segment-zero's COM frame, the attachment must therefore be strictly
    # between the plug's two axial faces rather than moved to its rear face.
    assert plug_points[:, 2].min() < attachment_z < plug_points[:, 2].max()
    # The cable's first deformable joint is outside the rear housing. Because
    # the plug shapes and this whole span share one body, no connector-relative
    # translation or rotation degree of freedom exists.
    assert first_flexible_joint_z - plug_points[:, 2].max() > 0.008
    assert plug_points[:, 2].min() < -0.03

    socket = socket_render_part()
    socket_points = torch.tensor(socket.points, dtype=torch.float64)
    assert socket_points.shape[0] > 0
    assert socket.face_vertex_counts
    assert socket.face_vertex_indices
    assert torch.all(torch.isfinite(socket_points))


def test_showroom_has_eight_gapless_front_facing_racks() -> None:
    """The presentation restores the requested polished GB300 row without physics."""
    contract = rizon_sharpa_showroom_contract()
    assert GB300_SHOWROOM_RACK_COUNT == 8
    assert len(GB300_SHOWROOM_RACK_TRANSLATIONS_E) == 8
    for first, second in zip(GB300_SHOWROOM_RACK_TRANSLATIONS_E, GB300_SHOWROOM_RACK_TRANSLATIONS_E[1:]):
        assert second[0] - first[0] == pytest.approx(GB300_SHOWROOM_RACK_SPACING_X_M)
        assert second[1:] == pytest.approx(first[1:])
    assert contract["rack_rotation_xyzw"] == pytest.approx((0.0, 0.0, -math.sqrt(0.5), math.sqrt(0.5)))
    assert contract["floor"]["height_m"] == 0.0
    assert contract["floor"]["roughness"] == pytest.approx(0.10)
    assert contract["pedestal"] == "physical-single-cuboid-authored-by-scene"


def test_newton_ik_and_dexterous_hand_form_one_29_dimensional_command() -> None:
    """The AVP wrist and every Sharpa joint have one stable action slot."""
    cfg = RizonSharpaCableEnvCfg()
    palm = cfg.actions.right_palm
    assert isinstance(palm, NewtonInverseKinematicsActionCfg)
    assert palm.controller.optimizer == "lm"
    assert palm.controller.jacobian_mode == "analytic"
    assert palm.controller.iterations == 4
    pose_objectives = [objective for objective in palm.objectives if isinstance(objective, NewtonIKPoseObjectiveCfg)]
    assert len(pose_objectives) == 1
    assert pose_objectives[0].body_name == RIZON_SHARPA_END_EFFECTOR_BODY_NAME == "r_palm_ctrl"
    assert pose_objectives[0].command_type == "pose"
    assert pose_objectives[0].use_relative_mode is False
    assert pose_objectives[0].body_offset_pos == (0.0, 0.0, 0.0)
    assert pose_objectives[0].body_offset_rot == (0.0, 0.0, 0.0, 1.0)
    posture_objectives = [
        objective for objective in palm.objectives if isinstance(objective, NewtonIKJointPostureObjectiveCfg)
    ]
    assert len(posture_objectives) == 1
    assert posture_objectives[0].joint_names == ["joint1", "joint2", "joint3", "joint4"]
    assert posture_objectives[0].target_positions == (0.0, -0.698, 0.0, 1.571)
    assert posture_objectives[0].weight == pytest.approx(0.01)
    hand = cfg.actions.right_hand
    assert isinstance(hand, mdp.JointPositionActionCfg)
    assert hand.joint_names == list(RIZON_SHARPA_RIGHT_HAND_JOINT_NAMES)
    assert hand.preserve_order is True
    assert hand.use_default_offset is False
    assert hand.clip == dict(zip(RIZON_SHARPA_RIGHT_HAND_JOINT_NAMES, RIZON_SHARPA_RIGHT_HAND_LIMITS_RAD, strict=True))
    control = rizon_sharpa_cable_contract()["control"]
    assert control["action_dim"] == 29
    assert control["pose_tracking"] == "absolute-position-and-orientation; dropout-hold"
    assert control["orientation"] == "openxr-wrist-to-canonical-r-palm-ctrl-absolute"
    assert control["palm_control_frame"] == "r_palm_ctrl"
    assert control["tracker_offsets_rpy_deg"] == SHARPA_OPENXR_TO_CANONICAL_PALM_RPY_DEG
    assert control["thumb_retargeting_gains"] == SHARPA_THUMB_RETARGETING_GAINS
    thumb = cfg.scene.robot.actuators["right_thumb"]
    fingers = cfg.scene.robot.actuators["right_fingers"]
    assert thumb.joint_names_expr == ["right_thumb_.*"]
    assert thumb.joint_effort_limit == pytest.approx(100.0)
    assert thumb.stiffness == pytest.approx(500.0)
    assert thumb.damping == pytest.approx(100.0)
    assert fingers.joint_names_expr == ["right_(?:index|middle|ring|pinky)_.*"]
    assert fingers.joint_effort_limit == pytest.approx(3.3)
    assert fingers.stiffness == pytest.approx(24.0)
    assert fingers.damping == pytest.approx(1.2)
    assert control["thumb_actuator"] == {
        "effort_limit_n_m": 100.0,
        "stiffness_n_m_rad": 500.0,
        "damping_n_m_s_rad": 100.0,
    }


def test_sharpa_dexpilot_assets_and_provenance_are_pinned() -> None:
    """Use NVIDIA's official Sharpa DexPilot contract and standalone URDF exactly."""
    config_path = sharpa_dexpilot_config_path()
    urdf_path = sharpa_dexpilot_urdf_path()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))["retargeting"]
    assert config["type"] == "DexPilot"
    assert config["wrist_link_name"] == RIZON_SHARPA_NATIVE_PALM_BODY_NAME
    assert tuple(config["target_joint_names"]) == RIZON_SHARPA_RIGHT_HAND_JOINT_NAMES
    assert config["low_pass_alpha"] == pytest.approx(0.2)
    assert config["scaling_factor"] == pytest.approx(1.2)

    urdf_root = ElementTree.parse(urdf_path).getroot()
    urdf_joints = {joint.get("name") for joint in urdf_root.iter("joint")}
    assert set(RIZON_SHARPA_RIGHT_HAND_JOINT_NAMES) <= urdf_joints
    assert urdf_root.find(f".//link[@name='{RIZON_SHARPA_NATIVE_PALM_BODY_NAME}']") is not None

    contract = sharpa_hand_retargeting_contract()
    assert contract["method"] == "DexPilot"
    assert contract["implementation_revision"] == ISAAC_TELEOP_REVISION
    assert contract["urdf_revision"] == SHARPA_URDF_REVISION
    assert contract["config_sha256"] == SHARPA_DEXPILOT_CONFIG_SHA256
    assert contract["urdf_sha256"] == SHARPA_DEXPILOT_URDF_SHA256
    assert contract["handtracking_to_baselink"] == SHARPA_HANDTRACKING_TO_BASELINK
    assert contract["finger_joint_names"] == RIZON_SHARPA_RIGHT_HAND_JOINT_NAMES


def test_openxr_to_canonical_palm_offset_matches_upstream_geometry() -> None:
    """The wrist offset is derived from official Sharpa axes, not hand tuning."""
    openxr_to_native = torch.tensor(SHARPA_HANDTRACKING_TO_BASELINK, dtype=torch.float64).reshape(3, 3).numpy()
    native_to_canonical = Rotation.from_euler("xyz", (1.54660563, -1.36673426, 1.54598752), degrees=False).as_matrix()
    expected = openxr_to_native.T @ native_to_canonical
    configured = Rotation.from_euler("XYZ", SHARPA_OPENXR_TO_CANONICAL_PALM_RPY_DEG, degrees=True).as_matrix()

    assert configured == pytest.approx(expected, abs=1.0e-9)


def test_sharpa_pipeline_uses_dexpilot_on_raw_hand_tracking() -> None:
    """Finger IK must follow NVIDIA's raw-hands topology, not the wrist's world transform."""
    from isaaclab_tasks.contrib.rizon_sharpa_cable.teleop import build_rizon_sharpa_teleop_pipeline

    pipeline = build_rizon_sharpa_teleop_pipeline()
    output_selector = pipeline.output_mapping["action"]
    reorder_subgraph = output_selector.module
    hand_subgraph = reorder_subgraph._input_connections["right_hand"].module
    palm_subgraph = reorder_subgraph._input_connections["right_palm"].module

    assert type(hand_subgraph._target_module).__name__ == "SharpaThumbDexHandRetargeter"
    assert type(hand_subgraph._input_connections["hand_right"].module).__name__ == "HandsSource"
    assert type(palm_subgraph._input_connections["hand_right"].module._target_module).__name__ == "HandTransform"
    palm_cfg = palm_subgraph._target_module._config
    assert (
        palm_cfg.target_offset_roll,
        palm_cfg.target_offset_pitch,
        palm_cfg.target_offset_yaw,
    ) == pytest.approx(SHARPA_OPENXR_TO_CANONICAL_PALM_RPY_DEG)


def test_coupler_has_only_robot_cable_and_one_directed_hand_proxy() -> None:
    """Keep ownership identical to the Waterhose-v2 two-entry pattern."""
    cfg = RizonSharpaCableEnvCfg()
    physics = cfg.sim.physics
    assert isinstance(physics, NewtonCfg)
    assert physics.use_cuda_graph is False
    assert cfg.sim.dt == pytest.approx(1.0 / 60.0)
    assert physics.num_substeps == 1
    assert physics.collision_decimation == 0
    assert cfg.decimation == 1
    assert isinstance(physics.solver_cfg, CouplerProxyCfg)
    assert physics.solver_cfg.iterations == 1
    assert len(physics.solver_cfg.entries) == 2
    robot, cable = physics.solver_cfg.entries
    assert isinstance(robot.solver_cfg, MJWarpSolverCfg)
    assert isinstance(cable.solver_cfg, VBDSolverCfg)
    assert robot.solver_cfg.ls_iterations == 10
    assert cable.solver_cfg.iterations == 10
    assert robot.name == "rizon_sharpa"
    assert cable.name == "hanging_cable"
    assert len(cable.bodies) == 2
    assert len(physics.solver_cfg.proxies) == 1
    proxy = physics.solver_cfg.proxies[0]
    assert (proxy.source, proxy.destination, proxy.mode) == ("rizon_sharpa", "hanging_cable", "staggered")
    assert proxy.proxy_relaxation == pytest.approx(0.0)
    assert proxy.mass_scale == pytest.approx(1_000.0)
    assert proxy.collide_interval == 1
    contract = rizon_sharpa_cable_contract()["physics"]
    assert contract["proxy"] == "complete-right-hand-to-one-plug-and-one-cable"
    assert contract["proxy_feedback_relaxation"] == pytest.approx(0.0)
    assert contract["proxy_mass_scale"] == pytest.approx(1_000.0)
    assert contract["connector_attachment"] == "same-body-rigid-strain-relief;first-flex-joint-behind-plug"


def test_reset_is_explicit_and_xr_does_not_autostart() -> None:
    """Operator start/stop/reset controls remain authoritative."""
    cfg = RizonSharpaCableEnvCfg()
    assert cfg.events.reset_scene.func is mdp.reset_scene_to_default
    assert cfg.events.reset_scene.params == {"reset_joint_targets": True}
    assert cfg.isaac_teleop.teleoperation_active_default is False
    assert cfg.isaac_teleop.target_frame_prim_path.endswith("/Robot/base_link")
    assert tuple(cfg.xr.anchor_pos) == XR_ANCHOR_POSITION_E == pytest.approx((-0.137, -0.083, 0.0))
    assert tuple(cfg.xr.anchor_rot) == XR_ANCHOR_ROTATION_XYZW == (0.0, 0.0, 0.0, 1.0)
    assert cfg.sim.default_visualizer_cfg.eye == CAMERA_EYE_E
    assert cfg.sim.default_visualizer_cfg.lookat == CAMERA_LOOKAT_E


def test_absolute_pose_tracking_co_locates_robot_and_debug_hand_and_holds_dropout() -> None:
    """Absolute XR translation must not retain the former acquisition offset."""
    current = torch.tensor([[0.42, -0.08, 0.91, 0.0, 0.0, 0.0, 1.0]])
    command = torch.tensor([[0.18, 0.24, 0.76, 0.0, 0.0, 0.0, -2.0]])

    target, valid = update_absolute_pose_with_dropout(command, current)

    assert target[:, :3] == pytest.approx(command[:, :3])
    assert target[:, 3:] == pytest.approx(torch.tensor([[0.0, 0.0, 0.0, 1.0]]))
    assert valid.tolist() == [True]

    invalid = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    held, valid = update_absolute_pose_with_dropout(invalid, target)
    assert held == pytest.approx(target)
    assert valid.tolist() == [False]


def test_thumb_gain_is_independent_and_joint_limit_safe() -> None:
    """Only thumb channels receive extra travel and every channel remains legal."""
    names = RIZON_SHARPA_RIGHT_HAND_JOINT_NAMES
    limits = RIZON_SHARPA_RIGHT_HAND_LIMITS_RAD
    commands = torch.full((len(names),), 0.1, dtype=torch.float64).numpy()
    commands[0] = 10.0

    remapped = apply_sharpa_thumb_retargeting_gain(commands, names, limits)

    assert remapped[0] == pytest.approx(limits[0][1])
    assert SHARPA_THUMB_RETARGETING_GAINS["right_thumb_CMC_AA"] == pytest.approx(1.0)
    assert SHARPA_THUMB_RETARGETING_GAINS["right_thumb_MCP_AA"] == pytest.approx(1.0)
    assert SHARPA_THUMB_RETARGETING_GAINS["right_thumb_CMC_FE"] > 1.0
    assert SHARPA_THUMB_RETARGETING_GAINS["right_thumb_MCP_FE"] > 1.0
    assert SHARPA_THUMB_RETARGETING_GAINS["right_thumb_IP"] > 1.0
    for index, name in enumerate(names):
        if name in SHARPA_THUMB_RETARGETING_GAINS:
            expected = min(
                max(commands[index] * SHARPA_THUMB_RETARGETING_GAINS[name], limits[index][0]),
                limits[index][1],
            )
            assert remapped[index] == pytest.approx(expected)
        else:
            assert remapped[index] == pytest.approx(commands[index])
