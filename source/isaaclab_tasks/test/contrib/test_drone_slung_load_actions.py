# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for drone-only slung-load actuation."""

import math
from types import SimpleNamespace

import pytest
import torch

from isaaclab.utils.math import quat_apply

from isaaclab_tasks.contrib.drone_slung_load.mdp.actions import (
    CollectiveThrustBodyRateAction,
    CollectiveThrustBodyRateActionCfg,
)
from isaaclab_tasks.contrib.drone_slung_load.mdp.controllers import (
    quadrotor_allocation_matrix,
    quadrotor_rotor_geometry,
    reconstruct_wrench,
    scale_flare_actions,
)

pytestmark = pytest.mark.unit


class _RecordingComposer:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, torch.Tensor | bool]]] = []

    def _record(self, method: str, kwargs: dict[str, torch.Tensor | bool]) -> None:
        captured = {key: value.clone() if isinstance(value, torch.Tensor) else value for key, value in kwargs.items()}
        self.calls.append((method, captured))

    def set_forces_and_torques_index(self, **kwargs) -> None:
        self._record("set", kwargs)

    def add_forces_and_torques_index(self, **kwargs) -> None:
        self._record("add", kwargs)


def _cache_action_scaling_parameters(action: CollectiveThrustBodyRateAction) -> None:
    action._drone_mass_kg = action._drone_mass()
    action._gravity = action.cfg.gravity
    action._half_max_thrust_to_weight = 0.5 * action.cfg.max_thrust_to_weight
    action._max_body_rate_tensor = torch.as_tensor(
        action.cfg.max_body_rates, device=action._processed_actions.device, dtype=action._processed_actions.dtype
    )
    residual_body_rate_limits = getattr(action.cfg, "residual_body_rate_limits", None)
    action._residual_body_rate_tensor = torch.as_tensor(
        action.cfg.max_body_rates if residual_body_rate_limits is None else residual_body_rate_limits,
        device=action._processed_actions.device,
        dtype=action._processed_actions.dtype,
    )
    action._vertical_velocity_damping_gain = 0.0
    action._tilt_compensation = False
    action._minimum_tilt_cosine = math.cos(0.5)
    action._supported_mass_kg = action._drone_mass_kg.clone()
    action._maximum_collective_thrust = action._drone_mass_kg * action.cfg.gravity * action.cfg.max_thrust_to_weight
    action._tilt_cosine = torch.ones_like(action._drone_mass_kg)
    action._vertical_velocity = torch.zeros_like(action._drone_mass_kg)
    action._path_velocity_command = None
    action._path_velocity_cross_track_gain = 0.0
    action._path_velocity_maximum_cross_track_speed = 0.75
    action._path_velocity_curvature_feedforward_gain = 0.0
    action._path_speed_reference = torch.zeros_like(action._drone_mass_kg)
    action._path_tangent_w = torch.zeros(action._drone_mass_kg.shape[0], 3)
    action._path_curvature_w = torch.zeros_like(action._path_tangent_w)
    action._desired_path_velocity_w = torch.zeros_like(action._path_tangent_w)
    action._desired_path_acceleration_w = torch.zeros_like(action._path_tangent_w)
    action._path_velocity_error_w = torch.zeros_like(action._path_tangent_w)
    action._path_scalar_buffer = torch.zeros_like(action._drone_mass_kg)
    action._path_vector_scale = torch.ones(action._drone_mass_kg.shape[0], 1)


def _minimal_action_env() -> SimpleNamespace:
    asset = SimpleNamespace(data=SimpleNamespace(body_mass=SimpleNamespace(torch=torch.tensor([[[0.305]]]))))
    return SimpleNamespace(num_envs=1, device="cpu", scene={"robot": asset})


def _path_tracking_action(
    linear_velocity_w: torch.Tensor,
    path_speed_reference: torch.Tensor,
    path_tangent_e: torch.Tensor,
    *,
    quaternion_xyzw: torch.Tensor | None = None,
    path_cross_track_error_e: torch.Tensor | None = None,
    path_curvature_e: torch.Tensor | None = None,
    horizontal_gain: float = 1.0,
    vertical_gain: float = 0.0,
    cross_track_gain: float = 0.0,
    maximum_cross_track_speed: float = 0.75,
    curvature_feedforward_gain: float = 0.0,
    residual_body_rate_limits: tuple[float, float, float] | None = None,
) -> tuple[CollectiveThrustBodyRateAction, SimpleNamespace]:
    """Construct the path-prior hot path without simulator/controller dependencies."""
    num_envs = linear_velocity_w.shape[0]
    if quaternion_xyzw is None:
        quaternion_xyzw = torch.tensor([0.0, 0.0, 0.0, 1.0]).expand(num_envs, -1).clone()
    if path_cross_track_error_e is None:
        path_cross_track_error_e = torch.zeros(num_envs, 3)
    if path_curvature_e is None:
        path_curvature_e = torch.zeros(num_envs, 3)
    spatial_velocity_w = torch.cat((linear_velocity_w, torch.zeros(num_envs, 3)), dim=-1)
    action = object.__new__(CollectiveThrustBodyRateAction)
    action._env = SimpleNamespace(num_envs=num_envs)
    action._asset = SimpleNamespace(
        data=SimpleNamespace(
            body_mass=SimpleNamespace(torch=torch.full((num_envs, 1, 1), 0.305)),
            body_link_pose_w=SimpleNamespace(
                torch=torch.cat((torch.zeros(num_envs, 3), quaternion_xyzw), dim=-1)[:, None]
            ),
            body_com_vel_w=SimpleNamespace(torch=spatial_velocity_w[:, None]),
        )
    )
    action._raw_actions = torch.zeros(num_envs, 4)
    action._processed_actions = torch.zeros(num_envs, 4)
    action._needs_motor_init = torch.ones(num_envs, dtype=torch.bool)
    action._controller = SimpleNamespace(motor_thrusts=torch.zeros(num_envs, 4))
    action.cfg = SimpleNamespace(
        gravity=9.81,
        max_thrust_to_weight=3.5,
        max_body_rates=(15.0, 15.0, 5.0),
        residual_body_rate_limits=residual_body_rate_limits,
    )
    _cache_action_scaling_parameters(action)
    action._attitude_hold_gain = 2.0
    action._horizontal_velocity_damping_gain = horizontal_gain
    action._vertical_velocity_damping_gain = vertical_gain
    action._tilt_compensation = vertical_gain > 0.0 or curvature_feedforward_gain > 0.0
    action._maximum_horizontal_acceleration = 9.81 * math.tan(0.25)
    action._world_up = torch.tensor([[0.0, 0.0, 1.0]]).expand(num_envs, -1).clone()
    action._desired_thrust_axis_w = action._world_up.clone()
    action._velocity_hold_scale = torch.ones(num_envs, 1)
    action._attitude_rate_correction = torch.zeros(num_envs, 2)
    action._path_velocity_cross_track_gain = cross_track_gain
    action._path_velocity_maximum_cross_track_speed = maximum_cross_track_speed
    action._path_velocity_curvature_feedforward_gain = curvature_feedforward_gain
    term = SimpleNamespace(
        path_speed_reference=path_speed_reference,
        path_tangent_e=path_tangent_e,
        path_cross_track_error_e=path_cross_track_error_e,
        path_curvature_e=path_curvature_e,
    )
    action._path_velocity_command = term
    return action, term


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"vertical_velocity_damping_gain": float("nan")}, "finite and nonnegative"),
        ({"suspended_mass": -0.1}, "finite and nonnegative"),
        ({"vertical_velocity_damping_gain": 1.0}, "requires tilt_compensation=True"),
        ({"max_body_rates": (15.0, 0.0, 5.0)}, "max_body_rates must contain only positive"),
        ({"residual_body_rate_limits": (12.0, 0.0, 2.5)}, "residual_body_rate_limits must contain only positive"),
        ({"residual_body_rate_limits": (16.0, 12.0, 2.5)}, "must not exceed max_body_rates"),
        ({"path_velocity_command_name": ""}, "nonempty string or None"),
        ({"path_velocity_cross_track_gain": -1.0}, "finite and nonnegative"),
        (
            {"path_velocity_cross_track_gain": 1.0, "path_velocity_maximum_cross_track_speed": 0.0},
            "must be positive when path_velocity_cross_track_gain is",
        ),
        ({"path_velocity_curvature_feedforward_gain": float("inf")}, "finite and nonnegative"),
        ({"path_velocity_cross_track_gain": 1.0}, "require path_velocity_command_name"),
        (
            {
                "attitude_hold_gain": 2.0,
                "horizontal_velocity_damping_gain": 1.0,
                "tilt_compensation": True,
                "maximum_velocity_hold_tilt": 0.5,
                "maximum_tilt_compensation_angle": 0.25,
            },
            "must cover maximum_velocity_hold_tilt",
        ),
    ],
)
def test_velocity_hold_configuration_rejects_invalid_physical_contract(overrides, error):
    cfg = CollectiveThrustBodyRateActionCfg()
    for name, value in overrides.items():
        setattr(cfg, name, value)

    with pytest.raises(ValueError, match=error):
        CollectiveThrustBodyRateAction(cfg, _minimal_action_env())


def test_path_velocity_configuration_resolves_live_command_tensor_contract():
    term = SimpleNamespace(
        path_speed_reference=torch.zeros(1),
        path_tangent_e=torch.tensor([[1.0, 0.0, 0.0]]),
        path_cross_track_error_e=torch.zeros(1, 3),
        path_curvature_e=torch.zeros(1, 3),
    )
    env = _minimal_action_env()
    env.command_manager = SimpleNamespace(get_term=lambda name: term if name == "route" else None)
    cfg = CollectiveThrustBodyRateActionCfg(
        path_velocity_command_name="route",
        attitude_hold_gain=2.0,
        horizontal_velocity_damping_gain=1.0,
    )

    action = CollectiveThrustBodyRateAction(cfg, env)

    assert action._path_velocity_command is term

    term.path_tangent_e = torch.zeros(1, 2)
    with pytest.raises(ValueError, match=r"path_tangent_e must be a tensor with shape \(1, 3\)"):
        CollectiveThrustBodyRateAction(cfg, env)


def test_body_rate_action_applies_four_rotors_only_to_drone_and_reconstructs_wrench():
    """Each rotor acts at its physical site; cable and payload have no actuation handle."""
    thrusts = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    allocation = quadrotor_allocation_matrix(arm_length=0.13, yaw_coeff=0.07)
    output = SimpleNamespace(rotor_thrusts=thrusts, wrench=reconstruct_wrench(thrusts, allocation))
    controller_calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def _compute(command, body_rate):
        controller_calls.append((command.clone(), body_rate.clone()))
        return output

    composer = _RecordingComposer()
    rotor_positions_b, yaw_directions = quadrotor_rotor_geometry(arm_length=0.13, rotor_z=0.03)
    action = object.__new__(CollectiveThrustBodyRateAction)
    action._processed_actions = torch.tensor([[12.0, 1.0, -2.0, 0.5]])
    action._rotor_positions_b = rotor_positions_b[:, None, None, :].clone()
    action._rotor_yaw_coefficients = yaw_directions * 0.07
    action._rotor_forces_b = torch.zeros(4, 1, 1, 3)
    action._rotor_torques_b = torch.zeros_like(action._rotor_forces_b)
    action._controller = SimpleNamespace(compute=_compute)
    action._body_rate_b = lambda: torch.tensor([[0.25, -0.5, 0.75]])
    action._asset = SimpleNamespace(permanent_wrench_composer=composer)

    # The action deliberately has no cable or payload object, so the drone composer
    # is the only possible force path.
    action.apply_actions()

    assert len(controller_calls) == 1
    torch.testing.assert_close(controller_calls[0][0], action._processed_actions)
    torch.testing.assert_close(controller_calls[0][1], torch.tensor([[0.25, -0.5, 0.75]]))
    assert [method for method, _ in composer.calls] == ["set", "add", "add", "add"]

    applied_forces = torch.cat([call["forces"] for _, call in composer.calls], dim=1).squeeze(0)
    applied_torques = torch.cat([call["torques"] for _, call in composer.calls], dim=1).squeeze(0)
    applied_positions = torch.cat([call["positions"] for _, call in composer.calls], dim=1).squeeze(0)
    torch.testing.assert_close(applied_positions, rotor_positions_b)
    torch.testing.assert_close(applied_forces[:, :2], torch.zeros(4, 2))
    torch.testing.assert_close(applied_forces[:, 2], thrusts.squeeze(0))
    torch.testing.assert_close(applied_torques[:, :2], torch.zeros(4, 2))
    torch.testing.assert_close(applied_torques[:, 2], yaw_directions * 0.07 * thrusts.squeeze(0))
    assert all(call["is_global"] is False for _, call in composer.calls)

    realized_force_b = applied_forces.sum(dim=0)
    realized_torque_b = (torch.linalg.cross(applied_positions, applied_forces) + applied_torques).sum(dim=0)
    realized_wrench = torch.cat((realized_force_b[2:3], realized_torque_b))
    torch.testing.assert_close(realized_wrench, output.wrench.squeeze(0))


def test_body_rate_action_scales_motor_capacity_from_drone_mass_only():
    action = object.__new__(CollectiveThrustBodyRateAction)
    action._env = SimpleNamespace(num_envs=2)
    action._asset = SimpleNamespace(
        data=SimpleNamespace(body_mass=SimpleNamespace(torch=torch.tensor([[[0.305]], [[0.500]]])))
    )
    action._raw_actions = torch.zeros(2, 4)
    action._processed_actions = torch.zeros(2, 4)
    action._needs_motor_init = torch.zeros(2, dtype=torch.bool)
    action._controller = SimpleNamespace(motor_thrusts=torch.zeros(2, 4))
    action._attitude_hold_gain = 0.0
    action._horizontal_velocity_damping_gain = 0.0
    action.cfg = SimpleNamespace(
        gravity=9.81,
        max_thrust_to_weight=3.5,
        max_body_rates=(15.0, 15.0, 5.0),
    )
    _cache_action_scaling_parameters(action)

    action.process_actions(torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]))

    expected_capacity = torch.tensor([0.305, 0.500]) * 9.81 * 3.5
    torch.testing.assert_close(action._processed_actions[:, 0], expected_capacity)


def test_body_rate_action_fast_scaling_and_motor_initialization_match_reference():
    action = object.__new__(CollectiveThrustBodyRateAction)
    action._env = SimpleNamespace(num_envs=3)
    action._asset = SimpleNamespace(
        data=SimpleNamespace(body_mass=SimpleNamespace(torch=torch.tensor([[[0.305]], [[0.410]], [[0.500]]])))
    )
    action._raw_actions = torch.zeros(3, 4)
    action._processed_actions = torch.zeros(3, 4)
    action._needs_motor_init = torch.tensor([True, False, True])
    original_motor_thrusts = torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [0.9, 1.0, 1.1, 1.2]])
    action._controller = SimpleNamespace(motor_thrusts=original_motor_thrusts.clone())
    action._attitude_hold_gain = 0.0
    action._horizontal_velocity_damping_gain = 0.0
    action.cfg = SimpleNamespace(
        gravity=9.81,
        max_thrust_to_weight=3.5,
        max_body_rates=(15.0, 15.0, 5.0),
    )
    _cache_action_scaling_parameters(action)
    raw_actions = torch.tensor(
        [
            [float("nan"), 0.5, -2.0, float("inf")],
            [-1.0, -0.25, 0.75, -0.5],
            [0.4, 2.0, -0.3, -float("inf")],
        ]
    )
    expected_processed = scale_flare_actions(
        raw_actions,
        action._drone_mass_kg,
        gravity=action.cfg.gravity,
        max_thrust_to_weight=action.cfg.max_thrust_to_weight,
        max_body_rates=action.cfg.max_body_rates,
    )
    expected_motor_thrusts = torch.where(
        action._needs_motor_init[:, None], expected_processed[:, :1] / 4.0, original_motor_thrusts
    )

    action.process_actions(raw_actions)

    assert torch.equal(action._processed_actions, expected_processed)
    torch.testing.assert_close(action._controller.motor_thrusts, expected_motor_thrusts)
    assert not action._needs_motor_init.any()


def test_enhanced_attitude_hold_levels_roll_and_pitch_without_affecting_yaw():
    action = object.__new__(CollectiveThrustBodyRateAction)
    action._env = SimpleNamespace(num_envs=3)
    half_angle = 0.05
    quaternion = torch.tensor(
        [
            [math.sin(half_angle), 0.0, 0.0, math.cos(half_angle)],
            [0.0, math.sin(half_angle), 0.0, math.cos(half_angle)],
            [0.0, 0.0, math.sin(half_angle), math.cos(half_angle)],
        ]
    )
    action._asset = SimpleNamespace(
        data=SimpleNamespace(
            body_mass=SimpleNamespace(torch=torch.full((3, 1, 1), 0.305)),
            body_link_pose_w=SimpleNamespace(torch=torch.cat((torch.zeros(3, 3), quaternion), dim=-1)[:, None]),
        )
    )
    action._raw_actions = torch.zeros(3, 4)
    action._processed_actions = torch.zeros(3, 4)
    action._needs_motor_init = torch.ones(3, dtype=torch.bool)
    action._controller = SimpleNamespace(motor_thrusts=torch.zeros(3, 4))
    action.cfg = SimpleNamespace(gravity=9.81, max_thrust_to_weight=3.5, max_body_rates=(15.0, 15.0, 5.0))
    _cache_action_scaling_parameters(action)
    action._attitude_hold_gain = 2.0
    action._horizontal_velocity_damping_gain = 0.0
    action._world_up = torch.tensor([[0.0, 0.0, 1.0]]).expand(3, -1).clone()
    action._attitude_rate_correction = torch.zeros(3, 2)

    action.process_actions(torch.zeros(3, 4))

    expected = 2.0 * math.sin(2.0 * half_angle)
    assert action._processed_actions[0, 1].item() == pytest.approx(-expected)
    assert action._processed_actions[1, 2].item() == pytest.approx(-expected)
    torch.testing.assert_close(action._processed_actions[2, 1:3], torch.zeros(2), atol=1.0e-7, rtol=0.0)
    torch.testing.assert_close(action._processed_actions[:, 3], torch.zeros(3))


def test_policy_rate_residual_is_scaled_before_prior_and_complete_command_is_clamped():
    action = object.__new__(CollectiveThrustBodyRateAction)
    action._env = SimpleNamespace(num_envs=2)
    half_right_angle = 0.25 * math.pi
    quaternion = torch.tensor(
        [
            [-math.sin(half_right_angle), 0.0, 0.0, math.cos(half_right_angle)],
            [0.0, math.sin(half_right_angle), 0.0, math.cos(half_right_angle)],
        ]
    )
    action._asset = SimpleNamespace(
        data=SimpleNamespace(
            body_mass=SimpleNamespace(torch=torch.full((2, 1, 1), 0.305)),
            body_link_pose_w=SimpleNamespace(torch=torch.cat((torch.zeros(2, 3), quaternion), dim=-1)[:, None]),
        )
    )
    action._raw_actions = torch.zeros(2, 4)
    action._processed_actions = torch.zeros(2, 4)
    action._needs_motor_init = torch.ones(2, dtype=torch.bool)
    action._controller = SimpleNamespace(motor_thrusts=torch.zeros(2, 4))
    action.cfg = SimpleNamespace(
        gravity=9.81,
        max_thrust_to_weight=3.5,
        max_body_rates=(15.0, 15.0, 5.0),
        residual_body_rate_limits=(12.0, 12.0, 2.5),
    )
    _cache_action_scaling_parameters(action)
    action._attitude_hold_gain = 4.0
    action._horizontal_velocity_damping_gain = 0.0
    action._world_up = torch.tensor([[0.0, 0.0, 1.0]]).expand(2, -1).clone()
    action._attitude_rate_correction = torch.zeros(2, 2)

    action.process_actions(torch.tensor([[0.0, 1.0, 0.0, 1.0], [0.0, 0.0, -1.0, -1.0]]))

    torch.testing.assert_close(
        action._processed_actions[:, 1:4],
        torch.tensor([[15.0, 0.0, 2.5], [0.0, -15.0, -2.5]]),
        atol=1.0e-6,
        rtol=0.0,
    )


def test_enhanced_velocity_hold_brakes_world_velocity_with_yaw_equivariant_body_rates():
    action = object.__new__(CollectiveThrustBodyRateAction)
    action._env = SimpleNamespace(num_envs=4)
    yaw_90 = torch.tensor([0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)])
    quaternion = torch.stack((torch.tensor([0.0, 0.0, 0.0, 1.0]),) * 2 + (yaw_90,) * 2)
    # Rows 0/2 are the same +body-X velocity at yaw 0/90 degrees. Rows
    # 1/3 are the same +body-Y velocity under the same yaw rotation.
    linear_velocity_w = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
        ]
    )
    spatial_velocity_w = torch.cat((linear_velocity_w, torch.zeros(4, 3)), dim=-1)
    action._asset = SimpleNamespace(
        data=SimpleNamespace(
            body_mass=SimpleNamespace(torch=torch.full((4, 1, 1), 0.305)),
            body_link_pose_w=SimpleNamespace(torch=torch.cat((torch.zeros(4, 3), quaternion), dim=-1)[:, None]),
            body_com_vel_w=SimpleNamespace(torch=spatial_velocity_w[:, None]),
        )
    )
    action._raw_actions = torch.zeros(4, 4)
    action._processed_actions = torch.zeros(4, 4)
    action._needs_motor_init = torch.ones(4, dtype=torch.bool)
    action._controller = SimpleNamespace(motor_thrusts=torch.zeros(4, 4))
    action.cfg = SimpleNamespace(gravity=9.81, max_thrust_to_weight=3.5, max_body_rates=(15.0, 15.0, 5.0))
    _cache_action_scaling_parameters(action)
    action._attitude_hold_gain = 2.0
    action._horizontal_velocity_damping_gain = 1.0
    action._maximum_horizontal_acceleration = 9.81 * math.tan(0.25)
    action._world_up = torch.tensor([[0.0, 0.0, 1.0]]).expand(4, -1).clone()
    action._desired_thrust_axis_w = action._world_up.clone()
    action._velocity_hold_scale = torch.ones(4, 1)
    action._attitude_rate_correction = torch.zeros(4, 2)

    action.process_actions(torch.zeros(4, 4))

    expected_rate = 2.0 / math.sqrt(9.81**2 + 1.0)
    # +body-X velocity requests negative pitch; +body-Y requests positive roll.
    torch.testing.assert_close(
        action._processed_actions[:, 1:3],
        torch.tensor(
            [
                [0.0, -expected_rate],
                [expected_rate, 0.0],
                [0.0, -expected_rate],
                [expected_rate, 0.0],
            ]
        ),
        atol=1.0e-6,
        rtol=0.0,
    )
    torch.testing.assert_close(action._processed_actions[:, 3], torch.zeros(4))

    spatial_velocity_w[0, 0] = 100.0
    action.process_actions(torch.zeros(4, 4))
    assert action._processed_actions[0, 2].item() == pytest.approx(-2.0 * math.sin(0.25))


def test_path_velocity_prior_accelerates_brakes_and_damps_transverse_speed_yaw_equivariantly():
    yaw_90 = torch.tensor([0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)])
    quaternion = torch.stack(
        (
            torch.tensor([0.0, 0.0, 0.0, 1.0]),
            yaw_90,
            torch.tensor([0.0, 0.0, 0.0, 1.0]),
            torch.tensor([0.0, 0.0, 0.0, 1.0]),
            torch.tensor([0.0, 0.0, 0.0, 1.0]),
        )
    )
    action, _ = _path_tracking_action(
        torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
            ]
        ),
        torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0]),
        torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ]
        ),
        quaternion_xyzw=quaternion,
    )

    action.process_actions(torch.zeros(5, 4))

    unit_acceleration_rate = 2.0 / math.sqrt(9.81**2 + 1.0)
    # The same world path rotated with yaw produces the same body-frame pitch.
    assert action.processed_actions[0, 2].item() == pytest.approx(unit_acceleration_rate)
    assert action.processed_actions[1, 2].item() == pytest.approx(unit_acceleration_rate)
    # Overspeed and a zero target both brake; transverse velocity is damped.
    assert action.processed_actions[2, 2].item() == pytest.approx(-unit_acceleration_rate)
    assert action.processed_actions[3, 1].item() > 0.0
    assert action.processed_actions[3, 2].item() > 0.0
    assert action.processed_actions[4, 2].item() == pytest.approx(-unit_acceleration_rate)


def test_path_cross_track_convergence_uses_live_indexed_error_with_correct_sign_and_independent_cap():
    action, term = _path_tracking_action(
        torch.zeros(2, 3),
        torch.zeros(2),
        torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        path_cross_track_error_e=torch.tensor([[0.0, 10.0, 0.0], [0.0, -10.0, 0.0]]),
        cross_track_gain=4.0,
        maximum_cross_track_speed=0.75,
    )
    # A figure-eight can have a nearer point on the other branch. The action
    # deliberately consumes only the command term's indexed active-branch error.
    term.other_branch_cross_track_error_e = -term.path_cross_track_error_e

    action.process_actions(torch.zeros(2, 4))

    expected_rate = 2.0 * 0.75 / math.sqrt(9.81**2 + 0.75**2)
    torch.testing.assert_close(
        action.processed_actions[:, 1],
        torch.tensor([expected_rate, -expected_rate]),
        atol=1.0e-6,
        rtol=0.0,
    )
    torch.testing.assert_close(
        torch.linalg.vector_norm(action._desired_path_velocity_w, dim=-1),
        torch.full((2,), 0.75),
    )

    # The reference is queried live: switching the indexed branch switches the
    # controller correction on the very next control step.
    term.path_cross_track_error_e.neg_()
    action.process_actions(torch.zeros(2, 4))
    torch.testing.assert_close(
        action.processed_actions[:, 1],
        torch.tensor([-expected_rate, expected_rate]),
        atol=1.0e-6,
        rtol=0.0,
    )


def test_path_prior_tracks_vertical_tangent_and_adds_bounded_circle_centripetal_feedforward():
    root_half = math.sqrt(0.5)
    action, _ = _path_tracking_action(
        torch.zeros(3, 3),
        torch.tensor([2.0, 2.0, 2.0]),
        torch.tensor(
            [
                [root_half, 0.0, root_half],
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
            ]
        ),
        path_curvature_e=torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.5],
                [0.0, 0.0, 0.0],
            ]
        ),
        vertical_gain=1.0,
        curvature_feedforward_gain=1.0,
    )
    base_collective = scale_flare_actions(torch.zeros(3, 4), action._drone_mass_kg)[:, 0]

    action.process_actions(torch.zeros(3, 4))

    expected_vertical_increment = action._supported_mass_kg[0] * 2.0 * root_half
    assert action.processed_actions[0, 0].item() == pytest.approx(
        (base_collective[0] + expected_vertical_increment).item()
    )
    # Counter-clockwise circle at +X: tangent +Y and curvature binormal +Z,
    # so cross(k_b, tangent) is the inward -X acceleration.
    assert action.processed_actions[1, 2].item() < 0.0
    # A straight path has exactly zero centripetal feedforward.
    assert action.processed_actions[2, 1].item() == pytest.approx(0.0, abs=1.0e-7)
    assert action.processed_actions[2, 2].item() > 0.0
    assert torch.isfinite(action.processed_actions).all()


def test_path_prior_sanitizes_curved_geometry_and_actor_residual_cannot_bypass_final_rate_clamp():
    action, _ = _path_tracking_action(
        torch.tensor([[-100.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        torch.tensor([100.0, 2.0]),
        torch.tensor([[1.0, 0.0, 0.0], [float("nan"), 1.0, 0.0]]),
        path_curvature_e=torch.tensor([[0.0, 0.0, 100.0], [float("inf"), 0.0, 1.0]]),
        curvature_feedforward_gain=1.0,
        residual_body_rate_limits=(12.0, 12.0, 2.5),
    )
    action._attitude_hold_gain = 20.0

    action.process_actions(torch.tensor([[0.0, -1.0, 1.0, 1.0], [0.0, -1.0, -1.0, -1.0]]))

    assert torch.isfinite(action.processed_actions).all()
    assert torch.all(action.processed_actions[:, 1:4] <= torch.tensor([15.0, 15.0, 5.0]))
    assert torch.all(action.processed_actions[:, 1:4] >= torch.tensor([-15.0, -15.0, -5.0]))
    assert action.processed_actions[0, 1].item() == -15.0


def test_enhanced_vertical_hold_uses_supported_mass_and_compensates_tilt():
    action = object.__new__(CollectiveThrustBodyRateAction)
    action._env = SimpleNamespace(num_envs=4)
    roll = 0.2
    rolled = torch.tensor([math.sin(0.5 * roll), 0.0, 0.0, math.cos(0.5 * roll)])
    identity = torch.tensor([0.0, 0.0, 0.0, 1.0])
    quaternion = torch.stack((identity, identity, rolled, identity))
    linear_velocity_w = torch.tensor(
        [
            [0.0, 0.0, -1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -0.5],
            [0.0, 0.0, 0.0],
        ]
    )
    action._asset = SimpleNamespace(
        data=SimpleNamespace(
            body_mass=SimpleNamespace(torch=torch.full((4, 1, 1), 0.305)),
            body_link_pose_w=SimpleNamespace(torch=torch.cat((torch.zeros(4, 3), quaternion), dim=-1)[:, None]),
            body_com_vel_w=SimpleNamespace(torch=torch.cat((linear_velocity_w, torch.zeros(4, 3)), dim=-1)[:, None]),
        )
    )
    action._raw_actions = torch.zeros(4, 4)
    action._processed_actions = torch.zeros(4, 4)
    action._needs_motor_init = torch.ones(4, dtype=torch.bool)
    action._controller = SimpleNamespace(motor_thrusts=torch.zeros(4, 4))
    action._attitude_hold_gain = 0.0
    action._horizontal_velocity_damping_gain = 0.0
    action.cfg = SimpleNamespace(gravity=9.81, max_thrust_to_weight=3.5, max_body_rates=(15.0, 15.0, 5.0))
    _cache_action_scaling_parameters(action)
    action._vertical_velocity_damping_gain = 1.0
    action._tilt_compensation = True
    suspended_mass = 0.071
    action._supported_mass_kg.add_(suspended_mass)

    raw_actions = torch.tensor([[-0.2, 0.1, -0.1, 0.2]]).expand(4, -1).clone()
    reference = scale_flare_actions(
        raw_actions,
        action._drone_mass_kg,
        gravity=action.cfg.gravity,
        max_thrust_to_weight=action.cfg.max_thrust_to_weight,
        max_body_rates=action.cfg.max_body_rates,
    )
    action.process_actions(raw_actions)

    supported_mass = 0.305 + suspended_mass
    expected_collective = reference[:, 0].clone()
    expected_collective[0] += supported_mass
    expected_collective[1] -= supported_mass
    expected_collective[2] = (expected_collective[2] + 0.5 * supported_mass) / math.cos(roll)
    torch.testing.assert_close(action._processed_actions[:, 0], expected_collective)
    torch.testing.assert_close(action._processed_actions[:, 1:], reference[:, 1:])
    # At level attitude and zero vertical speed the v5 mapping is exactly the
    # calibrated policy collective, with no hidden trim or force offset.
    assert action._processed_actions[3, 0].item() == reference[3, 0].item()


def test_tilt_compensation_matches_xyzw_body_axis_projection():
    action = object.__new__(CollectiveThrustBodyRateAction)
    action._env = SimpleNamespace(num_envs=3)
    roll, pitch, yaw = 0.2, -0.3, 0.7
    quaternion_xyzw = torch.tensor(
        [
            [math.sin(0.5 * roll), 0.0, 0.0, math.cos(0.5 * roll)],
            [0.0, math.sin(0.5 * pitch), 0.0, math.cos(0.5 * pitch)],
            [0.0, 0.0, math.sin(0.5 * yaw), math.cos(0.5 * yaw)],
        ]
    )
    action._asset = SimpleNamespace(
        data=SimpleNamespace(
            body_mass=SimpleNamespace(torch=torch.full((3, 1, 1), 0.305)),
            body_link_pose_w=SimpleNamespace(torch=torch.cat((torch.zeros(3, 3), quaternion_xyzw), dim=-1)[:, None]),
        )
    )
    action._raw_actions = torch.zeros(3, 4)
    action._processed_actions = torch.zeros(3, 4)
    action._needs_motor_init = torch.ones(3, dtype=torch.bool)
    action._controller = SimpleNamespace(motor_thrusts=torch.zeros(3, 4))
    action._attitude_hold_gain = 0.0
    action._horizontal_velocity_damping_gain = 0.0
    action.cfg = SimpleNamespace(gravity=9.81, max_thrust_to_weight=3.5, max_body_rates=(15.0, 15.0, 5.0))
    _cache_action_scaling_parameters(action)
    action._tilt_compensation = True

    action.process_actions(torch.zeros(3, 4))

    body_z = torch.tensor([[0.0, 0.0, 1.0]]).expand(3, -1)
    expected_projection = quat_apply(quaternion_xyzw, body_z)[:, 2].clamp_min(math.cos(0.5))
    torch.testing.assert_close(action._tilt_cosine, expected_projection)


def test_enhanced_vertical_hold_sanitizes_state_and_clamps_collective():
    action = object.__new__(CollectiveThrustBodyRateAction)
    action._env = SimpleNamespace(num_envs=5)
    identity = torch.tensor([0.0, 0.0, 0.0, 1.0])
    excessive_roll = torch.tensor([math.sin(0.4), 0.0, 0.0, math.cos(0.4)])
    quaternion = torch.stack((identity, identity, identity, identity, excessive_roll))
    quaternion[2] = float("nan")
    linear_velocity_w = torch.tensor(
        [
            [0.0, 0.0, -100.0],
            [0.0, 0.0, 100.0],
            [0.0, 0.0, float("nan")],
            [0.0, 0.0, float("inf")],
            [0.0, 0.0, 0.0],
        ]
    )
    action._asset = SimpleNamespace(
        data=SimpleNamespace(
            body_mass=SimpleNamespace(torch=torch.full((5, 1, 1), 0.305)),
            body_link_pose_w=SimpleNamespace(torch=torch.cat((torch.zeros(5, 3), quaternion), dim=-1)[:, None]),
            body_com_vel_w=SimpleNamespace(torch=torch.cat((linear_velocity_w, torch.zeros(5, 3)), dim=-1)[:, None]),
        )
    )
    action._raw_actions = torch.zeros(5, 4)
    action._processed_actions = torch.zeros(5, 4)
    action._needs_motor_init = torch.ones(5, dtype=torch.bool)
    action._controller = SimpleNamespace(motor_thrusts=torch.zeros(5, 4))
    action._attitude_hold_gain = 0.0
    action._horizontal_velocity_damping_gain = 0.0
    action.cfg = SimpleNamespace(gravity=9.81, max_thrust_to_weight=3.5, max_body_rates=(15.0, 15.0, 5.0))
    _cache_action_scaling_parameters(action)
    action._vertical_velocity_damping_gain = 1.0
    action._tilt_compensation = True
    action._supported_mass_kg.add_(0.071)

    raw_actions = torch.tensor(
        [
            [0.9, 0.0, 0.0, 0.0],
            [-0.9, 0.0, 0.0, 0.0],
            [-0.2, 0.0, 0.0, 0.0],
            [-0.2, 0.0, 0.0, 0.0],
            [-0.2, 0.0, 0.0, 0.0],
        ]
    )
    base = scale_flare_actions(raw_actions, action._drone_mass_kg)[:, 0]
    action.process_actions(raw_actions)

    maximum = action._maximum_collective_thrust
    assert torch.isfinite(action._processed_actions).all()
    assert torch.all((action._processed_actions[:, 0] >= 0.0) & (action._processed_actions[:, 0] <= maximum))
    assert action._processed_actions[0, 0].item() == maximum[0].item()
    assert action._processed_actions[1, 0].item() == 0.0
    assert action._processed_actions[2, 0].item() == base[2].item()
    assert action._processed_actions[3, 0].item() == base[3].item()
    assert action._processed_actions[4, 0].item() == pytest.approx(base[4].item() / math.cos(0.5))


def test_body_rate_action_reset_clears_commands_rotor_wrenches_and_controller_state():
    reset_ids: list[torch.Tensor | None] = []
    action = object.__new__(CollectiveThrustBodyRateAction)
    action._raw_actions = torch.ones(2, 4)
    action._processed_actions = torch.ones(2, 4)
    action._rotor_forces_b = torch.ones(4, 2, 1, 3)
    action._rotor_torques_b = torch.ones(4, 2, 1, 3)
    action._world_up = torch.tensor([[0.0, 0.0, 1.0]]).expand(2, -1).clone()
    action._desired_thrust_axis_w = torch.ones(2, 3)
    action._velocity_hold_scale = torch.full((2, 1), 2.0)
    action._attitude_rate_correction = torch.ones(2, 2)
    action._tilt_cosine = torch.zeros(2)
    action._vertical_velocity = torch.ones(2)
    action._path_speed_reference = torch.ones(2)
    action._path_tangent_w = torch.ones(2, 3)
    action._path_curvature_w = torch.ones(2, 3)
    action._desired_path_velocity_w = torch.ones(2, 3)
    action._desired_path_acceleration_w = torch.ones(2, 3)
    action._path_velocity_error_w = torch.ones(2, 3)
    action._path_scalar_buffer = torch.ones(2)
    action._path_vector_scale = torch.full((2, 1), 2.0)
    action._needs_motor_init = torch.zeros(2, dtype=torch.bool)
    action._controller = SimpleNamespace(reset=lambda env_ids: reset_ids.append(env_ids))

    action.reset(torch.tensor([1]))

    torch.testing.assert_close(action._raw_actions[0], torch.ones(4))
    torch.testing.assert_close(action._processed_actions[0], torch.ones(4))
    torch.testing.assert_close(action._rotor_forces_b[:, 0], torch.ones(4, 1, 3))
    torch.testing.assert_close(action._rotor_torques_b[:, 0], torch.ones(4, 1, 3))
    torch.testing.assert_close(action._raw_actions[1], torch.zeros(4))
    torch.testing.assert_close(action._processed_actions[1], torch.zeros(4))
    torch.testing.assert_close(action._rotor_forces_b[:, 1], torch.zeros(4, 1, 3))
    torch.testing.assert_close(action._rotor_torques_b[:, 1], torch.zeros(4, 1, 3))
    torch.testing.assert_close(action._desired_thrust_axis_w[1], torch.tensor([0.0, 0.0, 1.0]))
    torch.testing.assert_close(action._velocity_hold_scale[1], torch.ones(1))
    torch.testing.assert_close(action._attitude_rate_correction[1], torch.zeros(2))
    assert action._tilt_cosine[1].item() == 1.0
    assert action._vertical_velocity[1].item() == 0.0
    assert action._path_speed_reference[1].item() == 0.0
    torch.testing.assert_close(action._path_tangent_w[1], torch.zeros(3))
    torch.testing.assert_close(action._path_curvature_w[1], torch.zeros(3))
    torch.testing.assert_close(action._desired_path_velocity_w[1], torch.zeros(3))
    torch.testing.assert_close(action._desired_path_acceleration_w[1], torch.zeros(3))
    torch.testing.assert_close(action._path_velocity_error_w[1], torch.zeros(3))
    assert action._path_scalar_buffer[1].item() == 0.0
    torch.testing.assert_close(action._path_vector_scale[1], torch.ones(1))
    assert action._needs_motor_init.tolist() == [False, True]
    assert len(reset_ids) == 1
    torch.testing.assert_close(reset_ids[0], torch.tensor([1]))
