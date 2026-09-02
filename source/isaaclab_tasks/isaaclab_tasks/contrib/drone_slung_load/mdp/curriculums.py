# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Curriculum terms for precise, high-speed slung-load path tracking."""

from __future__ import annotations

import copy
import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import CurriculumTermCfg, ManagerTermBase

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import EventTermCfg, RewardTermCfg, TerminationTermCfg


class PrecisionSpeedCurriculum(ManagerTermBase):
    """Warm up, then linearly tighten path precision and rewarded path speed.

    The schedule is derived only from :attr:`ManagerBasedRLEnv.common_step_counter`,
    so it is independent of which environments reset and naturally follows a
    restored global step counter. Evaluation environments can disable the
    schedule by omitting this curriculum term.
    """

    cfg: CurriculumTermCfg
    """Configuration for the curriculum term."""

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        """Initialize and validate the precision/speed schedule.

        Args:
            cfg: Curriculum term configuration.
            env: Manager-based RL environment.
        """
        super().__init__(cfg, env)
        parameters = {
            "command_name": "route",
            "path_progress_reward_name": "path_progress",
            "path_precision_reward_name": "path_precision",
            "path_corridor_termination_name": "path_corridor",
            "warmup_steps": 20_000,
            "ramp_steps": 100_000,
            "initial_acceptance_radius": 0.50,
            "final_acceptance_radius": 0.15,
            "initial_maximum_rate": 1.25,
            "final_maximum_rate": 3.50,
            "initial_cross_track_scale": 0.50,
            "final_cross_track_scale": 0.20,
            "initial_transverse_velocity_scale": 1.00,
            "final_transverse_velocity_scale": 0.40,
            "initial_precision_reward_weight": -2.0,
            "final_precision_reward_weight": -15.0,
            "initial_corridor_distance": 1.50,
            "final_corridor_distance": 0.75,
        }
        parameters.update(cfg.params)
        self._validate_parameters(**parameters)
        self._get_bound_configs(
            env,
            command_name=parameters["command_name"],
            path_progress_reward_name=parameters["path_progress_reward_name"],
            path_precision_reward_name=parameters["path_precision_reward_name"],
            path_corridor_termination_name=parameters["path_corridor_termination_name"],
        )

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        _env_ids: Sequence[int],
        command_name: str = "route",
        path_progress_reward_name: str = "path_progress",
        path_precision_reward_name: str = "path_precision",
        path_corridor_termination_name: str = "path_corridor",
        warmup_steps: int = 20_000,
        ramp_steps: int = 100_000,
        initial_acceptance_radius: float = 0.50,
        final_acceptance_radius: float = 0.15,
        initial_maximum_rate: float = 1.25,
        final_maximum_rate: float = 3.50,
        initial_cross_track_scale: float = 0.50,
        final_cross_track_scale: float = 0.20,
        initial_transverse_velocity_scale: float = 1.00,
        final_transverse_velocity_scale: float = 0.40,
        initial_precision_reward_weight: float = -2.0,
        final_precision_reward_weight: float = -15.0,
        initial_corridor_distance: float = 1.50,
        final_corridor_distance: float = 0.75,
    ) -> dict[str, float]:
        """Apply the schedule value at the current global control step.

        Args:
            env: Manager-based RL environment.
            _env_ids: Environment indices being reset. The global schedule is
                shared by all environments, so these indices are not used.
            command_name: Waypoint command term to update.
            path_progress_reward_name: Path-progress reward term to update.
            path_precision_reward_name: Path-precision reward term to update.
            path_corridor_termination_name: Path-corridor termination term to update.
            warmup_steps: Number of global control steps held at the initial values.
            ramp_steps: Number of global control steps over which to ramp after warmup.
            initial_acceptance_radius: Initial waypoint acceptance radius [m].
            final_acceptance_radius: Final waypoint acceptance radius [m].
            initial_maximum_rate: Initial maximum rewarded path-progress rate [m/s].
            final_maximum_rate: Final maximum rewarded path-progress rate [m/s].
            initial_cross_track_scale: Initial cross-track precision scale [m].
            final_cross_track_scale: Final cross-track precision scale [m].
            initial_transverse_velocity_scale: Initial transverse-velocity scale [m/s].
            final_transverse_velocity_scale: Final transverse-velocity scale [m/s].
            initial_precision_reward_weight: Initial negative precision-cost weight.
            final_precision_reward_weight: Final negative precision-cost weight.
            initial_corridor_distance: Initial spline-corridor half-width [m].
            final_corridor_distance: Final spline-corridor half-width [m].

        Returns:
            Current schedule values for curriculum logging.
        """
        self._validate_parameters(
            command_name=command_name,
            path_progress_reward_name=path_progress_reward_name,
            path_precision_reward_name=path_precision_reward_name,
            path_corridor_termination_name=path_corridor_termination_name,
            warmup_steps=warmup_steps,
            ramp_steps=ramp_steps,
            initial_acceptance_radius=initial_acceptance_radius,
            final_acceptance_radius=final_acceptance_radius,
            initial_maximum_rate=initial_maximum_rate,
            final_maximum_rate=final_maximum_rate,
            initial_cross_track_scale=initial_cross_track_scale,
            final_cross_track_scale=final_cross_track_scale,
            initial_transverse_velocity_scale=initial_transverse_velocity_scale,
            final_transverse_velocity_scale=final_transverse_velocity_scale,
            initial_precision_reward_weight=initial_precision_reward_weight,
            final_precision_reward_weight=final_precision_reward_weight,
            initial_corridor_distance=initial_corridor_distance,
            final_corridor_distance=final_corridor_distance,
        )
        step = self._global_step(env)
        fraction = min(max(step - warmup_steps, 0) / ramp_steps, 1.0)
        values = {
            "acceptance_radius": self._lerp(initial_acceptance_radius, final_acceptance_radius, fraction),
            "maximum_rate": self._lerp(initial_maximum_rate, final_maximum_rate, fraction),
            "cross_track_scale": self._lerp(initial_cross_track_scale, final_cross_track_scale, fraction),
            "transverse_velocity_scale": self._lerp(
                initial_transverse_velocity_scale, final_transverse_velocity_scale, fraction
            ),
            "precision_reward_weight": self._lerp(
                initial_precision_reward_weight, final_precision_reward_weight, fraction
            ),
            "corridor_distance": self._lerp(initial_corridor_distance, final_corridor_distance, fraction),
        }

        command_cfg, progress_cfg, precision_cfg, corridor_cfg = self._get_bound_configs(
            env,
            command_name=command_name,
            path_progress_reward_name=path_progress_reward_name,
            path_precision_reward_name=path_precision_reward_name,
            path_corridor_termination_name=path_corridor_termination_name,
        )
        if command_cfg.acceptance_radius != values["acceptance_radius"]:
            command_cfg.acceptance_radius = values["acceptance_radius"]

        if progress_cfg.params["maximum_rate"] != values["maximum_rate"]:
            progress_cfg = self._copy_reward_cfg(progress_cfg)
            progress_cfg.params["maximum_rate"] = values["maximum_rate"]
            env.reward_manager.set_term_cfg(path_progress_reward_name, progress_cfg)

        precision_changed = (
            precision_cfg.params["cross_track_scale"] != values["cross_track_scale"]
            or precision_cfg.params["transverse_velocity_scale"] != values["transverse_velocity_scale"]
            or precision_cfg.weight != values["precision_reward_weight"]
        )
        if precision_changed:
            precision_cfg = self._copy_reward_cfg(precision_cfg)
            precision_cfg.params["cross_track_scale"] = values["cross_track_scale"]
            precision_cfg.params["transverse_velocity_scale"] = values["transverse_velocity_scale"]
            precision_cfg.weight = values["precision_reward_weight"]
            env.reward_manager.set_term_cfg(path_precision_reward_name, precision_cfg)

        if corridor_cfg.params["maximum_distance"] != values["corridor_distance"]:
            corridor_cfg = self._copy_termination_cfg(corridor_cfg)
            corridor_cfg.params["maximum_distance"] = values["corridor_distance"]
            env.termination_manager.set_term_cfg(path_corridor_termination_name, corridor_cfg)

        return values

    @staticmethod
    def _copy_reward_cfg(cfg: RewardTermCfg) -> RewardTermCfg:
        """Copy a reward configuration without copying its callable term state."""
        copied = copy.copy(cfg)
        copied.params = cfg.params.copy()
        return copied

    @staticmethod
    def _copy_termination_cfg(cfg: TerminationTermCfg) -> TerminationTermCfg:
        """Copy a termination configuration without copying callable term state."""
        copied = copy.copy(cfg)
        copied.params = cfg.params.copy()
        return copied

    @staticmethod
    def _get_bound_configs(
        env: ManagerBasedRLEnv,
        command_name: str,
        path_progress_reward_name: str,
        path_precision_reward_name: str,
        path_corridor_termination_name: str,
    ) -> tuple[object, RewardTermCfg, RewardTermCfg, TerminationTermCfg]:
        """Resolve and validate bound command, reward, and termination configurations."""
        try:
            command_cfg = env.command_manager.get_term(command_name).cfg
        except (KeyError, ValueError) as error:
            raise ValueError(f"Command term '{command_name}' required by the curriculum was not found.") from error
        if not hasattr(command_cfg, "acceptance_radius"):
            raise ValueError(f"Command term '{command_name}' has no mutable 'acceptance_radius' configuration.")

        try:
            progress_cfg = env.reward_manager.get_term_cfg(path_progress_reward_name)
        except (KeyError, ValueError) as error:
            raise ValueError(
                f"Reward term '{path_progress_reward_name}' required by the curriculum was not found."
            ) from error
        if "maximum_rate" not in progress_cfg.params:
            raise ValueError(f"Reward term '{path_progress_reward_name}' has no 'maximum_rate' parameter.")

        try:
            precision_cfg = env.reward_manager.get_term_cfg(path_precision_reward_name)
        except (KeyError, ValueError) as error:
            raise ValueError(
                f"Reward term '{path_precision_reward_name}' required by the curriculum was not found."
            ) from error
        missing_parameters = {"cross_track_scale", "transverse_velocity_scale"} - precision_cfg.params.keys()
        if missing_parameters:
            missing = ", ".join(sorted(missing_parameters))
            raise ValueError(f"Reward term '{path_precision_reward_name}' is missing parameters: {missing}.")
        if not hasattr(precision_cfg, "weight"):
            raise ValueError(f"Reward term '{path_precision_reward_name}' has no mutable weight.")

        try:
            corridor_cfg = env.termination_manager.get_term_cfg(path_corridor_termination_name)
        except (KeyError, ValueError) as error:
            raise ValueError(
                f"Termination term '{path_corridor_termination_name}' required by the curriculum was not found."
            ) from error
        if "maximum_distance" not in corridor_cfg.params:
            raise ValueError(
                f"Termination term '{path_corridor_termination_name}' has no 'maximum_distance' parameter."
            )
        if getattr(corridor_cfg, "time_out", False):
            raise ValueError(f"Termination term '{path_corridor_termination_name}' must not be a time-out.")
        return command_cfg, progress_cfg, precision_cfg, corridor_cfg

    @staticmethod
    def _validate_parameters(
        command_name: str,
        path_progress_reward_name: str,
        path_precision_reward_name: str,
        path_corridor_termination_name: str,
        warmup_steps: int,
        ramp_steps: int,
        initial_acceptance_radius: float,
        final_acceptance_radius: float,
        initial_maximum_rate: float,
        final_maximum_rate: float,
        initial_cross_track_scale: float,
        final_cross_track_scale: float,
        initial_transverse_velocity_scale: float,
        final_transverse_velocity_scale: float,
        initial_precision_reward_weight: float,
        final_precision_reward_weight: float,
        initial_corridor_distance: float,
        final_corridor_distance: float,
    ) -> None:
        """Validate schedule names, duration, bounds, and directions."""
        names = {
            "command_name": command_name,
            "path_progress_reward_name": path_progress_reward_name,
            "path_precision_reward_name": path_precision_reward_name,
            "path_corridor_termination_name": path_corridor_termination_name,
        }
        for parameter_name, value in names.items():
            if not isinstance(value, str) or not value:
                raise ValueError(f"{parameter_name} must be a non-empty string.")
        if isinstance(warmup_steps, bool) or not isinstance(warmup_steps, int) or warmup_steps < 0:
            raise ValueError("warmup_steps must be a non-negative integer.")
        if isinstance(ramp_steps, bool) or not isinstance(ramp_steps, int) or ramp_steps <= 0:
            raise ValueError("ramp_steps must be a positive integer.")

        endpoints = {
            "initial_acceptance_radius": initial_acceptance_radius,
            "final_acceptance_radius": final_acceptance_radius,
            "initial_maximum_rate": initial_maximum_rate,
            "final_maximum_rate": final_maximum_rate,
            "initial_cross_track_scale": initial_cross_track_scale,
            "final_cross_track_scale": final_cross_track_scale,
            "initial_transverse_velocity_scale": initial_transverse_velocity_scale,
            "final_transverse_velocity_scale": final_transverse_velocity_scale,
            "initial_corridor_distance": initial_corridor_distance,
            "final_corridor_distance": final_corridor_distance,
        }
        for parameter_name, value in endpoints.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(value)
                or value <= 0.0
            ):
                raise ValueError(f"{parameter_name} must be positive and finite.")

        weights = {
            "initial_precision_reward_weight": initial_precision_reward_weight,
            "final_precision_reward_weight": final_precision_reward_weight,
        }
        for parameter_name, value in weights.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(value)
                or value >= 0.0
            ):
                raise ValueError(f"{parameter_name} must be negative and finite.")

        if final_acceptance_radius > initial_acceptance_radius:
            raise ValueError("final_acceptance_radius must not exceed initial_acceptance_radius.")
        if final_maximum_rate < initial_maximum_rate:
            raise ValueError("final_maximum_rate must not be less than initial_maximum_rate.")
        if final_cross_track_scale > initial_cross_track_scale:
            raise ValueError("final_cross_track_scale must not exceed initial_cross_track_scale.")
        if final_transverse_velocity_scale > initial_transverse_velocity_scale:
            raise ValueError("final_transverse_velocity_scale must not exceed initial_transverse_velocity_scale.")
        if final_precision_reward_weight > initial_precision_reward_weight:
            raise ValueError("final_precision_reward_weight must not exceed initial_precision_reward_weight.")
        if final_corridor_distance > initial_corridor_distance:
            raise ValueError("final_corridor_distance must not exceed initial_corridor_distance.")

    @staticmethod
    def _global_step(env: ManagerBasedRLEnv) -> int:
        """Return a validated global environment control step."""
        step = env.common_step_counter
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError("env.common_step_counter must be a non-negative integer.")
        return step

    @staticmethod
    def _lerp(initial: float, final: float, fraction: float) -> float:
        """Linearly interpolate a scalar schedule endpoint."""
        if fraction <= 0.0:
            return initial
        if fraction >= 1.0:
            return final
        return initial + fraction * (final - initial)


class PrecisionSpeedCurriculumV13(PrecisionSpeedCurriculum):
    """Stage precision, precision strength, and speed with a competence gate.

    This additive v13 schedule leaves :class:`PrecisionSpeedCurriculum`
    backward compatible. It first tightens acceptance, tracking scales, and the
    safety corridor at the initial speed. It then strengthens the precision
    cost while all scales are fixed, holds that final-precision task, and only
    then raises both the signed-progress cap and the target cruise speed. The
    split avoids multiplying a 2.5x scale tightening and a 2x reward-weight
    increase in the same ramp.

    The optional speed gate is disabled by default, making the schedule
    deterministic and identical across asynchronously resetting distributed
    ranks. In a single-rank experiment it can be enabled to use completed-
    episode command metrics from the resetting environments. Requiring both
    route traversal and low cross-track RMS prevents a stationary policy near
    the route anchor from opening the gate. If a restored run has no serialized
    curriculum state, a competent batch safely starts a fresh speed ramp at the
    restored global step instead of jumping directly to a high target speed.
    """

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        ManagerTermBase.__init__(self, cfg, env)
        parameters = {
            "command_name": "route",
            "path_progress_reward_name": "path_progress",
            "path_precision_reward_name": "path_precision",
            "path_corridor_termination_name": "path_corridor",
            "warmup_steps": 20_000,
            "precision_geometry_ramp_steps": 40_000,
            "precision_weight_ramp_steps": 20_000,
            "precision_hold_steps": 20_000,
            "speed_ramp_steps": 60_000,
            "initial_acceptance_radius": 0.50,
            "final_acceptance_radius": 0.15,
            "initial_maximum_rate": 1.25,
            "final_maximum_rate": 3.50,
            "initial_target_cruise_speed": 1.25,
            "final_target_cruise_speed": 3.50,
            "initial_cross_track_scale": 0.50,
            "final_cross_track_scale": 0.20,
            "initial_transverse_velocity_scale": 1.00,
            "final_transverse_velocity_scale": 0.40,
            "initial_precision_reward_weight": -2.0,
            "final_precision_reward_weight": -4.0,
            "initial_corridor_distance": 1.50,
            "final_corridor_distance": 0.75,
            "performance_gate_enabled": False,
            "gate_cross_track_metric_name": "cross_track_error_rms",
            "gate_cross_track_threshold": 0.30,
            "gate_progress_metric_name": "route_traversal_fraction",
            "gate_progress_threshold": 0.25,
            "minimum_gate_pass_fraction": 0.50,
        }
        parameters.update(cfg.params)
        self._validate_v13_parameters(**parameters)
        command_term, *_ = self._get_v13_bound_configs(
            env,
            command_name=parameters["command_name"],
            path_progress_reward_name=parameters["path_progress_reward_name"],
            path_precision_reward_name=parameters["path_precision_reward_name"],
            path_corridor_termination_name=parameters["path_corridor_termination_name"],
        )
        if parameters["performance_gate_enabled"]:
            self._validate_gate_bindings(
                command_term,
                parameters["command_name"],
                parameters["gate_cross_track_metric_name"],
                parameters["gate_progress_metric_name"],
            )
        self._speed_unlocked_at: int | None = None
        self._last_gate_pass_fraction = 0.0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int] | torch.Tensor | slice,
        command_name: str = "route",
        path_progress_reward_name: str = "path_progress",
        path_precision_reward_name: str = "path_precision",
        path_corridor_termination_name: str = "path_corridor",
        warmup_steps: int = 20_000,
        precision_geometry_ramp_steps: int = 40_000,
        precision_weight_ramp_steps: int = 20_000,
        precision_hold_steps: int = 20_000,
        speed_ramp_steps: int = 60_000,
        initial_acceptance_radius: float = 0.50,
        final_acceptance_radius: float = 0.15,
        initial_maximum_rate: float = 1.25,
        final_maximum_rate: float = 3.50,
        initial_target_cruise_speed: float = 1.25,
        final_target_cruise_speed: float = 3.50,
        initial_cross_track_scale: float = 0.50,
        final_cross_track_scale: float = 0.20,
        initial_transverse_velocity_scale: float = 1.00,
        final_transverse_velocity_scale: float = 0.40,
        initial_precision_reward_weight: float = -2.0,
        final_precision_reward_weight: float = -4.0,
        initial_corridor_distance: float = 1.50,
        final_corridor_distance: float = 0.75,
        performance_gate_enabled: bool = False,
        gate_cross_track_metric_name: str = "cross_track_error_rms",
        gate_cross_track_threshold: float = 0.30,
        gate_progress_metric_name: str = "route_traversal_fraction",
        gate_progress_threshold: float = 0.25,
        minimum_gate_pass_fraction: float = 0.50,
    ) -> dict[str, float]:
        """Apply the staged schedule at the current global control step."""
        parameters = locals().copy()
        parameters.pop("self")
        parameters.pop("env")
        parameters.pop("env_ids")
        self._validate_v13_parameters(**parameters)

        command_term, command_cfg, progress_cfg, precision_cfg, corridor_cfg = self._get_v13_bound_configs(
            env,
            command_name=command_name,
            path_progress_reward_name=path_progress_reward_name,
            path_precision_reward_name=path_precision_reward_name,
            path_corridor_termination_name=path_corridor_termination_name,
        )
        step = self._global_step(env)
        geometry_start = warmup_steps
        weight_start = geometry_start + precision_geometry_ramp_steps
        hold_start = weight_start + precision_weight_ramp_steps
        nominal_speed_start = hold_start + precision_hold_steps
        geometry_fraction = self._smooth_fraction(step, geometry_start, precision_geometry_ramp_steps)
        weight_fraction = self._smooth_fraction(step, weight_start, precision_weight_ramp_steps)

        if performance_gate_enabled:
            self._validate_gate_bindings(
                command_term,
                command_name,
                gate_cross_track_metric_name,
                gate_progress_metric_name,
            )
            gate_pass_fraction = self._gate_pass_fraction(
                command_term,
                env_ids,
                gate_cross_track_metric_name,
                gate_cross_track_threshold,
                gate_progress_metric_name,
                gate_progress_threshold,
            )
            if gate_pass_fraction is not None:
                self._last_gate_pass_fraction = gate_pass_fraction
                if self._speed_unlocked_at is None and gate_pass_fraction >= minimum_gate_pass_fraction:
                    self._speed_unlocked_at = max(step, nominal_speed_start)
        else:
            self._last_gate_pass_fraction = 1.0
            self._speed_unlocked_at = nominal_speed_start

        speed_fraction = 0.0
        if self._speed_unlocked_at is not None:
            speed_fraction = self._smooth_fraction(step, self._speed_unlocked_at, speed_ramp_steps)
        values = {
            "stage": self._stage(
                step,
                geometry_start,
                weight_start,
                hold_start,
                self._speed_unlocked_at,
                speed_fraction,
            ),
            "precision_geometry_fraction": geometry_fraction,
            "precision_weight_fraction": weight_fraction,
            "speed_fraction": speed_fraction,
            "gate_pass_fraction": self._last_gate_pass_fraction,
            "gate_open": float(self._speed_unlocked_at is not None),
            "speed_stage_start_step": float(-1 if self._speed_unlocked_at is None else self._speed_unlocked_at),
            "acceptance_radius": self._lerp(initial_acceptance_radius, final_acceptance_radius, geometry_fraction),
            "maximum_rate": self._lerp(initial_maximum_rate, final_maximum_rate, speed_fraction),
            "target_cruise_speed": self._lerp(initial_target_cruise_speed, final_target_cruise_speed, speed_fraction),
            "cross_track_scale": self._lerp(initial_cross_track_scale, final_cross_track_scale, geometry_fraction),
            "transverse_velocity_scale": self._lerp(
                initial_transverse_velocity_scale, final_transverse_velocity_scale, geometry_fraction
            ),
            "precision_reward_weight": self._lerp(
                initial_precision_reward_weight, final_precision_reward_weight, weight_fraction
            ),
            "corridor_distance": self._lerp(initial_corridor_distance, final_corridor_distance, geometry_fraction),
        }

        if command_cfg.acceptance_radius != values["acceptance_radius"]:
            command_cfg.acceptance_radius = values["acceptance_radius"]
        if command_cfg.target_cruise_speed != values["target_cruise_speed"]:
            command_cfg.target_cruise_speed = values["target_cruise_speed"]
        if progress_cfg.params["maximum_rate"] != values["maximum_rate"]:
            progress_cfg = self._copy_reward_cfg(progress_cfg)
            progress_cfg.params["maximum_rate"] = values["maximum_rate"]
            env.reward_manager.set_term_cfg(path_progress_reward_name, progress_cfg)
        precision_changed = (
            precision_cfg.params["cross_track_scale"] != values["cross_track_scale"]
            or precision_cfg.params["transverse_velocity_scale"] != values["transverse_velocity_scale"]
            or precision_cfg.weight != values["precision_reward_weight"]
        )
        if precision_changed:
            precision_cfg = self._copy_reward_cfg(precision_cfg)
            precision_cfg.params["cross_track_scale"] = values["cross_track_scale"]
            precision_cfg.params["transverse_velocity_scale"] = values["transverse_velocity_scale"]
            precision_cfg.weight = values["precision_reward_weight"]
            env.reward_manager.set_term_cfg(path_precision_reward_name, precision_cfg)
        if corridor_cfg.params["maximum_distance"] != values["corridor_distance"]:
            corridor_cfg = self._copy_termination_cfg(corridor_cfg)
            corridor_cfg.params["maximum_distance"] = values["corridor_distance"]
            env.termination_manager.set_term_cfg(path_corridor_termination_name, corridor_cfg)
        return values

    @staticmethod
    def _get_v13_bound_configs(
        env: ManagerBasedRLEnv,
        command_name: str,
        path_progress_reward_name: str,
        path_precision_reward_name: str,
        path_corridor_termination_name: str,
    ) -> tuple[object, object, RewardTermCfg, RewardTermCfg, TerminationTermCfg]:
        command_cfg, progress_cfg, precision_cfg, corridor_cfg = PrecisionSpeedCurriculum._get_bound_configs(
            env,
            command_name,
            path_progress_reward_name,
            path_precision_reward_name,
            path_corridor_termination_name,
        )
        command_term = env.command_manager.get_term(command_name)
        if not hasattr(command_cfg, "target_cruise_speed"):
            raise ValueError(f"Command term '{command_name}' has no mutable 'target_cruise_speed' configuration.")
        return command_term, command_cfg, progress_cfg, precision_cfg, corridor_cfg

    @staticmethod
    def _validate_gate_bindings(
        command_term: object,
        command_name: str,
        cross_track_metric_name: str,
        progress_metric_name: str,
    ) -> None:
        metrics = getattr(command_term, "metrics", None)
        if metrics is None:
            raise ValueError(f"Command term '{command_name}' has no episode metrics for the performance gate.")
        missing = {cross_track_metric_name, progress_metric_name} - metrics.keys()
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"Command term '{command_name}' is missing performance-gate metrics: {names}.")

    @staticmethod
    def _gate_pass_fraction(
        command_term: object,
        env_ids: Sequence[int] | torch.Tensor | slice,
        cross_track_metric_name: str,
        cross_track_threshold: float,
        progress_metric_name: str,
        progress_threshold: float,
    ) -> float | None:
        cross_track = command_term.metrics[cross_track_metric_name][env_ids].reshape(-1)
        progress = command_term.metrics[progress_metric_name][env_ids].reshape(-1)
        if cross_track.numel() == 0:
            return None
        finite = torch.isfinite(cross_track) & torch.isfinite(progress)
        passing = finite & (cross_track <= cross_track_threshold) & (progress >= progress_threshold)
        return passing.float().mean().item()

    @staticmethod
    def _smooth_fraction(step: int, start: int, duration: int) -> float:
        linear = min(max(step - start, 0) / duration, 1.0)
        return linear * linear * (3.0 - 2.0 * linear)

    @staticmethod
    def _stage(
        step: int,
        geometry_start: int,
        weight_start: int,
        hold_start: int,
        speed_start: int | None,
        speed_fraction: float,
    ) -> float:
        if step < geometry_start:
            return 0.0
        if step < weight_start:
            return 1.0
        if step < hold_start:
            return 2.0
        if speed_start is None or step < speed_start:
            return 3.0
        if speed_fraction < 1.0:
            return 4.0
        return 5.0

    @staticmethod
    def _validate_v13_parameters(
        command_name: str,
        path_progress_reward_name: str,
        path_precision_reward_name: str,
        path_corridor_termination_name: str,
        warmup_steps: int,
        precision_geometry_ramp_steps: int,
        precision_weight_ramp_steps: int,
        precision_hold_steps: int,
        speed_ramp_steps: int,
        initial_acceptance_radius: float,
        final_acceptance_radius: float,
        initial_maximum_rate: float,
        final_maximum_rate: float,
        initial_target_cruise_speed: float,
        final_target_cruise_speed: float,
        initial_cross_track_scale: float,
        final_cross_track_scale: float,
        initial_transverse_velocity_scale: float,
        final_transverse_velocity_scale: float,
        initial_precision_reward_weight: float,
        final_precision_reward_weight: float,
        initial_corridor_distance: float,
        final_corridor_distance: float,
        performance_gate_enabled: bool,
        gate_cross_track_metric_name: str,
        gate_cross_track_threshold: float,
        gate_progress_metric_name: str,
        gate_progress_threshold: float,
        minimum_gate_pass_fraction: float,
    ) -> None:
        names = {
            "command_name": command_name,
            "path_progress_reward_name": path_progress_reward_name,
            "path_precision_reward_name": path_precision_reward_name,
            "path_corridor_termination_name": path_corridor_termination_name,
            "gate_cross_track_metric_name": gate_cross_track_metric_name,
            "gate_progress_metric_name": gate_progress_metric_name,
        }
        for name, value in names.items():
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string.")
        durations = {
            "warmup_steps": warmup_steps,
            "precision_geometry_ramp_steps": precision_geometry_ramp_steps,
            "precision_weight_ramp_steps": precision_weight_ramp_steps,
            "precision_hold_steps": precision_hold_steps,
            "speed_ramp_steps": speed_ramp_steps,
        }
        for name, value in durations.items():
            minimum = 0 if name in {"warmup_steps", "precision_hold_steps"} else 1
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                qualifier = "non-negative" if minimum == 0 else "positive"
                raise ValueError(f"{name} must be a {qualifier} integer.")
        endpoints = {
            "initial_acceptance_radius": initial_acceptance_radius,
            "final_acceptance_radius": final_acceptance_radius,
            "initial_maximum_rate": initial_maximum_rate,
            "final_maximum_rate": final_maximum_rate,
            "initial_target_cruise_speed": initial_target_cruise_speed,
            "final_target_cruise_speed": final_target_cruise_speed,
            "initial_cross_track_scale": initial_cross_track_scale,
            "final_cross_track_scale": final_cross_track_scale,
            "initial_transverse_velocity_scale": initial_transverse_velocity_scale,
            "final_transverse_velocity_scale": final_transverse_velocity_scale,
            "initial_corridor_distance": initial_corridor_distance,
            "final_corridor_distance": final_corridor_distance,
            "gate_cross_track_threshold": gate_cross_track_threshold,
        }
        for name, value in endpoints.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(value)
                or value <= 0.0
            ):
                raise ValueError(f"{name} must be positive and finite.")
        for name, value in {
            "gate_progress_threshold": gate_progress_threshold,
            "minimum_gate_pass_fraction": minimum_gate_pass_fraction,
        }.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(value)
                or not 0.0 < value <= 1.0
            ):
                raise ValueError(f"{name} must be finite and in (0, 1].")
        if not isinstance(performance_gate_enabled, bool):
            raise ValueError("performance_gate_enabled must be a bool.")
        for name, value in {
            "initial_precision_reward_weight": initial_precision_reward_weight,
            "final_precision_reward_weight": final_precision_reward_weight,
        }.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(value)
                or value >= 0.0
            ):
                raise ValueError(f"{name} must be negative and finite.")

        decreasing = {
            "acceptance_radius": (initial_acceptance_radius, final_acceptance_radius),
            "cross_track_scale": (initial_cross_track_scale, final_cross_track_scale),
            "transverse_velocity_scale": (
                initial_transverse_velocity_scale,
                final_transverse_velocity_scale,
            ),
            "corridor_distance": (initial_corridor_distance, final_corridor_distance),
        }
        for name, (initial, final) in decreasing.items():
            if final > initial:
                raise ValueError(f"final_{name} must not exceed initial_{name}.")
        increasing = {
            "maximum_rate": (initial_maximum_rate, final_maximum_rate),
            "target_cruise_speed": (initial_target_cruise_speed, final_target_cruise_speed),
        }
        for name, (initial, final) in increasing.items():
            if final < initial:
                raise ValueError(f"final_{name} must not be less than initial_{name}.")
        if final_precision_reward_weight > initial_precision_reward_weight:
            raise ValueError("final_precision_reward_weight must not exceed initial_precision_reward_weight.")


class DirectCTBRCurriculumV14(PrecisionSpeedCurriculum):
    """Stage direct-CTBR speed, precision, and reset-domain difficulty.

    The default 100 Hz schedule first exposes a route-completable speed and then
    the final aggressive speed while the tracking tube is still loose. It
    tightens geometric precision only after the speed target is fully exposed,
    strengthens the precision cost last, and then holds the complete task. The
    reset tilt and initial cable swing increase smoothly over the first 80,000
    control steps so a hover-initialized direct-rate actor is not given the
    residual controller's full reset disturbance on its first rollout.

    All values depend only on ``common_step_counter``. Event configurations are
    copied before their nested parameter dictionaries are changed, so reset
    callbacks cannot observe a partially mutated configuration and repeated
    calls at the same step are idempotent.
    """

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        ManagerTermBase.__init__(self, cfg, env)
        parameters = {
            "command_name": "route",
            "path_progress_reward_name": "path_progress",
            "path_precision_reward_name": "path_precision",
            "path_corridor_termination_name": "path_corridor",
            "reset_base_event_name": "reset_base",
            "reset_slung_load_event_name": "reset_slung_load",
            "warmup_steps": 10_000,
            "completion_speed_ramp_steps": 20_000,
            "aggressive_speed_ramp_steps": 50_000,
            "precision_geometry_ramp_steps": 40_000,
            "precision_weight_ramp_steps": 20_000,
            "reset_ramp_steps": 80_000,
            "initial_speed": 1.25,
            "completion_speed": 2.25,
            "final_speed": 4.50,
            "initial_acceptance_radius": 0.50,
            "final_acceptance_radius": 0.15,
            "initial_cross_track_scale": 0.50,
            "final_cross_track_scale": 0.20,
            "initial_transverse_velocity_scale": 1.00,
            "final_transverse_velocity_scale": 0.40,
            "initial_precision_reward_weight": -1.0,
            "final_precision_reward_weight": -4.0,
            "initial_corridor_distance": 1.50,
            "final_corridor_distance": 0.75,
            "initial_reset_tilt_limit": 0.005,
            "final_reset_tilt_limit": 0.050,
            "initial_max_initial_swing": 0.020,
            "final_max_initial_swing": 0.100,
        }
        parameters.update(cfg.params)
        self._validate_v14_parameters(**parameters)
        self._get_v14_bound_configs(
            env,
            command_name=parameters["command_name"],
            path_progress_reward_name=parameters["path_progress_reward_name"],
            path_precision_reward_name=parameters["path_precision_reward_name"],
            path_corridor_termination_name=parameters["path_corridor_termination_name"],
            reset_base_event_name=parameters["reset_base_event_name"],
            reset_slung_load_event_name=parameters["reset_slung_load_event_name"],
        )

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        _env_ids: Sequence[int] | torch.Tensor | slice,
        command_name: str = "route",
        path_progress_reward_name: str = "path_progress",
        path_precision_reward_name: str = "path_precision",
        path_corridor_termination_name: str = "path_corridor",
        reset_base_event_name: str = "reset_base",
        reset_slung_load_event_name: str = "reset_slung_load",
        warmup_steps: int = 10_000,
        completion_speed_ramp_steps: int = 20_000,
        aggressive_speed_ramp_steps: int = 50_000,
        precision_geometry_ramp_steps: int = 40_000,
        precision_weight_ramp_steps: int = 20_000,
        reset_ramp_steps: int = 80_000,
        initial_speed: float = 1.25,
        completion_speed: float = 2.25,
        final_speed: float = 4.50,
        initial_acceptance_radius: float = 0.50,
        final_acceptance_radius: float = 0.15,
        initial_cross_track_scale: float = 0.50,
        final_cross_track_scale: float = 0.20,
        initial_transverse_velocity_scale: float = 1.00,
        final_transverse_velocity_scale: float = 0.40,
        initial_precision_reward_weight: float = -1.0,
        final_precision_reward_weight: float = -4.0,
        initial_corridor_distance: float = 1.50,
        final_corridor_distance: float = 0.75,
        initial_reset_tilt_limit: float = 0.005,
        final_reset_tilt_limit: float = 0.050,
        initial_max_initial_swing: float = 0.020,
        final_max_initial_swing: float = 0.100,
    ) -> dict[str, float]:
        """Apply and return the direct-CTBR schedule at the current step."""
        parameters = locals().copy()
        parameters.pop("self")
        parameters.pop("env")
        parameters.pop("_env_ids")
        self._validate_v14_parameters(**parameters)

        (
            _command_term,
            command_cfg,
            progress_cfg,
            precision_cfg,
            corridor_cfg,
            reset_base_cfg,
            reset_slung_load_cfg,
        ) = self._get_v14_bound_configs(
            env,
            command_name=command_name,
            path_progress_reward_name=path_progress_reward_name,
            path_precision_reward_name=path_precision_reward_name,
            path_corridor_termination_name=path_corridor_termination_name,
            reset_base_event_name=reset_base_event_name,
            reset_slung_load_event_name=reset_slung_load_event_name,
        )

        step = self._global_step(env)
        completion_speed_start = warmup_steps
        aggressive_speed_start = completion_speed_start + completion_speed_ramp_steps
        precision_geometry_start = aggressive_speed_start + aggressive_speed_ramp_steps
        precision_weight_start = precision_geometry_start + precision_geometry_ramp_steps
        hold_start = precision_weight_start + precision_weight_ramp_steps

        completion_speed_fraction = PrecisionSpeedCurriculumV13._smooth_fraction(
            step, completion_speed_start, completion_speed_ramp_steps
        )
        aggressive_speed_fraction = PrecisionSpeedCurriculumV13._smooth_fraction(
            step, aggressive_speed_start, aggressive_speed_ramp_steps
        )
        precision_geometry_fraction = PrecisionSpeedCurriculumV13._smooth_fraction(
            step, precision_geometry_start, precision_geometry_ramp_steps
        )
        precision_weight_fraction = PrecisionSpeedCurriculumV13._smooth_fraction(
            step, precision_weight_start, precision_weight_ramp_steps
        )
        reset_fraction = PrecisionSpeedCurriculumV13._smooth_fraction(step, 0, reset_ramp_steps)

        if step < aggressive_speed_start:
            speed = self._lerp(initial_speed, completion_speed, completion_speed_fraction)
        else:
            speed = self._lerp(completion_speed, final_speed, aggressive_speed_fraction)
        reset_tilt_limit = self._lerp(initial_reset_tilt_limit, final_reset_tilt_limit, reset_fraction)
        max_initial_swing = self._lerp(initial_max_initial_swing, final_max_initial_swing, reset_fraction)
        values = {
            "stage": self._v14_stage(
                step,
                completion_speed_start,
                aggressive_speed_start,
                precision_geometry_start,
                precision_weight_start,
                hold_start,
            ),
            "completion_speed_fraction": completion_speed_fraction,
            "aggressive_speed_fraction": aggressive_speed_fraction,
            "precision_geometry_fraction": precision_geometry_fraction,
            "precision_weight_fraction": precision_weight_fraction,
            "reset_fraction": reset_fraction,
            "acceptance_radius": self._lerp(
                initial_acceptance_radius, final_acceptance_radius, precision_geometry_fraction
            ),
            "maximum_rate": speed,
            "target_cruise_speed": speed,
            "cross_track_scale": self._lerp(
                initial_cross_track_scale, final_cross_track_scale, precision_geometry_fraction
            ),
            "transverse_velocity_scale": self._lerp(
                initial_transverse_velocity_scale,
                final_transverse_velocity_scale,
                precision_geometry_fraction,
            ),
            "precision_reward_weight": self._lerp(
                initial_precision_reward_weight,
                final_precision_reward_weight,
                precision_weight_fraction,
            ),
            "corridor_distance": self._lerp(
                initial_corridor_distance, final_corridor_distance, precision_geometry_fraction
            ),
            "reset_tilt_limit": reset_tilt_limit,
            "max_initial_swing": max_initial_swing,
        }

        if command_cfg.acceptance_radius != values["acceptance_radius"]:
            command_cfg.acceptance_radius = values["acceptance_radius"]
        if command_cfg.target_cruise_speed != values["target_cruise_speed"]:
            command_cfg.target_cruise_speed = values["target_cruise_speed"]
        if progress_cfg.params["maximum_rate"] != values["maximum_rate"]:
            progress_cfg = self._copy_reward_cfg(progress_cfg)
            progress_cfg.params["maximum_rate"] = values["maximum_rate"]
            env.reward_manager.set_term_cfg(path_progress_reward_name, progress_cfg)

        precision_changed = (
            precision_cfg.params["cross_track_scale"] != values["cross_track_scale"]
            or precision_cfg.params["transverse_velocity_scale"] != values["transverse_velocity_scale"]
            or precision_cfg.weight != values["precision_reward_weight"]
        )
        if precision_changed:
            precision_cfg = self._copy_reward_cfg(precision_cfg)
            precision_cfg.params["cross_track_scale"] = values["cross_track_scale"]
            precision_cfg.params["transverse_velocity_scale"] = values["transverse_velocity_scale"]
            precision_cfg.weight = values["precision_reward_weight"]
            env.reward_manager.set_term_cfg(path_precision_reward_name, precision_cfg)
        if corridor_cfg.params["maximum_distance"] != values["corridor_distance"]:
            corridor_cfg = self._copy_termination_cfg(corridor_cfg)
            corridor_cfg.params["maximum_distance"] = values["corridor_distance"]
            env.termination_manager.set_term_cfg(path_corridor_termination_name, corridor_cfg)

        reset_range = (-reset_tilt_limit, reset_tilt_limit)
        if reset_base_cfg.params["roll_range"] != reset_range or reset_base_cfg.params["pitch_range"] != reset_range:
            reset_base_cfg = self._copy_event_cfg(reset_base_cfg)
            reset_base_cfg.params["roll_range"] = reset_range
            reset_base_cfg.params["pitch_range"] = reset_range
            env.event_manager.set_term_cfg(reset_base_event_name, reset_base_cfg)
        if reset_slung_load_cfg.params["max_initial_swing"] != max_initial_swing:
            reset_slung_load_cfg = self._copy_event_cfg(reset_slung_load_cfg)
            reset_slung_load_cfg.params["max_initial_swing"] = max_initial_swing
            env.event_manager.set_term_cfg(reset_slung_load_event_name, reset_slung_load_cfg)
        return values

    @staticmethod
    def _copy_event_cfg(cfg: EventTermCfg) -> EventTermCfg:
        """Copy an event configuration and its mutable parameter dictionary."""
        copied = copy.copy(cfg)
        copied.params = cfg.params.copy()
        return copied

    @staticmethod
    def _get_v14_bound_configs(
        env: ManagerBasedRLEnv,
        command_name: str,
        path_progress_reward_name: str,
        path_precision_reward_name: str,
        path_corridor_termination_name: str,
        reset_base_event_name: str,
        reset_slung_load_event_name: str,
    ) -> tuple[object, object, RewardTermCfg, RewardTermCfg, TerminationTermCfg, EventTermCfg, EventTermCfg]:
        """Resolve every live configuration before any schedule mutation."""
        bound = PrecisionSpeedCurriculumV13._get_v13_bound_configs(
            env,
            command_name,
            path_progress_reward_name,
            path_precision_reward_name,
            path_corridor_termination_name,
        )
        try:
            reset_base_cfg = env.event_manager.get_term_cfg(reset_base_event_name)
        except (AttributeError, KeyError, ValueError) as error:
            raise ValueError(
                f"Reset event term '{reset_base_event_name}' required by the curriculum was not found."
            ) from error
        try:
            reset_slung_load_cfg = env.event_manager.get_term_cfg(reset_slung_load_event_name)
        except (AttributeError, KeyError, ValueError) as error:
            raise ValueError(
                f"Reset event term '{reset_slung_load_event_name}' required by the curriculum was not found."
            ) from error

        for parameter_name in ("roll_range", "pitch_range"):
            if parameter_name not in reset_base_cfg.params:
                raise ValueError(f"Reset event term '{reset_base_event_name}' is missing parameter '{parameter_name}'.")
            value = reset_base_cfg.params[parameter_name]
            if (
                not isinstance(value, Sequence)
                or isinstance(value, str | bytes)
                or len(value) != 2
                or any(
                    isinstance(endpoint, bool) or not isinstance(endpoint, int | float) or not math.isfinite(endpoint)
                    for endpoint in value
                )
            ):
                raise ValueError(
                    f"Reset event term '{reset_base_event_name}' parameter '{parameter_name}' "
                    "must contain two finite numbers."
                )
        if "max_initial_swing" not in reset_slung_load_cfg.params:
            raise ValueError(
                f"Reset event term '{reset_slung_load_event_name}' is missing parameter 'max_initial_swing'."
            )
        max_initial_swing = reset_slung_load_cfg.params["max_initial_swing"]
        if (
            isinstance(max_initial_swing, bool)
            or not isinstance(max_initial_swing, int | float)
            or not math.isfinite(max_initial_swing)
            or max_initial_swing < 0.0
        ):
            raise ValueError(
                f"Reset event term '{reset_slung_load_event_name}' parameter 'max_initial_swing' "
                "must be finite and nonnegative."
            )
        return (*bound, reset_base_cfg, reset_slung_load_cfg)

    @staticmethod
    def _v14_stage(
        step: int,
        completion_speed_start: int,
        aggressive_speed_start: int,
        precision_geometry_start: int,
        precision_weight_start: int,
        hold_start: int,
    ) -> float:
        if step < completion_speed_start:
            return 0.0
        if step < aggressive_speed_start:
            return 1.0
        if step < precision_geometry_start:
            return 2.0
        if step < precision_weight_start:
            return 3.0
        if step < hold_start:
            return 4.0
        return 5.0

    @staticmethod
    def _validate_v14_parameters(
        command_name: str,
        path_progress_reward_name: str,
        path_precision_reward_name: str,
        path_corridor_termination_name: str,
        reset_base_event_name: str,
        reset_slung_load_event_name: str,
        warmup_steps: int,
        completion_speed_ramp_steps: int,
        aggressive_speed_ramp_steps: int,
        precision_geometry_ramp_steps: int,
        precision_weight_ramp_steps: int,
        reset_ramp_steps: int,
        initial_speed: float,
        completion_speed: float,
        final_speed: float,
        initial_acceptance_radius: float,
        final_acceptance_radius: float,
        initial_cross_track_scale: float,
        final_cross_track_scale: float,
        initial_transverse_velocity_scale: float,
        final_transverse_velocity_scale: float,
        initial_precision_reward_weight: float,
        final_precision_reward_weight: float,
        initial_corridor_distance: float,
        final_corridor_distance: float,
        initial_reset_tilt_limit: float,
        final_reset_tilt_limit: float,
        initial_max_initial_swing: float,
        final_max_initial_swing: float,
    ) -> None:
        names = {
            "command_name": command_name,
            "path_progress_reward_name": path_progress_reward_name,
            "path_precision_reward_name": path_precision_reward_name,
            "path_corridor_termination_name": path_corridor_termination_name,
            "reset_base_event_name": reset_base_event_name,
            "reset_slung_load_event_name": reset_slung_load_event_name,
        }
        for name, value in names.items():
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string.")

        durations = {
            "warmup_steps": warmup_steps,
            "completion_speed_ramp_steps": completion_speed_ramp_steps,
            "aggressive_speed_ramp_steps": aggressive_speed_ramp_steps,
            "precision_geometry_ramp_steps": precision_geometry_ramp_steps,
            "precision_weight_ramp_steps": precision_weight_ramp_steps,
            "reset_ramp_steps": reset_ramp_steps,
        }
        for name, value in durations.items():
            minimum = 0 if name == "warmup_steps" else 1
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                qualifier = "non-negative" if minimum == 0 else "positive"
                raise ValueError(f"{name} must be a {qualifier} integer.")

        positive_endpoints = {
            "initial_speed": initial_speed,
            "completion_speed": completion_speed,
            "final_speed": final_speed,
            "initial_acceptance_radius": initial_acceptance_radius,
            "final_acceptance_radius": final_acceptance_radius,
            "initial_cross_track_scale": initial_cross_track_scale,
            "final_cross_track_scale": final_cross_track_scale,
            "initial_transverse_velocity_scale": initial_transverse_velocity_scale,
            "final_transverse_velocity_scale": final_transverse_velocity_scale,
            "initial_corridor_distance": initial_corridor_distance,
            "final_corridor_distance": final_corridor_distance,
            "initial_reset_tilt_limit": initial_reset_tilt_limit,
            "final_reset_tilt_limit": final_reset_tilt_limit,
            "initial_max_initial_swing": initial_max_initial_swing,
            "final_max_initial_swing": final_max_initial_swing,
        }
        for name, value in positive_endpoints.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(value)
                or value <= 0.0
            ):
                raise ValueError(f"{name} must be positive and finite.")

        for name, value in {
            "initial_precision_reward_weight": initial_precision_reward_weight,
            "final_precision_reward_weight": final_precision_reward_weight,
        }.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(value)
                or value >= 0.0
            ):
                raise ValueError(f"{name} must be negative and finite.")

        if completion_speed < initial_speed:
            raise ValueError("completion_speed must not be less than initial_speed.")
        if final_speed < completion_speed:
            raise ValueError("final_speed must not be less than completion_speed.")
        for name, initial, final in (
            ("acceptance_radius", initial_acceptance_radius, final_acceptance_radius),
            ("cross_track_scale", initial_cross_track_scale, final_cross_track_scale),
            (
                "transverse_velocity_scale",
                initial_transverse_velocity_scale,
                final_transverse_velocity_scale,
            ),
            ("corridor_distance", initial_corridor_distance, final_corridor_distance),
        ):
            if final > initial:
                raise ValueError(f"final_{name} must not exceed initial_{name}.")
        if final_precision_reward_weight > initial_precision_reward_weight:
            raise ValueError("final_precision_reward_weight must not exceed initial_precision_reward_weight.")
        if final_reset_tilt_limit < initial_reset_tilt_limit:
            raise ValueError("final_reset_tilt_limit must not be less than initial_reset_tilt_limit.")
        if final_max_initial_swing < initial_max_initial_swing:
            raise ValueError("final_max_initial_swing must not be less than initial_max_initial_swing.")
        if final_reset_tilt_limit >= 0.5 * math.pi:
            raise ValueError("final_reset_tilt_limit must be less than pi / 2.")
        if final_max_initial_swing >= 0.5 * math.pi:
            raise ValueError("final_max_initial_swing must be less than pi / 2.")


class DirectCTBRRouteCurriculum(ManagerTermBase):
    """Expose smooth routes first, then add bounded sharp-turn routes.

    The schedule starts from planar random-heading straight routes at 1.0 m/s,
    switches to planar ellipses, then tightens waypoint acceptance and the
    safety corridor, introduces figure-eights and vertical motion, and raises
    speed. The first 200,000 control steps preserve the v17 schedule exactly.
    A continuation then consolidates figure-eights at a reduced speed, ramps in
    bounded random corners and their turn severity, restores full speed, and
    holds the final task. Every value is a pure function of the global control-
    step counter, making the schedule deterministic across distributed ranks
    and restartable without serialized curriculum state.

    Route-family changes update the command configuration and take effect on
    each environment's next reset. Active routes are not rebuilt mid-episode,
    which preserves indexed progress and reward-potential continuity.

    Unlike :class:`DirectCTBRCurriculumV14`, this shared route curriculum has no
    event-manager or payload-reset dependency. Drone-only and slung-load tasks
    bind the same command, progress-reward, and corridor-termination API.
    """

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        parameters = {
            "command_name": "route",
            "path_progress_reward_name": "path_progress",
            "path_corridor_termination_name": "path_corridor",
            "foundation_steps": 40_000,
            "straight_foundation_steps": 20_000,
            "acceptance_ramp_steps": 30_000,
            "corridor_ramp_steps": 20_000,
            "figure_eight_ramp_steps": 30_000,
            "vertical_ramp_steps": 20_000,
            "speed_ramp_steps": 40_000,
            "easy_hold_steps": 20_000,
            "hard_bridge_steps": 20_000,
            "hard_geometry_ramp_steps": 40_000,
            "hard_speed_ramp_steps": 60_000,
            "hard_hold_steps": 20_000,
            "initial_acceptance_radius": 1.00,
            "final_acceptance_radius": 0.50,
            "initial_corridor_distance": 2.50,
            "final_corridor_distance": 1.50,
            "initial_figure_eight_probability": 0.0,
            "final_figure_eight_probability": 0.50,
            "minimum_vertical_amplitude": 0.0,
            "initial_maximum_vertical_amplitude": 0.0,
            "final_maximum_vertical_amplitude": 0.15,
            "initial_speed": 1.00,
            "final_speed": 3.50,
            "initial_hard_speed": 2.25,
            "initial_hard_figure_eight_probability": 1.00,
            "final_hard_figure_eight_probability": 0.50,
            "hard_xy_bound": 4.80,
            "hard_z_bound": 0.40,
            "hard_minimum_waypoint_separation": 0.90,
            "hard_maximum_waypoint_separation": 1.35,
            "hard_maximum_vertical_step": 0.15,
            "hard_random_sampling_attempts": 32,
            "hard_route_sampling_attempts": 16,
            "hard_heading_change_interval": 3,
            "initial_hard_nominal_heading_change": math.radians(40.0),
            "final_hard_nominal_heading_change": math.radians(100.0),
            "initial_hard_maximum_heading_change": math.radians(60.0),
            "final_hard_maximum_heading_change": math.radians(110.0),
        }
        parameters.update(cfg.params)
        self._validate_route_parameters(**parameters)
        self._get_route_bound_configs(
            env,
            command_name=parameters["command_name"],
            path_progress_reward_name=parameters["path_progress_reward_name"],
            path_corridor_termination_name=parameters["path_corridor_termination_name"],
        )

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        _env_ids: Sequence[int] | torch.Tensor | slice,
        command_name: str = "route",
        path_progress_reward_name: str = "path_progress",
        path_corridor_termination_name: str = "path_corridor",
        foundation_steps: int = 40_000,
        straight_foundation_steps: int = 20_000,
        acceptance_ramp_steps: int = 30_000,
        corridor_ramp_steps: int = 20_000,
        figure_eight_ramp_steps: int = 30_000,
        vertical_ramp_steps: int = 20_000,
        speed_ramp_steps: int = 40_000,
        easy_hold_steps: int = 20_000,
        hard_bridge_steps: int = 20_000,
        hard_geometry_ramp_steps: int = 40_000,
        hard_speed_ramp_steps: int = 60_000,
        hard_hold_steps: int = 20_000,
        initial_acceptance_radius: float = 1.00,
        final_acceptance_radius: float = 0.50,
        initial_corridor_distance: float = 2.50,
        final_corridor_distance: float = 1.50,
        initial_figure_eight_probability: float = 0.0,
        final_figure_eight_probability: float = 0.50,
        minimum_vertical_amplitude: float = 0.0,
        initial_maximum_vertical_amplitude: float = 0.0,
        final_maximum_vertical_amplitude: float = 0.15,
        initial_speed: float = 1.00,
        final_speed: float = 3.50,
        initial_hard_speed: float = 2.25,
        initial_hard_figure_eight_probability: float = 1.00,
        final_hard_figure_eight_probability: float = 0.50,
        hard_xy_bound: float = 4.80,
        hard_z_bound: float = 0.40,
        hard_minimum_waypoint_separation: float = 0.90,
        hard_maximum_waypoint_separation: float = 1.35,
        hard_maximum_vertical_step: float = 0.15,
        hard_random_sampling_attempts: int = 32,
        hard_route_sampling_attempts: int = 16,
        hard_heading_change_interval: int = 3,
        initial_hard_nominal_heading_change: float = math.radians(40.0),
        final_hard_nominal_heading_change: float = math.radians(100.0),
        initial_hard_maximum_heading_change: float = math.radians(60.0),
        final_hard_maximum_heading_change: float = math.radians(110.0),
    ) -> dict[str, float]:
        """Apply and return the route-first schedule at the global step."""
        parameters = locals().copy()
        parameters.pop("self")
        parameters.pop("env")
        parameters.pop("_env_ids")
        self._validate_route_parameters(**parameters)
        command_cfg, progress_cfg, corridor_cfg = self._get_route_bound_configs(
            env,
            command_name=command_name,
            path_progress_reward_name=path_progress_reward_name,
            path_corridor_termination_name=path_corridor_termination_name,
        )

        step = PrecisionSpeedCurriculum._global_step(env)
        acceptance_start = foundation_steps
        corridor_start = acceptance_start + acceptance_ramp_steps
        figure_eight_start = corridor_start + corridor_ramp_steps
        vertical_start = figure_eight_start + figure_eight_ramp_steps
        speed_start = vertical_start + vertical_ramp_steps
        easy_hold_start = speed_start + speed_ramp_steps
        hard_route_start = easy_hold_start + easy_hold_steps
        hard_geometry_start = hard_route_start + hard_bridge_steps
        hard_speed_start = hard_geometry_start + hard_geometry_ramp_steps
        hard_hold_start = hard_speed_start + hard_speed_ramp_steps
        hard_end = hard_hold_start + hard_hold_steps
        acceptance_fraction = PrecisionSpeedCurriculumV13._smooth_fraction(
            step, acceptance_start, acceptance_ramp_steps
        )
        corridor_fraction = PrecisionSpeedCurriculumV13._smooth_fraction(step, corridor_start, corridor_ramp_steps)
        figure_eight_fraction = PrecisionSpeedCurriculumV13._smooth_fraction(
            step, figure_eight_start, figure_eight_ramp_steps
        )
        vertical_fraction = PrecisionSpeedCurriculumV13._smooth_fraction(step, vertical_start, vertical_ramp_steps)
        speed_fraction = PrecisionSpeedCurriculumV13._smooth_fraction(step, speed_start, speed_ramp_steps)
        hard_geometry_fraction = PrecisionSpeedCurriculumV13._smooth_fraction(
            step, hard_geometry_start, hard_geometry_ramp_steps
        )
        hard_speed_fraction = PrecisionSpeedCurriculumV13._smooth_fraction(
            step, hard_speed_start, hard_speed_ramp_steps
        )
        if step < hard_route_start:
            figure_eight_probability = PrecisionSpeedCurriculum._lerp(
                initial_figure_eight_probability,
                final_figure_eight_probability,
                figure_eight_fraction,
            )
            target_speed = PrecisionSpeedCurriculum._lerp(initial_speed, final_speed, speed_fraction)
            nominal_heading_change = 0.0
            maximum_heading_change = 0.0
        else:
            figure_eight_probability = PrecisionSpeedCurriculum._lerp(
                initial_hard_figure_eight_probability,
                final_hard_figure_eight_probability,
                hard_geometry_fraction,
            )
            target_speed = PrecisionSpeedCurriculum._lerp(initial_hard_speed, final_speed, hard_speed_fraction)
            nominal_heading_change = PrecisionSpeedCurriculum._lerp(
                initial_hard_nominal_heading_change,
                final_hard_nominal_heading_change,
                hard_geometry_fraction,
            )
            maximum_heading_change = PrecisionSpeedCurriculum._lerp(
                initial_hard_maximum_heading_change,
                final_hard_maximum_heading_change,
                hard_geometry_fraction,
            )
        values = {
            "stage": self._route_stage(
                step,
                straight_foundation_steps,
                acceptance_start,
                corridor_start,
                figure_eight_start,
                vertical_start,
                speed_start,
                easy_hold_start,
                hard_route_start,
                hard_geometry_start,
                hard_speed_start,
                hard_hold_start,
                hard_end,
            ),
            "acceptance_fraction": acceptance_fraction,
            "corridor_fraction": corridor_fraction,
            "figure_eight_fraction": figure_eight_fraction,
            "vertical_fraction": vertical_fraction,
            "speed_fraction": speed_fraction,
            "hard_geometry_fraction": hard_geometry_fraction,
            "hard_speed_fraction": hard_speed_fraction,
            "acceptance_radius": PrecisionSpeedCurriculum._lerp(
                initial_acceptance_radius, final_acceptance_radius, acceptance_fraction
            ),
            "corridor_distance": PrecisionSpeedCurriculum._lerp(
                initial_corridor_distance, final_corridor_distance, corridor_fraction
            ),
            "figure_eight_probability": figure_eight_probability,
            "maximum_vertical_amplitude": PrecisionSpeedCurriculum._lerp(
                initial_maximum_vertical_amplitude,
                final_maximum_vertical_amplitude,
                vertical_fraction,
            ),
            "target_cruise_speed": target_speed,
            "maximum_rate": target_speed,
            "nominal_heading_change": nominal_heading_change,
            "maximum_heading_change": maximum_heading_change,
        }

        if step < straight_foundation_steps:
            route_family = "random_walk"
        elif step < hard_route_start:
            route_family = "bounded_template_mix"
        else:
            route_family = "bounded_hard_mix"
        if command_cfg.route_family != route_family:
            command_cfg.route_family = route_family
        if command_cfg.acceptance_radius != values["acceptance_radius"]:
            command_cfg.acceptance_radius = values["acceptance_radius"]
        if command_cfg.figure_eight_probability != values["figure_eight_probability"]:
            command_cfg.figure_eight_probability = values["figure_eight_probability"]
        vertical_range = (minimum_vertical_amplitude, values["maximum_vertical_amplitude"])
        if command_cfg.vertical_amplitude_range != vertical_range:
            command_cfg.vertical_amplitude_range = vertical_range
        if command_cfg.target_cruise_speed != values["target_cruise_speed"]:
            command_cfg.target_cruise_speed = values["target_cruise_speed"]
        if step >= hard_route_start:
            hard_ranges = command_cfg.random_waypoint_ranges
            hard_pos_x = (-hard_xy_bound, hard_xy_bound)
            hard_pos_y = (-hard_xy_bound, hard_xy_bound)
            hard_pos_z = (-hard_z_bound, hard_z_bound)
            if hard_ranges.pos_x != hard_pos_x:
                hard_ranges.pos_x = hard_pos_x
            if hard_ranges.pos_y != hard_pos_y:
                hard_ranges.pos_y = hard_pos_y
            if hard_ranges.pos_z != hard_pos_z:
                hard_ranges.pos_z = hard_pos_z
            if command_cfg.minimum_waypoint_separation != hard_minimum_waypoint_separation:
                command_cfg.minimum_waypoint_separation = hard_minimum_waypoint_separation
            if command_cfg.maximum_waypoint_separation != hard_maximum_waypoint_separation:
                command_cfg.maximum_waypoint_separation = hard_maximum_waypoint_separation
            if command_cfg.maximum_vertical_step != hard_maximum_vertical_step:
                command_cfg.maximum_vertical_step = hard_maximum_vertical_step
            if command_cfg.random_sampling_attempts != hard_random_sampling_attempts:
                command_cfg.random_sampling_attempts = hard_random_sampling_attempts
            if command_cfg.route_sampling_attempts != hard_route_sampling_attempts:
                command_cfg.route_sampling_attempts = hard_route_sampling_attempts
            if command_cfg.random_heading_change_interval != hard_heading_change_interval:
                command_cfg.random_heading_change_interval = hard_heading_change_interval
            if command_cfg.nominal_heading_change != values["nominal_heading_change"]:
                command_cfg.nominal_heading_change = values["nominal_heading_change"]
            if command_cfg.maximum_heading_change != values["maximum_heading_change"]:
                command_cfg.maximum_heading_change = values["maximum_heading_change"]
        if progress_cfg.params["maximum_rate"] != values["maximum_rate"]:
            progress_cfg = PrecisionSpeedCurriculum._copy_reward_cfg(progress_cfg)
            progress_cfg.params["maximum_rate"] = values["maximum_rate"]
            env.reward_manager.set_term_cfg(path_progress_reward_name, progress_cfg)
        if corridor_cfg.params["maximum_distance"] != values["corridor_distance"]:
            corridor_cfg = PrecisionSpeedCurriculum._copy_termination_cfg(corridor_cfg)
            corridor_cfg.params["maximum_distance"] = values["corridor_distance"]
            env.termination_manager.set_term_cfg(path_corridor_termination_name, corridor_cfg)
        return values

    @staticmethod
    def _get_route_bound_configs(
        env: ManagerBasedRLEnv,
        command_name: str,
        path_progress_reward_name: str,
        path_corridor_termination_name: str,
    ) -> tuple[object, RewardTermCfg, TerminationTermCfg]:
        """Resolve the payload-independent live configurations before mutation."""
        try:
            command_cfg = env.command_manager.get_term(command_name).cfg
        except (AttributeError, KeyError, ValueError) as error:
            raise ValueError(f"Command term '{command_name}' required by the curriculum was not found.") from error
        required_attributes = (
            "acceptance_radius",
            "figure_eight_probability",
            "route_family",
            "target_cruise_speed",
            "vertical_amplitude_range",
            "random_waypoint_ranges",
            "minimum_waypoint_separation",
            "maximum_waypoint_separation",
            "nominal_heading_change",
            "maximum_heading_change",
            "maximum_vertical_step",
            "random_sampling_attempts",
            "route_sampling_attempts",
            "random_heading_change_interval",
        )
        missing_attributes = {name for name in required_attributes if not hasattr(command_cfg, name)}
        if missing_attributes:
            missing = ", ".join(sorted(missing_attributes))
            raise ValueError(f"Command term '{command_name}' is missing mutable route parameters: {missing}.")
        if command_cfg.route_family not in {"random_walk", "bounded_template_mix", "bounded_hard_mix"}:
            raise ValueError(
                f"Command term '{command_name}' must use route_family='random_walk' or "
                "one of the bounded staged route mixes."
            )
        vertical_range = command_cfg.vertical_amplitude_range
        if (
            not isinstance(vertical_range, Sequence)
            or isinstance(vertical_range, str | bytes)
            or len(vertical_range) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value)
                for value in vertical_range
            )
        ):
            raise ValueError(f"Command term '{command_name}' vertical_amplitude_range must contain two finite numbers.")

        try:
            progress_cfg = env.reward_manager.get_term_cfg(path_progress_reward_name)
        except (AttributeError, KeyError, ValueError) as error:
            raise ValueError(
                f"Reward term '{path_progress_reward_name}' required by the curriculum was not found."
            ) from error
        if "maximum_rate" not in progress_cfg.params:
            raise ValueError(f"Reward term '{path_progress_reward_name}' has no 'maximum_rate' parameter.")

        try:
            corridor_cfg = env.termination_manager.get_term_cfg(path_corridor_termination_name)
        except (AttributeError, KeyError, ValueError) as error:
            raise ValueError(
                f"Termination term '{path_corridor_termination_name}' required by the curriculum was not found."
            ) from error
        if "maximum_distance" not in corridor_cfg.params:
            raise ValueError(
                f"Termination term '{path_corridor_termination_name}' has no 'maximum_distance' parameter."
            )
        if getattr(corridor_cfg, "time_out", False):
            raise ValueError(f"Termination term '{path_corridor_termination_name}' must not be a time-out.")
        return command_cfg, progress_cfg, corridor_cfg

    @staticmethod
    def _route_stage(
        step: int,
        straight_foundation_end: int,
        acceptance_start: int,
        corridor_start: int,
        figure_eight_start: int,
        vertical_start: int,
        speed_start: int,
        easy_hold_start: int,
        hard_route_start: int,
        hard_geometry_start: int,
        hard_speed_start: int,
        hard_hold_start: int,
        hard_end: int,
    ) -> float:
        if step < straight_foundation_end:
            return 0.0
        if step < acceptance_start:
            return 1.0
        if step < corridor_start:
            return 2.0
        if step < figure_eight_start:
            return 3.0
        if step < vertical_start:
            return 4.0
        if step < speed_start:
            return 5.0
        if step < easy_hold_start:
            return 6.0
        if step < hard_route_start:
            return 7.0
        if step < hard_geometry_start:
            return 8.0
        if step < hard_speed_start:
            return 9.0
        if step < hard_hold_start:
            return 10.0
        if step < hard_end:
            return 11.0
        return 12.0

    @staticmethod
    def _validate_route_parameters(
        command_name: str,
        path_progress_reward_name: str,
        path_corridor_termination_name: str,
        foundation_steps: int,
        straight_foundation_steps: int,
        acceptance_ramp_steps: int,
        corridor_ramp_steps: int,
        figure_eight_ramp_steps: int,
        vertical_ramp_steps: int,
        speed_ramp_steps: int,
        easy_hold_steps: int,
        hard_bridge_steps: int,
        hard_geometry_ramp_steps: int,
        hard_speed_ramp_steps: int,
        hard_hold_steps: int,
        initial_acceptance_radius: float,
        final_acceptance_radius: float,
        initial_corridor_distance: float,
        final_corridor_distance: float,
        initial_figure_eight_probability: float,
        final_figure_eight_probability: float,
        minimum_vertical_amplitude: float,
        initial_maximum_vertical_amplitude: float,
        final_maximum_vertical_amplitude: float,
        initial_speed: float,
        final_speed: float,
        initial_hard_speed: float,
        initial_hard_figure_eight_probability: float,
        final_hard_figure_eight_probability: float,
        hard_xy_bound: float,
        hard_z_bound: float,
        hard_minimum_waypoint_separation: float,
        hard_maximum_waypoint_separation: float,
        hard_maximum_vertical_step: float,
        hard_random_sampling_attempts: int,
        hard_route_sampling_attempts: int,
        hard_heading_change_interval: int,
        initial_hard_nominal_heading_change: float,
        final_hard_nominal_heading_change: float,
        initial_hard_maximum_heading_change: float,
        final_hard_maximum_heading_change: float,
    ) -> None:
        for name, value in {
            "command_name": command_name,
            "path_progress_reward_name": path_progress_reward_name,
            "path_corridor_termination_name": path_corridor_termination_name,
        }.items():
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string.")
        durations = {
            "foundation_steps": foundation_steps,
            "straight_foundation_steps": straight_foundation_steps,
            "acceptance_ramp_steps": acceptance_ramp_steps,
            "corridor_ramp_steps": corridor_ramp_steps,
            "figure_eight_ramp_steps": figure_eight_ramp_steps,
            "vertical_ramp_steps": vertical_ramp_steps,
            "speed_ramp_steps": speed_ramp_steps,
            "easy_hold_steps": easy_hold_steps,
            "hard_bridge_steps": hard_bridge_steps,
            "hard_geometry_ramp_steps": hard_geometry_ramp_steps,
            "hard_speed_ramp_steps": hard_speed_ramp_steps,
            "hard_hold_steps": hard_hold_steps,
        }
        for name, value in durations.items():
            minimum = (
                0
                if name in {"foundation_steps", "straight_foundation_steps", "easy_hold_steps", "hard_hold_steps"}
                else 1
            )
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                qualifier = "non-negative" if minimum == 0 else "positive"
                raise ValueError(f"{name} must be a {qualifier} integer.")
        if straight_foundation_steps > foundation_steps:
            raise ValueError("straight_foundation_steps must not exceed foundation_steps.")
        for name, value in {
            "hard_random_sampling_attempts": hard_random_sampling_attempts,
            "hard_route_sampling_attempts": hard_route_sampling_attempts,
            "hard_heading_change_interval": hard_heading_change_interval,
        }.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        for name, value in {
            "initial_acceptance_radius": initial_acceptance_radius,
            "final_acceptance_radius": final_acceptance_radius,
            "initial_corridor_distance": initial_corridor_distance,
            "final_corridor_distance": final_corridor_distance,
            "initial_speed": initial_speed,
            "final_speed": final_speed,
            "initial_hard_speed": initial_hard_speed,
            "hard_xy_bound": hard_xy_bound,
            "hard_minimum_waypoint_separation": hard_minimum_waypoint_separation,
            "hard_maximum_waypoint_separation": hard_maximum_waypoint_separation,
        }.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(value)
                or value <= 0.0
            ):
                raise ValueError(f"{name} must be positive and finite.")
        for name, value in {
            "initial_figure_eight_probability": initial_figure_eight_probability,
            "final_figure_eight_probability": final_figure_eight_probability,
            "initial_hard_figure_eight_probability": initial_hard_figure_eight_probability,
            "final_hard_figure_eight_probability": final_hard_figure_eight_probability,
        }.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(value)
                or not 0.0 <= value <= 1.0
            ):
                raise ValueError(f"{name} must be finite and in [0, 1].")
        for name, value in {
            "minimum_vertical_amplitude": minimum_vertical_amplitude,
            "initial_maximum_vertical_amplitude": initial_maximum_vertical_amplitude,
            "final_maximum_vertical_amplitude": final_maximum_vertical_amplitude,
            "hard_z_bound": hard_z_bound,
            "hard_maximum_vertical_step": hard_maximum_vertical_step,
        }.items():
            if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")
        if final_acceptance_radius > initial_acceptance_radius:
            raise ValueError("final_acceptance_radius must not exceed initial_acceptance_radius.")
        if final_corridor_distance > initial_corridor_distance:
            raise ValueError("final_corridor_distance must not exceed initial_corridor_distance.")
        if final_figure_eight_probability < initial_figure_eight_probability:
            raise ValueError("final_figure_eight_probability must not be less than initial_figure_eight_probability.")
        if initial_maximum_vertical_amplitude < minimum_vertical_amplitude:
            raise ValueError("initial_maximum_vertical_amplitude must not be less than minimum_vertical_amplitude.")
        if final_maximum_vertical_amplitude < initial_maximum_vertical_amplitude:
            raise ValueError(
                "final_maximum_vertical_amplitude must not be less than initial_maximum_vertical_amplitude."
            )
        if final_speed < initial_speed:
            raise ValueError("final_speed must not be less than initial_speed.")
        if initial_hard_speed > final_speed:
            raise ValueError("initial_hard_speed must not exceed final_speed.")
        if final_hard_figure_eight_probability > initial_hard_figure_eight_probability:
            raise ValueError(
                "final_hard_figure_eight_probability must not exceed initial_hard_figure_eight_probability."
            )
        if hard_maximum_waypoint_separation < hard_minimum_waypoint_separation:
            raise ValueError("hard_maximum_waypoint_separation must not be less than hard_minimum_waypoint_separation.")
        if hard_maximum_vertical_step > hard_maximum_waypoint_separation:
            raise ValueError("hard_maximum_vertical_step must not exceed hard_maximum_waypoint_separation.")
        heading_endpoints = {
            "initial_hard_nominal_heading_change": initial_hard_nominal_heading_change,
            "final_hard_nominal_heading_change": final_hard_nominal_heading_change,
            "initial_hard_maximum_heading_change": initial_hard_maximum_heading_change,
            "final_hard_maximum_heading_change": final_hard_maximum_heading_change,
        }
        for name, value in heading_endpoints.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(value)
                or not 0.0 < value <= math.pi
            ):
                raise ValueError(f"{name} must be finite and in (0, pi].")
        if final_hard_nominal_heading_change < initial_hard_nominal_heading_change:
            raise ValueError(
                "final_hard_nominal_heading_change must not be less than initial_hard_nominal_heading_change."
            )
        if final_hard_maximum_heading_change < initial_hard_maximum_heading_change:
            raise ValueError(
                "final_hard_maximum_heading_change must not be less than initial_hard_maximum_heading_change."
            )
        if initial_hard_nominal_heading_change > initial_hard_maximum_heading_change:
            raise ValueError("initial_hard_nominal_heading_change must not exceed initial_hard_maximum_heading_change.")
        if final_hard_nominal_heading_change > final_hard_maximum_heading_change:
            raise ValueError("final_hard_nominal_heading_change must not exceed final_hard_maximum_heading_change.")
