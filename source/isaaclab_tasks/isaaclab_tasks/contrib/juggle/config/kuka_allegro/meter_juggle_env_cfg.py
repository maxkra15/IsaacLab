# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""One-metre toss-and-catch specialization of the KUKA-Allegro task."""

import math
from collections.abc import Sequence
from numbers import Real

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.contrib.juggle import mdp
from isaaclab_tasks.contrib.stack.spawners import ColoredCuboidCfg

from .juggle_env_cfg import RewardsCfg, _KukaAllegroJuggleBaseEnvCfg

_METER_ARM_EFFORT_LIMITS = {
    "iiwa7_joint_(1|2)": 352.0,
    "iiwa7_joint_(3|4|5)": 220.0,
    "iiwa7_joint_(6|7)": 80.0,
}
"""Simulation-only arm limits that make a one-metre ballistic release reachable [N m]."""

_METER_APEX_HORIZONTAL_DISPLACEMENT = 0.15
"""Maximum ball displacement from the launch origin at the qualified apex [m]."""

_METER_WORKSPACE_UPPER_Z = 2.00
"""Vertical escape bound with headroom above the required one-metre apex [m]."""

_METER_BALL_CONTACT_STIFFNESS = 5.0e4
"""Ball contact stiffness for dynamic hand impacts [N/m]."""

_METER_ARM_ACTION_SCALE = 2.0
"""Maximum measured-state KUKA target residual at unit XYZ input [rad]."""

_METER_TASK_SPACE_DAMPING = 2.5e-3
"""Regularization for the Juggle task's translational Jacobian mapping."""

_METER_HAND_OPENING_DELTA = tuple(
    opened - preload
    for opened, preload in zip(
        mdp.JUGGLE_SPHERE_OPEN_HAND_POSITION,
        mdp.JUGGLE_SPHERE_PRELOAD_HAND_POSITION,
        strict=True,
    )
)
_METER_HAND_OPENING_SCALE = max(abs(delta) for delta in _METER_HAND_OPENING_DELTA)
_METER_HAND_SYNERGY_DIRECTIONS = tuple(delta / _METER_HAND_OPENING_SCALE for delta in _METER_HAND_OPENING_DELTA)
"""Row-aligned Allegro opening direction with unit maximum magnitude."""


def _visual_play_stage_box(
    prim_path: str,
    size: tuple[float, float, float],
    position: tuple[float, float, float],
    color: tuple[float, float, float],
) -> AssetBaseCfg:
    """Create a visual-only stage box for continuous playback."""
    return AssetBaseCfg(
        prim_path=prim_path,
        init_state=AssetBaseCfg.InitialStateCfg(pos=position),
        spawn=ColoredCuboidCfg(
            size=size,
            display_color=color,
        ),
    )


@configclass
class MeterTossRewardsCfg(RewardsCfg):
    """Physical progress, exact milestones, recatch, and drop cost."""

    # Difference-of-potential shaping supplies a local physics signal without
    # actions, demonstrations, or authored trajectory targets. Matching PPO's
    # gamma makes the shaping internally consistent with its discounted return;
    # a workspace failure repays accumulated progress on that step.
    physical_progress = RewTerm(
        func=mdp.JugglePhysicalProgressReward,
        weight=1.0,
        params={
            "gamma": 0.998,
            "target_height_gain": 1.0,
            "apex_maximum_horizontal_displacement": _METER_APEX_HORIZONTAL_DISPLACEMENT,
            "catch_distance_scale": 0.12,
            "catch_relative_speed_scale": 0.45,
            "canonical_launch_fraction": 0.5,
        },
    )

    # Local goals remain curriculum outcome labels, not a second objective that
    # can dominate the complete throw-catch-rethrow behavior.
    local_transition = None
    apex_height = RewTerm(func=mdp.apex_height_pulse, weight=1.0)
    full_cycle = RewTerm(func=mdp.full_cycle_pulse, weight=2.0)
    # Keep a first throw-then-drop negative even after discounting the later
    # terminal pulse across the longest configured training horizon. A stable
    # recatch still retains the +1 height and +2 full-cycle return.
    dropped_ball = RewTerm(func=mdp.ball_out_of_workspace_pulse, weight=-2.0)


@configclass
class KukaAllegroJuggleRLEnvCfg(_KukaAllegroJuggleBaseEnvCfg):
    """Continuously throw a ball one metre above the hand and recatch it."""

    _EXPECTED_EPISODE_LENGTH_S = 5.0

    rewards: MeterTossRewardsCfg = MeterTossRewardsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        target_height = mdp.METER_TOSS_RESET_PROFILE.minimum_apex_height_gain

        self.episode_length_s = self._EXPECTED_EPISODE_LENGTH_S
        self.events.reset_from_catalog.params["profile"] = mdp.METER_TOSS_RESET_PROFILE.name
        self.events.reset_from_catalog.params["sampling_mode"] = "continuous"
        self.events.reset_from_catalog.params["continuous_seed"] = 17
        self.events.reset_from_catalog.params["rows_per_phase"] = 128
        self.curriculum.reset_sampling.params["sampling_mode"] = "continuous"
        self.curriculum.reset_sampling.params["canonical_fraction"] = 0.35
        self.curriculum.reset_sampling.params["continuous_sampler"].coverage_fraction = 0.15 / 0.65
        self.terminations.progress_context.params["apex_height_gain"] = target_height
        self.terminations.progress_context.params["apex_maximum_horizontal_displacement"] = (
            _METER_APEX_HORIZONTAL_DISPLACEMENT
        )
        self.terminations.progress_context.params["track_supported_release_reference"] = True
        self.terminations.progress_context.params["rearm_after_stable_catch"] = True
        # Every reset slice continues through later phases. Local success stays
        # latched for adaptive sampling and metrics, while only physical failure
        # or the neutral horizon starts a new episode.
        self.terminations.local_goal_success = None
        self.terminations.success = None
        self.terminations.ball_out_of_workspace.params = {
            "workspace_lower": (0.20, -0.40, 0.08),
            "workspace_upper": (0.80, 0.40, _METER_WORKSPACE_UPPER_Z),
        }
        # The catch is trained explicitly; a ground plane would turn a drop
        # into an unmodelled bounce instead of the intended terminal failure.
        self.scene.ground = None
        # Meter-scale impacts retain dissipation and friction with a stiffer
        # spring, avoiding visibly over-damped hand contact.
        self.scene.ball.spawn.physics_material.contact_stiffness = _METER_BALL_CONTACT_STIFFNESS

        # Compact XYZ/aperture controls make the policy 108-in/4-out.
        self.observations.policy.ball_height_and_velocity = ObsTerm(
            func=mdp.ball_height_above_release_hand_and_velocity,
            params={"target_height_gain": target_height},
        )
        # Three translation coordinates replace seven independently explored
        # arm residuals.  The current palm-center Jacobian supplies only robot
        # geometry; PPO owns every XYZ direction and magnitude.  The inherited
        # measured-relative controller retains workspace, speed, and gravity
        # bounds.
        full_arm_action = self.actions.arm_action
        self.actions.arm_action = mdp.JuggleTaskSpaceTranslationActionCfg(
            asset_name=full_arm_action.asset_name,
            joint_names=list(full_arm_action.joint_names),
            preserve_order=full_arm_action.preserve_order,
            scale=_METER_ARM_ACTION_SCALE,
            offset=full_arm_action.offset,
            clip=full_arm_action.clip,
            debug_vis=full_arm_action.debug_vis,
            max_delta=_METER_ARM_ACTION_SCALE,
            joint_limit_margin=full_arm_action.joint_limit_margin,
            workspace_lower=tuple(full_arm_action.workspace_lower),
            workspace_upper=tuple(full_arm_action.workspace_upper),
            gravity_compensation=full_arm_action.gravity_compensation,
            body_name="palm_link",
            tool_offset=mdp.JUGGLE_SPHERE_CENTER_OFFSET,
            damping=_METER_TASK_SPACE_DAMPING,
        )
        # A single physical aperture replaces 16 independently explored finger
        # residuals. Zero holds the measured reset pose, positive commands open
        # along the calibrated hand-pose direction, and max_delta preserves the
        # finger speed.
        full_hand_action = self.actions.hand_action
        self.actions.hand_action = mdp.JuggleHandSynergyActionCfg(
            asset_name=full_hand_action.asset_name,
            joint_names=list(full_hand_action.joint_names),
            preserve_order=full_hand_action.preserve_order,
            scale=full_hand_action.scale,
            max_delta=full_hand_action.max_delta,
            joint_limit_margin=full_hand_action.joint_limit_margin,
            reset_preload_joint_names=tuple(full_hand_action.reset_preload_joint_names),
            reset_preload_commands_by_pair=tuple(full_hand_action.reset_preload_commands_by_pair),
            reset_open_commands_by_pair=tuple(full_hand_action.reset_open_commands_by_pair),
            preload_release_threshold=full_hand_action.preload_release_threshold,
            preload_release_steps=full_hand_action.preload_release_steps,
            release_preload_after_first_action=True,
            joint_directions=_METER_HAND_SYNERGY_DIRECTIONS,
        )
        self.observations.policy.actions = ObsTerm(func=mdp.last_action)
        actuator_name = "kuka_allegro_actuators"
        actuator = self.scene.robot.actuators[actuator_name]
        if not isinstance(actuator.effort_limit_sim, dict):
            raise TypeError("The one-metre task requires per-joint simulated effort limits.")
        # A live Newton feasibility bracket found that the stock iiwa effort
        # limits top out below a one-metre ballistic release, while 2x is the
        # smallest tested tier that makes the required throw physically reachable.
        # This deliberately makes the variant a simulated juggling sport, not
        # a stock-hardware or sim-to-real claim. Allegro limits remain untouched.
        meter_actuator = actuator.replace(
            effort_limit_sim={**actuator.effort_limit_sim, **_METER_ARM_EFFORT_LIMITS},
        )
        self.scene.robot = self.scene.robot.replace(
            actuators={**self.scene.robot.actuators, actuator_name: meter_actuator},
        )

        self.sim.default_visualizer_cfg.eye = (2.2, 2.2, 1.75)
        self.sim.default_visualizer_cfg.lookat = (0.50, 0.0, 0.80)

    def play_mode(self) -> None:
        """Evaluate continuous rallies and reset only after physical failure."""
        super().play_mode()
        self.terminations.local_goal_success = None
        self.terminations.success = None
        self.terminations.time_out = None
        # A physical ground would let failed catches bounce back into play. The
        # stage is visual-only, preserving drop termination and task physics.
        self.scene.play_stage_deck = _visual_play_stage_box(
            "{ENV_REGEX_NS}/PlayStage/Deck",
            size=(1.60, 1.40, 0.08),
            position=(0.25, 0.0, -0.16),
            color=(0.055, 0.070, 0.095),
        )
        self.scene.play_workspace_mat = _visual_play_stage_box(
            "{ENV_REGEX_NS}/PlayStage/WorkspaceMat",
            size=(0.60, 0.80, 0.008),
            position=(0.50, 0.0, -0.116),
            color=(0.035, 0.24, 0.36),
        )
        self.scene.play_robot_pedestal = _visual_play_stage_box(
            "{ENV_REGEX_NS}/PlayStage/RobotPedestal",
            size=(0.34, 0.34, 0.12),
            position=(0.0, 0.0, -0.06),
            color=(0.20, 0.24, 0.30),
        )
        self.scene.light.spawn.color = (0.90, 0.94, 1.00)
        self.scene.light.spawn.intensity = 3200.0
        self.sim.default_visualizer_cfg.eye = (1.75, 1.75, 1.35)
        self.sim.default_visualizer_cfg.lookat = (0.45, 0.0, 0.70)

    def validate_config(self) -> None:
        """Validate the one-metre task contract in addition to the base task."""
        super().validate_config()
        target_height = mdp.METER_TOSS_RESET_PROFILE.minimum_apex_height_gain
        if self.events.reset_from_catalog.params.get("profile") != mdp.METER_TOSS_RESET_PROFILE.name:
            raise ValueError("The one-metre task requires the meter-toss reset profile.")
        if self.events.reset_from_catalog.params.get("sampling_mode") != "continuous" or (
            self.curriculum is not None and self.curriculum.reset_sampling.params.get("sampling_mode") != "continuous"
        ):
            raise ValueError("The one-metre task requires continuous success-model reset sampling.")
        if self.terminations.progress_context.params.get("apex_height_gain") != target_height:
            raise ValueError("The one-metre task must qualify a one-metre first apex.")
        if (
            self.terminations.progress_context.params.get("apex_maximum_horizontal_displacement")
            != _METER_APEX_HORIZONTAL_DISPLACEMENT
        ):
            raise ValueError("The one-metre task requires its validated launch-relative apex corridor.")
        if self.terminations.progress_context.params.get("track_supported_release_reference") is not True:
            raise ValueError("The one-metre task must latch the last supported release-hand reference.")
        if self.scene.ground is not None:
            raise ValueError("The one-metre task requires the ground plane to remain disabled.")
        if self.scene.ball.spawn.physics_material.contact_stiffness != _METER_BALL_CONTACT_STIFFNESS:
            raise ValueError("The one-metre task requires its dynamic-impact ball contact stiffness.")
        apex_reward = getattr(self.rewards, "apex_height", None)
        if apex_reward is None or apex_reward.func is not mdp.apex_height_pulse or apex_reward.weight != 1.0:
            raise ValueError("The one-metre task requires its unit one-shot apex-height reward.")
        progress_reward = getattr(self.rewards, "physical_progress", None)
        if (
            progress_reward is None
            or progress_reward.func is not mdp.JugglePhysicalProgressReward
            or progress_reward.weight != 1.0
            or progress_reward.params.get("gamma") != 0.998
            or progress_reward.params.get("target_height_gain") != target_height
            or progress_reward.params.get("apex_maximum_horizontal_displacement") != _METER_APEX_HORIZONTAL_DISPLACEMENT
        ):
            raise ValueError("The one-metre task requires its reset-aware physical-progress reward.")
        dropped_ball_reward = getattr(self.rewards, "dropped_ball", None)
        if (
            dropped_ball_reward is None
            or dropped_ball_reward.func is not mdp.ball_out_of_workspace_pulse
            or dropped_ball_reward.weight != -2.0
        ):
            raise ValueError("The one-metre task requires its negative terminal dropped-ball reward.")
        workspace_lower = self.terminations.ball_out_of_workspace.params.get("workspace_lower")
        workspace_upper = self.terminations.ball_out_of_workspace.params.get("workspace_upper")
        for name, bounds in (("workspace_lower", workspace_lower), ("workspace_upper", workspace_upper)):
            if (
                not isinstance(bounds, Sequence)
                or isinstance(bounds, (str, bytes))
                or len(bounds) != 3
                or any(isinstance(value, bool) or not isinstance(value, Real) for value in bounds)
            ):
                raise ValueError(f"{name} must be a three-dimensional numeric workspace vector.")
        if not all(math.isfinite(value) for value in (*workspace_lower, *workspace_upper)) or any(
            lower >= upper for lower, upper in zip(workspace_lower, workspace_upper, strict=True)
        ):
            raise ValueError("The one-metre task requires finite, ordered three-dimensional workspace bounds.")
        if workspace_upper[2] < 1.90:
            raise ValueError("The one-metre apex requires at least 1.90 m of vertical workspace.")
        if self.episode_length_s != self._EXPECTED_EPISODE_LENGTH_S:
            raise ValueError("Juggling requires its validated five-second episode horizon.")
        if not isinstance(self.actions.arm_action, mdp.JuggleTaskSpaceTranslationActionCfg):
            raise ValueError("The one-metre task requires three-dimensional task-space arm translation control.")
        if (
            self.actions.arm_action.scale != _METER_ARM_ACTION_SCALE
            or self.actions.arm_action.max_delta != _METER_ARM_ACTION_SCALE
            or self.actions.arm_action.body_name != "palm_link"
            or self.actions.arm_action.tool_offset != mdp.JUGGLE_SPHERE_CENTER_OFFSET
            or self.actions.arm_action.damping != _METER_TASK_SPACE_DAMPING
        ):
            raise ValueError("The one-metre task requires its validated palm-center task-space arm mapping.")
        if not isinstance(self.actions.hand_action, mdp.JuggleHandSynergyActionCfg):
            raise ValueError("The one-metre task requires one-dimensional Allegro-hand aperture control.")
        if (
            self.actions.hand_action.scale != 0.10
            or self.actions.hand_action.max_delta != 0.10
            or self.actions.hand_action.joint_directions != _METER_HAND_SYNERGY_DIRECTIONS
        ):
            raise ValueError("The one-metre task requires its calibrated Allegro-hand opening synergy.")
        if not self.actions.hand_action.release_preload_after_first_action:
            raise ValueError("The one-metre task may use reset preload assistance for only its first action.")
        height_observation = self.observations.policy.ball_height_and_velocity
        if (
            height_observation.func is not mdp.ball_height_above_release_hand_and_velocity
            or height_observation.params.get("target_height_gain") != target_height
        ):
            raise ValueError("The one-metre policy must observe height relative to the latched release hand.")
        if self.observations.policy.actions.func is not mdp.last_action:
            raise ValueError("The one-metre policy must observe its four-dimensional live action.")
        if workspace_lower != (0.20, -0.40, 0.08) or workspace_upper != (0.80, 0.40, _METER_WORKSPACE_UPPER_Z):
            raise ValueError("The one-metre task requires its validated workspace bounds.")
        actuator = self.scene.robot.actuators["kuka_allegro_actuators"]
        if not isinstance(actuator.effort_limit_sim, dict) or any(
            actuator.effort_limit_sim.get(expression) != effort
            for expression, effort in _METER_ARM_EFFORT_LIMITS.items()
        ):
            raise ValueError("The one-metre task requires its validated simulated arm-effort tier.")
        if actuator.effort_limit_sim.get("(index|middle|ring|thumb)_joint_(0|1|2|3)") != 0.7:
            raise ValueError("The one-metre arm override must not change the Allegro effort limit.")
        self._validate_continuous_episode_contract()

    def _validate_continuous_episode_contract(self) -> None:
        """Validate shared continuous-cycle and mode-specific reset boundaries."""
        if self.terminations.progress_context.params.get("rearm_after_stable_catch") is not True:
            raise ValueError("Juggling must re-arm every stable catch in place.")
        if self.terminations.success is not None:
            raise ValueError("Juggling must not terminate at the first completed cycle.")
        if self.terminations.local_goal_success is not None:
            raise ValueError("Juggling must not terminate at phase-local success.")
        if self.rewards.local_transition is not None:
            raise ValueError("Phase-local success must remain an adaptive-reset label, not a reward.")
        if self.curriculum is not None:
            if self.terminations.time_out is None:
                raise ValueError("Training must retain its neutral timeout episode boundary.")
            return
        self._validate_play_mode()

    def _validate_play_mode(self) -> None:
        """Validate endless-rally playback overrides."""
        if self.events.reset_from_catalog.params.get("fixed_phase") != int(mdp.JugglePhase.HELD_PRETHROW) or (
            self.events.reset_from_catalog.params.get("static_held_only") is not True
        ):
            raise ValueError("Playback must begin from canonical static held starts.")
        if self.terminations.local_goal_success is not None or self.terminations.time_out is not None:
            raise ValueError("Playback must reset only after physical failure.")
        if self.terminations.ball_out_of_workspace is None or self.terminations.nonfinite_state is None:
            raise ValueError("Playback must retain physical-failure and numerical-safety terminations.")
        stage_assets = (self.scene.play_stage_deck, self.scene.play_workspace_mat, self.scene.play_robot_pedestal)
        if any(
            not isinstance(asset, AssetBaseCfg)
            or not isinstance(asset.spawn, sim_utils.CuboidCfg)
            or asset.spawn.rigid_props is not None
            or asset.spawn.collision_props is not None
            or asset.spawn.mass_props is not None
            or asset.spawn.physics_material is not None
            for asset in stage_assets
        ):
            raise ValueError("Playback stage geometry must remain visual-only.")
