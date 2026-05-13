# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
from pathlib import Path

from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg


def _default_ur10_urdf_path() -> str:
    isaac_sim_source_dir = Path(
        os.environ.get("ISAAC_SIM_SOURCE_DIR", Path(__file__).resolve().parents[6] / "omni_isaac_sim")
    )
    return str(
        isaac_sim_source_dir
        / "source"
        / "extensions"
        / "isaacsim.asset.importer.urdf"
        / "data"
        / "urdf"
        / "robots"
        / "ur10"
        / "urdf"
        / "ur10.urdf"
    )


@configclass
class UR10ParticleScoopEnvCfg(DirectRLEnvCfg):
    """Pure Newton direct RL prototype: UR10 moves MPM particles into a side bin."""

    # env
    decimation = 1
    episode_length_s = 12.0
    action_space = 6
    heightmap_size = 20
    heightmap_channels = 2
    proprio_dim = 55
    privileged_dim = 13
    observation_space = heightmap_size * heightmap_size * heightmap_channels + proprio_dim
    state_space = observation_space + privileged_dim

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 60,
        render_interval=decimation,
        physics=NewtonCfg(
            solver_cfg=MJWarpSolverCfg(
                use_mujoco_contacts=False,
                njmax=160,
                nconmax=320,
                iterations=80,
            ),
            num_substeps=4,
            use_cuda_graph=True,
        ),
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=3.0, replicate_physics=True)
    viewer = ViewerCfg(eye=(1, -1, 0.7), lookat=(0, 0.0, 0.75))

    # Newton UR10 import
    ur10_urdf_path = _default_ur10_urdf_path()
    robot_base_pos = (0.0, -0.55, 0.775)
    ee_body_name = "ee_link"
    arm_joint_names = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ]

    # generated Newton workspace
    paddle_size = (0.26, 0.32, 0.025)
    paddle_ee_offset = (0.16, 0.0, 0.0)
    paddle_collision_margin = 0.035
    table_center = (0.35, 0.0, 0.75)
    table_size = (0.8, 0.90, 0.05)
    table_top_z = table_center[2] + 0.5 * table_size[2]
    table_leg_size = (0.045, 0.045, 0.75)
    # Deep side catch bin: open toward the table edge, with floor below the tabletop.
    bin_center = (1.07, -0.05, table_top_z - 0.06)
    bin_inner_half_extents = (0.32, 0.32, 0.24)
    bin_wall_thickness = 0.035
    bin_wall_height = 0.52
    bin_front_wall_height = table_center[2] - 0.5 * table_size[2] - (bin_center[2] - bin_inner_half_extents[2])
    bin_rim_height = 0.045
    bin_rim_thickness = 0.045

    # Newton MPM pile
    voxel_size = 0.055
    particles_per_cell = 2.0
    mpm_iterations = 40
    mpm_grid_padding = 24
    mpm_max_active_cell_count = 1 << 15
    sand_density = 1800.0
    sand_friction = 0.75
    sand_damping = 0.0
    sand_young_modulus = 1.0e15
    sand_yield_pressure = 1.0e15
    sand_tensile_yield_ratio = 0.0
    pile_lo = (0.24, -0.24, table_top_z + 0.015)
    pile_hi = (0.52, 0.04, table_top_z + 0.18)
    # Fixed policy grid in the environment frame, covering the table and side bin.
    heightmap_x_bounds = (-0.10, 1.25)
    heightmap_y_bounds = (-0.50, 0.50)
    heightmap_z_min = table_top_z - 0.28
    heightmap_z_range = 0.78
    heightmap_occupied_cell_value = 0.05
    heightmap_density_norm = 6.0
    bin_particle_min_height = 0.02

    # control and rewards
    # Cartesian actions are end-effector deltas in the environment frame: xyz translation + axis-angle rotation.
    cartesian_position_action_scale = 0.35
    cartesian_rotation_action_scale = 1.25
    ik_damping = 0.08
    ik_fd_epsilon = 1.0e-3
    max_ik_delta_q = 0.08
    success_fraction = 0.80
    reward_bin_fraction_scale = 12.0
    reward_delta_bin_fraction_scale = 48.0
    reward_particle_progress_scale = 1.0
    reward_mouth_entry_scale = 0.12
    reward_bin_proximity_scale = 0.05
    reward_spill_penalty_scale = 0.25
    reward_paddle_proximity_scale = 0.05
    reward_paddle_bin_proximity_scale = 0.04
    reward_paddle_orientation_scale = 0.03
    reward_paddle_low_penalty_scale = 0.25
    reward_paddle_speed_penalty_scale = 0.0
    reward_success_bonus = 50.0
    action_penalty_scale = 0.0
    action_rate_penalty_scale = 0.0005
    joint_velocity_penalty_scale = 0.0
    paddle_min_height = table_top_z + 0.05
    max_paddle_speed = 5.0
    max_joint_velocity = 10.0

    # curriculum
    curriculum_enabled = True
    curriculum_stage_success_fractions = (0.06, 0.12, 0.25, 0.45, 0.65, 0.80)
    curriculum_success_rate_thresholds = (0.25, 0.30, 0.35, 0.45, 0.55)
    curriculum_min_resets_per_stage = 256
    curriculum_success_ema_alpha = 0.05
    curriculum_pile_center_x_ranges = (
        (0.54, 0.60),
        (0.50, 0.58),
        (0.44, 0.54),
        (0.34, 0.54),
        (0.28, 0.52),
        (0.24, 0.50),
    )
    curriculum_pile_center_y_ranges = (
        (-0.08, -0.02),
        (-0.10, 0.00),
        (-0.12, 0.02),
        (-0.20, 0.08),
        (-0.24, 0.10),
        (-0.26, 0.10),
    )
    curriculum_pile_scale_ranges = (
        (0.45, 0.60),
        (0.55, 0.70),
        (0.60, 0.75),
        (0.75, 0.95),
        (0.85, 1.05),
        (0.90, 1.10),
    )
    curriculum_robot_init_enabled = True
    curriculum_robot_init_iterations = 16
    curriculum_robot_start_x_offset_ranges = (
        (0.00, 0.04),
        (0.02, 0.06),
        (0.12, 0.18),
        (0.18, 0.32),
        (0.22, 0.40),
        (0.24, 0.48),
    )
    curriculum_robot_start_y_noise_ranges = (
        (-0.005, 0.005),
        (-0.01, 0.01),
        (-0.03, 0.03),
        (-0.08, 0.08),
        (-0.12, 0.12),
        (-0.16, 0.16),
    )
    curriculum_robot_start_z_offsets = (0.065, 0.075, 0.11, 0.14, 0.16, 0.18)
    max_ik_reset_delta_q = 0.16


@configclass
class UR10ParticleScoopEnvCfg_PLAY(UR10ParticleScoopEnvCfg):
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4, env_spacing=3.0, replicate_physics=True)
