# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from isaaclab_newton.physics import CoupledSolverCfg, NewtonCfg


@configclass
class UR10ParticleScoopEnvCfg(DirectRLEnvCfg):
    """Pure Newton direct RL prototype: UR10 moves MPM particles into a side bin."""

    # env
    decimation = 4
    episode_length_s = 12.0
    action_space = 6
    state_space = 0
    heightmap_size = 32
    proprio_dim = 22
    observation_space = heightmap_size * heightmap_size + proprio_dim

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physics=NewtonCfg(
            solver_cfg=CoupledSolverCfg(),
            num_substeps=1,
            use_cuda_graph=True,
        ),
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=3.0, replicate_physics=True)
    viewer = ViewerCfg(eye=(2.8, -2.4, 2.0), lookat=(0.55, 0.0, 0.75))

    # Newton UR10 import
    ur10_urdf_path = "/home/horde/omni_isaac_sim/source/extensions/isaacsim.asset.importer.urdf/data/urdf/robots/ur10/urdf/ur10.urdf"
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
    paddle_size = (0.28, 0.36, 0.035)
    paddle_ee_offset = (0.16, 0.0, 0.0)
    table_center = (0.6, 0.0, 0.75)
    table_size = (1.45, 1.05, 0.05)
    table_top_z = table_center[2] + 0.5 * table_size[2]
    # Side catch bin: open toward the table, bottom roughly level with the tabletop.
    bin_center = (1.50, 0.0, table_top_z + 0.10)
    bin_inner_half_extents = (0.22, 0.28, 0.16)

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
    heightmap_z_range = 0.45

    # control and rewards
    action_scale = 0.55
    reward_count_scale = 2.0
    reward_delta_count_scale = 4.0
    reward_particle_progress_scale = 0.5
    reward_paddle_proximity_scale = 0.15
    action_penalty_scale = 0.005


@configclass
class UR10ParticleScoopEnvCfg_PLAY(UR10ParticleScoopEnvCfg):
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4, env_spacing=3.0, replicate_physics=True)
