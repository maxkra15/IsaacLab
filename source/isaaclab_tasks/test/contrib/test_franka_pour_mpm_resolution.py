# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Completed terminal cohorts control real MPM resource transitions."""

from types import SimpleNamespace

import newton
import pytest
import torch
from isaaclab_newton.physics import MPMSolverCfg, NewtonCfg, NewtonManager
from isaaclab_newton.physics.newton_backend import NewtonBackend
from isaaclab_newton.physics.newton_backend_cfg import NewtonBackendCfg
from newton.solvers import SolverImplicitMPM

from isaaclab.managers import CurriculumTermCfg

from isaaclab_contrib.coupling import CouplerEntryCfg, CouplerProxyCfg

from isaaclab_tasks.contrib.franka_pour.mdp.mpm_resolution import PourMPMResolutionCurriculum
from isaaclab_tasks.contrib.franka_pour.pour_env_cfg import FrankaPourResetDatasetEnvCfg, PourMPMResolutionStageCfg

pytestmark = pytest.mark.kitless


def test_resolution_uses_terminal_success_cohorts_not_explicit_resets():
    builder = newton.ModelBuilder()
    SolverImplicitMPM.register_custom_attributes(builder)
    builder.add_particle(pos=(0.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=0.01, radius=0.02)
    solver_cfg = MPMSolverCfg(grid_type="dense", grid_padding=1, voxel_size=0.1, max_iterations=1)
    physics = NewtonCfg(
        use_cuda_graph=False,
        solver_cfg=CouplerProxyCfg(
            entries=[
                CouplerEntryCfg(name="media", particles=[0], in_place=True, solver_cfg=solver_cfg),
            ]
        ),
    )
    manager = NewtonManager(physics)
    manager.backend = NewtonBackend(NewtonBackendCfg(device="cpu", physics_cfg=physics), builder, dt=0.001)
    manager.backend.init_solver()
    cfg = FrankaPourResetDatasetEnvCfg()
    cfg.mpm_resolution_min_episodes = 4
    cfg.mpm_resolution_stages = (PourMPMResolutionStageCfg(success_rate=0.75, voxel_size=0.05),)
    env = SimpleNamespace(
        cfg=cfg,
        num_envs=4,
        device="cpu",
        common_step_counter=7,
        episode_length_buf=torch.ones(4, dtype=torch.long),
        episode_succeeded=torch.ones(4, dtype=torch.bool),
        termination_manager=SimpleNamespace(dones=torch.zeros(4, dtype=torch.bool)),
        sim=SimpleNamespace(backend_manager=manager),
    )
    curriculum = PourMPMResolutionCurriculum(CurriculumTermCfg(func=PourMPMResolutionCurriculum), env)
    native = manager.backend.solver.solver("media")
    curriculum(env, slice(None))
    assert native.voxel_size == 0.1
    env.termination_manager.dones.fill_(True)
    env.episode_succeeded[:] = torch.tensor([True, False, False, False])
    curriculum(env, [0, 1])
    assert native.voxel_size == 0.1
    curriculum(env, [2, 3])
    assert native.voxel_size == 0.1
    env.episode_succeeded[:] = torch.tensor([True, True, True, False])
    cfg.curriculum_freeze = True
    curriculum(env, slice(None))
    assert native.voxel_size == 0.1
    cfg.curriculum_freeze = False
    metrics = curriculum(env, slice(None))
    assert native.voxel_size == 0.05
    assert metrics["stage"] == 1 and metrics["last_transition_policy_step"] == 7
    manager.backend.step()
    assert torch.isfinite(torch.from_numpy(manager.backend.state_0.particle_q.numpy())).all()


@pytest.mark.parametrize("values", ({}, {"voxel_size": 0.0}, {"voxel_size": 0.02, "success_rate": float("nan")}))
def test_resolution_invalid_stage_rejected_before_simulation(values):
    cfg = FrankaPourResetDatasetEnvCfg(mpm_resolution_stages=(PourMPMResolutionStageCfg(**values),))
    with pytest.raises(ValueError, match="MPM resolution"):
        cfg.validate_config()
