# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Asset-free regression of the teapot demo's coupled moving boundary."""

import sys
from pathlib import Path

import newton
import numpy as np
import pytest
import warp as wp
from newton.solvers import SolverImplicitMPM, SolverXPBD
from newton.solvers.experimental.coupled import SolverCoupled
from scipy.spatial.transform import Rotation, Slerp

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "examples"))
from rizon_sharpa_teapot import _TeapotColliderSyncSolver


@pytest.mark.parametrize("substeps", (1, 2, 4))
def test_teapot_boundary_substeps(substeps):
    """Interpolate a rotating off-center body through the actual fluid microsteps."""
    records = []

    class RecordingMPM(SolverImplicitMPM):
        def step(self, state_in, state_out, control, contacts, dt):
            start = state_in.body_q.numpy().copy()
            velocity = state_in.body_qd.numpy().copy()
            super().step(state_in, state_out, control, contacts, dt)
            records.append((start, velocity, self._particle_collider_end_poses.numpy().copy(), dt))

    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    SolverImplicitMPM.register_custom_attributes(builder)
    initial_rotation = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), 0.6)
    pot = builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), initial_rotation),
        com=wp.vec3(0.02, 0.03, 0.04),
        mass=1.0,
        inertia=wp.mat33(np.eye(3, dtype=np.float32)),
        lock_inertia=True,
        label="/World/envs/env_0/PourContainer",
    )
    shape = builder.add_shape_box(
        body=pot,
        hx=0.01,
        hy=0.1,
        hz=0.1,
        cfg=newton.ModelBuilder.ShapeConfig(density=0.0, margin=0.0),
        label="/World/envs/env_0/PourContainer/geometry/shell",
    )
    bowl = builder.add_body(is_kinematic=True, label="/World/envs/env_0/CatchBowl")
    particle_pos = wp.transform_point(builder.body_q[pot], wp.vec3(0.035, 0.0, 0.0))
    builder.add_particle(pos=particle_pos, vel=wp.vec3(0.0), mass=0.001, radius=0.001)
    model = builder.finalize(device="cpu")
    solver = _TeapotColliderSyncSolver(
        model,
        [
            SolverCoupled.Entry(
                name="robot",
                solver=lambda view: SolverXPBD(view, iterations=1),
                bodies=(pot,),
                shapes=(shape,),
            ),
            SolverCoupled.Entry(
                name="fluid",
                solver=lambda view: RecordingMPM(
                    view,
                    SolverImplicitMPM.Config(fidelity="particles", particle_iterations=1),
                    verbose=False,
                ),
                bodies=(bowl,),
                particles=(0,),
                in_place=True,
                substeps=substeps,
            ),
        ],
    )
    state_in, state_out = model.state(), model.state()
    velocity = state_in.body_qd.numpy()
    velocity[pot] = (0.2, -0.1, 0.05, 0.0, 0.0, 0.4)
    state_in.body_qd.assign(velocity)
    initial_pose = state_in.body_q.numpy()[pot]
    solver.step(state_in, state_out, None, None, 0.1)
    final_pose = state_out.body_q.numpy()[pot]
    com = model.body_com.numpy()[pot]
    initial_com = initial_pose[:3] + Rotation.from_quat(initial_pose[3:]).apply(com)
    final_com = final_pose[:3] + Rotation.from_quat(final_pose[3:]).apply(com)
    rotation_path = Slerp([0.0, 1.0], Rotation.from_quat([initial_pose[3:], final_pose[3:]]))
    np.testing.assert_allclose(np.linalg.norm(final_pose[3:]), 1.0, atol=1.0e-6)
    assert np.linalg.norm(final_com - initial_com) > 0.01
    assert (Rotation.from_quat(final_pose[3:]) * Rotation.from_quat(initial_pose[3:]).inv()).magnitude() > 0.02
    assert len(records) == substeps
    for index, (poses, velocities, predicted_poses, dt) in enumerate(records):
        np.testing.assert_allclose(dt, 0.1 / substeps)
        np.testing.assert_allclose(velocities[solver._fluid_pot, :3], (final_com - initial_com) / 0.1, atol=1.0e-6)
        # Check the actual pose consumed by the native solver, then its actual
        # COM-aware forward prediction, rather than duplicating the sync kernel.
        for pose, fraction in (
            (poses[solver._fluid_pot], index / substeps),
            (predicted_poses[solver._fluid_pot], (index + 1) / substeps),
        ):
            expected_com = initial_com + fraction * (final_com - initial_com)
            rotation = Rotation.from_quat(pose[3:])
            np.testing.assert_allclose(pose[:3] + rotation.apply(com), expected_com, atol=2.0e-7)
            np.testing.assert_allclose(rotation.as_matrix(), rotation_path([fraction]).as_matrix()[0], atol=2.0e-6)
    np.testing.assert_allclose(records[-1][2][solver._fluid_pot], final_pose, atol=2.0e-7)
    assert np.isfinite(state_out.particle_q.numpy()).all()
    assert np.isfinite(state_out.particle_qd.numpy()).all()
