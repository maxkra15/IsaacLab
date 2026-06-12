# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def create_mpm_particle_visualization(
    prim_path: str,
    positions: np.ndarray,
    widths: np.ndarray,
    color: Sequence[float],
) -> list[str]:
    """Create one ``UsdGeom.PointInstancer`` per environment for Kit MPM particle rendering.

    A ``PointInstancer`` (sphere prototype, one instance per particle) is used
    instead of a bare ``UsdGeom.Points`` prim because the Fabric Scene Delegate
    refreshes an instancer's ``positions`` from Fabric every frame, whereas a
    bare ``Points`` prim's ``points`` array is not. The created prims are static
    USD containers: per-frame position updates are written straight into the
    Fabric ``positions`` array on the GPU by
    :meth:`isaaclab_newton.physics.NewtonManager.sync_particles_to_usd` for prims
    registered via
    :meth:`isaaclab_newton.physics.NewtonManager.register_particle_visual_prim`.

    Args:
        prim_path: Base prim path; one instancer is created per environment at
            ``{prim_path}/env_{idx}``.
        positions: Initial world-frame particle positions [m], shape
            ``(num_envs, particles_per_env, 3)``.
        widths: Particle display widths (diameters) [m], one per particle.
        color: RGB display color of the particles.

    Returns:
        The created ``PointInstancer`` prim paths, one per environment.
    """
    from pxr import Gf, Sdf, UsdGeom, Vt  # noqa: PLC0415

    import isaaclab.sim as sim_utils

    stage = sim_utils.get_current_stage()
    prim_paths = [f"{prim_path}/env_{env_idx}" for env_idx in range(positions.shape[0])]

    num_particles = positions.shape[1]
    radius = float(np.mean(np.ascontiguousarray(widths, dtype=np.float32))) * 0.5
    proto_indices_vt = Vt.IntArray.FromNumpy(np.zeros(num_particles, dtype=np.int32))
    color_vt = Vt.Vec3fArray([Gf.Vec3f(*(float(value) for value in color))])

    for env_idx, path in enumerate(prim_paths):
        instancer = UsdGeom.PointInstancer.Define(stage, path)
        prototype = UsdGeom.Sphere.Define(stage, f"{path}/proto")
        prototype.CreateRadiusAttr(radius)
        prototype.CreateDisplayColorAttr(color_vt)
        instancer.CreatePrototypesRel().SetTargets([prototype.GetPath()])
        with Sdf.ChangeBlock():
            instancer.GetPositionsAttr().Set(Vt.Vec3fArray.FromNumpy(positions[env_idx]))
            instancer.GetProtoIndicesAttr().Set(proto_indices_vt)

    return prim_paths
