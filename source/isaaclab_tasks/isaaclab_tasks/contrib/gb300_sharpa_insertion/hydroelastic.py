# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hydroelastic hand/connector contact with a physically attached VBD cable."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from isaaclab_newton.physics import HydroelasticSDFCfg, NewtonCfg, NewtonCollisionPipelineCfg

from isaaclab_contrib.coupling import CouplerAdmmCfg

if TYPE_CHECKING:
    import newton

# Newton exposes this builder attribute but does not publish its mode enum.
_MJWARP_SOLREF_MODE_FORCE_SPACE = 0


def configure_hydroelastic_physics(physics: NewtonCfg) -> None:
    """Keep hydroelastic grasp contacts in MJWarp; exchange cable loads through ADMM."""
    rigid, cable = physics.solver_cfg.entries
    rigid.bodies.extend((r"/World/envs/env_.*/Cable/head", r"/World/envs/env_.*/Cable/latch"))
    cable.bodies = [r"/World/envs/env_.*/Cable/(?!(?:head|latch)(?:/|$)).*"]
    # Distributed contacts require more rows than the point-contact default.
    rigid.solver_cfg.nconmax = 1024
    # Up to six rows per frictional contact, plus articulation joint limits.
    rigid.solver_cfg.njmax = 6 * rigid.solver_cfg.nconmax + 128
    cable.solver_cfg.rigid_contact_history = False
    physics.solver_cfg = CouplerAdmmCfg(
        entries=[rigid, cable],
        iterations=5,
        rho=50.0,
        # MJWarp cannot refresh proximal inertia for changing contact candidates.
        # Keeping this optional term off avoids enormous artificial robot masses.
        gamma=0.0,
        baumgarte=0.01,
        joint_stiffness=1.0e4,
        joint_angular_stiffness=1.0e4,
        joint_proximal_destination_entries=["rigid"],
        rigid_contact_matching="sticky",
    )
    # Validated at 1/120 s: 16 substeps kept the free-swing attachment gap below 1 mm.
    physics.num_substeps = max(physics.num_substeps, 16)
    physics.collision_cfg = NewtonCollisionPipelineCfg(
        sdf_hydroelastic_config=HydroelasticSDFCfg(
            output_contact_surface=True,
            # The detailed Sharpa pads generated 6336 broad-phase candidates.
            buffer_fraction=1.0,
            # A 1 mm² speculative patch; penetrating contacts retain their SDF area.
            margin_contact_area=1.0e-6,
        )
    )
    physics.collision_decimation = 1


def configure_hydroelastic_shapes(
    builder: newton.ModelBuilder,
    grasp_body_patterns: tuple[str, ...],
    *,
    thumb_latch_force_space: bool = False,
    thumb_latch_hydroelastic: bool = False,
    pad_pressure_gradient: float = 5.0e8,
) -> None:
    """Cook fingertip/plug SDFs while retaining ordinary contacts for every other pair.

    Massless fingertip copies collide only with the loose plug. The original
    colliders retain finger self-contact and contact with the socket and latch.
    Both copies occupy the same surface in the body's local frame [m].
    Ordinary connector/rack contacts use force-space stiffness [N/m] and
    damping [N s/m], without changing the hydroelastic pad response.

    ``thumb_latch_force_space`` additionally opts the native thumb pad into that
    material convention for clip depression. MJWarp requires both endpoints;
    the hydroelastic duplicate remains plug-only and unchanged.
    ``thumb_latch_hydroelastic`` additionally routes thumb/clip contact through
    the existing massless thumb copy. Other pads retain ordinary clip contact.
    ``pad_pressure_gradient`` controls the compliant covering [N/m^3], without
    changing the rigid connector material or adding mass to the pad copies.
    """
    import newton

    if not np.isfinite(pad_pressure_gradient) or pad_pressure_gradient <= 0.0:
        raise ValueError("pad_pressure_gradient must be finite and positive.")
    solref_mode = builder.custom_attributes.get("mujoco:solref_mode")
    if solref_mode is None:
        raise RuntimeError("MJWarp did not register its per-shape mujoco:solref_mode attribute.")
    if solref_mode.values is None:
        solref_mode.values = {}
    if not isinstance(solref_mode.values, dict):
        raise RuntimeError("MJWarp's per-shape mujoco:solref_mode storage must be sparse.")
    plugs = {
        shape
        for shape, label in enumerate(builder.shape_label)
        if builder.shape_world[shape] == builder.current_world and str(label).endswith("/Cable/head/grip")
    }
    latches = {
        shape
        for shape, label in enumerate(builder.shape_label)
        if builder.shape_world[shape] == builder.current_world and str(label).endswith("/Cable/latch/contact")
    }
    grasp_shapes = []
    thumb_grasp_shapes = set()
    for shape, label in enumerate(tuple(builder.shape_label)):
        if builder.shape_world[shape] != builder.current_world:
            continue
        if not builder.shape_flags[shape] & int(newton.ShapeFlags.COLLIDE_SHAPES):
            continue
        if shape in plugs or str(label).endswith("/Cable/latch/contact") or "/Fixtures/GB300/" in str(label):
            # Both endpoints must opt in; averaging legacy solref values makes
            # the differently damped clip and socket contacts nearly powerless.
            solref_mode.values[shape] = _MJWARP_SOLREF_MODE_FORCE_SPACE
        components = str(label).split("/")
        soft_pad = any(component.endswith("_elastomer") for component in components) and any(
            pattern in str(label) for pattern in grasp_body_patterns
        )
        if (
            thumb_latch_force_space
            and soft_pad
            and any(component.endswith("thumb_elastomer") for component in components)
        ):
            solref_mode.values[shape] = _MJWARP_SOLREF_MODE_FORCE_SPACE
        if not (soft_pad or shape in plugs or (thumb_latch_hydroelastic and shape in latches)):
            continue
        if soft_pad:
            original = shape
            # Separate pair routing without removing ordinary finger self-contact
            # or adding mass and inertia for the coincident collision surface.
            shape = builder.add_shape(
                body=builder.shape_body[original],
                type=builder.shape_type[original],
                xform=builder.shape_transform[original],
                scale=builder.shape_scale[original],
                src=builder.shape_source[original],
                cfg=newton.ModelBuilder.ShapeConfig(
                    density=0.0,
                    ke=builder.shape_material_ke[original],
                    kd=builder.shape_material_kd[original],
                    kf=builder.shape_material_kf[original],
                    ka=builder.shape_material_ka[original],
                    mu=builder.shape_material_mu[original],
                    restitution=builder.shape_material_restitution[original],
                    mu_torsional=builder.shape_material_mu_torsional[original],
                    mu_rolling=builder.shape_material_mu_rolling[original],
                    has_particle_collision=False,
                    is_visible=False,
                ),
                label=f"{label}/hydroelastic_grasp",
            )
            grasp_shapes.append(shape)
            for plug in plugs:
                builder.add_shape_collision_filter_pair(original, plug)
            if thumb_latch_hydroelastic and any(component.endswith("thumb_elastomer") for component in components):
                thumb_grasp_shapes.add(shape)
                for latch in latches:
                    builder.add_shape_collision_filter_pair(original, latch)
        if builder.shape_type[shape] in (newton.GeoType.MESH, newton.GeoType.CONVEX_MESH):
            mesh = builder.shape_source[shape]
            scale = np.asarray(builder.shape_scale[shape], dtype=np.float32)
            if soft_pad or not np.allclose(scale, 1.0):
                mesh = mesh.copy(vertices=mesh.vertices * scale, recompute_inertia=True)
                builder.shape_source[shape] = mesh
                builder.shape_scale[shape] = (1.0, 1.0, 1.0)
            if mesh.sdf is None:
                mesh.build_sdf(max_resolution=128, narrow_band_range=(-0.003, 0.003), margin=0.0003)
        else:
            builder.shape_sdf_max_resolution[shape] = 128
            builder.shape_sdf_narrow_band_range[shape] = (-0.003, 0.003)
        builder.shape_flags[shape] |= int(newton.ShapeFlags.HYDROELASTIC)
        # Linear pressure gradients [N/m^3]: compliant finger covering and hard
        # connector housing, estimated rather than measured hardware properties.
        builder.shape_material_kh[shape] = pad_pressure_gradient if soft_pad else 1.0e11
        builder.shape_gap[shape] = 0.0003
        builder.shape_margin[shape] = 0.0
    for shape in grasp_shapes:
        for other in range(builder.shape_count):
            thumb_clip = shape in thumb_grasp_shapes and other in latches
            if other != shape and other not in plugs and not thumb_clip:
                builder.add_shape_collision_filter_pair(shape, other)
