# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reusable, fast Warp kernels for spawning granular media in a natural settled-pile shape.

A pile of cohesionless granular media poured onto a flat surface settles into a cone whose side
slope equals the material's angle of repose. :func:`sample_conical_pile` fills a cone (apex up,
base on the surface) with particles sampled uniformly by volume, so the initial particle cloud is
already close to the MPM equilibrium and only needs to settle slightly. The angle of repose is a
material property (roughly ``atan(friction)`` for dry granular media), so the same kernel produces
the correct pile for any configured friction.
"""

from __future__ import annotations

import math

import numpy as np
import warp as wp

_TWO_PI = float(2.0 * math.pi)


@wp.kernel(enable_backward=False)
def _sample_conical_pile_kernel(
    seed: int,
    center: wp.vec3,
    base_radius: float,
    height: float,
    jitter: float,
    out_pos: wp.array(dtype=wp.vec3),
):
    """Write one particle position per thread inside a cone (apex up) sampled uniformly by volume."""
    i = wp.tid()
    rng = wp.rand_init(seed, i)
    # Height sampled so the cloud is uniform in the cone's VOLUME (cross-section shrinks with height):
    # for a cone of unit height, P(z<=t) = 1-(1-t)^3, so t = 1 - (1-u)^(1/3) biases toward the wide base.
    t = 1.0 - wp.pow(1.0 - wp.randf(rng), 1.0 / 3.0)
    z = height * t
    r_max = base_radius * (1.0 - t)
    # Uniform within the disk of radius r_max at this height.
    r = r_max * wp.sqrt(wp.randf(rng))
    ang = _TWO_PI * wp.randf(rng)
    x = r * wp.cos(ang)
    y = r * wp.sin(ang)
    # Small isotropic surface noise so the pile is not a perfectly smooth analytic cone.
    x += (wp.randf(rng) - 0.5) * jitter
    y += (wp.randf(rng) - 0.5) * jitter
    z += (wp.randf(rng) - 0.5) * jitter
    out_pos[i] = center + wp.vec3(x, y, wp.max(z, 0.0))


def sample_conical_pile(
    num_particles: int,
    center,
    *,
    height: float,
    angle_of_repose: float | None = None,
    base_radius: float | None = None,
    jitter: float = 0.0,
    seed: int = 0,
    device: str = "cpu",
) -> np.ndarray:
    """Sample particle positions forming a natural settled cone of granular media.

    Args:
        num_particles: Number of particle positions to generate.
        center: Cone base centre ``(x, y, z)`` on the surface [m].
        height: Cone (pile) height [m].
        angle_of_repose: Side slope of the pile [rad]; the base radius is ``height / tan(angle)``.
            Ignored if ``base_radius`` is given.
        base_radius: Explicit cone base radius [m]; overrides ``angle_of_repose``.
        jitter: Isotropic uniform position noise added per particle [m].
        seed: Random seed for reproducible piles.
        device: Warp device to run the kernel on.

    Returns:
        Particle positions, shape ``(num_particles, 3)``, float32.
    """
    if base_radius is None:
        if angle_of_repose is None:
            raise ValueError("Provide either angle_of_repose or base_radius.")
        base_radius = float(height) / max(math.tan(float(angle_of_repose)), 1.0e-3)
    cx, cy, cz = (float(center[0]), float(center[1]), float(center[2]))
    out = wp.zeros(int(num_particles), dtype=wp.vec3, device=device)
    wp.launch(
        _sample_conical_pile_kernel,
        dim=int(num_particles),
        inputs=[int(seed), wp.vec3(cx, cy, cz), float(base_radius), float(height), float(jitter), out],
        device=device,
    )
    return out.numpy().astype(np.float32, copy=False)
