# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton implicit MPM snowball-smash demo.

Three snowballs are thrown from the side into a pyramid of colored crates,
striking one after another. The snow splats with elastic-plastic snow rheology
and the impact impulses knock the crates over through Newton proxy coupling
(MuJoCo-Warp rigid bodies two-way coupled with the implicit MPM solver).

The demo shows the Isaac Lab MPM scene path:

* spawn snowballs as :class:`~isaaclab_newton.assets.mpm_object.MPMObject` assets
  with per-particle snow material and a non-zero initial velocity;
* spawn the crates as standard dynamic :class:`~isaaclab.assets.RigidObject` assets;
* configure :class:`~isaaclab_newton.physics.CoupledSolverCfg` so the crates are
  advanced by MuJoCo-Warp while the snow exchanges impulses with them through a
  lagged proxy mapping.

.. code-block:: bash

    ./isaaclab.sh -p scripts/demos/mpm/snowball_smash.py --device cuda:0 --visualizer newton
"""

from __future__ import annotations

import argparse

import numpy as np

from isaaclab.app import add_launcher_args, launch_simulation

DEFAULT_VOXEL_SIZE = 0.05

parser = argparse.ArgumentParser(description="Newton implicit MPM snowball-smash demo.")
parser.add_argument("--max-steps", type=int, default=900, help="Stop after this many frames; negative runs forever.")
parser.add_argument(
    "--voxel-size",
    type=float,
    default=DEFAULT_VOXEL_SIZE,
    help=f"MPM grid voxel size in meters. Defaults to {DEFAULT_VOXEL_SIZE:g}.",
)
parser.add_argument("--throw-speed", type=float, default=6.5, help="Horizontal speed of the thrown snowball [m/s].")
parser.add_argument(
    "--crate-mass",
    type=float,
    default=20.0,
    help="Mass of each crate [kg]. Below ~15 kg the late snow flow can fling resting crates.",
)
parser.add_argument("--mpm-iterations", type=int, default=100, help="Maximum MPM rheology iterations.")
parser.add_argument("--rigid-substeps", type=int, default=4, help="MuJoCo-Warp substeps inside each coupled step.")
parser.add_argument(
    "--proxy-relaxation",
    type=float,
    default=0.5,
    help="Lagged proxy feedback relaxation; values below 1 damp the snow-crate impulse exchange.",
)
parser.add_argument(
    "--log-interval", type=int, default=60, help="Print simulation progress every N frames; 0 disables."
)
parser.add_argument("--disable-cuda-graph", action="store_true", help="Disable Newton CUDA graph capture.")
add_launcher_args(parser)
parser.set_defaults(visualizer=["newton"])
args_cli = parser.parse_args()


FPS = 60.0
VOXEL_SIZE = float(args_cli.voxel_size)
# Two snow particles per grid cell along each axis, as in Newton's snow-ball example.
SNOW_SPACING = 0.5 * VOXEL_SIZE

# A fixed grid keeps a static topology so the coupled step can be captured in a
# CUDA graph. ``grid_padding`` sizes the frozen grid around the initial snowball
# bounds; the balls start well to the side of the crates, so the padding must
# cover the horizontal flight to the stack plus the splat spread beyond it.
GRID_TYPE = "fixed"
GRID_PADDING = 96
MAX_ACTIVE_CELL_COUNT = 1 << 17

RIGID_ENTRY = "crates"
SNOW_ENTRY = "snow"
CRATE_BODY_PATTERN = r"Crate_.*"

# Snow rheology from Newton's example_mpm_snow_ball: soft elastic-plastic
# material with hardening and dilatancy, so the balls splat and compact on impact.
SNOW_DENSITY = 400.0
SNOW_YOUNG_MODULUS = 1.4e6
SNOW_POISSON_RATIO = 0.3
SNOW_FRICTION = 0.5
SNOW_DAMPING = 0.01
SNOW_YIELD_PRESSURE = 1.4e6
SNOW_TENSILE_YIELD_RATIO = 0.2
SNOW_HARDENING = 5.0
SNOW_DILATANCY = 1.0
SNOW_COLOR = (0.88, 0.92, 0.98)

CRATE_SIZE = 0.6
CRATE_MASS = float(args_cli.crate_mass)
CRATE_FRICTION = 0.6

# Crate pyramid at the origin: ``(center, color)`` per crate, stacked 3-2-1.
CRATE_STACK = (
    ((0.0, -0.62, 0.30), (0.85, 0.18, 0.18)),
    ((0.0, 0.0, 0.30), (0.95, 0.55, 0.10)),
    ((0.0, 0.62, 0.30), (0.93, 0.80, 0.15)),
    ((0.0, -0.31, 0.92), (0.20, 0.65, 0.30)),
    ((0.0, 0.31, 0.92), (0.15, 0.40, 0.80)),
    ((0.0, 0.0, 1.54), (0.55, 0.25, 0.70)),
)

# Snowballs: all thrown sideways from the -x side at staggered distances, so
# they strike the stack one after another (~0.45 s, ~0.6 s, ~0.75 s). The
# slight upward and lateral velocity components arc each ball into a different
# part of the pyramid.
THROW_SPEED = float(args_cli.throw_speed)
SNOWBALLS = (
    {
        "name": "SnowballFirst",
        "asset": "snowball_first",
        "radius": 0.25,
        "center": (-2.8, 0.0, 1.2),
        "velocity": (THROW_SPEED, 0.0, 1.8),
        "seed": 11,
    },
    {
        "name": "SnowballSecond",
        "asset": "snowball_second",
        "radius": 0.22,
        "center": (-3.8, -0.5, 1.8),
        "velocity": (THROW_SPEED, 0.9, 1.8),
        "seed": 12,
    },
    {
        "name": "SnowballThird",
        "asset": "snowball_third",
        "radius": 0.20,
        "center": (-4.8, 0.5, 2.0),
        "velocity": (THROW_SPEED, -0.8, 2.2),
        "seed": 13,
    },
)
SNOWBALL_ASSET_NAMES = tuple(ball["asset"] for ball in SNOWBALLS)

GROUND_COLOR = (0.32, 0.34, 0.38)

# View from the -x side the snowballs are thrown from, so the camera looks at
# the pyramid face the balls slam into, with the throws crossing the frame.
CAMERA_EYE = (-4.2, -3.2, 2.2)
CAMERA_TARGET = (0.0, 0.0, 0.9)


def create_visualizer_cfgs():
    """Create demo-specific visualizer configs for requested backends."""
    if "newton" not in (args_cli.visualizer or []):
        return []

    from isaaclab_visualizers.newton import NewtonVisualizerCfg

    return [
        NewtonVisualizerCfg(
            show_particles=True,
            particle_color=SNOW_COLOR,
        )
    ]


def create_snowball_points(radius: float, seed: int) -> tuple[np.ndarray, float, float]:
    """Return jittered local-space sphere points with per-particle mass and radius."""
    dim = int(2.0 * radius / SNOW_SPACING) + 1
    axis = (np.arange(dim) - 0.5 * (dim - 1)) * SNOW_SPACING
    points = np.stack(np.meshgrid(axis, axis, axis, indexing="ij")).reshape(3, -1).T
    points = points[np.linalg.norm(points, axis=1) <= radius]
    if points.shape[0] == 0:
        raise RuntimeError("Snowball generation produced no particles; reduce --voxel-size.")

    rng = np.random.default_rng(seed)
    points += (rng.random(points.shape) - 0.5) * SNOW_SPACING

    particle_mass = float(SNOW_SPACING**3 * SNOW_DENSITY)
    particle_radius = 0.5 * SNOW_SPACING
    return points.astype(np.float32), particle_mass, particle_radius


def create_sim_cfg():
    """Create the Isaac Lab simulation config with the proxy-coupled solver."""
    from isaaclab_newton.physics import (
        CoupledProxyCfg,
        CoupledSolverCfg,
        CoupledSolverEntryCfg,
        MJWarpSolverCfg,
        MPMSolverCfg,
        NewtonCfg,
        ProxyCouplingCfg,
    )

    import isaaclab.sim as sim_utils

    if not str(args_cli.device).startswith("cuda"):
        raise RuntimeError("Newton implicit MPM coupling requires a CUDA device.")

    solver_cfg = CoupledSolverCfg(
        coupling_type="proxy",
        entries=[
            CoupledSolverEntryCfg(
                name=RIGID_ENTRY,
                # MuJoCo handles crate-crate and crate-ground contacts; the proxy
                # coupling below handles the snow-crate impulse exchange.
                solver_cfg=MJWarpSolverCfg(use_mujoco_contacts=True, njmax=500, nconmax=400),
                body_name_patterns=[CRATE_BODY_PATTERN],
                include_static_shapes=True,
                substeps=args_cli.rigid_substeps,
            ),
            CoupledSolverEntryCfg(
                name=SNOW_ENTRY,
                solver_cfg=MPMSolverCfg(
                    voxel_size=VOXEL_SIZE,
                    grid_type=GRID_TYPE,
                    grid_padding=GRID_PADDING,
                    max_active_cell_count=MAX_ACTIVE_CELL_COUNT,
                    strain_basis="P0",
                    transfer_scheme="apic",
                    max_iterations=args_cli.mpm_iterations,
                    # Keep the ballistic snowballs crisp; the default drag of 1.0
                    # visibly slows the thrown ball mid-flight.
                    air_drag=0.2,
                ),
                all_particles=True,
                in_place=True,
            ),
        ],
        use_collision_pipeline=False,
        proxy_coupling=ProxyCouplingCfg(
            proxies=[
                CoupledProxyCfg(
                    source=RIGID_ENTRY,
                    destination=SNOW_ENTRY,
                    body_name_patterns=[CRATE_BODY_PATTERN],
                    mode="lagged",
                    # Under-relax the lagged feedback so the snow wave plowing a
                    # crate along the ground cannot pump it to unphysical speeds.
                    proxy_relaxation=float(args_cli.proxy_relaxation),
                )
            ],
        ),
    )

    return sim_utils.SimulationCfg(
        dt=1.0 / FPS,
        device=args_cli.device,
        gravity=(0.0, 0.0, -9.81),
        visualizer_cfgs=create_visualizer_cfgs(),
        physics=NewtonCfg(
            solver_cfg=solver_cfg,
            num_substeps=1,
            use_cuda_graph=not args_cli.disable_cuda_graph,
        ),
    )


def preview_material(color):
    """Return a preview-surface material for Kit runs; Kit-less runs spawn no USD materials."""
    if "kit" not in (args_cli.visualizer or []):
        return None

    import isaaclab.sim as sim_utils

    return sim_utils.PreviewSurfaceCfg(diffuse_color=color)


def create_scene_cfg():
    """Create the snowball-smash scene using declarative Isaac Lab assets."""
    from isaaclab_newton.assets.mpm_object import MPMObjectCfg
    from isaaclab_newton.sim.spawners.mpm import MPMParticleMaterialCfg, MPMPointsCfg

    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.utils.configclass import configclass

    def crate_cfg(index: int) -> RigidObjectCfg:
        center, color = CRATE_STACK[index]
        return RigidObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Crate_{index}",
            spawn=sim_utils.CuboidCfg(
                size=(CRATE_SIZE, CRATE_SIZE, CRATE_SIZE),
                rigid_props=sim_utils.NewtonRigidBodyPropertiesCfg(rigid_body_enabled=True),
                mass_props=sim_utils.MassPropertiesCfg(mass=CRATE_MASS),
                collision_props=sim_utils.NewtonCollisionPropertiesCfg(collision_enabled=True),
                physics_material=sim_utils.NewtonMaterialPropertiesCfg(
                    static_friction=CRATE_FRICTION,
                    dynamic_friction=CRATE_FRICTION,
                ),
                physics_material_path="physicsMaterial",
                visual_material=preview_material(color),
                visual_material_path="visualMaterial",
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=center),
        )

    def snowball_cfg(ball: dict) -> MPMObjectCfg:
        points, particle_mass, particle_radius = create_snowball_points(ball["radius"], ball["seed"])
        return MPMObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/{ball['name']}",
            spawn=MPMPointsCfg(
                positions=points.tolist(),
                velocities=[list(ball["velocity"])] * points.shape[0],
                mass=particle_mass,
                radius=particle_radius,
                material=MPMParticleMaterialCfg(
                    density=SNOW_DENSITY,
                    young_modulus=SNOW_YOUNG_MODULUS,
                    poisson_ratio=SNOW_POISSON_RATIO,
                    friction=SNOW_FRICTION,
                    damping=SNOW_DAMPING,
                    yield_pressure=SNOW_YIELD_PRESSURE,
                    tensile_yield_ratio=SNOW_TENSILE_YIELD_RATIO,
                    hardening=SNOW_HARDENING,
                    dilatancy=SNOW_DILATANCY,
                ),
                visual_color=SNOW_COLOR,
            ),
            init_state=MPMObjectCfg.InitialStateCfg(pos=ball["center"]),
        )

    @configclass
    class SnowballSmashSceneCfg(InteractiveSceneCfg):
        """Scene with a crate pyramid, three MPM snowballs, ground, and lighting."""

        ground = AssetBaseCfg(
            prim_path="/World/Ground",
            spawn=sim_utils.GroundPlaneCfg(size=(12.0, 12.0), color=GROUND_COLOR),
        )

        dome_light = AssetBaseCfg(
            prim_path="/World/DomeLight",
            spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.78, 0.78, 0.78)),
        )

        crate_0 = crate_cfg(0)
        crate_1 = crate_cfg(1)
        crate_2 = crate_cfg(2)
        crate_3 = crate_cfg(3)
        crate_4 = crate_cfg(4)
        crate_5 = crate_cfg(5)

        snowball_first = snowball_cfg(SNOWBALLS[0])
        snowball_second = snowball_cfg(SNOWBALLS[1])
        snowball_third = snowball_cfg(SNOWBALLS[2])

    return SnowballSmashSceneCfg(num_envs=1, env_spacing=0.0)


def particle_count(scene) -> int:
    """Return the number of MPM particles in the scene."""
    return sum(scene[name].num_instances * scene[name].particles_per_object for name in SNOWBALL_ASSET_NAMES)


def keep_running(sim, count: int) -> bool:
    """Return whether the demo loop should continue this frame."""
    if args_cli.max_steps >= 0 and count >= args_cli.max_steps:
        return False
    return sim.is_headless_or_exist_active_visualizer()


def read_snow_reaction_force() -> float | None:
    """Return the total force magnitude the snow currently applies to the crates."""
    import warp as wp
    from isaaclab_newton.physics import NewtonCoupledManager

    wrenches = NewtonCoupledManager.get_proxy_body_wrenches(RIGID_ENTRY, SNOW_ENTRY)
    if wrenches is None:
        return None
    forces = wp.to_torch(wrenches)[:, 0:3]
    return float(forces.norm(dim=1).sum().item())


def log_force_plot(sim, force: float | None) -> None:
    """Push the snow impact force into the Newton viewer plot window."""
    if force is None:
        return
    for visualizer in sim.visualizers:
        viewer = getattr(visualizer, "_viewer", None)
        if viewer is not None and hasattr(viewer, "log_scalar"):
            viewer.log_scalar("Snow impact |F| [N]", force, smoothing=4)


def crate_displacement(scene) -> tuple[float, float]:
    """Return the largest crate displacement from its initial pose and the largest crate speed."""
    import torch

    displacements = []
    speeds = []
    positions = []
    for index, (center, _) in enumerate(CRATE_STACK):
        pos = scene[f"crate_{index}"].data.root_pos_w.torch[0]
        vel = scene[f"crate_{index}"].data.root_lin_vel_w.torch[0]
        init = torch.tensor(center, dtype=pos.dtype, device=pos.device)
        displacements.append(float(torch.linalg.norm(pos - init).item()))
        speeds.append(float(torch.linalg.norm(vel).item()))
        positions.append(pos)
    fastest = max(range(len(speeds)), key=speeds.__getitem__)
    pos_text = ", ".join(f"{float(v):.2f}" for v in positions[fastest])
    return max(displacements), max(speeds), f"crate_{fastest}@({pos_text})"


def log_progress(scene, count: int, force: float | None) -> None:
    """Print a compact heartbeat showing impacts and crate motion."""
    if args_cli.log_interval <= 0 or count % args_cli.log_interval != 0:
        return
    force_text = f"{force:.1f}" if force is not None else "n/a"
    displacement, speed, fastest = crate_displacement(scene)
    print(
        f"[INFO]: step {count:05d} t={count / FPS:.2f}s"
        f" snow->crates |F|={force_text}N"
        f" max crate displacement={displacement:.3f}m"
        f" max crate speed={speed:.2f}m/s ({fastest})",
        flush=True,
    )


def run_simulator(sim, scene) -> None:
    """Run the snowball-smash coupled MPM loop."""
    sim_dt = sim.get_physics_dt()
    count = 0
    while keep_running(sim, count):
        sim.step(render=False)
        scene.update(sim_dt)
        force = read_snow_reaction_force()
        log_force_plot(sim, force)
        log_progress(scene, count, force)
        if sim.is_rendering:
            sim.render()
        count += 1


def main() -> None:
    """Set up and run the Isaac Lab Newton MPM snowball-smash demo."""
    sim_cfg = create_sim_cfg()
    with launch_simulation(sim_cfg, args_cli):
        import isaaclab.sim as sim_utils
        from isaaclab.scene import InteractiveScene

        sim = sim_utils.SimulationContext(sim_cfg)
        scene = InteractiveScene(create_scene_cfg())
        sim.reset()
        sim.set_camera_view(eye=CAMERA_EYE, target=CAMERA_TARGET)

        print(
            "[INFO]: Isaac Lab Newton snowball-smash MPM demo ready."
            f" Spawned {particle_count(scene)} MPM particles across {len(SNOWBALLS)} snowballs;"
            f" voxel size {VOXEL_SIZE:.4g} m;"
            f" the first snowball hits the crates after ~0.45s.",
            flush=True,
        )
        run_simulator(sim, scene)


if __name__ == "__main__":
    main()
