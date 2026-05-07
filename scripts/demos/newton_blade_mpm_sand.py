# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Rigid blade cutting through MPM sand with Newton proxy coupling.

This is a compact Isaac Lab version of Newton's coupled-solver examples: Isaac
Lab owns the app, simulation context, visualizer lifecycle, and Newton manager,
while Newton's ``SolverProxyCoupled`` exchanges impulses between a MuJoCo Warp
rigid blade and an implicit MPM sand bed.

.. code-block:: bash

    # Headless, suitable for training/regression runs
    ./isaaclab.sh -p scripts/demos/newton_blade_mpm_sand.py

    # Native Newton visualizer with particles
    ./isaaclab.sh -p scripts/demos/newton_blade_mpm_sand.py --viz newton

    # Kit and Newton visualizers
    ./isaaclab.sh -p scripts/demos/newton_blade_mpm_sand.py --viz kit,newton
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Newton proxy-coupled blade cutting through MPM sand.")
parser.add_argument("--fps", type=float, default=60.0, help="Simulation/control frames per second.")
parser.add_argument("--max-steps", type=int, default=600, help="Stop after this many frames; negative runs forever.")
parser.add_argument("--blade-speed", type=float, default=0.45, help="Commanded blade speed along +Y [m/s].")
parser.add_argument("--voxel-size", type=float, default=0.05, help="MPM grid voxel size in meters.")
parser.add_argument("--particles-per-cell", type=float, default=1.2, help="Sand particles per grid cell.")
parser.add_argument("--mpm-iterations", type=int, default=50, help="Maximum MPM rheology iterations.")
parser.add_argument("--proxy-iterations", type=int, default=1, help="Proxy relaxation passes per coupled step.")
parser.add_argument("--proxy-mass-relaxation", type=float, default=1.0, help="Scale proxy blade mass inside MPM.")
parser.add_argument("--rigid-substeps", type=int, default=4, help="MJWarp substeps inside each coupled step.")
parser.add_argument("--disable-cuda-graph", action="store_true", help="Disable Newton CUDA graph capture.")
parser.add_argument("--kit-sand-stride", type=int, default=2, help="Render every Nth particle in Kit.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import numpy as np
import torch
import warp as wp

import newton
from newton.solvers import SolverImplicitMPM

import isaaclab.sim as sim_utils
from isaaclab_newton.physics import (
    CoupledProxyCfg,
    CoupledSolverCfg,
    CoupledSolverEntryCfg,
    MJWarpSolverCfg,
    MPMSolverCfg,
    NewtonCfg,
    NewtonManager,
    ProxyCouplingCfg,
)


BLADE_BODY_PATH = "/World/Blade"


class KitSandPoints:
    """Minimal USD ``Points`` helper for optional Kit particle visualization."""

    def __init__(self, prim_path: str, widths: np.ndarray):
        from pxr import Gf, UsdGeom, Vt  # noqa: PLC0415

        stage = sim_utils.get_current_stage()
        self._points = UsdGeom.Points.Define(stage, prim_path)
        self._points_attr = self._points.GetPointsAttr()
        self._widths_attr = self._points.CreateWidthsAttr(Vt.FloatArray(widths.tolist()))
        self._color_attr = self._points.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(0.72, 0.60, 0.38)]))
        self._widths = Vt.FloatArray(widths.tolist())
        self._colors = Vt.Vec3fArray([Gf.Vec3f(0.72, 0.60, 0.38)] * len(widths))
        self._particle_count = len(widths)
        self._widths_attr.Set(self._widths)
        self._color_attr.Set(self._colors)

    def update(self, positions: torch.Tensor) -> None:
        from pxr import Gf, Sdf, Vt  # noqa: PLC0415

        positions_np = positions.detach().cpu().numpy().astype(np.float32, copy=False)
        with Sdf.ChangeBlock():
            self._points_attr.Set(Vt.Vec3fArray.FromNumpy(positions_np))
            if positions_np.shape[0] != self._particle_count:
                self._particle_count = int(positions_np.shape[0])
                self._colors = Vt.Vec3fArray([Gf.Vec3f(0.72, 0.60, 0.38)] * self._particle_count)
                self._color_attr.Set(self._colors)


def setup_kit_scene(sim: sim_utils.SimulationContext) -> None:
    """Create simple Kit-only visuals for the blade, sand, ground, and light."""
    if "kit" not in sim.resolve_visualizer_types():
        return

    sim_utils.create_prim(BLADE_BODY_PATH, "Xform")
    sim_utils.create_prim(
        f"{BLADE_BODY_PATH}/visual",
        "Cube",
        scale=(0.45, 0.025, 0.30),
        attributes={"displayColor": [(0.18, 0.18, 0.20)]},
    )

    if not sim_utils.get_current_stage().GetPrimAtPath("/World/Ground").IsValid():
        ground_cfg = sim_utils.GroundPlaneCfg(size=(6.0, 6.0), color=(0.46, 0.38, 0.24))
        ground_cfg.func("/World/Ground", ground_cfg)

    if not sim_utils.get_current_stage().GetPrimAtPath("/World/DomeLight").IsValid():
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/DomeLight", light_cfg)


def spawn_sand(builder: newton.ModelBuilder) -> tuple[int, int]:
    """Add an MPM sand bed and return the particle index range."""
    density = 2500.0
    sand_lo = np.array([-0.65, -0.45, 0.0])
    sand_hi = np.array([0.65, 0.95, 0.25])
    resolution = np.maximum(np.ceil(args_cli.particles_per_cell * (sand_hi - sand_lo) / args_cli.voxel_size), 1).astype(
        int
    )
    cell_size = (sand_hi - sand_lo) / resolution
    radius = float(np.max(cell_size) * 0.5)
    mass = float(np.prod(cell_size) * density)
    particle_start = builder.particle_count
    builder.add_particle_grid(
        pos=wp.vec3(sand_lo),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0),
        dim_x=int(resolution[0]) + 1,
        dim_y=int(resolution[1]) + 1,
        dim_z=int(resolution[2]) + 1,
        cell_x=float(cell_size[0]),
        cell_y=float(cell_size[1]),
        cell_z=float(cell_size[2]),
        mass=mass,
        jitter=2.0 * radius,
        radius_mean=radius,
        custom_attributes={"mpm:friction": 0.75},
    )
    return particle_start, builder.particle_count


def build_blade_sand_model() -> tuple[newton.ModelBuilder, CoupledSolverCfg, int]:
    """Build the Newton model and matching proxy-coupled solver cfg."""
    builder = NewtonManager.create_builder()
    SolverImplicitMPM.register_custom_attributes(builder)
    builder.default_shape_cfg.ke = 5.0e4
    builder.default_shape_cfg.kd = 5.0e2
    builder.default_shape_cfg.kf = 1.0e3
    builder.default_shape_cfg.mu = 0.75

    blade_body = builder.add_body(
        xform=wp.transform(wp.vec3(0.0, -0.75, 0.34), wp.quat_identity()),
        mass=25.0,
        label=BLADE_BODY_PATH,
    )
    blade_cfg = newton.ModelBuilder.ShapeConfig(density=0.0, mu=0.65, ke=5.0e4, kd=5.0e2, kf=1.0e3)
    builder.add_shape_box(blade_body, hx=0.45, hy=0.025, hz=0.30, cfg=blade_cfg, color=(0.18, 0.18, 0.20))
    builder.add_ground_plane(color=(0.46, 0.38, 0.24))

    joint_qd_start = builder.joint_qd_start[0]
    builder.joint_qd[joint_qd_start + 1] = args_cli.blade_speed

    particle_start, particle_end = spawn_sand(builder)
    particle_indices = list(range(particle_start, particle_end))

    solver_cfg = CoupledSolverCfg(
        entries=[
            CoupledSolverEntryCfg(
                name="rigid",
                solver_cfg=MJWarpSolverCfg(
                    use_mujoco_contacts=False,
                    njmax=200,
                    nconmax=200,
                    iterations=50,
                    ls_iterations=25,
                ),
                bodies=[blade_body],
                joints=list(range(builder.joint_count)),
                substeps=args_cli.rigid_substeps,
            ),
            CoupledSolverEntryCfg(
                name="sand",
                solver_cfg=MPMSolverCfg(
                    voxel_size=args_cli.voxel_size,
                    grid_type="fixed",
                    grid_padding=32,
                    max_active_cell_count=1 << 15,
                    strain_basis="P0",
                    transfer_scheme="pic",
                    max_iterations=args_cli.mpm_iterations,
                    critical_fraction=0.0,
                    collider_velocity_mode="backward",
                ),
                particles=particle_indices,
            ),
        ],
        proxy_coupling=ProxyCouplingCfg(
            proxies=[
                CoupledProxyCfg(
                    source="rigid",
                    destination="sand",
                    bodies=[blade_body],
                    mass_scale=args_cli.proxy_mass_relaxation,
                    mode="lagged",
                )
            ],
            iterations=args_cli.proxy_iterations,
        ),
    )
    return builder, solver_cfg, blade_body


def create_sand_points(sim: sim_utils.SimulationContext, model: newton.Model) -> KitSandPoints | None:
    """Create Kit points for MPM particles when Kit visualization is active."""
    if "kit" not in sim.resolve_visualizer_types():
        return None
    if args_cli.kit_sand_stride < 1:
        raise ValueError("--kit-sand-stride must be >= 1.")
    particle_radius = wp.to_torch(model.particle_radius)
    widths = 2.0 * particle_radius[:: args_cli.kit_sand_stride].detach().cpu().numpy().astype(np.float32, copy=False)
    sim_utils.create_prim("/World/Visuals", "Xform")
    return KitSandPoints("/World/Visuals/SandParticles", widths=widths)


def update_sand_points(points: KitSandPoints | None, state: newton.State) -> None:
    """Push particle positions into the Kit USD points cloud."""
    if points is None:
        return
    particle_q = wp.to_torch(state.particle_q)
    points.update(particle_q[:: args_cli.kit_sand_stride])


def enable_newton_particle_visualization(sim: sim_utils.SimulationContext) -> None:
    """Turn on native particle rendering for active Newton-family visualizers."""
    for visualizer in sim.visualizers:
        viewer = getattr(visualizer, "_viewer", None)
        if viewer is not None and hasattr(viewer, "show_particles"):
            viewer.show_particles = True


def drive_blade_velocity(state: newton.State) -> None:
    """Keep the blade moving at the commanded speed through the bed."""
    joint_qd = wp.to_torch(state.joint_qd)
    joint_qd[0:3] = torch.tensor((0.0, args_cli.blade_speed, 0.0), device=joint_qd.device, dtype=joint_qd.dtype)


def run_simulator(sim: sim_utils.SimulationContext, sand_points: KitSandPoints | None) -> None:
    """Run the blade/sand simulation loop."""
    count = 0
    while simulation_app.is_running():
        if args_cli.max_steps >= 0 and count >= args_cli.max_steps:
            break
        drive_blade_velocity(NewtonManager.get_state_0())
        sim.step(render=False)
        if sim.is_rendering:
            update_sand_points(sand_points, NewtonManager.get_state_0())
            sim.render()
        count += 1


def main() -> None:
    """Set up and run the Isaac Lab blade/sand coupled demo."""
    if not str(args_cli.device).startswith("cuda"):
        raise RuntimeError("Newton implicit MPM coupling requires a CUDA device.")

    builder, solver_cfg, _blade_body = build_blade_sand_model()
    sim_cfg = sim_utils.SimulationCfg(
        dt=1.0 / args_cli.fps,
        device=args_cli.device,
        gravity=(0.0, 0.0, -9.81),
        physics=NewtonCfg(
            solver_cfg=solver_cfg,
            num_substeps=1,
            use_cuda_graph=not args_cli.disable_cuda_graph,
        ),
    )
    sim = sim_utils.SimulationContext(sim_cfg)
    setup_kit_scene(sim)
    sim.set_camera_view(eye=[2.8, -2.7, 1.4], target=[0.0, 0.2, 0.25])
    NewtonManager.set_builder(builder)
    sim.reset()
    enable_newton_particle_visualization(sim)

    sand_points = create_sand_points(sim, NewtonManager.get_model())
    update_sand_points(sand_points, NewtonManager.get_state_0())
    print("[INFO]: Isaac Lab Newton proxy-coupled blade/sand demo ready.")
    run_simulator(sim, sand_points)


if __name__ == "__main__":
    main()
    simulation_app.close()
