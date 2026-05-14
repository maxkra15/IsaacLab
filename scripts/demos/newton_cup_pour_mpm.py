# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

"""Newton MPM cup-pouring demo with Isaac Sim tableware assets.

This demo places a table, bowl, and mug/cup visual in Isaac Lab, fills an
analytic Newton cup collider with small MPM particles, and kinematically tilts
the cup so the material pours into the bowl.

.. code-block:: bash

    ./isaaclab.sh -p scripts/demos/newton_cup_pour_mpm.py --viz newton
"""

import argparse
import math
from dataclasses import dataclass
from types import SimpleNamespace

from isaaclab_tasks.utils.sim_launcher import add_launcher_args, launch_simulation


parser = argparse.ArgumentParser(description="Newton MPM cup pour demo.")
parser.add_argument("--fps", type=float, default=120.0, help="Simulation/control frames per second.")
parser.add_argument("--max-steps", type=int, default=900, help="Stop after this many frames; negative runs forever.")
parser.add_argument("--sim-substeps", type=int, default=1, help="Newton MPM substeps per rendered/control frame.")
parser.add_argument("--voxel-size", type=float, default=0.01, help="MPM grid voxel size in meters.")
parser.add_argument("--particles-per-cell", type=float, default=2.5, help="Particle lattice density per MPM grid cell.")
parser.add_argument("--mpm-iterations", type=int, default=160, help="Maximum MPM rheology iterations.")
parser.add_argument("--density", type=float, default=1000.0, help="Particle material density in kg/m^3.")
parser.add_argument("--viscosity", type=float, default=50.0, help="MPM plastic viscosity; higher values look thicker.")
parser.add_argument("--fluid-friction", type=float, default=0.0, help="MPM particle friction coefficient.")
parser.add_argument("--fill-fraction", type=float, default=1.0, help="Fraction of the cup interior height initially filled.")
parser.add_argument(
    "--fluid-wall-clearance",
    type=float,
    default=0.014,
    help="Minimum distance between initial particles and the cup inner wall/bottom/rim.",
)
parser.add_argument(
    "--fluid-bottom-clearance",
    type=float,
    default=0.0,
    help="Minimum distance between initial particles and the cup floor.",
)
parser.add_argument(
    "--fluid-top-clearance",
    type=float,
    default=0.0,
    help="Minimum distance between initial particles and the cup rim.",
)
parser.add_argument("--tensile-yield-ratio", type=float, default=1.0, help="MPM tensile yield ratio for cohesion.")
parser.add_argument("--yield-pressure", type=float, default=1.0e15, help="MPM compressive yield pressure.")
parser.add_argument("--yield-stress", type=float, default=0.0, help="MPM deviatoric yield stress.")
parser.add_argument("--young-modulus", type=float, default=1.0e15, help="MPM Young's modulus.")
parser.add_argument("--damping", type=float, default=0.0, help="MPM elastic damping relaxation time.")
parser.add_argument("--collider-margin", type=float, default=0.002, help="Collider margin/thickness used by MPM.")
parser.add_argument("--cup-friction", type=float, default=0.0, help="Cup collider friction.")
parser.add_argument("--bowl-friction", type=float, default=0.05, help="Bowl collider friction.")
parser.add_argument("--table-friction", type=float, default=0.5, help="Table/ground collider friction.")
parser.add_argument("--hold-time", type=float, default=0.45, help="Seconds to hold the filled cup upright.")
parser.add_argument("--tilt-time", type=float, default=2.2, help="Seconds over which the cup tilts.")
parser.add_argument("--pour-angle-deg", type=float, default=112.0, help="Final cup tilt angle toward the bowl.")
parser.add_argument("--log-interval", type=int, default=60, help="Print simulation progress every N steps; 0 disables.")
parser.add_argument("--disable-cuda-graph", action="store_true", help="Disable Newton CUDA graph capture.")
parser.add_argument("--kit-particle-stride", type=int, default=2, help="Render every Nth particle in Kit; 1 renders all.")
parser.add_argument("--grains-per-particle", type=int, default=5, help="Newton viewer grain samples per MPM particle.")
parser.add_argument("--grain-radius-scale", type=float, default=1.0, help="Scale factor for Newton viewer grain radii.")
parser.add_argument("--table-usd", type=str, default=None, help="Override the table USD asset path.")
parser.add_argument("--bowl-usd", type=str, default=None, help="Override the bowl USD asset path.")
parser.add_argument("--cup-usd", type=str, default=None, help="Override the cup/mug USD asset path.")
parser.add_argument(
    "--cup-visual",
    type=str,
    choices=("procedural", "usd"),
    default="procedural",
    help="Use the matching procedural flat cup visual or a USD mug/beaker visual.",
)
parser.add_argument(
    "--cup-kind",
    type=str,
    choices=("mug", "beaker"),
    default="mug",
    help="Default IsaacLab cup-like asset to use when --cup-usd is not provided.",
)
parser.add_argument("--table-visual-z", type=float, default=0.68, help="Z translation for the table USD visual.")
parser.add_argument("--asset-scale", type=float, default=1.0, help="Uniform scale applied to bowl/cup visual assets.")
add_launcher_args(parser)
args_cli = parser.parse_args()


np = torch = wp = newton = sim_utils = None
SolverImplicitMPM = None
MPMSolverCfg = NewtonCfg = NewtonManager = None


TABLE_PATH = "/World/Table"
BOWL_PATH = "/World/Bowl"
CUP_BODY_PATH = "/World/Cup"
GROUND_PATH = "/World/Ground"
VISUALS_PATH = "/World/Visuals"

TABLE_TOP_Z = 0.85
TABLE_HALF_EXTENTS = (0.85, 0.55, 0.03)
BOWL_BASE_POS = (0.22, 0.0, TABLE_TOP_Z + 0.02)
BOWL_HEIGHT = 0.13
BOWL_RIM_Z = BOWL_BASE_POS[2] + BOWL_HEIGHT
CUP_BASE_POS = (0.02, 0.0, BOWL_RIM_Z + 0.24)
CUP_INNER_RADIUS = 0.075
CUP_WALL_THICKNESS = 0.008
CUP_HEIGHT = 0.18
CUP_BOTTOM_THICKNESS = 0.024


@dataclass(frozen=True)
class AssetPaths:
    """Resolved USD asset paths for the Kit-only visual scene."""

    table: str
    bowl: str
    cup: str


def import_runtime_dependencies() -> None:
    """Import Newton/Isaac Lab modules after Kit has launched when requested."""

    global np, torch, wp, newton, sim_utils, SolverImplicitMPM
    global MPMSolverCfg, NewtonCfg, NewtonManager

    import numpy as np_module
    import torch as torch_module
    import warp as wp_module

    import newton as newton_module
    from newton.solvers import SolverImplicitMPM as SolverImplicitMPMClass

    import isaaclab.sim as sim_utils_module
    from isaaclab_newton.physics import MPMSolverCfg as MPMSolverCfgClass
    from isaaclab_newton.physics import NewtonCfg as NewtonCfgClass
    from isaaclab_newton.physics import NewtonManager as NewtonManagerClass

    np = np_module
    torch = torch_module
    wp = wp_module
    newton = newton_module
    SolverImplicitMPM = SolverImplicitMPMClass
    sim_utils = sim_utils_module
    MPMSolverCfg = MPMSolverCfgClass
    NewtonCfg = NewtonCfgClass
    NewtonManager = NewtonManagerClass


def resolve_asset_paths() -> AssetPaths:
    """Resolve default IsaacLab/Isaac Sim USD assets, honoring CLI overrides."""

    from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

    cup_default = (
        f"{ISAACLAB_NUCLEUS_DIR}/Mimic/nut_pour_task/nut_pour_assets/sorting_beaker_red.usd"
        if args_cli.cup_kind == "beaker"
        else f"{ISAACLAB_NUCLEUS_DIR}/Objects/Mug/mug.usd"
    )
    return AssetPaths(
        table=args_cli.table_usd or f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
        bowl=args_cli.bowl_usd
        or f"{ISAACLAB_NUCLEUS_DIR}/Mimic/nut_pour_task/nut_pour_assets/sorting_bowl_yellow.usd",
        cup=args_cli.cup_usd or cup_default,
    )


def quat_y(angle_rad: float) -> tuple[float, float, float, float]:
    """Return an XYZW quaternion for a rotation about +Y."""

    half = 0.5 * angle_rad
    return (0.0, math.sin(half), 0.0, math.cos(half))


def smoothstep(value: float) -> float:
    """Smoothly remap a 0..1 value to ease the cup tilt."""

    t = max(0.0, min(1.0, value))
    return t * t * (3.0 - 2.0 * t)


def cup_pose_at_time(sim_time: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return cup position, orientation, and angular velocity for the scripted pour."""

    if args_cli.tilt_time <= 0.0:
        alpha = 1.0
        alpha_dot = 0.0
    else:
        raw = (sim_time - args_cli.hold_time) / args_cli.tilt_time
        clamped = max(0.0, min(1.0, raw))
        alpha = smoothstep(clamped)
        alpha_dot = 0.0
        if 0.0 < raw < 1.0:
            alpha_dot = (6.0 * clamped * (1.0 - clamped)) / args_cli.tilt_time

    final_angle = math.radians(args_cli.pour_angle_deg)
    angle = final_angle * alpha
    angular_speed = final_angle * alpha_dot
    pos = np.array(CUP_BASE_POS, dtype=np.float32)
    quat = np.array(quat_y(angle), dtype=np.float32)
    # Spatial vectors store linear velocity followed by angular velocity.
    qd = np.array([0.0, 0.0, 0.0, 0.0, angular_speed, 0.0], dtype=np.float32)
    return pos, quat, qd


def create_open_cup_mesh(
    *,
    inner_radius: float,
    wall_thickness: float,
    height: float,
    bottom_thickness: float,
    num_segments: int = 80,
    include_bottom: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a thick, open-top cup mesh in local coordinates."""

    theta = np.linspace(0.0, 2.0 * math.pi, num_segments, endpoint=False)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    outer_radius = inner_radius + wall_thickness

    def ring(radius: float, z: float) -> np.ndarray:
        return np.column_stack([radius * cos_t, radius * sin_t, np.full(num_segments, z)])

    inner_bottom = ring(inner_radius, 0.0)
    inner_top = ring(inner_radius, height)
    outer_top = ring(outer_radius, height)
    outer_bottom = ring(outer_radius, -bottom_thickness)
    rings = [inner_bottom, inner_top, outer_top, outer_bottom]
    if include_bottom:
        rings.extend(
            [
                np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
                np.array([[0.0, 0.0, -bottom_thickness]], dtype=np.float32),
            ]
        )
    vertices = np.vstack(rings).astype(np.float32)
    inner_center_id = 4 * num_segments
    outer_center_id = inner_center_id + 1

    indices: list[int] = []
    for i in range(num_segments):
        j = (i + 1) % num_segments
        ib_i, ib_j = i, j
        it_i, it_j = i + num_segments, j + num_segments
        ot_i, ot_j = i + 2 * num_segments, j + 2 * num_segments
        ob_i, ob_j = i + 3 * num_segments, j + 3 * num_segments

        # Inner wall normals face inward toward the fluid volume.
        indices.extend([ib_i, it_i, ib_j])
        indices.extend([ib_j, it_i, it_j])
        # Outer wall normals face away from the solid cup wall.
        indices.extend([ob_i, ob_j, ot_i])
        indices.extend([ot_i, ob_j, ot_j])
        # Top rim.
        indices.extend([it_i, ot_i, it_j])
        indices.extend([it_j, ot_i, ot_j])
        if include_bottom:
            # Visual-only flat bottom; physics uses a separate thick cylinder plug.
            indices.extend([inner_center_id, ib_i, ib_j])
            indices.extend([outer_center_id, ob_j, ob_i])

    return vertices, np.array(indices, dtype=np.int32)


def create_open_bowl_mesh(
    *,
    inner_bottom_radius: float,
    inner_top_radius: float,
    wall_thickness: float,
    height: float,
    bottom_thickness: float,
    num_segments: int = 96,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a simple flared open bowl mesh in local coordinates."""

    theta = np.linspace(0.0, 2.0 * math.pi, num_segments, endpoint=False)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    outer_bottom_radius = inner_bottom_radius + wall_thickness
    outer_top_radius = inner_top_radius + wall_thickness

    def ring(radius: float, z: float) -> np.ndarray:
        return np.column_stack([radius * cos_t, radius * sin_t, np.full(num_segments, z)])

    inner_bottom = ring(inner_bottom_radius, bottom_thickness)
    inner_top = ring(inner_top_radius, height)
    outer_top = ring(outer_top_radius, height)
    outer_bottom = ring(outer_bottom_radius, 0.0)
    inner_center = np.array([[0.0, 0.0, bottom_thickness]], dtype=np.float32)
    outer_center = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)

    vertices = np.vstack([inner_bottom, inner_top, outer_top, outer_bottom, inner_center, outer_center]).astype(
        np.float32
    )
    inner_center_id = 4 * num_segments
    outer_center_id = inner_center_id + 1

    indices: list[int] = []
    for i in range(num_segments):
        j = (i + 1) % num_segments
        ib_i, ib_j = i, j
        it_i, it_j = i + num_segments, j + num_segments
        ot_i, ot_j = i + 2 * num_segments, j + 2 * num_segments
        ob_i, ob_j = i + 3 * num_segments, j + 3 * num_segments

        indices.extend([ib_i, it_i, ib_j])
        indices.extend([ib_j, it_i, it_j])
        indices.extend([ob_i, ob_j, ot_i])
        indices.extend([ot_i, ob_j, ot_j])
        indices.extend([it_i, ot_i, it_j])
        indices.extend([it_j, ot_i, ot_j])
        indices.extend([inner_center_id, ib_i, ib_j])
        indices.extend([outer_center_id, ob_j, ob_i])

    return vertices, np.array(indices, dtype=np.int32)


def emit_fluid_particles(builder: newton.ModelBuilder) -> tuple[int, int]:
    """Seed tiny MPM particles inside the upright cup volume."""

    # Match Newton's MPM examples: keep initial particles at least one voxel
    # away from collider surfaces so the SDF projection starts well-conditioned.
    wall_clearance = max(args_cli.fluid_wall_clearance, args_cli.voxel_size, 3.0 * args_cli.collider_margin)
    bottom_clearance = max(args_cli.fluid_bottom_clearance, args_cli.voxel_size, 3.0 * args_cli.collider_margin)
    top_clearance = max(args_cli.fluid_top_clearance, 0.5 * args_cli.voxel_size, 1.5 * args_cli.collider_margin)
    fill_height = max(0.0, min(1.0, args_cli.fill_fraction)) * CUP_HEIGHT
    fill_height = min(fill_height, CUP_HEIGHT - top_clearance)

    particle_lo = np.array(
        [-CUP_INNER_RADIUS + wall_clearance, -CUP_INNER_RADIUS + wall_clearance, bottom_clearance]
    )
    particle_hi = np.array([CUP_INNER_RADIUS - wall_clearance, CUP_INNER_RADIUS - wall_clearance, fill_height])
    if np.any(particle_hi <= particle_lo):
        raise RuntimeError(
            "Particle initialization has no valid cup interior. Reduce --fluid-wall-clearance or --voxel-size."
        )
    resolution = np.maximum(
        np.ceil(args_cli.particles_per_cell * (particle_hi - particle_lo) / args_cli.voxel_size), 1
    ).astype(int)
    cell_size = (particle_hi - particle_lo) / resolution
    cell_volume = float(np.prod(cell_size))
    radius = float(np.max(cell_size) * 0.45)
    mass = float(cell_volume * args_cli.density)

    px = np.arange(int(resolution[0]) + 1) * cell_size[0]
    py = np.arange(int(resolution[1]) + 1) * cell_size[1]
    pz = np.arange(int(resolution[2]) + 1) * cell_size[2]
    points = np.stack(np.meshgrid(px, py, pz, indexing="ij")).reshape(3, -1).T

    rng = np.random.default_rng(7)
    points += (rng.random(points.shape) - 0.5) * (0.10 * np.max(cell_size))
    points += particle_lo

    r_xy = np.linalg.norm(points[:, 0:2], axis=1)
    inside = (r_xy < CUP_INNER_RADIUS - wall_clearance) & (points[:, 2] < fill_height)
    points = points[inside] + np.array(CUP_BASE_POS)

    if points.shape[0] == 0:
        raise RuntimeError("Particle initialization produced no particles; reduce --voxel-size or --collider-margin.")

    particle_start = builder.particle_count
    builder.add_particles(
        pos=points.tolist(),
        vel=np.zeros_like(points).tolist(),
        mass=[mass] * points.shape[0],
        radius=[radius] * points.shape[0],
        custom_attributes={
            "mpm:viscosity": args_cli.viscosity,
            "mpm:friction": args_cli.fluid_friction,
            "mpm:tensile_yield_ratio": args_cli.tensile_yield_ratio,
            "mpm:yield_pressure": args_cli.yield_pressure,
            "mpm:yield_stress": args_cli.yield_stress,
            "mpm:young_modulus": args_cli.young_modulus,
            "mpm:damping": args_cli.damping,
        },
    )
    return particle_start, builder.particle_count


def build_cup_pour_model() -> tuple[newton.ModelBuilder, int, tuple[int, int]]:
    """Build the Newton MPM model with analytic colliders and fluid particles."""

    builder = NewtonManager.create_builder()
    SolverImplicitMPM.register_custom_attributes(builder)
    builder.default_shape_cfg.mu = 0.2

    table_cfg = newton.ModelBuilder.ShapeConfig(mu=args_cli.table_friction, margin=args_cli.collider_margin)
    builder.add_shape_box(
        -1,
        xform=wp.transform(
            wp.vec3(0.0, 0.0, TABLE_TOP_Z - TABLE_HALF_EXTENTS[2]),
            wp.quat_identity(),
        ),
        hx=TABLE_HALF_EXTENTS[0],
        hy=TABLE_HALF_EXTENTS[1],
        hz=TABLE_HALF_EXTENTS[2],
        cfg=table_cfg,
        color=(0.48, 0.38, 0.26),
        label="/World/TableCollider",
    )

    bowl_vertices, bowl_indices = create_open_bowl_mesh(
        inner_bottom_radius=0.045,
        inner_top_radius=0.19,
        wall_thickness=0.008,
        height=BOWL_HEIGHT,
        bottom_thickness=0.012,
    )
    bowl_mesh = newton.Mesh(bowl_vertices, bowl_indices, compute_inertia=False, is_solid=False)
    builder.add_shape_mesh(
        body=-1,
        xform=wp.transform(wp.vec3(*BOWL_BASE_POS), wp.quat_identity()),
        mesh=bowl_mesh,
        cfg=newton.ModelBuilder.ShapeConfig(mu=args_cli.bowl_friction, margin=args_cli.collider_margin),
        color=(0.95, 0.82, 0.16),
        label="/World/BowlCollider",
    )

    cup_pos, cup_quat, _ = cup_pose_at_time(0.0)
    cup_body = builder.add_body(
        xform=wp.transform(wp.vec3(*cup_pos.tolist()), wp.quat(*cup_quat.tolist())),
        mass=0.0,
        label=CUP_BODY_PATH,
        lock_inertia=True,
        is_kinematic=True,
    )
    cup_vertices, cup_indices = create_open_cup_mesh(
        inner_radius=CUP_INNER_RADIUS,
        wall_thickness=CUP_WALL_THICKNESS,
        height=CUP_HEIGHT,
        bottom_thickness=CUP_BOTTOM_THICKNESS,
        include_bottom=True,
    )
    cup_mesh = newton.Mesh(cup_vertices, cup_indices, compute_inertia=False, is_solid=False)
    builder.add_shape_mesh(
        body=cup_body,
        mesh=cup_mesh,
        cfg=newton.ModelBuilder.ShapeConfig(mu=args_cli.cup_friction, margin=args_cli.collider_margin),
        color=(0.85, 0.88, 0.92),
        label="/World/CupPhysicsMesh",
    )

    builder.add_ground_plane(
        cfg=newton.ModelBuilder.ShapeConfig(mu=args_cli.table_friction, margin=args_cli.collider_margin),
        color=(0.30, 0.30, 0.30),
    )
    particle_range = emit_fluid_particles(builder)
    return builder, cup_body, particle_range


def _spawn_usd_visual(
    prim_path: str,
    usd_path: str,
    *,
    translation: tuple[float, float, float],
    orientation: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
    fallback=None,
) -> None:
    """Spawn a USD visual, falling back to a procedural primitive if loading fails."""

    stage = sim_utils.get_current_stage()
    if stage.GetPrimAtPath(prim_path).IsValid():
        return

    cfg = sim_utils.UsdFileCfg(usd_path=usd_path, scale=scale)
    try:
        cfg.func(prim_path, cfg, translation=translation, orientation=orientation)
    except Exception as exc:
        print(f"[WARN]: Could not spawn USD asset '{usd_path}' at '{prim_path}': {exc}")
        if fallback is not None:
            fallback.func(prim_path, fallback, translation=translation, orientation=orientation)


def setup_procedural_cup_visual(translation: np.ndarray, orientation: np.ndarray) -> None:
    """Create a simple flat-bottom cup visual that matches the Newton collider."""

    from pxr import Gf, Sdf, UsdGeom, Vt  # noqa: PLC0415

    stage = sim_utils.get_current_stage()
    cup_prim = stage.GetPrimAtPath(CUP_BODY_PATH)
    if not cup_prim.IsValid():
        cup_prim = UsdGeom.Xform.Define(stage, CUP_BODY_PATH).GetPrim()

    xformable = UsdGeom.Xformable(cup_prim)
    xformable.ClearXformOpOrder()
    xformable.SetResetXformStack(True)
    xform_op = xformable.AddTransformOp(UsdGeom.XformOp.PrecisionDouble, "newton_world")

    matrix = Gf.Matrix4d(1.0)
    matrix.SetRotate(
        Gf.Quatd(float(orientation[3]), Gf.Vec3d(float(orientation[0]), float(orientation[1]), float(orientation[2])))
    )
    matrix.SetTranslateOnly(Gf.Vec3d(float(translation[0]), float(translation[1]), float(translation[2])))
    xform_op.Set(matrix)

    mesh_path = f"{CUP_BODY_PATH}/FlatCupMesh"
    vertices, indices = create_open_cup_mesh(
        inner_radius=CUP_INNER_RADIUS,
        wall_thickness=CUP_WALL_THICKNESS,
        height=CUP_HEIGHT,
        bottom_thickness=CUP_BOTTOM_THICKNESS,
        include_bottom=True,
    )
    mesh = UsdGeom.Mesh.Define(stage, mesh_path)
    with Sdf.ChangeBlock():
        mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(vertices.astype(np.float32, copy=False)))
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * (len(indices) // 3)))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(indices.astype(np.int32, copy=False).tolist()))
        mesh.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(0.85, 0.88, 0.92)]))


def setup_kit_scene(sim: sim_utils.SimulationContext) -> None:
    """Create Kit USD visuals for tableware and lighting."""

    if "kit" not in sim.resolve_visualizer_types():
        return

    assets = resolve_asset_paths()
    initial_pos, initial_quat, _ = cup_pose_at_time(0.0)

    table_fallback = sim_utils.CuboidCfg(
        size=(2.0 * TABLE_HALF_EXTENTS[0], 2.0 * TABLE_HALF_EXTENTS[1], 2.0 * TABLE_HALF_EXTENTS[2]),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.48, 0.38, 0.26)),
    )
    _spawn_usd_visual(
        TABLE_PATH,
        assets.table,
        translation=(0.0, 0.0, args_cli.table_visual_z),
        scale=(1.0, 1.0, 0.75),
        fallback=table_fallback,
    )

    bowl_scale = (args_cli.asset_scale, args_cli.asset_scale, args_cli.asset_scale)
    bowl_fallback = sim_utils.CylinderCfg(
        radius=0.19,
        height=0.12,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.82, 0.16)),
    )
    _spawn_usd_visual(
        BOWL_PATH,
        assets.bowl,
        translation=BOWL_BASE_POS,
        scale=bowl_scale,
        fallback=bowl_fallback,
    )

    if args_cli.cup_visual == "procedural":
        setup_procedural_cup_visual(initial_pos, initial_quat)
    else:
        cup_scale = (args_cli.asset_scale, args_cli.asset_scale, args_cli.asset_scale)
        cup_fallback = sim_utils.CylinderCfg(
            radius=CUP_INNER_RADIUS + CUP_WALL_THICKNESS,
            height=CUP_HEIGHT,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.88, 0.92)),
        )
        _spawn_usd_visual(
            CUP_BODY_PATH,
            assets.cup,
            translation=tuple(float(v) for v in initial_pos),
            orientation=tuple(float(v) for v in initial_quat),
            scale=cup_scale,
            fallback=cup_fallback,
        )

    stage = sim_utils.get_current_stage()
    if not stage.GetPrimAtPath(GROUND_PATH).IsValid():
        ground_cfg = sim_utils.GroundPlaneCfg(size=(4.0, 4.0), color=(0.30, 0.30, 0.30))
        ground_cfg.func(GROUND_PATH, ground_cfg)
    if not stage.GetPrimAtPath("/World/DomeLight").IsValid():
        light_cfg = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.78, 0.78, 0.78))
        light_cfg.func("/World/DomeLight", light_cfg)


class KitBodyVisual:
    """Fallback USD transform synchronizer for a single Newton body."""

    def __init__(self, prim_path: str, body_id: int):
        from pxr import UsdGeom  # noqa: PLC0415

        stage = sim_utils.get_current_stage()
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise RuntimeError(f"Kit visual prim does not exist: {prim_path}")
        xformable = UsdGeom.Xformable(prim)
        xformable.ClearXformOpOrder()
        xformable.SetResetXformStack(True)
        self._xform_op = xformable.AddTransformOp(UsdGeom.XformOp.PrecisionDouble, "newton_world")
        self._body_id = body_id

    def update(self, state: newton.State) -> None:
        """Write the Newton body pose into the Kit USD Xform."""

        from pxr import Gf, Sdf  # noqa: PLC0415

        transform = state.body_q.numpy()[self._body_id]
        pos = transform[:3]
        quat = transform[3:7]
        matrix = Gf.Matrix4d(1.0)
        matrix.SetRotate(Gf.Quatd(float(quat[3]), Gf.Vec3d(float(quat[0]), float(quat[1]), float(quat[2]))))
        matrix.SetTranslateOnly(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))
        with Sdf.ChangeBlock():
            self._xform_op.Set(matrix)


class KitParticlePoints:
    """USD ``Points`` helper for visualizing MPM particles in Kit."""

    def __init__(self, prim_path: str, widths: np.ndarray):
        from pxr import Gf, UsdGeom, Vt  # noqa: PLC0415

        stage = sim_utils.get_current_stage()
        self._points = UsdGeom.Points.Define(stage, prim_path)
        self._widths_np = widths.astype(np.float32, copy=False)
        self._color = Gf.Vec3f(0.45, 0.82, 1.0)
        self._points_attr = self._points.GetPointsAttr()
        self._widths_attr = self._points.CreateWidthsAttr(Vt.FloatArray())
        self._color_attr = self._points.CreateDisplayColorAttr(Vt.Vec3fArray())
        self._particle_count = -1
        self._widths = Vt.FloatArray()
        self._colors = Vt.Vec3fArray()

    def update(self, positions: torch.Tensor) -> None:
        from pxr import Sdf, Vt  # noqa: PLC0415

        positions_np = positions.detach().cpu().numpy().astype(np.float32, copy=False)
        particle_count = int(positions_np.shape[0])
        with Sdf.ChangeBlock():
            self._points_attr.Set(Vt.Vec3fArray.FromNumpy(positions_np))
            if particle_count != self._particle_count:
                self._particle_count = particle_count
                self._widths = Vt.FloatArray(self._widths_np[:particle_count].tolist())
                self._colors = Vt.Vec3fArray([self._color] * particle_count)
                self._widths_attr.Set(self._widths)
                self._color_attr.Set(self._colors)


def create_kit_body_visual(sim: sim_utils.SimulationContext, body_id: int) -> KitBodyVisual | None:
    """Create the Kit-side cup transform sync helper."""

    if "kit" not in sim.resolve_visualizer_types():
        return None
    return KitBodyVisual(CUP_BODY_PATH, body_id)


def create_particle_points(sim: sim_utils.SimulationContext, model: newton.Model) -> KitParticlePoints | None:
    """Create Kit points for MPM particles when Kit visualization is active."""

    if "kit" not in sim.resolve_visualizer_types():
        return None
    if args_cli.kit_particle_stride < 1:
        raise ValueError("--kit-particle-stride must be >= 1.")

    particle_radius = wp.to_torch(model.particle_radius)
    rendered_radius = particle_radius if args_cli.kit_particle_stride == 1 else particle_radius[:: args_cli.kit_particle_stride]
    widths = 2.0 * rendered_radius.detach().cpu().numpy().astype(np.float32, copy=False)
    sim_utils.create_prim(VISUALS_PATH, "Xform")
    return KitParticlePoints(f"{VISUALS_PATH}/FluidParticles", widths=widths)


def update_particle_points(points: KitParticlePoints | None, state: newton.State) -> None:
    """Push particle positions into the Kit USD points cloud."""

    if points is None:
        return
    particle_q = wp.to_torch(state.particle_q)
    if args_cli.kit_particle_stride == 1:
        points.update(particle_q)
    else:
        points.update(particle_q[:: args_cli.kit_particle_stride])


def update_kit_body_visual(body_visual: KitBodyVisual | None, state: newton.State) -> None:
    """Push the cup pose into the Kit USD visual."""

    if body_visual is not None:
        body_visual.update(state)


class NewtonGrainRenderer:
    """High-resolution grain renderer for Newton MPM particles."""

    def __init__(self, model: newton.Model, state: newton.State):
        self._solver = getattr(NewtonManager, "_solver", None)
        self._enabled = (
            self._solver is not None
            and args_cli.grains_per_particle > 0
            and hasattr(self._solver, "sample_render_grains")
        )
        if not self._enabled:
            self._grains = None
            self._grain_radii = None
            self._grain_colors = None
            self._grain_offsets = None
            return

        self._grains = self._solver.sample_render_grains(state, args_cli.grains_per_particle)
        grain_positions = wp.to_torch(self._grains)
        particle_positions = wp.to_torch(state.particle_q)
        self._grain_offsets = (grain_positions - particle_positions[:, None, :]).detach().clone()
        grain_radius = args_cli.grain_radius_scale * args_cli.voxel_size / (3.0 * args_cli.grains_per_particle)
        self._grain_radii = wp.full(self._grains.size, value=grain_radius, dtype=float, device=model.device)
        self._grain_colors = wp.full(
            self._grains.size,
            value=wp.vec3(0.45, 0.82, 1.0),
            dtype=wp.vec3,
            device=model.device,
        )

    @property
    def enabled(self) -> bool:
        """Whether grain rendering is active."""

        return self._enabled

    def capture_previous(self, state: newton.State) -> None:
        """Save the pre-step MPM state for grain advection."""

        del state

    def update_and_log(self, sim: sim_utils.SimulationContext, state: newton.State, dt: float) -> None:
        """Advect grains and log them to all active native Newton viewers."""

        del dt
        if not self._enabled:
            return
        with torch.no_grad():
            grain_positions = wp.to_torch(self._grains)
            particle_positions = wp.to_torch(state.particle_q)
            grain_positions.copy_(particle_positions[:, None, :] + self._grain_offsets)
        for visualizer in sim.visualizers:
            viewer = getattr(visualizer, "_viewer", None)
            if viewer is not None and hasattr(viewer, "log_points"):
                viewer.log_points(
                    "/fluid/grains",
                    points=self._grains.flatten(),
                    radii=self._grain_radii,
                    colors=self._grain_colors,
                    hidden=False,
                )


def configure_newton_viewer(sim: sim_utils.SimulationContext) -> None:
    """Enable particle/contact rendering in native Newton visualizers."""

    for visualizer in sim.visualizers:
        viewer = getattr(visualizer, "_viewer", None)
        if viewer is None:
            continue
        if hasattr(viewer, "show_particles"):
            viewer.show_particles = args_cli.grains_per_particle <= 0
        if hasattr(viewer, "show_contacts"):
            viewer.show_contacts = True


def _copy_body_transform(body_q_arr, body_id: int, pos: np.ndarray, quat: np.ndarray) -> None:
    """Copy one body transform into a Warp transform array."""

    body_q_np = body_q_arr.numpy()
    body_q_np[body_id, 0:3] = pos
    body_q_np[body_id, 3:7] = quat
    wp.copy(body_q_arr, wp.array(body_q_np, dtype=wp.transform, device=body_q_arr.device))


def _copy_body_velocity(body_qd_arr, body_id: int, qd: np.ndarray) -> None:
    """Copy one body velocity into a Warp spatial-vector array."""

    body_qd_np = body_qd_arr.numpy()
    body_qd_np[body_id, :] = qd
    wp.copy(body_qd_arr, wp.array(body_qd_np, dtype=wp.spatial_vector, device=body_qd_arr.device))


def write_cup_kinematic_state(cup_body: int, pos: np.ndarray, quat: np.ndarray, qd: np.ndarray) -> None:
    """Write scripted cup state into model and active Newton state buffers."""

    model = NewtonManager.get_model()
    state_0 = NewtonManager.get_state_0()
    state_1 = NewtonManager.get_state_1()
    for body_q_arr in (model.body_q, state_0.body_q, state_1.body_q):
        _copy_body_transform(body_q_arr, cup_body, pos, quat)
    for body_qd_arr in (model.body_qd, state_0.body_qd, state_1.body_qd):
        _copy_body_velocity(body_qd_arr, cup_body, qd)


def keep_running(sim: sim_utils.SimulationContext, count: int) -> bool:
    """Return whether the demo loop should continue."""

    if args_cli.max_steps >= 0 and count >= args_cli.max_steps:
        return False
    if not sim.visualizers:
        return True
    return any(not viz.is_closed and viz.is_running() for viz in sim.visualizers)


def log_progress(count: int, state: newton.State, cup_body: int) -> None:
    """Print a compact heartbeat for particle motion and cup pose."""

    if args_cli.log_interval <= 0 or count % args_cli.log_interval != 0:
        return
    particle_q = wp.to_torch(state.particle_q)
    cup_pos = wp.to_torch(state.body_q)[cup_body, 0:3].detach().cpu().numpy()
    fluid_min = particle_q.min(dim=0).values.detach().cpu().numpy()
    fluid_max = particle_q.max(dim=0).values.detach().cpu().numpy()
    print(
        "[INFO]: step "
        f"{count:06d} t={count / args_cli.fps:.2f}s "
        f"cup=({cup_pos[0]:.3f}, {cup_pos[1]:.3f}, {cup_pos[2]:.3f}) "
        f"fluid_x=[{fluid_min[0]:.3f}, {fluid_max[0]:.3f}] "
        f"fluid_z=[{fluid_min[2]:.3f}, {fluid_max[2]:.3f}]",
        flush=True,
    )


def run_simulator(
    sim: sim_utils.SimulationContext,
    cup_body: int,
    cup_visual: KitBodyVisual | None,
    particle_points: KitParticlePoints | None,
    grain_renderer: NewtonGrainRenderer,
) -> None:
    """Run the scripted cup pour simulation loop."""

    count = 0
    while keep_running(sim, count):
        sim_time = count / args_cli.fps
        pos, quat, qd = cup_pose_at_time(sim_time)
        write_cup_kinematic_state(cup_body, pos, quat, qd)
        grain_renderer.capture_previous(NewtonManager.get_state_0())

        sim.step(render=False)
        state = NewtonManager.get_state_0()
        grain_renderer.update_and_log(sim, state, dt=1.0 / args_cli.fps)
        log_progress(count, state, cup_body)

        if sim.is_rendering:
            update_kit_body_visual(cup_visual, state)
            update_particle_points(particle_points, state)
            sim.render()
        count += 1


def create_launcher_sim_cfg():
    """Create the minimal config used to decide whether Kit is required."""

    import isaaclab.sim as sim_utils_module
    from isaaclab_newton.physics import NewtonCfg as NewtonCfgClass

    device = str(args_cli.device)
    if not device.startswith("cuda"):
        raise RuntimeError("Newton implicit MPM requires a CUDA device.")
    return sim_utils_module.SimulationCfg(
        dt=1.0 / args_cli.fps,
        device=device,
        gravity=(0.0, 0.0, -9.81),
        physics=NewtonCfgClass(num_substeps=args_cli.sim_substeps, use_cuda_graph=not args_cli.disable_cuda_graph),
    )


def main() -> None:
    """Set up and run the Isaac Lab Newton cup-pour MPM demo."""

    sim_cfg = create_launcher_sim_cfg()

    with launch_simulation(SimpleNamespace(sim=sim_cfg), args_cli):
        import_runtime_dependencies()
        builder, cup_body, particle_range = build_cup_pour_model()
        sim_cfg.physics = NewtonCfg(
            solver_cfg=MPMSolverCfg(
                voxel_size=args_cli.voxel_size,
                grid_type="fixed",
                grid_padding=48,
                max_active_cell_count=1 << 16,
                strain_basis="P0",
                velocity_basis="Q1",
                collider_basis="S2",
                transfer_scheme="apic",
                max_iterations=args_cli.mpm_iterations,
                critical_fraction=0.0,
                air_drag=0.2,
                collider_velocity_mode="backward",
                solver="gauss-seidel",
            ),
            num_substeps=args_cli.sim_substeps,
            use_cuda_graph=not args_cli.disable_cuda_graph,
        )
        sim = sim_utils.SimulationContext(sim_cfg)
        try:
            sim.set_camera_view(eye=[1.45, -1.65, 1.65], target=[0.07, 0.0, 1.12])
            setup_kit_scene(sim)
            NewtonManager.set_builder(builder)
            sim.reset()
            configure_newton_viewer(sim)

            model = NewtonManager.get_model()
            state = NewtonManager.get_state_0()
            cup_visual = create_kit_body_visual(sim, cup_body)
            update_kit_body_visual(cup_visual, state)
            particle_points = create_particle_points(sim, model)
            update_particle_points(particle_points, state)
            grain_renderer = NewtonGrainRenderer(model, state)
            grain_renderer.update_and_log(sim, state, dt=1.0 / args_cli.fps)

            print("[INFO]: Isaac Lab Newton cup-pour MPM demo ready.")
            print(
                "[INFO]: "
                f"Spawned {particle_range[1] - particle_range[0]} MPM particles; "
                "the cup will tilt after the hold interval.",
                flush=True,
            )
            run_simulator(sim, cup_body, cup_visual, particle_points, grain_renderer)
        finally:
            sim.clear_instance()


if __name__ == "__main__":
    main()
