# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run ANYmal-C walking over implicit MPM sand through Isaac Lab.

This mirrors Newton's ``mpm_anymal`` example, but the application, simulation
context, visualizer selection, and Newton manager lifecycle are owned by Isaac
Lab.  The robot is advanced by Isaac Lab's Newton MJWarp manager, while a
``SolverImplicitMPM`` instance is attached to the same Newton model to update
the sand after each robot step.

.. code-block:: bash

    ./isaaclab.sh -p scripts/demos/newton_anymal_mpm_sand.py --viz kit,newton

The demo spawns Isaac Lab's ANYmal-C USD for Kit visuals and drives those prims
from Newton body transforms.  MPM particles are shown in Kit as a USD
``Points`` cloud, and the Newton visualizer can render the native Newton model
and particles directly.
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="ANYmal-C walking over Newton implicit MPM sand.")
parser.add_argument("--voxel-size", type=float, default=0.08, help="MPM grid voxel size in meters.")
parser.add_argument("--particles-per-cell", type=float, default=1.0, help="Sand particles per grid cell.")
parser.add_argument("--grid-type", choices=["sparse", "dense", "fixed"], default="sparse", help="MPM grid type.")
parser.add_argument("--tolerance", type=float, default=1.0e-6, help="MPM rheology solver tolerance.")
parser.add_argument("--mpm-iterations", type=int, default=50, help="Maximum MPM rheology iterations.")
parser.add_argument("--robot-substeps", type=int, default=4, help="MJWarp substeps per policy/control frame.")
parser.add_argument("--fps", type=float, default=50.0, help="Control/render frames per second.")
parser.add_argument("--command-forward", type=float, default=1.0, help="Commanded forward velocity.")
parser.add_argument("--command-lateral", type=float, default=0.0, help="Commanded lateral velocity.")
parser.add_argument("--command-yaw", type=float, default=0.0, help="Commanded yaw velocity.")
parser.add_argument("--max-steps", type=int, default=-1, help="Stop after this many frames; negative runs forever.")
parser.add_argument("--policy", type=str, default=None, help="Optional path to a TorchScript ANYmal policy.")
parser.add_argument("--disable-cuda-graph", action="store_true", help="Disable MJWarp CUDA graph capture.")
parser.add_argument("--kit-sand-stride", type=int, default=1, help="Render every Nth sand particle in Kit.")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(visualizer=["kit", "newton"])
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import warnings

import numpy as np
import torch
import warp as wp

import newton
import newton.utils
from newton.solvers import SolverImplicitMPM, SolverMuJoCo

import isaaclab.sim as sim_utils
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonManager

from isaaclab_assets.robots.anymal import ANYMAL_C_CFG  # isort:skip


LAB_TO_MUJOCO = [0, 6, 3, 9, 1, 7, 4, 10, 2, 8, 5, 11]
MUJOCO_TO_LAB = [0, 4, 8, 2, 6, 10, 1, 5, 9, 3, 7, 11]


def quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vector ``v`` by inverse quaternion ``q``. Quaternions use XYZW order."""
    q_w = q[..., 3]
    q_vec = q[..., :3]
    a = v * (2.0 * q_w**2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * torch.bmm(q_vec.view(q.shape[0], 1, 3), v.view(q.shape[0], 3, 1)).squeeze(-1) * 2.0
    return a - b + c


def compute_obs(
    actions: torch.Tensor,
    state: newton.State,
    joint_pos_initial: torch.Tensor,
    indices: torch.Tensor,
    gravity_vec: torch.Tensor,
    command: torch.Tensor,
) -> torch.Tensor:
    """Build the observation vector expected by the Newton ANYmal walking policy."""
    joint_q = wp.to_torch(state.joint_q)
    joint_qd = wp.to_torch(state.joint_qd)
    root_quat_w = joint_q[3:7].unsqueeze(0)
    root_lin_vel_w = joint_qd[:3].unsqueeze(0)
    root_ang_vel_w = joint_qd[3:6].unsqueeze(0)
    joint_pos_current = joint_q[7:].unsqueeze(0)
    joint_vel_current = joint_qd[6:].unsqueeze(0)

    vel_b = quat_rotate_inverse(root_quat_w, root_lin_vel_w)
    ang_vel_b = quat_rotate_inverse(root_quat_w, root_ang_vel_w)
    grav_b = quat_rotate_inverse(root_quat_w, gravity_vec)
    joint_pos_rel = torch.index_select(joint_pos_current - joint_pos_initial, 1, indices)
    joint_vel_rel = torch.index_select(joint_vel_current, 1, indices)
    return torch.cat([vel_b, ang_vel_b, grav_b, command, joint_pos_rel, joint_vel_rel, actions], dim=1)


def spawn_sand(builder: newton.ModelBuilder, voxel_size: float, particles_per_cell: float) -> None:
    """Add a shallow sand bed in front of ANYmal."""
    density = 2500.0
    particle_lo = np.array([-0.5, -0.5, 0.0])
    particle_hi = np.array([0.5, 2.5, 0.15])
    particle_res = np.array(np.ceil(particles_per_cell * (particle_hi - particle_lo) / voxel_size), dtype=int)
    cell_size = (particle_hi - particle_lo) / particle_res
    cell_volume = float(np.prod(cell_size))
    radius = float(np.max(cell_size) * 0.5)
    mass = float(cell_volume * density)

    builder.add_particle_grid(
        pos=wp.vec3(particle_lo),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0),
        dim_x=int(particle_res[0]) + 1,
        dim_y=int(particle_res[1]) + 1,
        dim_z=int(particle_res[2]) + 1,
        cell_x=float(cell_size[0]),
        cell_y=float(cell_size[1]),
        cell_z=float(cell_size[2]),
        mass=mass,
        jitter=2.0 * radius,
        radius_mean=radius,
    )


def create_kit_body_prims(builder: newton.ModelBuilder) -> None:
    """Spawn ANYmal-C USD visuals and relabel Newton bodies to valid USD paths."""
    from pxr import UsdGeom  # noqa: PLC0415

    ANYMAL_C_CFG.spawn.func(
        "/World/Robot",
        ANYMAL_C_CFG.spawn,
        translation=(0.0, 0.0, 0.62),
        orientation=(0.0, 0.0, 0.70710678, 0.70710678),
    )
    stage = sim_utils.get_current_stage()

    for body_id, body_label in enumerate(builder.body_label):
        body_name = body_label.rsplit("/", 1)[-1]
        body_prim_path = f"/World/Robot/{body_name}"
        if not stage.GetPrimAtPath(body_prim_path).IsValid():
            UsdGeom.Xform.Define(stage, body_prim_path)
        builder.body_label[body_id] = body_prim_path


def build_anymal_sand_model() -> str:
    """Populate ``NewtonManager`` with ANYmal-C, ground, and MPM particles."""
    builder = NewtonManager.create_builder()
    SolverMuJoCo.register_custom_attributes(builder)
    SolverImplicitMPM.register_custom_attributes(builder)

    builder.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
        armature=0.06,
        limit_ke=1.0e3,
        limit_kd=1.0e1,
    )
    builder.default_shape_cfg.ke = 5.0e4
    builder.default_shape_cfg.kd = 5.0e2
    builder.default_shape_cfg.kf = 1.0e3
    builder.default_shape_cfg.mu = 0.75

    asset_path = newton.utils.download_asset("anybotics_anymal_c")
    urdf_path = str(asset_path / "urdf" / "anymal.urdf")
    builder.add_urdf(
        urdf_path,
        xform=wp.transform(
            wp.vec3(0.0, 0.0, 0.62),
            wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), wp.pi * 0.5),
        ),
        floating=True,
        enable_self_collisions=False,
        collapse_fixed_joints=True,
        ignore_inertial_definitions=False,
    )
    create_kit_body_prims(builder)

    # Only the shank collision shapes interact with particles, matching Newton's
    # reference ANYmal-MPM setup.
    for body_id, body_label in enumerate(builder.body_label):
        if "SHANK" not in body_label:
            for shape_id in builder.body_shapes[body_id]:
                builder.shape_flags[shape_id] &= ~newton.ShapeFlags.COLLIDE_PARTICLES

    builder.add_ground_plane()

    initial_q = {
        "RH_HAA": 0.0,
        "RH_HFE": -0.4,
        "RH_KFE": 0.8,
        "LH_HAA": 0.0,
        "LH_HFE": -0.4,
        "LH_KFE": 0.8,
        "RF_HAA": 0.0,
        "RF_HFE": 0.4,
        "RF_KFE": -0.8,
        "LF_HAA": 0.0,
        "LF_HFE": 0.4,
        "LF_KFE": -0.8,
    }
    for name, value in initial_q.items():
        joint_index = next(i for i, label in enumerate(builder.joint_label) if label.endswith(f"/{name}"))
        builder.joint_q[joint_index + 6] = value

    for joint_dof_index in range(builder.joint_dof_count):
        builder.joint_target_ke[joint_dof_index] = 150.0
        builder.joint_target_kd[joint_dof_index] = 5.0

    spawn_sand(builder, args_cli.voxel_size, args_cli.particles_per_cell)
    NewtonManager.set_builder(builder)
    return str(asset_path / "rl_policies" / "anymal_walking_policy_physx.pt")


def create_mpm_solver(model: newton.Model, state: newton.State) -> SolverImplicitMPM:
    """Create and configure the one-way coupled sand solver."""
    mpm_cfg = SolverImplicitMPM.Config()
    mpm_cfg.voxel_size = args_cli.voxel_size
    mpm_cfg.tolerance = args_cli.tolerance
    mpm_cfg.transfer_scheme = "pic"
    mpm_cfg.grid_type = args_cli.grid_type
    mpm_cfg.grid_padding = 50 if args_cli.grid_type == "fixed" else 0
    mpm_cfg.max_active_cell_count = 1 << 15 if args_cli.grid_type == "fixed" else -1
    mpm_cfg.strain_basis = "P0"
    mpm_cfg.max_iterations = args_cli.mpm_iterations
    mpm_cfg.critical_fraction = 0.0
    mpm_cfg.air_drag = 1.0
    mpm_cfg.collider_velocity_mode = "backward"
    # The installed Newton 1.2.0.dev0 package expects the long solver name.
    mpm_cfg.solver = "gauss-seidel"

    mpm_solver = SolverImplicitMPM(model, mpm_cfg)
    mpm_solver.setup_collider(body_mass=wp.zeros_like(model.body_mass), body_q=state.body_q)
    return mpm_solver


class KitSandPoints:
    """Small USD ``Points`` helper for visualizing MPM particles in Kit."""

    def __init__(self, prim_path: str, radius: float):
        from pxr import Gf, UsdGeom, Vt  # noqa: PLC0415

        stage = sim_utils.get_current_stage()
        self._points = UsdGeom.Points.Define(stage, prim_path)
        self._radius = float(radius)
        self._color = Gf.Vec3f(0.72, 0.60, 0.38)
        self._points.CreateWidthsAttr(Vt.FloatArray())
        self._points.CreateDisplayColorAttr(Vt.Vec3fArray())

    def update(self, positions: torch.Tensor) -> None:
        from pxr import Vt  # noqa: PLC0415

        positions_np = positions.detach().cpu().numpy().astype(np.float32, copy=False)
        self._points.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(positions_np))
        # RTX treats point widths as vertex data; author one width per point.
        self._points.GetWidthsAttr().Set(Vt.FloatArray([self._radius] * positions_np.shape[0]))
        self._points.GetDisplayColorAttr().Set(Vt.Vec3fArray([self._color] * positions_np.shape[0]))


def create_sand_points(sim: sim_utils.SimulationContext) -> KitSandPoints | None:
    """Create Kit points for the MPM particles when Kit visualization is active."""
    if "kit" not in sim.resolve_visualizer_types():
        return None
    if args_cli.kit_sand_stride < 1:
        raise ValueError("--kit-sand-stride must be >= 1.")
    sim_utils.create_prim("/World/Visuals", "Xform")
    return KitSandPoints("/World/Visuals/SandParticles", radius=args_cli.voxel_size * 0.25)


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


def load_policy(default_policy_path: str, device: torch.device) -> torch.jit.ScriptModule:
    """Load the walking policy used by Newton's ANYmal example."""
    policy_path = args_cli.policy or default_policy_path
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"`torch\.jit\.load` is deprecated\. Please switch to `torch\.export`\.",
            category=DeprecationWarning,
        )
        return torch.jit.load(policy_path, map_location=device)


def apply_policy(
    policy: torch.jit.ScriptModule,
    control: newton.Control,
    state: newton.State,
    joint_pos_initial: torch.Tensor,
    action: torch.Tensor,
    lab_to_mujoco_indices: torch.Tensor,
    mujoco_to_lab_indices: torch.Tensor,
    gravity_vec: torch.Tensor,
    command: torch.Tensor,
) -> torch.Tensor:
    """Run the walking policy and write joint targets into Newton control."""
    obs = compute_obs(action, state, joint_pos_initial, lab_to_mujoco_indices, gravity_vec, command)
    with torch.no_grad():
        action = policy(obs)
        rearranged_action = torch.gather(action, 1, mujoco_to_lab_indices.unsqueeze(0))
        target = joint_pos_initial + 0.5 * rearranged_action
        target_with_free_joint = torch.cat(
            [torch.zeros(6, device=target.device, dtype=torch.float32), target.squeeze(0)]
        )
        wp.copy(control.joint_target_pos, wp.from_torch(target_with_free_joint, dtype=wp.float32, requires_grad=False))
    return action


def run_simulator(
    sim: sim_utils.SimulationContext,
    mpm_solver: SolverImplicitMPM,
    policy: torch.jit.ScriptModule,
    sand_points: KitSandPoints | None,
) -> None:
    """Run the coupled robot/sand simulation loop."""
    model = NewtonManager.get_model()
    state = NewtonManager.get_state_0()
    control = NewtonManager.get_control()
    torch_device = wp.device_to_torch(model.device)

    joint_pos_initial = wp.to_torch(state.joint_q)[7:].unsqueeze(0).detach().clone()
    action = torch.zeros(1, 12, device=torch_device, dtype=torch.float32)
    lab_to_mujoco_indices = torch.tensor(LAB_TO_MUJOCO, device=torch_device)
    mujoco_to_lab_indices = torch.tensor(MUJOCO_TO_LAB, device=torch_device)
    gravity_vec = torch.tensor([[0.0, 0.0, -1.0]], device=torch_device, dtype=torch.float32)
    command = torch.tensor(
        [[args_cli.command_forward, args_cli.command_lateral, args_cli.command_yaw]],
        device=torch_device,
        dtype=torch.float32,
    )

    count = 0
    frame_dt = 1.0 / args_cli.fps
    while simulation_app.is_running():
        if args_cli.max_steps >= 0 and count >= args_cli.max_steps:
            break

        state = NewtonManager.get_state_0()
        action = apply_policy(
            policy,
            control,
            state,
            joint_pos_initial,
            action,
            lab_to_mujoco_indices,
            mujoco_to_lab_indices,
            gravity_vec,
            command,
        )

        sim.step(render=False)
        state = NewtonManager.get_state_0()
        mpm_solver.step(state, state, control=None, contacts=None, dt=frame_dt)
        update_sand_points(sand_points, state)
        sim.render()
        count += 1


def main() -> None:
    """Set up and run the Isaac Lab ANYmal-on-sand demo."""
    if not str(args_cli.device).startswith("cuda"):
        raise RuntimeError("Newton implicit MPM ANYmal demo requires a CUDA device.")

    frame_dt = 1.0 / args_cli.fps
    sim_cfg = sim_utils.SimulationCfg(
        dt=frame_dt,
        device=args_cli.device,
        gravity=(0.0, 0.0, -9.81),
        physics=NewtonCfg(
            solver_cfg=MJWarpSolverCfg(
                njmax=50,
                nconmax=100,
                ls_iterations=50,
                use_mujoco_contacts=True,
            ),
            num_substeps=args_cli.robot_substeps,
            use_cuda_graph=not args_cli.disable_cuda_graph,
        ),
    )
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[4.5, -4.5, 2.2], target=[0.0, 0.8, 0.4])

    default_policy_path = build_anymal_sand_model()
    sim.reset()
    enable_newton_particle_visualization(sim)

    model = NewtonManager.get_model()
    state = NewtonManager.get_state_0()
    mpm_solver = create_mpm_solver(model, state)
    sand_points = create_sand_points(sim)
    update_sand_points(sand_points, state)
    policy = load_policy(default_policy_path, wp.device_to_torch(model.device))

    print("[INFO]: Isaac Lab Newton ANYmal MPM sand demo ready.")
    print("[INFO]: Use --command-forward/--command-lateral/--command-yaw to change the fixed velocity command.")
    run_simulator(sim, mpm_solver, policy, sand_points)


if __name__ == "__main__":
    main()
    simulation_app.close()
