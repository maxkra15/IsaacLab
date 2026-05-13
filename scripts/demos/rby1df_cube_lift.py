# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Lazy-launch smoke/demo runner for the Newton MJWarp RBY1DF cube-lift task.

Examples:

.. code-block:: bash

    # Kitless Newton simulation.
    ./isaaclab.sh -p scripts/demos/rby1df_cube_lift.py

    # Native Newton visualizer, still without launching Kit.
    ./isaaclab.sh -p scripts/demos/rby1df_cube_lift.py --viz newton

    # Isaac Sim Kit viewport plus Newton visualizer.
    ./isaaclab.sh -p scripts/demos/rby1df_cube_lift.py --viz kit,newton
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import time
from types import SimpleNamespace

from isaaclab_tasks.utils import add_launcher_args, launch_simulation


parser = argparse.ArgumentParser(description="Run the direct Newton MJWarp RBY1DF cube smoke demo.")
parser.add_argument("--num-envs", type=int, default=1, help="Number of environments to simulate. Only 1 is supported.")
parser.add_argument("--steps", type=int, default=300, help="Number of policy steps to run; negative runs forever.")
parser.add_argument("--real-time", action="store_true", help="Throttle stepping to the environment timestep.")
add_launcher_args(parser)
args_cli = parser.parse_args()

np = torch = wp = newton = sim_utils = None
MJWarpSolverCfg = NewtonCfg = NewtonManager = SolverMuJoCo = None

RBY1DF_URDF_PATH = Path(__file__).resolve().parents[2] / "source/isaaclab_assets/data/Robots/RBY1DF/urdf/robot_edited.urdf"
ROBOT_BASE_POS = (0.0, 0.0, 0.0)
CUBE_SIZE = 0.06
CUBE_POS = (0.55, -0.25, 0.055)
GRIPPER_DRIVER_DOFS = {13, 23}
GRIPPER_FINGER_DOFS = {14, 15, 24, 25}
INITIAL_JOINT_Q = [
    0.048646115,
    -0.11358134,
    0.28509942,
    0.30236751,
    -0.043634601,
    0.009673167,
    -0.85306484,
    -1.0891527,
    0.66765565,
    -2.0121396,
    -1.0203781,
    1.5501461,
    0.56562239,
    0.08,
    -0.04,
    0.04,
    -0.70531148,
    1.0506693,
    -0.44851208,
    -1.9159117,
    1.0035634,
    1.5637023,
    -0.84481186,
    0.08,
    -0.04,
    0.04,
    0.0,
    0.0,
]


def requested_visualizers() -> set[str]:
    """Return visualizers requested through launcher args."""
    visualizers = args_cli.visualizer or []
    return {str(name).lower() for name in visualizers}


def import_direct_runtime_dependencies() -> None:
    """Import Newton/Isaac Lab modules only after the lazy launcher has made the Kit decision."""
    global np, torch, wp, newton, sim_utils
    global MJWarpSolverCfg, NewtonCfg, NewtonManager, SolverMuJoCo

    import numpy as np_module
    import torch as torch_module
    import warp as wp_module

    import newton as newton_module
    from newton.solvers import SolverMuJoCo as SolverMuJoCoClass

    import isaaclab.sim as sim_utils_module
    from isaaclab_newton.physics import MJWarpSolverCfg as MJWarpSolverCfgClass
    from isaaclab_newton.physics import NewtonCfg as NewtonCfgClass
    from isaaclab_newton.physics import NewtonManager as NewtonManagerClass

    np = np_module
    torch = torch_module
    wp = wp_module
    newton = newton_module
    sim_utils = sim_utils_module
    MJWarpSolverCfg = MJWarpSolverCfgClass
    NewtonCfg = NewtonCfgClass
    NewtonManager = NewtonManagerClass
    SolverMuJoCo = SolverMuJoCoClass


def direct_sim_cfg():
    """Create a minimal Newton MJWarp sim config for the direct, kitless path."""
    import isaaclab.sim as sim_utils_module
    from isaaclab_newton.physics import MJWarpSolverCfg as MJWarpSolverCfgClass
    from isaaclab_newton.physics import NewtonCfg as NewtonCfgClass

    visualizer_cfgs = []
    if "newton" in requested_visualizers():
        from isaaclab_visualizers.newton import NewtonVisualizerCfg

        visualizer_cfgs.append(NewtonVisualizerCfg(headless=os.environ.get("DISPLAY") is None))

    return sim_utils_module.SimulationCfg(
        dt=0.01,
        render_interval=2,
        device=str(args_cli.device),
        gravity=(0.0, 0.0, -9.81),
        visualizer_cfgs=visualizer_cfgs,
        physics=NewtonCfgClass(
            solver_cfg=MJWarpSolverCfgClass(
                solver="newton",
                integrator="implicitfast",
                njmax=300,
                nconmax=120,
                impratio=10.0,
                cone="elliptic",
                update_data_interval=2,
                iterations=100,
                ls_iterations=15,
                use_mujoco_contacts=True,
                ccd_iterations=5000,
            ),
            num_substeps=2,
        ),
    )


def solid_box_inertia(mass: float, side_length: float):
    """Return the COM inertia tensor for a uniform cube."""
    diagonal = (1.0 / 6.0) * mass * side_length * side_length
    return wp.mat33(diagonal, 0.0, 0.0, 0.0, diagonal, 0.0, 0.0, 0.0, diagonal)


def patch_dummy_body_inertias(builder) -> None:
    """Patch Newton builder state for zero-mass gripper dummy bodies without editing the URDF."""
    for body_id, label in enumerate(builder.body_label):
        if label.endswith("gripper_dummy") and builder.body_mass[body_id] == 0.0:
            builder.body_mass[body_id] = 1.0e-6
            builder.body_inv_mass[body_id] = 1.0e6
            builder.body_inertia[body_id] = wp.mat33(np.eye(3, dtype=np.float32) * 1.0e-10)
            builder.body_inv_inertia[body_id] = wp.inverse(builder.body_inertia[body_id])


def configure_robot_dofs(builder) -> None:
    """Apply stable MJWarp drive gains to the imported RBY1DF robot."""
    for dof in range(builder.joint_dof_count):
        if dof in GRIPPER_DRIVER_DOFS or dof in GRIPPER_FINGER_DOFS:
            continue
        builder.joint_target_ke[dof] = 120000.0
        builder.joint_target_kd[dof] = 12000.0
        builder.joint_effort_limit[dof] = 10000.0
        builder.joint_armature[dof] = 0.2

    for dof in GRIPPER_DRIVER_DOFS:
        builder.joint_target_ke[dof] = 10000.0
        builder.joint_target_kd[dof] = 1000.0
        builder.joint_effort_limit[dof] = 100000.0
        builder.joint_armature[dof] = 0.5

    for dof in GRIPPER_FINGER_DOFS:
        builder.joint_target_ke[dof] = 500000.0
        builder.joint_target_kd[dof] = 10000.0
        builder.joint_effort_limit[dof] = 500000.0
        builder.joint_armature[dof] = 0.5

    builder.joint_q = INITIAL_JOINT_Q


def build_direct_newton_model():
    """Build RBY1DF, cube, and ground directly through Newton, bypassing Kit's URDF converter."""
    builder = NewtonManager.create_builder()
    SolverMuJoCo.register_custom_attributes(builder)
    builder.default_shape_cfg.mu = 0.8
    builder.default_joint_cfg = newton.ModelBuilder.JointDofConfig(armature=0.1, limit_ke=1.0e3, limit_kd=1.0e1)

    builder.begin_world(label="env_0")
    builder.add_urdf(
        str(RBY1DF_URDF_PATH),
        xform=wp.transform(wp.vec3(*ROBOT_BASE_POS), wp.quat_identity()),
        floating=False,
        enable_self_collisions=False,
        parse_visuals_as_colliders=False,
        ignore_inertial_definitions=True,
    )
    patch_dummy_body_inertias(builder)
    configure_robot_dofs(builder)

    cube_mass = 0.08
    cube_body = builder.add_body(
        xform=wp.transform(wp.vec3(*CUBE_POS), wp.quat_identity()),
        mass=cube_mass,
        inertia=solid_box_inertia(cube_mass, CUBE_SIZE),
        label="/World/envs/env_0/Object",
    )
    builder.add_shape_box(
        cube_body,
        hx=0.5 * CUBE_SIZE,
        hy=0.5 * CUBE_SIZE,
        hz=0.5 * CUBE_SIZE,
        cfg=newton.ModelBuilder.ShapeConfig(density=0.0, mu=0.8),
        color=(0.8, 0.1, 0.1),
    )
    builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.9), color=(0.4, 0.4, 0.4))
    builder.end_world()
    return builder


def configure_newton_viewer(sim) -> None:
    """Enable useful overlays in active Newton visualizers."""
    for visualizer in sim.visualizers:
        viewer = getattr(visualizer, "_viewer", None)
        if viewer is None:
            continue
        if hasattr(viewer, "show_contacts"):
            viewer.show_contacts = True


def direct_keep_running(sim, step_count: int) -> bool:
    """Return whether the direct Newton run should continue."""
    if args_cli.steps >= 0 and step_count >= args_cli.steps:
        return False
    if not sim.visualizers:
        return True
    return any(not viz.is_closed and viz.is_running() for viz in sim.visualizers)


def run_direct_newton() -> None:
    """Run the kitless/direct Newton RBY1DF smoke path."""
    if args_cli.num_envs != 1:
        raise ValueError("The direct Newton RBY1DF demo currently supports only --num-envs 1.")
    sim_cfg = direct_sim_cfg()
    with launch_simulation(SimpleNamespace(sim=sim_cfg), args_cli):
        import_direct_runtime_dependencies()
        builder = build_direct_newton_model()
        sim = sim_utils.SimulationContext(sim_cfg)
        try:
            NewtonManager.set_builder(builder)
            sim.reset()
            configure_newton_viewer(sim)
            control = NewtonManager.get_control()
            wp.copy(
                control.joint_target_pos,
                NewtonManager.get_state_0().joint_q,
                count=control.joint_target_pos.shape[0],
            )

            step_count = 0
            sleep_dt = sim_cfg.dt if args_cli.real_time else 0.0
            while direct_keep_running(sim, step_count):
                start_time = time.perf_counter()
                sim.step(render=False)
                if sim.is_rendering:
                    sim.render()
                step_count += 1
                if sleep_dt > 0.0:
                    time.sleep(max(0.0, sleep_dt - (time.perf_counter() - start_time)))
        finally:
            sim.clear_instance()


def main() -> None:
    """Run the direct Newton RBY1DF cube smoke demo."""
    run_direct_newton()


if __name__ == "__main__":
    main()
