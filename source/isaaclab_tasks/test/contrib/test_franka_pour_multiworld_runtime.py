# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Headless runtime integration for isolated multi-world Franka Pour MPM."""

from __future__ import annotations

import inspect
import os
import re
from unittest import mock

import pytest

_RUNTIME_UNAVAILABLE_REASON = "Isaac Sim runtime is unavailable because EXP_PATH is not set."
_RUNTIME_AVAILABLE = bool(os.environ.get("EXP_PATH"))

if _RUNTIME_AVAILABLE:
    from isaaclab.app import AppLauncher

    # Launch Kit before importing simulation-dependent modules.
    app_launcher = AppLauncher(headless=True)
    simulation_app = app_launcher.app

    import gymnasium as gym
    import newton
    import numpy as np
    import torch
    import warp as wp
    import warp.fem as fem
    from isaaclab_newton.physics import NewtonCoupledManager, NewtonManager

    import isaaclab.sim as sim_utils

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.contrib.franka_pour.pour_env_cfg import MPM_ENTRY, RIGID_ENTRY
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

pytestmark = [pytest.mark.isaacsim_ci, pytest.mark.newton_ci]

_TASK_ID = "Isaac-Pour-Franka-v0"
_MPM_HISTORY_FIELDS = (
    "particle_elastic_strain",
    "particle_transform",
    "particle_qd_grad",
    "particle_stress",
    "particle_Jp",
)
_CONSTITUTIVE_HISTORY_FIELDS = (
    "particle_elastic_strain",
    "particle_transform",
    "particle_stress",
    "particle_Jp",
)


def _require_sparse_capture_stack() -> None:
    """Skip only when an explicit CUDA/rebuildable-Warp capability is absent."""
    if not wp.is_cuda_available():
        pytest.skip("Franka Pour multi-world runtime integration requires a CUDA device.")

    device = wp.get_device("cuda:0")
    if not wp.is_mempool_enabled(device):
        pytest.skip("Sparse MPM CUDA graph capture requires the Warp CUDA memory pool.")
    with wp.ScopedDevice(device):
        if not wp.is_conditional_graph_supported():
            pytest.skip("Sparse MPM CUDA graph capture requires Warp conditional-graph support.")

    try:
        volume_allocate = inspect.signature(wp.Volume.allocate_by_voxels).parameters
        volume_rebuild = inspect.signature(wp.Volume.rebuild).parameters
        nanogrid_init = inspect.signature(fem.Nanogrid).parameters
        nanogrid_rebuild = inspect.signature(fem.Nanogrid.rebuild).parameters
    except (AttributeError, TypeError, ValueError):
        pytest.skip("Installed Warp lacks the rebuildable Volume/Nanogrid API required by sparse MPM capture.")
    rebuildable_api = (
        {"rebuildable", "status", "point_mask"} <= volume_allocate.keys()
        and {"status", "point_mask"} <= volume_rebuild.keys()
        and "rebuildable" in nanogrid_init
        and {"status", "point_mask"} <= nanogrid_rebuild.keys()
    )
    if not rebuildable_api:
        pytest.skip("Installed Warp lacks the rebuildable Volume/Nanogrid API required by sparse MPM capture.")
    if not getattr(fem.Nanogrid, "REBUILDABLE_EDGE_TOPOLOGY", False):
        pytest.skip("Installed Warp lacks rebuildable NanoVDB edge topology required by MPM colliders.")


def _particle_snapshot(state, particle_slice: slice) -> dict[str, np.ndarray]:
    """Copy core and implicit-MPM particle state for one contiguous world."""
    arrays = {
        "particle_q": state.particle_q,
        "particle_qd": state.particle_qd,
        **{name: getattr(state.mpm, name) for name in _MPM_HISTORY_FIELDS},
    }
    return {name: np.ascontiguousarray(array.numpy()[particle_slice]).copy() for name, array in arrays.items()}


def _assert_snapshot_bitwise_equal(actual: dict[str, np.ndarray], expected: dict[str, np.ndarray]) -> None:
    """Assert exact storage equality, including floating-point sign bits."""
    assert actual.keys() == expected.keys()
    for name in expected:
        assert actual[name].shape == expected[name].shape, name
        assert actual[name].dtype == expected[name].dtype, name
        np.testing.assert_array_equal(actual[name].view(np.uint8), expected[name].view(np.uint8), err_msg=name)


def _snapshot_changed(actual: dict[str, np.ndarray], expected: dict[str, np.ndarray]) -> bool:
    return any(not np.array_equal(actual[name].view(np.uint8), expected[name].view(np.uint8)) for name in expected)


def _assert_rollout_world_equivalent(
    actual: dict[str, np.ndarray], expected: dict[str, np.ndarray], *, context: str
) -> None:
    """Compare physical MPM outputs from independently constructed seeded rollouts."""
    np.testing.assert_allclose(
        actual["particle_q"],
        expected["particle_q"],
        rtol=1.0e-6,
        atol=5.0e-7,
        err_msg=f"{context}, field=particle_q",
    )
    np.testing.assert_allclose(
        actual["particle_qd"],
        expected["particle_qd"],
        rtol=5.0e-4,
        atol=3.0e-5,
        err_msg=f"{context}, field=particle_qd",
    )
    for name in _CONSTITUTIVE_HISTORY_FIELDS:
        np.testing.assert_array_equal(actual[name], expected[name], err_msg=f"{context}, field={name}")

    # Sparse-grid/FEM accumulation order makes particle_qd_grad vary even
    # between independently constructed eager, seeded rollouts. The two-step
    # q/qd comparison exercises its downstream effect; the single-environment
    # selective-reset test still checks this field bitwise.
    assert np.all(np.isfinite(actual["particle_qd_grad"])), context
    assert np.all(np.isfinite(expected["particle_qd_grad"])), context


def _assert_media_collider_ownership(model, media_view) -> None:
    """Check the entry's effective particle colliders and their Newton worlds."""
    assert media_view.parent is model
    particle_collision = int(newton.ShapeFlags.COLLIDE_PARTICLES)
    shape_flags = media_view.shape_flags.numpy()
    shape_bodies = media_view.shape_body.numpy()
    body_worlds = media_view.body_world.numpy()
    actual_by_world: dict[int, set[str]] = {0: set(), 1: set()}
    collider_count = 0

    for shape_id, flags in enumerate(shape_flags):
        if int(flags) & particle_collision == 0:
            continue
        collider_count += 1
        body_id = int(shape_bodies[shape_id])
        assert body_id >= 0, f"Media entry unexpectedly exposes global particle collider {shape_id}."
        world = int(body_worlds[body_id])
        label = str(media_view.body_label[body_id])
        match = re.search(r"/env_(\d+)(?:/|$)", label)
        assert match is not None, f"Particle collider body has no replicated-world label: {label!r}."
        assert world == int(match.group(1)), f"Collider {label!r} is assigned to Newton world {world}."
        assert world in actual_by_world, f"Collider {label!r} references unexpected world {world}."
        actual_by_world[world].add(label.rsplit("/", 1)[-1])

    expected = {"SourceCup", "TargetCup", "SpillFloor"}
    assert collider_count == 2 * len(expected)
    assert actual_by_world == {0: expected, 1: expected}


def _assert_scene_solver_roles(model) -> None:
    """Check exact per-world task bodies and solver-only collision roles."""
    body_world = model.body_world.numpy()
    body_mass = model.body_mass.numpy()
    body_inv_mass = model.body_inv_mass.numpy()
    body_flags = model.body_flags.numpy()
    shape_body = model.shape_body.numpy()
    shape_flags = model.shape_flags.numpy()
    collide_shapes = int(newton.ShapeFlags.COLLIDE_SHAPES)
    collide_particles = int(newton.ShapeFlags.COLLIDE_PARTICLES)
    visible = int(newton.ShapeFlags.VISIBLE)

    for world in range(2):
        bodies_by_name: dict[str, list[int]] = {}
        for body_id, label in enumerate(model.body_label):
            if int(body_world[body_id]) == world:
                bodies_by_name.setdefault(str(label).rsplit("/", 1)[-1], []).append(body_id)
        for name in ("SourceCup", "TargetCup", "TargetCupRigid", "SpillFloor"):
            assert len(bodies_by_name.get(name, [])) == 1, (world, name, bodies_by_name.get(name))

        target_body = bodies_by_name["TargetCup"][0]
        hidden_target_body = bodies_by_name["TargetCupRigid"][0]
        for body_id in (target_body, hidden_target_body):
            assert int(body_flags[body_id]) & int(newton.BodyFlags.KINEMATIC)
            assert float(body_mass[body_id]) == 0.0
            assert float(body_inv_mass[body_id]) == 0.0

        expected_shapes = {
            "SourceCup": ("/SourceCup/ParticleCollider", False, True, False),
            "TargetCup": ("/TargetCup/ParticleCollider", False, True, False),
            "TargetCupRigid": ("/TargetCupRigid/Collision", True, False, False),
            "SpillFloor": ("/SpillFloor/Collision", False, True, False),
        }
        for body_name, (suffix, rigid, particles, is_visible) in expected_shapes.items():
            body_id = bodies_by_name[body_name][0]
            matches = [
                shape_id
                for shape_id, label in enumerate(model.shape_label)
                if int(shape_body[shape_id]) == body_id and str(label).endswith(suffix)
            ]
            assert len(matches) == 1, (world, body_name, matches)
            flags = int(shape_flags[matches[0]])
            assert bool(flags & collide_shapes) is rigid
            assert bool(flags & collide_particles) is particles
            assert bool(flags & visible) is is_visible


def _make_runtime_cfg(*, use_cuda_graph: bool = True, env_spacing: float = 0.0):
    cfg = parse_env_cfg(_TASK_ID, device="cuda:0", num_envs=2)
    cfg.seed = 37
    cfg.scene.env_spacing = env_spacing
    cfg.decimation = 1
    cfg.num_substeps = 1
    cfg.sim.render_interval = 1
    cfg.sim.physics.num_substeps = 1
    cfg.sim.physics.use_cuda_graph = use_cuda_graph

    entries = {entry.name: entry for entry in cfg.sim.physics.solver_cfg.entries}
    entries[RIGID_ENTRY].substeps = 1
    entries[MPM_ENTRY].solver_cfg.max_iterations = 2
    assert entries[MPM_ENTRY].in_place
    return cfg


def _move_world_0_cup_collider(task, offset: tuple[float, float, float]) -> None:
    """Move only world 0's scene-owned source cup through its public writer."""
    model = NewtonManager.get_model()
    offset_t = torch.as_tensor(offset, device=task.device, dtype=torch.float32)
    env_ids = torch.tensor([0], device=task.device, dtype=torch.long)
    cup_pose = task.scene["source_cup"].data.root_link_pose_w.torch[env_ids].clone()
    cup_pose[:, :3] += offset_t
    task.scene["source_cup"].write_root_pose_to_sim_index(root_pose=cup_pose, env_ids=env_ids)
    _ = task.scene["source_cup"].data.body_link_pose_w

    world_mask_t = torch.zeros(task.num_envs, device=task.device, dtype=torch.bool)
    world_mask_t[0] = True
    NewtonManager.reset_solver_state(
        world_mask=wp.from_torch(world_mask_t, dtype=wp.bool),
        flags=newton.StateFlags.BODY_Q,
    )
    wp.synchronize_device(model.device)
    torch.testing.assert_close(
        task.scene["source_cup"].data.root_link_pose_w.torch[env_ids],
        cup_pose,
        rtol=0.0,
        atol=0.0,
    )


def _run_particle_rollout(
    *,
    use_cuda_graph: bool,
    cup_offset_world_0: tuple[float, float, float] | None = None,
    steps: int = 2,
) -> tuple[list[list[dict[str, np.ndarray]]], bool]:
    """Run a seeded two-world rollout and return per-step, per-world MPM state."""
    sim_utils.create_new_stage()
    env = None
    try:
        env = gym.make(_TASK_ID, cfg=_make_runtime_cfg(use_cuda_graph=use_cuda_graph))
        task = env.unwrapped
        task.sim._app_control_on_stop_handle = None
        env.reset()

        if cup_offset_world_0 is not None:
            _move_world_0_cup_collider(task, cup_offset_world_0)

        model = NewtonManager.get_model()
        media = task.scene[MPM_ENTRY]
        particle_offsets = media.particle_offsets.numpy().astype(np.int64, copy=False)
        particles_per_world = int(media.particles_per_object)
        actions = torch.zeros((task.num_envs, task.action_manager.total_action_dim), device=task.device)
        trajectory = []
        for _ in range(steps):
            env.step(actions)
            wp.synchronize_device(model.device)
            state = NewtonManager.get_state_0()
            trajectory.append(
                [
                    _particle_snapshot(state, slice(int(begin), int(begin) + particles_per_world))
                    for begin in particle_offsets
                ]
            )

        return trajectory, NewtonManager.is_cuda_graph_active()
    finally:
        if env is not None:
            env.close()


def _assert_pregrasp_state(task, *, arm_atol: float = 1.0e-5, gripper_atol: float = 1.0e-5) -> None:
    expected_arm = torch.as_tensor(task.cfg.arm_home, device=task.device).expand(task.num_envs, -1)
    arm_q = task._robot.data.joint_pos.torch[:, task._arm_joint_ids]
    torch.testing.assert_close(arm_q, expected_arm, rtol=0.0, atol=arm_atol)
    torch.testing.assert_close(
        task.gripper_width(),
        torch.full((task.num_envs,), 0.08, device=task.device),
        rtol=0.0,
        atol=gripper_atol,
    )
    tcp_error = torch.linalg.vector_norm(task.tcp_pos_e() - task.cup_grasp_point_e(), dim=-1)
    assert bool(torch.all(tcp_error < 0.005)), f"TCP-to-grasp error is too large: {tcp_error.tolist()}"


@pytest.mark.skipif(not _RUNTIME_AVAILABLE, reason=_RUNTIME_UNAVAILABLE_REASON)
def test_franka_pour_reset_preserves_pregrasp_immediately_and_after_one_step():
    """A task reset must not let the solver replace task-authored robot joints."""
    _require_sparse_capture_stack()
    sim_utils.create_new_stage()
    env = None
    try:
        env = gym.make(_TASK_ID, cfg=_make_runtime_cfg())
        task = env.unwrapped
        task.sim._app_control_on_stop_handle = None
        env.reset()

        _assert_pregrasp_state(task)
        actions = torch.zeros((task.num_envs, task.action_manager.total_action_dim), device=task.device)
        actions[:, -1] = 1.0
        env.step(actions)
        assert NewtonManager.is_cuda_graph_active()
        # Finite actuator gains allow sub-milliradian motion during a real
        # dynamics step; the reset itself remains exact above.
        _assert_pregrasp_state(task, arm_atol=1.0e-3, gripper_atol=1.0e-4)
    finally:
        if env is not None:
            env.close()


@pytest.mark.skipif(not _RUNTIME_AVAILABLE, reason=_RUNTIME_UNAVAILABLE_REASON)
def test_franka_pour_scene_owned_cups_use_public_state_and_leave_caller_cfg_unmodified():
    """The task resolves cloned cup assets without mutating its caller-owned config."""
    _require_sparse_capture_stack()
    sim_utils.create_new_stage()
    env = None
    caller_cfg = _make_runtime_cfg(use_cuda_graph=False, env_spacing=2.5)
    # Keep public views on the authoritative state after the eager step below.
    caller_cfg.sim.physics.num_substeps = 2
    assert caller_cfg.scene.source_cup is None
    assert caller_cfg.scene.target_cup is None
    assert caller_cfg.scene.media is None
    try:
        env = gym.make(_TASK_ID, cfg=caller_cfg)
        task = env.unwrapped
        task.sim._app_control_on_stop_handle = None
        env.reset()

        assert caller_cfg.scene.source_cup is None
        assert caller_cfg.scene.target_cup is None
        assert caller_cfg.scene.media is None
        assert task.cfg is not caller_cfg

        source_cup = task.scene["source_cup"]
        target_cup = task.scene["target_cup"]
        assert source_cup.num_instances == 2
        assert target_cup.num_instances == 2

        source_pose_e = source_cup.data.root_link_pose_w.torch.clone()
        source_pose_e[:, :3] -= task.scene.env_origins
        target_pose_e = target_cup.data.root_link_pose_w.torch.clone()
        target_pose_e[:, :3] -= task.scene.env_origins
        torch.testing.assert_close(task.cup_pose_e(), source_pose_e)
        torch.testing.assert_close(task.target_pose_e(), target_pose_e)
        _assert_scene_solver_roles(NewtonManager.get_model())

        for legacy_attribute in (
            "_cup_body_ids",
            "_target_body_ids",
            "_cup_joint_q",
            "_cup_joint_qd",
            "_cup_articulation_ids",
        ):
            assert not hasattr(task, legacy_attribute)

        target_pose_before = target_cup.data.root_link_pose_w.torch.clone()
        actions = torch.zeros((task.num_envs, task.action_manager.total_action_dim), device=task.device)
        actions[:, -1] = 1.0
        env.step(actions)
        wp.synchronize_device(NewtonManager.get_model().device)
        torch.testing.assert_close(
            target_cup.data.root_link_pose_w.torch,
            target_pose_before,
            rtol=0.0,
            atol=0.0,
        )
    finally:
        if env is not None:
            env.close()


@pytest.mark.skipif(not _RUNTIME_AVAILABLE, reason=_RUNTIME_UNAVAILABLE_REASON)
def test_franka_pour_offset_world_tables_support_both_cups():
    """Each rigid world collides with its own finite table at production spacing."""
    _require_sparse_capture_stack()
    sim_utils.create_new_stage()
    env = None
    try:
        env = gym.make(_TASK_ID, cfg=_make_runtime_cfg(use_cuda_graph=False, env_spacing=2.5))
        task = env.unwrapped
        task.sim._app_control_on_stop_handle = None
        env.reset()

        assert task.num_envs == 2
        assert not torch.equal(task.scene.env_origins[0], task.scene.env_origins[1])
        torch.testing.assert_close(task.cup_pose_e()[:, 2], torch.zeros(2, device=task.device), rtol=0.0, atol=0.0)

        actions = torch.zeros((task.num_envs, task.action_manager.total_action_dim), device=task.device)
        actions[:, -1] = 1.0  # keep the gripper open so this tests table support only
        for _ in range(30):
            env.step(actions)

        wp.synchronize_device(NewtonManager.get_model().device)
        cup_z = task.cup_pose_e()[:, 2]
        assert bool(torch.all(cup_z > -0.02)), f"A source cup fell through its local table: z={cup_z.tolist()}"
        torch.testing.assert_close(cup_z[0], cup_z[1], rtol=0.0, atol=2.0e-3)
    finally:
        if env is not None:
            env.close()


@pytest.mark.skipif(not _RUNTIME_AVAILABLE, reason=_RUNTIME_UNAVAILABLE_REASON)
def test_franka_pour_overlapping_worlds_step_selective_reset_and_step_again():
    """Exercise real isolated MPM ownership and an asynchronous one-world reset."""
    _require_sparse_capture_stack()
    sim_utils.create_new_stage()
    env = None
    try:
        env = gym.make(_TASK_ID, cfg=_make_runtime_cfg())
        task = env.unwrapped
        task.sim._app_control_on_stop_handle = None
        env.reset()

        assert task.num_envs == 2
        torch.testing.assert_close(task.scene.env_origins[0], task.scene.env_origins[1], rtol=0.0, atol=0.0)

        model = NewtonManager.get_model()
        media = task.scene[MPM_ENTRY]
        particle_offsets = media.particle_offsets.numpy().astype(np.int64, copy=False)
        particles_per_world = int(media.particles_per_object)
        particle_worlds = model.particle_world.numpy()
        particle_world_starts = model.particle_world_start.numpy()

        assert int(model.world_count) == 2
        assert int(model.particle_count) == 2 * particles_per_world
        np.testing.assert_array_equal(particle_offsets, particle_world_starts[:2])
        assert int(particle_world_starts[2]) == int(model.particle_count)
        assert int(particle_world_starts[-1]) == int(model.particle_count)
        for world, begin in enumerate(particle_offsets):
            end = int(begin) + particles_per_world
            np.testing.assert_array_equal(particle_worlds[int(begin) : end], np.full(particles_per_world, world))

        media_view = NewtonCoupledManager.get_entry_view(MPM_ENTRY)
        media_solver = NewtonCoupledManager.get_entry_solver(MPM_ENTRY)
        assert media_solver.model is media_view
        assert int(media_view.particle_count) == int(model.particle_count)
        assert media_view.particle_q is model.particle_q
        assert media_view.particle_world is model.particle_world
        np.testing.assert_array_equal(media_view.particle_world.numpy(), particle_worlds)
        _assert_media_collider_ownership(model, media_view)
        assert media_solver.supports_cuda_graph_capture

        actions = torch.zeros((task.num_envs, task.action_manager.total_action_dim), device=task.device)
        env.step(actions)
        wp.synchronize_device(model.device)
        assert NewtonManager.is_cuda_graph_active(), (
            "Franka Pour requested CUDA capture but no manager graph is active."
        )

        world_0 = slice(int(particle_offsets[0]), int(particle_offsets[0]) + particles_per_world)
        world_1 = slice(int(particle_offsets[1]), int(particle_offsets[1]) + particles_per_world)
        parent_states = []
        for state in (NewtonManager.get_state_0(), NewtonManager.get_state_1()):
            if all(state is not candidate for candidate in parent_states):
                parent_states.append(state)
        assert len(parent_states) == 2
        parent_world_0_before = [_particle_snapshot(state, world_0) for state in parent_states]
        parent_world_1_before = [_particle_snapshot(state, world_1) for state in parent_states]

        source_cup = task.scene["source_cup"]
        source_world_1_pose_before = source_cup.data.root_link_pose_w.torch[1].clone()
        source_world_1_velocity_before = source_cup.data.root_com_vel_w.torch[1].clone()
        robot_world_1_q_before = task._robot.data.joint_pos.torch[1].clone()
        robot_world_1_qd_before = task._robot.data.joint_vel.torch[1].clone()

        selected_env = torch.tensor([0], dtype=torch.long, device=task.device)
        perturbed_pose = source_cup.data.root_link_pose_w.torch[selected_env].clone()
        perturbed_pose[:, 0] += 0.05
        perturbed_velocity = torch.full((1, 6), 0.25, device=task.device)
        source_cup.write_root_pose_to_sim_index(root_pose=perturbed_pose, env_ids=selected_env)
        source_cup.write_root_velocity_to_sim_index(root_velocity=perturbed_velocity, env_ids=selected_env)
        _ = source_cup.data.body_link_pose_w

        entry_reset_observations: list[tuple[dict[str, np.ndarray], dict[str, np.ndarray]]] = []
        real_entry_reset = media_solver.reset

        def checked_entry_reset(state, world_mask=None, flags=None):
            assert world_mask is not None
            np.testing.assert_array_equal(world_mask.numpy(), np.array([True, False], dtype=np.bool_))
            assert flags == newton.StateFlags.BODY | newton.StateFlags.PARTICLE
            assert state.particle_q.shape == (model.particle_count,)
            before = _particle_snapshot(state, world_1)
            real_entry_reset(state, world_mask=world_mask, flags=flags)
            after = _particle_snapshot(state, world_1)
            entry_reset_observations.append((before, after))

        with mock.patch.object(media_solver, "reset", side_effect=checked_entry_reset):
            task.reset_pour_scene(selected_env)
        wp.synchronize_device(model.device)

        expected_source_pose = torch.tensor(
            [*task.cfg.cup_reset_pos, 0.0, 0.0, 0.0, 1.0],
            device=task.device,
        )
        expected_source_pose[:3] += task.scene.env_origins[0]
        torch.testing.assert_close(
            source_cup.data.root_link_pose_w.torch[0],
            expected_source_pose,
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            source_cup.data.root_com_vel_w.torch[0],
            torch.zeros(6, device=task.device),
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            source_cup.data.root_link_pose_w.torch[1],
            source_world_1_pose_before,
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            source_cup.data.root_com_vel_w.torch[1],
            source_world_1_velocity_before,
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            task._robot.data.joint_pos.torch[1],
            robot_world_1_q_before,
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            task._robot.data.joint_vel.torch[1],
            robot_world_1_qd_before,
            rtol=0.0,
            atol=0.0,
        )
        expected_refill = task._sample_cup_media(
            expected_source_pose[:3].unsqueeze(0),
            expected_source_pose[3:].unsqueeze(0),
        )[0]
        torch.testing.assert_close(
            media.data.particle_pos_w.torch[0],
            expected_refill,
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            media.data.particle_vel_w.torch[0],
            torch.zeros_like(expected_refill),
            rtol=0.0,
            atol=0.0,
        )

        assert len(entry_reset_observations) == len(parent_states)
        for entry_before, entry_after in entry_reset_observations:
            _assert_snapshot_bitwise_equal(entry_after, entry_before)
        for state, expected in zip(parent_states, parent_world_1_before, strict=True):
            _assert_snapshot_bitwise_equal(_particle_snapshot(state, world_1), expected)
        assert any(
            _snapshot_changed(_particle_snapshot(state, world_0), expected)
            for state, expected in zip(parent_states, parent_world_0_before, strict=True)
        ), "Selective reset did not change the selected world's particle state."

        env.step(actions)
        wp.synchronize_device(model.device)
    finally:
        if env is not None:
            env.close()


@pytest.mark.skipif(not _RUNTIME_AVAILABLE, reason=_RUNTIME_UNAVAILABLE_REASON)
def test_franka_pour_captured_matches_eager_and_moving_cup_stays_isolated():
    """Compare eager/captured physics and perturb only one overlapping material world."""
    _require_sparse_capture_stack()

    eager, eager_graph_active = _run_particle_rollout(use_cuda_graph=False)
    captured, captured_graph_active = _run_particle_rollout(use_cuda_graph=True)
    moved, moved_graph_active = _run_particle_rollout(
        use_cuda_graph=True,
        cup_offset_world_0=(0.025, 0.0, 0.0),
    )

    assert eager_graph_active is False
    assert captured_graph_active is True
    assert moved_graph_active is True
    for step, (eager_worlds, captured_worlds) in enumerate(zip(eager, captured, strict=True), start=1):
        for world, (eager_world, captured_world) in enumerate(zip(eager_worlds, captured_worlds, strict=True)):
            assert eager_world.keys() == captured_world.keys()
            _assert_rollout_world_equivalent(
                captured_world,
                eager_world,
                context=f"step={step}, world={world}, mode=captured-vs-eager",
            )

    for step, (moved_worlds, captured_worlds) in enumerate(zip(moved, captured, strict=True), start=1):
        _assert_rollout_world_equivalent(
            moved_worlds[1],
            captured_worlds[1],
            context=f"step={step}, world=1, mode=world-0-collider-perturbation",
        )
    world_0_position_delta = float(np.max(np.abs(moved[-1][0]["particle_q"] - captured[-1][0]["particle_q"])))
    assert world_0_position_delta > 1.0e-6, (
        "Moving world 0's cup did not measurably affect world 0's material: "
        f"max particle position delta={world_0_position_delta:.9g}."
    )
