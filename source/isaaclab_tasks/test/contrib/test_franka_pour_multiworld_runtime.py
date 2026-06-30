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

    expected = {"Cup", "TargetCup", "SpillFloor"}
    assert collider_count == 2 * len(expected)
    assert actual_by_world == {0: expected, 1: expected}


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
    """Move only world 0's cup articulation while preserving its particle coordinates."""
    model = NewtonManager.get_model()
    cup_q_start = int(task._cup_joint_q[0].item())
    offset_t = torch.as_tensor(offset, device=task.device, dtype=torch.float32)
    articulation_mask_t = torch.zeros(model.articulation_count, device=task.device, dtype=torch.bool)
    articulation_mask_t[task._cup_articulation_ids[0]] = True
    articulation_mask = wp.from_torch(articulation_mask_t, dtype=wp.bool)

    state_0 = NewtonManager.get_state_0()
    state_1 = NewtonManager.get_state_1()
    states = (state_0,) if state_0 is state_1 else (state_0, state_1)
    expected_cup_positions = []
    for state in states:
        cup_position = wp.to_torch(state.joint_q)[cup_q_start : cup_q_start + 3]
        cup_position += offset_t
        expected_cup_positions.append(cup_position.clone())
        newton.eval_fk(model, state.joint_q, state.joint_qd, state, articulation_mask)

    world_mask_t = torch.zeros(task.num_envs, device=task.device, dtype=torch.bool)
    world_mask_t[0] = True
    NewtonManager.reset_solver_state(
        world_mask=wp.from_torch(world_mask_t, dtype=wp.bool),
        flags=newton.StateFlags.BODY_Q,
    )
    wp.synchronize_device(model.device)
    for state, expected in zip(states, expected_cup_positions, strict=True):
        actual = wp.to_torch(state.joint_q)[cup_q_start : cup_q_start + 3]
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


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

        entry_reset_observations: list[tuple[dict[str, np.ndarray], dict[str, np.ndarray]]] = []
        real_entry_reset = media_solver.reset

        def checked_entry_reset(state, world_mask=None, flags=None):
            assert world_mask is not None
            np.testing.assert_array_equal(world_mask.numpy(), np.array([True, False], dtype=np.bool_))
            assert flags is None
            assert state.particle_q.shape == (model.particle_count,)
            before = _particle_snapshot(state, world_1)
            real_entry_reset(state, world_mask=world_mask, flags=flags)
            after = _particle_snapshot(state, world_1)
            entry_reset_observations.append((before, after))

        with mock.patch.object(media_solver, "reset", side_effect=checked_entry_reset):
            task.reset_pour_scene(torch.tensor([0], dtype=torch.long, device=task.device))
        wp.synchronize_device(model.device)

        assert len(entry_reset_observations) == len(parent_states)
        # The manager resets parent state_1 before authoritative state_0. The
        # coupled solver distributes each parent into the same entry state, so
        # correlate the observations by that public reset order, not identity.
        reset_expectations = reversed(parent_world_1_before)
        for (entry_before, entry_after), expected in zip(entry_reset_observations, reset_expectations, strict=True):
            _assert_snapshot_bitwise_equal(entry_before, expected)
            _assert_snapshot_bitwise_equal(entry_after, expected)
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
