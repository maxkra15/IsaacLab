# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration and pure task-logic tests for Franka RJ45 pick-and-insert."""

import math
from types import MethodType, SimpleNamespace

import pytest
import torch
import warp as wp
from isaaclab_newton.sim.schemas import MujocoJointCfg

from isaaclab.managers import ObservationTermCfg
from isaaclab.utils import math as math_utils

from isaaclab_contrib.coupling import NewtonCouplerManager

from isaaclab_tasks.contrib.franka_pour.reset_sampler import ResetDatasetSamplerCfg, _ResetDatasetSampler
from isaaclab_tasks.contrib.franka_rj45_insertion import mdp
from isaaclab_tasks.contrib.franka_rj45_insertion import pick_insert_env as pick_insert_env_module
from isaaclab_tasks.contrib.franka_rj45_insertion.asset_provenance import (
    FRANKA_RJ45_ASSET_CLOSURE_ENV,
    FRANKA_RJ45_ASSET_CLOSURE_TREE_SHA256,
    FRANKA_RJ45_FRANKA_LOGICAL_URI,
    FRANKA_RJ45_SEATTLE_TABLE_LOGICAL_URI,
    AssetClosureError,
    FrankaRJ45AssetClosure,
    franka_rj45_asset_contract,
)
from isaaclab_tasks.contrib.franka_rj45_insertion.config.franka.agents.pick_insert_rsl_rl_ppo_cfg import (
    FrankaRJ45PickInsertPPORunnerCfg,
)
from isaaclab_tasks.contrib.franka_rj45_insertion.franka_robot_cfg import (
    PICK_INSERT_ARM_TARGET_TRACKING_LIMITS,
    configure_franka_rj45_external_asset,
)
from isaaclab_tasks.contrib.franka_rj45_insertion.mdp import pick_insert_rewards
from isaaclab_tasks.contrib.franka_rj45_insertion.physics import (
    GRASP_FRICTION,
    make_rj45_task_layout,
)
from isaaclab_tasks.contrib.franka_rj45_insertion.pick_insert_env import (
    FrankaRJ45PickInsertEnv,
    _bilateral_grasp_proxy_contact_mask,
    _coupled_grasp_contact_alignment_mask,
    _sample_procedural_scene_poses,
    _symmetry_aware_grasp_alignment_mask,
)
from isaaclab_tasks.contrib.franka_rj45_insertion.pick_insert_env_cfg import (
    PICK_INSERT_CLOSED_FINGER_POSITION,
    PICK_INSERT_EFFECTIVE_GRASP_FRICTION,
    PICK_INSERT_GRASP_PROXY_FRICTION,
    PICK_INSERT_OPEN_FINGER_POSITION,
    PICK_INSERT_PHASE_0_REVERSE_CURRICULUM_AXIAL_RANGES_M,
    PICK_INSERT_PHASE_0_REVERSE_CURRICULUM_BAND_NAMES,
    PICK_INSERT_PHASE_0_REVERSE_CURRICULUM_WEIGHTS,
    PICK_INSERT_PHASE_NAMES,
    PICK_INSERT_RESET_PHASE_FRACTIONS,
    PICK_INSERT_RJ45_ENTRY_BODY_PATTERNS,
    PICK_INSERT_SUCCESS_MAX_PLUG_SPEED,
    FrankaRJ45PickInsertEnvCfg,
    PickInsertCurriculumCfg,
    pick_insert_phase_0_reverse_curriculum_sampling_contract,
    pick_insert_play_reset_seed_contract,
    pick_insert_reset_dataset_task_contract,
    pick_insert_topology_cfg,
)
from isaaclab_tasks.contrib.franka_rj45_insertion.pick_insert_reset_dataset_io import (
    PICK_INSERT_GOAL_MAX_PLUG_RELATIVE_LATCH_ANGLE_RAD,
    PICK_INSERT_GOAL_MAX_TASK_BODY_DRIFT_M,
)
from isaaclab_tasks.contrib.franka_rj45_insertion.play_seed import (
    load_play_reset_seed,
    rigidly_transform_task_pose,
    rigidly_transform_task_state,
    transform_play_reset_seed,
)
from isaaclab_tasks.contrib.franka_rj45_insertion.reset_dataset_io import reset_dataset_digest
from isaaclab_tasks.contrib.franka_rj45_insertion.rj45_env import FrankaRJ45InsertionEnv
from isaaclab_tasks.contrib.franka_rj45_insertion.rj45_env_cfg import (
    RJ45_ENTRY,
    FrankaRJ45InsertionEnvCfg,
)
from isaaclab_tasks.contrib.franka_rj45_insertion.table_scene_cfg import configure_seattle_table_external_asset


class _FakeArmAsset:
    def __init__(self, num_envs: int = 2) -> None:
        self.num_joints = 7
        self.data = SimpleNamespace(
            joint_pos=SimpleNamespace(torch=torch.zeros((num_envs, 7))),
            soft_joint_pos_limits=SimpleNamespace(
                torch=torch.tensor([[[-1.0, 1.0]] * 7] * num_envs, dtype=torch.float32)
            ),
        )
        self.command = torch.zeros((num_envs, 7))

    def find_joints(self, _names, *, preserve_order: bool, as_proxy: bool):
        assert preserve_order and as_proxy

        class JointIds:
            torch = torch.arange(7)

            def __len__(self) -> int:
                return 7

        return JointIds(), [f"panda_joint{index}" for index in range(1, 8)]

    def set_joint_position_target_index(self, *, target: torch.Tensor, joint_ids) -> None:
        del joint_ids
        self.command.copy_(target)


def _persistent_action(num_envs: int = 2):
    asset = _FakeArmAsset(num_envs)
    env = SimpleNamespace(num_envs=num_envs, device="cpu", scene={"robot": asset})
    cfg = mdp.PersistentResetTargetEMAJointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        preserve_order=True,
        scale=0.05,
        use_zero_offset=True,
        alpha=0.25,
        max_delta=0.05,
        joint_limit_margin=0.02,
        gravity_compensation=False,
    )
    return mdp.PersistentResetTargetEMAJointPositionAction(cfg, env), asset


def test_pick_pose_history_helpers_preserve_distinct_buffers_and_task_order(monkeypatch):
    task_body_ids = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
    origins = torch.tensor([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], dtype=torch.float32)
    previous_w = torch.zeros((4, 7), dtype=torch.float32)
    coupling_previous_w = torch.zeros_like(previous_w)
    previous_w[:, :3] = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [1.7, 2.8, 3.9], [2.0, 3.1, 4.2]])
    coupling_previous_w[:, :3] = previous_w[:, :3] + 0.125
    previous_w[:, 6] = 1.0
    coupling_previous_w[:, 6] = 1.0
    storage = {"previous": previous_w.clone(), "coupling": coupling_previous_w.clone()}
    selections: list[tuple[list[int], list[int]]] = []

    def capture(entry_name, body_ids, world_ids):
        assert entry_name == RJ45_ENTRY
        selected = torch.as_tensor(body_ids.numpy(), dtype=torch.long)
        selections.append((selected.tolist(), world_ids.numpy().tolist()))
        return (
            wp.from_torch(storage["previous"][selected].clone(), dtype=wp.transform),
            wp.from_torch(storage["coupling"][selected].clone(), dtype=wp.transform),
        )

    queued: dict[str, object] = {}

    def queue_restore(entry_name, body_ids, world_ids, previous, coupling):
        assert entry_name == RJ45_ENTRY
        selected = torch.as_tensor(body_ids.numpy(), dtype=torch.long)
        assert world_ids.numpy().tolist() == [0, 1]
        queued.update(
            selected=selected,
            previous=wp.to_torch(previous).clone(),
            coupling=wp.to_torch(coupling).clone(),
        )
        return SimpleNamespace(
            entry_name=RJ45_ENTRY,
            generation=7,
            body_ids=tuple(selected.tolist()),
            world_ids=(0, 1),
            expected_body_counts=(2, 2),
            pending_world_ids=(0, 1),
        )

    def restore_status(request):
        selected = queued["selected"]
        storage["previous"][selected] = queued["previous"]
        storage["coupling"][selected] = queued["coupling"]
        return SimpleNamespace(
            entry_name=request.entry_name,
            generation=request.generation,
            body_ids=request.body_ids,
            world_ids=request.world_ids,
            pending_world_ids=(),
            applied_world_ids=(0, 1),
            failed_world_ids=(),
            superseded_world_ids=(),
            application_count_deltas=(1, 1),
            expected_body_counts=(2, 2),
            body_application_count_deltas=(2, 2),
            pending=False,
            applied_exactly_once=True,
        )

    monkeypatch.setattr(NewtonCouplerManager, "capture_vbd_pose_history", capture)
    monkeypatch.setattr(NewtonCouplerManager, "queue_vbd_pose_history_restore", queue_restore)
    monkeypatch.setattr(NewtonCouplerManager, "get_vbd_pose_history_restore_status", restore_status)
    env = SimpleNamespace(
        num_envs=2,
        device="cpu",
        _task_body_ids=task_body_ids,
        _task_layout=SimpleNamespace(body_count=2, body_names=("socket", "plug")),
        env_origins=origins,
    )
    env._pose_history_selection = MethodType(FrankaRJ45PickInsertEnv._pose_history_selection, env)
    env.finalize_pending_task_pose_history_restores = MethodType(
        FrankaRJ45PickInsertEnv.finalize_pending_task_pose_history_restores,
        env,
    )

    previous_e, coupling_previous_e = FrankaRJ45PickInsertEnv.snapshot_task_pose_history_e(env)
    assert torch.allclose(previous_e[1, :, :3], previous_w.reshape(2, 2, 7)[1, :, :3] - origins[1])
    assert torch.allclose(
        coupling_previous_e[1, :, :3],
        coupling_previous_w.reshape(2, 2, 7)[1, :, :3] - origins[1],
    )
    assert not torch.equal(previous_e, coupling_previous_e)

    storage["previous"].zero_()
    storage["coupling"].zero_()
    evidence = FrankaRJ45PickInsertEnv.restore_task_pose_history_e(
        env,
        previous_e,
        coupling_previous_e,
    )

    assert selections[-1] == ([0, 1, 2, 3], [0, 1])
    assert evidence["body_order"] == ("socket", "plug")
    assert bool(evidence["restore_queued"].all())
    assert bool(evidence["pending_at_queue"].all())
    assert evidence["applied_exactly_once"] is None
    assert not bool(storage["previous"].any())
    assert not bool(storage["coupling"].any())

    env.finalize_pending_task_pose_history_restores()

    assert bool(evidence["applied_exactly_once"].all())
    assert not bool(evidence["pending_after_first_solve"].any())
    assert torch.equal(evidence["application_count_delta"], torch.ones(2, dtype=torch.int64))
    assert torch.equal(evidence["expected_body_count"], torch.full((2,), 2, dtype=torch.int64))
    assert torch.equal(evidence["body_application_count_delta"], torch.full((2,), 2, dtype=torch.int64))
    assert torch.equal(storage["previous"], previous_w)
    assert torch.equal(storage["coupling"], coupling_previous_w)


def test_pick_physics_ready_registers_post_solve_cable_projection_and_cleans_up(monkeypatch):
    body_ids = wp.array([3, 4, 5], dtype=wp.int32, device="cpu")
    calls: list[tuple[str, object]] = []

    class Handle:
        def __init__(self, name: str) -> None:
            self.name = name
            self.deregister_count = 0

        def deregister(self) -> None:
            self.deregister_count += 1

    old_handle = Handle("old")
    new_handle = Handle("new")
    runtime = SimpleNamespace(
        cable_preserved_input_body_ids=body_ids,
        align_after_step=lambda state: calls.append(("align", state)),
        prepare_step=lambda state: calls.append(("prepare", state)),
    )
    registration: dict[str, object] = {}

    def register(**kwargs):
        registration.update(kwargs)
        return new_handle

    monkeypatch.setattr(
        FrankaRJ45InsertionEnv,
        "_bind_rj45_physics_ready",
        lambda self, payload=None: calls.append(("base-bind", payload)),
    )
    monkeypatch.setattr(
        FrankaRJ45InsertionEnv,
        "_clear_rj45_callbacks",
        lambda self: calls.append(("base-clear", self)),
    )
    monkeypatch.setattr(
        NewtonCouplerManager,
        "register_vbd_preserved_input_pose_projection",
        register,
    )

    env = object.__new__(FrankaRJ45PickInsertEnv)
    env._is_closed = True
    env._rj45_runtime = runtime
    env._rj45_preserved_input_projection_handle = old_handle
    payload = object()
    env._bind_rj45_physics_ready(payload)

    assert calls == [("base-bind", payload)]
    assert old_handle.deregister_count == 1
    assert registration == {
        "name": "franka_rj45_pick_insert_cable_alignment",
        "entry_name": RJ45_ENTRY,
        "body_ids": body_ids,
        "callback": runtime.align_after_step,
    }
    assert env._rj45_preserved_input_projection_handle is new_handle

    state = object()
    env._prepare_rj45_substep(state)
    env._align_rj45_after_step()
    assert calls[-1] == ("prepare", state)
    assert not any(name == "align" for name, _ in calls)

    env._clear_rj45_callbacks()
    assert new_handle.deregister_count == 1
    assert env._rj45_preserved_input_projection_handle is None
    assert calls[-1][0] == "base-clear"


@pytest.mark.parametrize(
    ("pending", "failed", "superseded", "world_delta", "body_delta"),
    (
        (True, False, False, 0, 0),
        (False, True, False, 0, 0),
        (False, False, True, 0, 0),
        (False, False, False, 0, 48),
        (False, False, False, 2, 48),
        (False, False, False, 1, 47),
        (False, False, False, 1, 49),
    ),
)
def test_pick_pose_history_finalizer_rejects_nonexact_status(
    monkeypatch,
    pending: bool,
    failed: bool,
    superseded: bool,
    world_delta: int,
    body_delta: int,
):
    request = object()
    evidence = {
        "_request": request,
        "expected_body_count": torch.tensor([48]),
    }
    env = SimpleNamespace(
        device="cpu",
        _pending_task_pose_history_restores=[evidence],
    )
    status = SimpleNamespace(
        entry_name=RJ45_ENTRY,
        generation=3,
        world_ids=(0,),
        expected_body_counts=(48,),
        pending_world_ids=(0,) if pending else (),
        applied_world_ids=() if pending or failed or superseded else (0,),
        failed_world_ids=(0,) if failed else (),
        superseded_world_ids=(0,) if superseded else (),
        application_count_deltas=(world_delta,),
        body_application_count_deltas=(body_delta,),
        pending=pending,
        # Deliberately claim success for count/status mutations: the task
        # wrapper must recompute its own per-world exactness evidence.
        applied_exactly_once=not pending,
    )
    monkeypatch.setattr(
        NewtonCouplerManager,
        "get_vbd_pose_history_restore_status",
        lambda candidate: status if candidate is request else None,
    )

    with pytest.raises(RuntimeError, match="pose-history restore"):
        FrankaRJ45PickInsertEnv.finalize_pending_task_pose_history_restores(env)


def test_pick_pose_history_staging_coalesces_latest_row_per_environment():
    previous = torch.arange(4 * 2 * 7, dtype=torch.float32).reshape(4, 2, 7)
    coupling = previous + 1000.0
    env = SimpleNamespace(
        num_envs=3,
        device="cpu",
        _task_layout=SimpleNamespace(body_count=2),
        _reset_dataset_states={
            "task_body_previous_pose": previous,
            "task_body_coupling_previous_pose": coupling,
        },
    )

    FrankaRJ45PickInsertEnv._stage_task_pose_history_restore(
        env,
        torch.tensor([0, 2]),
        torch.tensor([0, 1]),
    )
    # A second reset before a real solve replaces only the latest requested
    # environment instead of queueing a conflicting core restore ticket.
    FrankaRJ45PickInsertEnv._stage_task_pose_history_restore(
        env,
        torch.tensor([2]),
        torch.tensor([3]),
    )

    assert env._task_pose_history_staging_mask.tolist() == [True, False, True]
    assert torch.equal(env._task_previous_pose_staging[0], previous[0])
    assert torch.equal(env._task_previous_pose_staging[2], previous[3])
    assert torch.equal(env._task_coupling_previous_pose_staging[0], coupling[0])
    assert torch.equal(env._task_coupling_previous_pose_staging[2], coupling[3])

    queued: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    def restore(previous_pose, coupling_pose, env_ids):
        queued.append((previous_pose.clone(), coupling_pose.clone(), env_ids.clone()))
        return {
            "restore_queued": torch.ones(2, dtype=torch.bool),
            "pending_at_queue": torch.ones(2, dtype=torch.bool),
            "previous_pose_queued": torch.ones(2, dtype=torch.bool),
            "coupling_previous_pose_queued": torch.ones(2, dtype=torch.bool),
            "body_order_exact": True,
            "world_order_exact": True,
        }

    env.finalize_pending_task_pose_history_restores = lambda *, require_complete: None
    env.scene = SimpleNamespace(write_data_to_sim=lambda: None)
    env.sim = SimpleNamespace(forward=lambda: None)
    env.restore_task_pose_history_e = restore
    evidence = FrankaRJ45PickInsertEnv._queue_staged_task_pose_history_restore(env)

    assert evidence is not None
    assert len(queued) == 1
    assert queued[0][2].tolist() == [0, 2]
    assert torch.equal(queued[0][0], torch.stack((previous[0], previous[3])))
    assert torch.equal(queued[0][1], torch.stack((coupling[0], coupling[3])))
    assert not bool(env._task_pose_history_staging_mask.any())


def test_pick_insert_reset_to_fails_closed_without_pose_history():
    with pytest.raises(RuntimeError, match="two validated VBD pose-history buffers"):
        FrankaRJ45PickInsertEnv.reset_to(
            SimpleNamespace(),
            state={},
            env_ids=torch.tensor([0]),
        )


def test_persistent_arm_action_integrates_ema_and_zero_holds_target_bitwise():
    action, asset = _persistent_action()

    action.process_actions(torch.ones((2, 7)))
    assert torch.allclose(action.position_targets, torch.full((2, 7), 0.0125))
    asset.data.joint_pos.torch.fill_(0.01)
    action.process_actions(torch.ones((2, 7)))
    assert torch.allclose(action.position_targets, torch.full((2, 7), 0.034375))

    held = action.position_targets.clone()
    asset.data.joint_pos.torch.fill_(-0.02)
    action.process_actions(torch.zeros((2, 7)))
    assert torch.equal(action.position_targets, held)
    assert torch.equal(action.processed_actions, torch.zeros_like(action.processed_actions))
    assert torch.equal(action._previous_delta, torch.zeros_like(action._previous_delta))
    action.apply_actions()
    assert torch.equal(asset.command, held)


def test_persistent_arm_action_reset_target_and_tracking_failure_are_bounded_per_world():
    action, asset = _persistent_action()
    target = torch.zeros((2, 7))
    target[0, 4] = 0.048
    action.set_reset_target(target, asset.data.joint_pos.torch)
    with pytest.raises(ValueError, match="tracking envelope"):
        rejected = target.clone()
        rejected[0, 4] += 1.0e-4
        action.set_reset_target(rejected, asset.data.joint_pos.torch)

    asset.data.joint_pos.torch[1, 6] = -0.3
    env = SimpleNamespace(
        action_manager=SimpleNamespace(get_term=lambda _name: action),
    )
    assert mdp.arm_target_tracking_failure(env).tolist() == [False, True]
    observed = mdp.arm_target_error_obs(env)
    assert tuple(observed.shape) == (2, 7)
    assert torch.isfinite(observed).all()


def test_pick_insert_config_is_distinct_long_horizon_six_stage_task():
    cfg = FrankaRJ45PickInsertEnvCfg()
    cfg.validate_config()
    layout = make_rj45_task_layout(pick_insert_topology_cfg(cfg))

    assert len(PICK_INSERT_PHASE_NAMES) == 6
    assert cfg.episode_length_s == 12.0
    assert cfg.is_finite_horizon is True
    assert cfg.actions.arm_action.max_delta == 0.05
    assert isinstance(cfg.actions.arm_action, mdp.PersistentResetTargetEMAJointPositionActionCfg)
    assert cfg.actions.arm_action.gravity_compensation is False
    assert tuple(cfg.actions.arm_action.tracking_error_limits) == PICK_INSERT_ARM_TARGET_TRACKING_LIMITS
    assert cfg.actions.gripper_action.default_position == cfg.actions.gripper_action.neutral_position
    assert cfg.actions.gripper_action.default_position == PICK_INSERT_OPEN_FINGER_POSITION == 0.04
    assert cfg.actions.gripper_action.close_position == PICK_INSERT_CLOSED_FINGER_POSITION == 0.0
    assert cfg.grasp_proxy_friction == PICK_INSERT_GRASP_PROXY_FRICTION == 4.5
    assert cfg.success_max_plug_speed == PICK_INSERT_SUCCESS_MAX_PLUG_SPEED == 0.10
    assert cfg.reset_dataset_rows_per_phase == 3334
    assert cfg.reset_dataset_rows_per_phase * len(PICK_INSERT_PHASE_NAMES) == 20_004
    assert cfg.reset_dataset_min_unique_full_pick_rows == 3000
    assert cfg.reset_dataset_phase_fractions == PICK_INSERT_RESET_PHASE_FRACTIONS
    assert cfg.reset_dataset_phase_fractions == (0.30, 0.15, 0.08, 0.06, 0.06, 0.35)
    assert PICK_INSERT_PHASE_0_REVERSE_CURRICULUM_AXIAL_RANGES_M == (
        (0.0010, 0.0016),
        (0.0016, 0.0035),
        (0.0035, 0.0120),
    )
    assert PICK_INSERT_PHASE_0_REVERSE_CURRICULUM_WEIGHTS == (0.35, 0.35, 0.30)
    assert math.fsum(PICK_INSERT_PHASE_0_REVERSE_CURRICULUM_WEIGHTS) == 1.0
    assert cfg.plug_grasp_offset == (0.0, -0.025, 0.010)
    assert tuple(cfg.scene.robot.init_state.joint_pos[f"panda_joint{index}"] for index in range(1, 8)) == (
        0.0444,
        -0.1894,
        -0.1107,
        -2.5148,
        0.0044,
        2.3775,
        0.6952,
    )
    assert str(cfg.scene.robot.spawn.usd_path) == FRANKA_RJ45_FRANKA_LOGICAL_URI
    assert str(cfg.scene.table.spawn.usd_path) == FRANKA_RJ45_SEATTLE_TABLE_LOGICAL_URI
    assert layout.body_count == 48
    assert layout.body_names[:3] == ("socket", "plug", "latch")
    assert layout.cable_segment_count == 45
    rj45_entry = next(entry for entry in cfg.sim.physics.solver_cfg.entries if entry.name == RJ45_ENTRY)
    assert tuple(rj45_entry.bodies) == PICK_INSERT_RJ45_ENTRY_BODY_PATTERNS
    assert cfg.observations.policy.cable_velocity.func is mdp.sampled_cable_linear_velocities_obs
    assert cfg.observations.policy.cable_velocity.scale == 0.1
    assert cfg.observations.policy.arm_target_error.func is mdp.arm_target_error_obs
    assert cfg.observations.policy.arm_target_error.scale == 4.0
    assert cfg.terminations.success.func is mdp.stable_pick_insert_success
    assert list(vars(cfg.terminations)) == [
        "stage_context",
        "nonfinite",
        "task_out_of_bounds",
        "arm_target_tracking",
        "lost_grasp",
        "success",
        "learning_progress_context",
        "time_out",
    ]


def test_pick_insert_play_mode_uses_procedural_full_pick_without_changing_task_contract():
    cfg = FrankaRJ45PickInsertEnvCfg()
    training_contract = pick_insert_reset_dataset_task_contract(cfg)
    assert cfg.reset_source == "dataset"
    assert isinstance(cfg.curriculum, PickInsertCurriculumCfg)
    assert cfg.events.reset_scene.func is mdp.reset_rj45_scene

    cfg.play_mode()
    cfg.validate_config()

    assert cfg.scene.num_envs == 1
    assert cfg.reset_source == "procedural"
    assert cfg.curriculum is None
    assert pick_insert_reset_dataset_task_contract(cfg) == training_contract


def test_detailed_network_switch_presentation_is_only_requested_for_kit():
    env = object.__new__(FrankaRJ45PickInsertEnv)
    env._is_closed = True
    env.sim = SimpleNamespace(resolve_visualizer_types=lambda: ["newton_gl"])
    assert not env._include_rj45_network_switch_presentation()
    env.sim = SimpleNamespace(resolve_visualizer_types=lambda: ["kit"])
    assert env._include_rj45_network_switch_presentation()


def test_procedural_scene_sampling_is_seeded_bounded_and_diverse():
    cfg = FrankaRJ45PickInsertEnvCfg()
    identity = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    zero = torch.zeros((1, 3))

    torch.manual_seed(137)
    first = _sample_procedural_scene_poses(
        cfg,
        512,
        device="cpu",
        dtype=torch.float32,
        socket_local_position=zero,
        socket_local_orientation=identity,
    )
    torch.manual_seed(137)
    repeated = _sample_procedural_scene_poses(
        cfg,
        512,
        device="cpu",
        dtype=torch.float32,
        socket_local_position=zero,
        socket_local_orientation=identity,
    )
    torch.manual_seed(138)
    different = _sample_procedural_scene_poses(
        cfg,
        512,
        device="cpu",
        dtype=torch.float32,
        socket_local_position=zero,
        socket_local_orientation=identity,
    )

    socket_pose, pickup_pose = first
    assert torch.equal(socket_pose, repeated[0])
    assert torch.equal(pickup_pose, repeated[1])
    assert not torch.equal(socket_pose, different[0])
    assert not torch.equal(pickup_pose, different[1])
    socket_lower = torch.tensor(cfg.socket_position_lower)
    socket_upper = torch.tensor(cfg.socket_position_upper)
    pickup_lower = torch.tensor(cfg.pickup_position_lower)
    pickup_upper = torch.tensor(cfg.pickup_position_upper)
    assert bool(((socket_pose[:, :3] >= socket_lower) & (socket_pose[:, :3] <= socket_upper)).all())
    assert bool(((pickup_pose[:, :3] >= pickup_lower) & (pickup_pose[:, :3] <= pickup_upper)).all())
    assert bool(
        (
            torch.linalg.vector_norm(pickup_pose[:, :2] - socket_pose[:, :2], dim=-1)
            >= cfg.minimum_pickup_socket_distance
        ).all()
    )
    assert bool((socket_pose[:, :2].amax(dim=0) - socket_pose[:, :2].amin(dim=0) > 0.10).all())
    assert bool((pickup_pose[:, :2].amax(dim=0) - pickup_pose[:, :2].amin(dim=0) > 0.14).all())
    assert torch.allclose(torch.linalg.vector_norm(socket_pose[:, 3:7], dim=-1), torch.ones(512), atol=1.0e-6)
    assert torch.allclose(torch.linalg.vector_norm(pickup_pose[:, 3:7], dim=-1), torch.ones(512), atol=1.0e-6)


def test_procedural_task_transform_preserves_whole_loose_assembly():
    task_pose = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [0.1, -0.2, 0.3, 0.0, 0.0, 0.0, 1.0],
                [-0.4, 0.5, 0.2, 0.0, 0.0, 0.0, 1.0],
            ]
        ],
        dtype=torch.float32,
    )
    source = task_pose[:, 1].clone()
    yaw = torch.tensor([0.7])
    destination = torch.cat(
        (
            torch.tensor([[0.55, -0.12, 0.02]]),
            math_utils.quat_from_euler_xyz(torch.zeros(1), torch.zeros(1), yaw),
        ),
        dim=-1,
    )

    transformed = rigidly_transform_task_pose(task_pose, source, destination)

    assert torch.allclose(transformed[:, 1], destination, atol=1.0e-6)
    assert torch.allclose(
        torch.cdist(task_pose[..., :3], task_pose[..., :3]),
        torch.cdist(transformed[..., :3], transformed[..., :3]),
        atol=1.0e-6,
    )
    assert torch.allclose(
        torch.linalg.vector_norm(transformed[..., 3:7], dim=-1),
        torch.ones((1, 3)),
        atol=1.0e-6,
    )


def test_packaged_play_reset_seed_matches_live_physics_and_transforms_velocity():
    cfg = FrankaRJ45PickInsertEnvCfg()
    layout = make_rj45_task_layout(pick_insert_topology_cfg(cfg))
    contract = pick_insert_play_reset_seed_contract(cfg)
    seed = load_play_reset_seed(
        expected_task_body_order=layout.body_names,
        expected_physics_contract=contract,
    )

    assert set(seed) == {
        "task_body_pose",
        "task_body_previous_pose",
        "task_body_coupling_previous_pose",
        "task_body_velocity",
        "goal_task_body_pose",
    }
    assert seed["task_body_pose"].shape == (layout.body_count, 7)
    pickup_height = float(seed["task_body_pose"][layout.plug_body_index, 2])
    assert cfg.pickup_position_lower[2] <= pickup_height <= cfg.pickup_position_upper[2]
    _, fixed_height_pickup = _sample_procedural_scene_poses(
        cfg,
        64,
        device="cpu",
        dtype=torch.float32,
        socket_local_position=torch.zeros((1, 3)),
        socket_local_orientation=torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
        pickup_height=pickup_height,
    )
    assert torch.equal(fixed_height_pickup[:, 2], torch.full((64,), pickup_height))
    with pytest.raises(ValueError, match="inside the declared sampling bounds"):
        _sample_procedural_scene_poses(
            cfg,
            1,
            device="cpu",
            dtype=torch.float32,
            socket_local_position=torch.zeros((1, 3)),
            socket_local_orientation=torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
            pickup_height=cfg.pickup_position_upper[2] + 0.001,
        )

    pose = seed["task_body_pose"].unsqueeze(0)
    velocity = seed["task_body_velocity"].unsqueeze(0)
    source = pose[:, layout.plug_body_index]
    destination = source.clone()
    destination[:, :2] += torch.tensor([[0.1, -0.2]])
    yaw = torch.tensor([0.5])
    destination[:, 3:7] = math_utils.quat_mul(
        math_utils.quat_from_euler_xyz(torch.zeros(1), torch.zeros(1), yaw),
        source[:, 3:7],
    )
    transformed_pose, transformed_velocity = rigidly_transform_task_state(
        pose,
        velocity,
        source,
        destination,
    )

    assert torch.allclose(transformed_pose[:, layout.plug_body_index], destination, atol=1.0e-6)
    assert torch.allclose(
        torch.linalg.vector_norm(transformed_velocity[..., :3], dim=-1),
        torch.linalg.vector_norm(velocity[..., :3], dim=-1),
        atol=1.0e-7,
    )
    assert torch.allclose(
        torch.linalg.vector_norm(transformed_velocity[..., 3:6], dim=-1),
        torch.linalg.vector_norm(velocity[..., 3:6], dim=-1),
        atol=1.0e-7,
    )


def test_packaged_play_reset_seed_rejects_incompatible_live_physics():
    cfg = FrankaRJ45PickInsertEnvCfg()
    layout = make_rj45_task_layout(pick_insert_topology_cfg(cfg))
    contract = pick_insert_play_reset_seed_contract(cfg)
    incompatible = dict(contract)
    incompatible["contract_version"] = 2

    with pytest.raises(ValueError, match="physics contract"):
        load_play_reset_seed(
            expected_task_body_order=layout.body_names,
            expected_physics_contract=incompatible,
        )


def test_packaged_play_reset_seed_two_stage_transform_preserves_history_and_socket():
    cfg = FrankaRJ45PickInsertEnvCfg()
    layout = make_rj45_task_layout(pick_insert_topology_cfg(cfg))
    seed = load_play_reset_seed(
        expected_task_body_order=layout.body_names,
        expected_physics_contract=pick_insert_play_reset_seed_contract(cfg),
    )
    socket_pose = seed["task_body_pose"][layout.socket_body_index].repeat(2, 1)
    socket_pose[:, :2] += torch.tensor([[0.04, -0.02], [-0.05, 0.06]])
    pickup_pose = seed["task_body_pose"][layout.plug_body_index].repeat(2, 1)
    pickup_pose[:, :2] += torch.tensor([[0.08, 0.03], [-0.06, -0.04]])
    pickup_yaw = torch.tensor([0.35, -0.55])
    pickup_pose[:, 3:7] = math_utils.quat_from_euler_xyz(
        torch.zeros(2),
        torch.zeros(2),
        pickup_yaw,
    )

    transformed = transform_play_reset_seed(
        seed,
        socket_body_index=layout.socket_body_index,
        plug_body_index=layout.plug_body_index,
        socket_pose=socket_pose,
        pickup_pose=pickup_pose,
    )

    assert torch.allclose(transformed["task_body_pose"][:, layout.socket_body_index], socket_pose, atol=1.0e-6)
    assert torch.allclose(transformed["goal_task_body_pose"][:, layout.socket_body_index], socket_pose, atol=1.0e-6)
    assert torch.allclose(transformed["task_body_pose"][:, layout.plug_body_index], pickup_pose, atol=1.0e-6)
    for history_name in ("task_body_previous_pose", "task_body_coupling_previous_pose"):
        seed_delta = torch.linalg.vector_norm(
            seed[history_name][..., :3] - seed["task_body_pose"][..., :3],
            dim=-1,
        )
        transformed_delta = torch.linalg.vector_norm(
            transformed[history_name][..., :3] - transformed["task_body_pose"][..., :3],
            dim=-1,
        )
        assert torch.allclose(transformed_delta, seed_delta.expand_as(transformed_delta), atol=1.0e-7)
    seed_speed = torch.linalg.vector_norm(seed["task_body_velocity"].reshape(layout.body_count, 2, 3), dim=-1)
    transformed_speed = torch.linalg.vector_norm(
        transformed["task_body_velocity"].reshape(2, layout.body_count, 2, 3), dim=-1
    )
    assert torch.allclose(transformed_speed, seed_speed.expand_as(transformed_speed), atol=1.0e-7)


def test_pick_insert_runtime_builder_forwards_proxy_friction_without_mutating_legacy_defaults():
    pick_cfg = FrankaRJ45PickInsertEnvCfg()
    legacy_cfg = FrankaRJ45InsertionEnvCfg()

    pick_builder = FrankaRJ45PickInsertEnv._create_rj45_builder(SimpleNamespace(), pick_cfg)
    legacy_builder = FrankaRJ45InsertionEnv._create_rj45_builder(SimpleNamespace(), legacy_cfg)

    assert GRASP_FRICTION == 2.0
    assert pick_builder.grasp_proxy_friction == PICK_INSERT_GRASP_PROXY_FRICTION
    assert (GRASP_FRICTION * pick_builder.grasp_proxy_friction) ** 0.5 == PICK_INSERT_EFFECTIVE_GRASP_FRICTION
    assert legacy_builder.grasp_proxy_friction == GRASP_FRICTION
    assert legacy_cfg.actions.gripper_action.close_position == 0.004
    assert legacy_cfg.success_max_plug_speed == 0.01


@pytest.mark.parametrize("value", (0.004, -0.001, float("nan"), True))
def test_pick_insert_config_rejects_noncanonical_close_target(value: object):
    cfg = FrankaRJ45PickInsertEnvCfg()
    cfg.actions.gripper_action.close_position = value

    with pytest.raises(ValueError, match="exact 0.0 m closed gripper target"):
        cfg.validate_config()


@pytest.mark.parametrize("value", (2.0, 4.4, float("nan"), True))
def test_pick_insert_config_rejects_noncanonical_proxy_friction(value: object):
    cfg = FrankaRJ45PickInsertEnvCfg()
    cfg.grasp_proxy_friction = value

    with pytest.raises(ValueError, match="exact 4.5 grasp-proxy friction"):
        cfg.validate_config()


def test_pick_insert_config_rejects_legacy_plug_success_speed():
    cfg = FrankaRJ45PickInsertEnvCfg()
    cfg.success_max_plug_speed = 0.01

    with pytest.raises(ValueError, match="exact 0.10 plug success-speed limit"):
        cfg.validate_config()


def test_pick_insert_config_rejects_timeout_bootstrapping():
    cfg = FrankaRJ45PickInsertEnvCfg()
    cfg.is_finite_horizon = False

    with pytest.raises(ValueError, match="finite task horizon"):
        cfg.validate_config()


def test_pick_insert_config_rejects_incoherent_reset_source_and_curriculum():
    cfg = FrankaRJ45PickInsertEnvCfg()
    cfg.reset_source = "procedural"
    with pytest.raises(ValueError, match="must not construct"):
        cfg.validate_config()

    cfg = FrankaRJ45PickInsertEnvCfg()
    cfg.curriculum = None
    with pytest.raises(ValueError, match="require.*curriculum"):
        cfg.validate_config()

    cfg = FrankaRJ45PickInsertEnvCfg()
    cfg.reset_source = "unknown"
    with pytest.raises(ValueError, match="reset_source"):
        cfg.validate_config()


@pytest.mark.parametrize("weight", (0.0, 1.0, -0.1, float("nan")))
def test_pick_insert_config_rejects_invalid_reach_orientation_weight(weight: float):
    cfg = FrankaRJ45PickInsertEnvCfg()
    cfg.reach_orientation_reward_weight = weight

    with pytest.raises(ValueError, match="reach_orientation_reward_weight"):
        cfg.validate_config()


def test_pick_insert_config_rejects_mutated_vbd_body_ownership():
    cfg = FrankaRJ45PickInsertEnvCfg()
    cfg.rj45_entry_body_patterns = (PICK_INSERT_RJ45_ENTRY_BODY_PATTERNS[0],)

    with pytest.raises(ValueError, match="exact validated.*VBD ownership selectors"):
        cfg.validate_config()


@pytest.mark.parametrize("value", (0, -1, True, 96.0))
def test_pick_insert_config_rejects_invalid_rows_per_phase(value: object):
    cfg = FrankaRJ45PickInsertEnvCfg()
    cfg.reset_dataset_rows_per_phase = value

    with pytest.raises(ValueError, match="positive plain integer"):
        cfg.validate_config()


@pytest.mark.parametrize(
    ("fractions", "match"),
    (
        ((1.0,), "exactly six"),
        ((0.30, 0.15, 0.08, 0.06, 0.06, float("nan")), "finite nonnegative"),
        ((-0.01, 0.16, 0.08, 0.06, 0.06, 0.65), "finite nonnegative"),
        ((0.29, 0.15, 0.08, 0.06, 0.06, 0.35), "sum exactly"),
        ((True, 0.15, 0.08, 0.06, 0.06, -0.35), "finite nonnegative"),
        ((0.31, 0.15, 0.08, 0.06, 0.06, 0.34), "phase-5 fraction"),
    ),
)
def test_pick_insert_config_rejects_invalid_phase_assignment_fractions(
    fractions: tuple[float, ...],
    match: str,
):
    cfg = FrankaRJ45PickInsertEnvCfg()
    cfg.reset_dataset_phase_fractions = fractions

    with pytest.raises(ValueError, match=match):
        cfg.validate_config()


def test_pick_insert_grasp_is_table_clearance_tilted_with_fingers_across_plug_width():
    cfg = FrankaRJ45PickInsertEnvCfg()
    orientation = torch.tensor(cfg.plug_grasp_orientation_xyzw).repeat(2, 1)
    tool_axes = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    rotated = math_utils.quat_apply(orientation, tool_axes)

    assert torch.allclose(rotated[0], torch.tensor([1.0, 0.0, 0.0]), atol=1.0e-6)
    assert torch.allclose(rotated[1], torch.tensor([0.0, 0.0, -1.0]), atol=1.0e-6)

    # The bundled Franka URDF assigns the left/right finger joints +Y/-Y in
    # tool coordinates.  The authored grasp therefore places them on the
    # proxy's +X/-X sides respectively.
    finger_directions_tool = torch.tensor([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]])
    finger_directions_plug = math_utils.quat_apply(orientation, finger_directions_tool)
    assert torch.allclose(
        finger_directions_plug,
        torch.tensor([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]),
        atol=1.0e-6,
    )


def test_pick_insert_grasp_alignment_is_top_down_and_half_turn_symmetric():
    zero = torch.zeros(5)
    current = math_utils.quat_from_euler_xyz(
        torch.tensor([0.0, 0.0, 0.0, math.radians(20.0), math.pi]),
        zero,
        torch.tensor([0.0, math.pi, math.pi / 2.0, 0.0, 0.0]),
    )
    target = math_utils.quat_from_euler_xyz(zero, zero, zero)

    acquisition = _symmetry_aware_grasp_alignment_mask(current, target, math.radians(15.0))
    retention = _symmetry_aware_grasp_alignment_mask(current, target, math.radians(25.0))

    assert acquisition.tolist() == [True, True, False, False, False]
    assert retention.tolist() == [True, True, False, True, False]


def test_pick_insert_grasp_couples_proxy_face_assignment_to_signed_closing_axis():
    target = math_utils.quat_from_euler_xyz(torch.zeros(4), torch.zeros(4), torch.zeros(4))
    current = math_utils.quat_from_euler_xyz(
        torch.zeros(4),
        torch.zeros(4),
        torch.tensor([0.0, math.pi, 0.0, math.pi]),
    )
    canonical_faces = torch.tensor([True, True, False, False])
    swapped_faces = ~canonical_faces

    coupled = _coupled_grasp_contact_alignment_mask(
        canonical_faces,
        swapped_faces,
        current,
        target,
        math.radians(15.0),
    )
    independent_product = (canonical_faces | swapped_faces) & _symmetry_aware_grasp_alignment_mask(
        current,
        target,
        math.radians(15.0),
    )

    # Positive Ryy matches canonical faces; negative Ryy matches swapped faces.
    # The middle two entries exercise and reject both mismatched Cartesian branches.
    assert independent_product.tolist() == [True, True, True, True]
    assert coupled.tolist() == [True, False, False, True]


@pytest.mark.parametrize(
    ("acquisition", "retention"),
    (
        (0.0, math.radians(25.0)),
        (math.radians(15.0), math.radians(15.0)),
        (math.radians(25.0), math.radians(15.0)),
        (float("nan"), math.radians(25.0)),
        (math.radians(15.0), math.pi / 2.0),
    ),
)
def test_pick_insert_config_rejects_invalid_grasp_axis_tolerances(acquisition: float, retention: float):
    cfg = FrankaRJ45PickInsertEnvCfg()
    cfg.grasp_acquisition_axis_tolerance_rad = acquisition
    cfg.grasp_retention_axis_tolerance_rad = retention

    with pytest.raises(ValueError, match="grasp axis tolerances"):
        cfg.validate_config()


@pytest.mark.parametrize("tolerance", (0.0, -1.0, float("nan"), 0.00475))
def test_pick_insert_config_rejects_invalid_grasp_proxy_face_tolerance(tolerance: float):
    cfg = FrankaRJ45PickInsertEnvCfg()
    cfg.grasp_proxy_face_tolerance_m = tolerance

    with pytest.raises(ValueError, match="grasp_proxy_face_tolerance_m"):
        cfg.validate_config()


def test_pick_insert_contract_round_trips_through_safe_torch_load(tmp_path):
    contract = pick_insert_reset_dataset_task_contract(FrankaRJ45PickInsertEnvCfg())
    artifact = tmp_path / "pick_insert_contract.pt"

    torch.save(contract, artifact)
    loaded = torch.load(artifact, map_location="cpu", weights_only=True)

    assert reset_dataset_digest(loaded) == reset_dataset_digest(contract)
    assert loaded["task_body_count"] == 48
    assert loaded["rj45_physics"]["socket_body_mode"] == "zero-mass-resettable"
    assert loaded["rj45_physics"]["task_support_plane_enabled"] is False
    assert tuple(loaded["coupler"]["rj45_entry"]["bodies"]) == PICK_INSERT_RJ45_ENTRY_BODY_PATTERNS
    assert loaded["contract_version"] == 8
    assert loaded["base_contract_version"] == 3
    assert loaded["external_assets"] == franka_rj45_asset_contract()
    assert loaded["robot"]["asset"] == FRANKA_RJ45_FRANKA_LOGICAL_URI
    assert loaded["robot"]["spawn"]["usd_path"] == FRANKA_RJ45_FRANKA_LOGICAL_URI
    assert loaded["static_scene"]["table_spawn"]["usd_path"] == FRANKA_RJ45_SEATTLE_TABLE_LOGICAL_URI
    assert loaded["pick_insert"]["semantics_version"] == 8
    assert loaded["pick_insert"]["goal_local_success_predicate_version"] == 1
    phase_0_sampling = loaded["pick_insert"]["phase_0_reverse_curriculum_sampling"]
    assert phase_0_sampling == pick_insert_phase_0_reverse_curriculum_sampling_contract()
    assert phase_0_sampling["sampler_version"] == 1
    assert phase_0_sampling["frame"] == "goal-plug-local"
    assert phase_0_sampling["band_names"] == PICK_INSERT_PHASE_0_REVERSE_CURRICULUM_BAND_NAMES
    assert phase_0_sampling["axial_offset_ranges_m"] == PICK_INSERT_PHASE_0_REVERSE_CURRICULUM_AXIAL_RANGES_M
    assert phase_0_sampling["band_weights"] == PICK_INSERT_PHASE_0_REVERSE_CURRICULUM_WEIGHTS
    assert phase_0_sampling["geometric_success_at_reset"] is False
    assert phase_0_sampling["rng_owner"] == "PickInsertResetDatasetGenerator.random"
    phase_4_sampling = loaded["pick_insert"]["phase_4_pregrasp_orientation_sampling"]
    assert phase_4_sampling["sampler_version"] == 1
    assert phase_4_sampling["clearance_height_m"] == 0.045
    assert phase_4_sampling["top_down_tilt_range_rad"] == (0.0, math.radians(25.0))
    assert phase_4_sampling["closing_axis_twist_range_rad"] == (
        -math.radians(60.0),
        math.radians(60.0),
    )
    assert loaded["validation_geometry"]["success_predicate_version"] == 1
    assert loaded["validation_geometry"]["grasp"] == {
        "contract_version": 2,
        "acquisition_axis_tolerance_rad": math.radians(15.0),
        "retention_axis_tolerance_rad": math.radians(25.0),
        "tool_axis": (0.0, 0.0, 1.0),
        "tool_axis_comparison": "signed",
        "finger_closing_axis": (0.0, 1.0, 0.0),
        "finger_closing_axis_comparison": "signed-coupled-to-proxy-face-assignment",
        "proxy_contact_faces": "exclusive-opposing-local-x-surface",
        "canonical_proxy_face_assignment": "left:+local-x,right:-local-x",
        "canonical_closing_axis_comparison": "relative-rotation-yy>=cos(tolerance)",
        "swapped_proxy_face_assignment": "left:-local-x,right:+local-x",
        "swapped_closing_axis_comparison": "relative-rotation-yy<=-cos(tolerance)",
        "proxy_face_tolerance_m": 5.0e-4,
    }
    assert loaded["rj45_physics"]["contract_version"] == 6
    assert loaded["rj45_physics"]["franka_finger_raw_friction"] == GRASP_FRICTION == 2.0
    assert loaded["rj45_physics"]["grasp_proxy_raw_friction"] == PICK_INSERT_GRASP_PROXY_FRICTION
    assert loaded["rj45_physics"]["grasp_contact_friction_combine_rule"] == "geometric-mean"
    assert loaded["rj45_physics"]["grasp_contact_effective_friction"] == PICK_INSERT_EFFECTIVE_GRASP_FRICTION
    assert loaded["reset_state_representation"] == {
        "contract_version": 2,
        "task_body_pose_frame": "environment-local-xyzw",
        "task_body_velocity_frame": "world-linear-angular",
        "vbd_entry_name": "rj45",
        "vbd_body_order_source": "task_body_order",
        "vbd_previous_pose_field": "task_body_previous_pose",
        "vbd_coupling_previous_pose_field": "task_body_coupling_previous_pose",
        "vbd_pose_history_frame": "environment-local-xyzw",
        "restore_semantics": "deferred-one-shot-after-input-and-proxy-rebaseline-before-first-vbd-solve",
        "preserved_input_task_body_range_half_open": (3, 47),
        "preserved_input_semantics": "scatter-history-without-pose-delta-velocity-injection-or-rewind",
    }
    assert loaded["pick_insert"]["finger_open_position"] == PICK_INSERT_OPEN_FINGER_POSITION
    assert loaded["pick_insert"]["finger_closed_position"] == PICK_INSERT_CLOSED_FINGER_POSITION
    assert loaded["pick_insert"]["goal_max_task_body_drift_m"] == PICK_INSERT_GOAL_MAX_TASK_BODY_DRIFT_M == 0.012
    assert (
        loaded["pick_insert"]["goal_max_plug_relative_latch_angle_rad"]
        == PICK_INSERT_GOAL_MAX_PLUG_RELATIVE_LATCH_ANGLE_RAD
        == 0.10
    )
    assert loaded["actions"]["gripper"]["neutral_position"] == PICK_INSERT_OPEN_FINGER_POSITION
    assert loaded["actions"]["gripper"]["default_position"] == PICK_INSERT_OPEN_FINGER_POSITION
    assert loaded["actions"]["gripper"]["close_position"] == PICK_INSERT_CLOSED_FINGER_POSITION
    assert loaded["validation_geometry"]["success_max_plug_speed"] == PICK_INSERT_SUCCESS_MAX_PLUG_SPEED
    assert loaded["pick_insert"]["reset_dataset_rows_per_phase"] == 3334
    assert loaded["pick_insert"]["reset_dataset_phase_fractions"] == PICK_INSERT_RESET_PHASE_FRACTIONS
    assert loaded["pick_insert"]["is_finite_horizon"] is True
    control = loaded["robot"]["reset_control_convention"]
    assert control["target_semantics"] == "persistent-absolute-integrated-once-per-policy-step"
    assert tuple(control["target_tracking_error_limits_rad"]) == PICK_INSERT_ARM_TARGET_TRACKING_LIMITS
    assert control["native_gravity_compensation"] == "mjwarp-joint-actuatorgravcomp"


def test_pick_insert_policy_interface_remains_checkpoint_compatible():
    """Pin the term order and dimensions consumed by the fine-tuned policy."""
    cfg = FrankaRJ45PickInsertEnvCfg()
    policy_widths = {
        "arm_q": 7,
        "arm_qd": 7,
        "arm_target_error": 7,
        "tcp_pose": 7,
        "tcp_velocity": 6,
        "plug_pose": 7,
        "socket_pose": 7,
        "goal_plug_pose": 7,
        "tcp_grasp_error": 6,
        "plug_goal_error": 7,
        "plug_velocity": 6,
        "cable_shape": 21,
        "cable_velocity": 21,
        "finger_position": 2,
        "finger_velocity": 2,
        "gripper_target": 1,
        "gripper_deflection": 2,
        "grasp_proxy_contact": 1,
        "grasp_stage": 2,
        "time_remaining": 1,
        "last_action": 8,
    }
    privileged_widths = {"reset_phase": 1, "reset_difficulty": 1, "success_dwell": 1}

    policy_terms = tuple(
        name for name, value in vars(cfg.observations.policy).items() if isinstance(value, ObservationTermCfg)
    )
    privileged_terms = tuple(
        name for name, value in vars(cfg.observations.privileged).items() if isinstance(value, ObservationTermCfg)
    )
    assert policy_terms == tuple(policy_widths)
    assert privileged_terms == tuple(privileged_widths)
    assert sum(policy_widths.values()) == 135
    assert sum(policy_widths.values()) + sum(privileged_widths.values()) == 138
    assert tuple(vars(cfg.actions)) == ("arm_action", "gripper_action")
    assert tuple(cfg.actions.arm_action.joint_names) == tuple(f"panda_joint{index}" for index in range(1, 8))

    reward_names = ("progress", "grasp_acquired", "success", "failure", "action_magnitude", "action_rate")
    assert tuple(vars(cfg.rewards)) == reward_names
    assert tuple(getattr(cfg.rewards, name).weight for name in reward_names) == pytest.approx(
        (1.0, 0.5, 10.0, -2.0, -5.0e-5, -1.0e-4)
    )
    assert tuple(vars(cfg.terminations)) == (
        "stage_context",
        "nonfinite",
        "task_out_of_bounds",
        "arm_target_tracking",
        "lost_grasp",
        "success",
        "learning_progress_context",
        "time_out",
    )

    runner_cfg = FrankaRJ45PickInsertPPORunnerCfg()
    assert runner_cfg.obs_groups == {"actor": ["policy"], "critic": ["policy", "privileged"]}
    assert runner_cfg.actor.hidden_dims == [512, 256, 128]
    assert runner_cfg.critic.hidden_dims == [512, 256, 128]
    assert runner_cfg.actor.obs_normalization is True
    assert runner_cfg.critic.obs_normalization is True
    assert runner_cfg.actor.distribution_cfg.init_std == pytest.approx(0.45)
    contract = pick_insert_reset_dataset_task_contract(cfg)
    assert set(contract["static_scene"]) == {
        "ground_initial_state",
        "ground_spawn",
        "table_contact_initial_state",
        "table_contact_spawn",
        "table_initial_state",
        "table_spawn",
    }


def test_pick_insert_contract_is_identical_after_verified_local_asset_binding(tmp_path):
    cfg = FrankaRJ45PickInsertEnvCfg()
    diagnostic_contract = pick_insert_reset_dataset_task_contract(cfg)
    closure = FrankaRJ45AssetClosure(
        root=tmp_path,
        franka_usd_path=tmp_path / "private/franka.usda",
        seattle_table_usd_path=tmp_path / "private/table.usd",
        tree_sha256=FRANKA_RJ45_ASSET_CLOSURE_TREE_SHA256,
    )
    configure_franka_rj45_external_asset(cfg.scene.robot, closure)
    configure_seattle_table_external_asset(cfg.scene.table, closure)

    production_contract = pick_insert_reset_dataset_task_contract(cfg)

    assert production_contract == diagnostic_contract
    assert str(tmp_path) not in str(production_contract)


def test_pick_insert_environment_requires_verified_assets_before_base_startup(monkeypatch, tmp_path):
    cfg = FrankaRJ45PickInsertEnvCfg()
    closure = FrankaRJ45AssetClosure(
        root=tmp_path,
        franka_usd_path=tmp_path / "franka.usda",
        seattle_table_usd_path=tmp_path / "table.usd",
        tree_sha256=FRANKA_RJ45_ASSET_CLOSURE_TREE_SHA256,
    )
    calls: list[object] = []

    def _configured(*, required: bool = False):
        calls.append(("verify", required))
        return closure

    def _base_init(self, base_cfg, render_mode=None, **kwargs):
        calls.append(("base", base_cfg, render_mode, kwargs))

    monkeypatch.setattr(pick_insert_env_module, "configured_franka_rj45_asset_closure", _configured)
    monkeypatch.setattr(FrankaRJ45InsertionEnv, "__init__", _base_init)
    env = object.__new__(FrankaRJ45PickInsertEnv)

    env.__init__(cfg, render_mode="rgb_array", marker="sentinel")

    assert calls[0] == ("verify", True)
    assert calls[1] == ("base", cfg, "rgb_array", {"marker": "sentinel"})
    assert env._external_asset_closure is closure
    assert str(cfg.scene.robot.spawn.usd_path) == str(closure.franka_usd_path)
    assert str(cfg.scene.table.spawn.usd_path) == str(closure.seattle_table_usd_path)


def test_pick_insert_environment_fails_before_base_startup_without_configured_assets(monkeypatch):
    cfg = FrankaRJ45PickInsertEnvCfg()
    base_started = False

    def _missing(*, required: bool = False):
        assert required is True
        raise AssetClosureError(f"{FRANKA_RJ45_ASSET_CLOSURE_ENV} is required")

    def _base_init(*_args, **_kwargs):
        nonlocal base_started
        base_started = True

    monkeypatch.setattr(pick_insert_env_module, "configured_franka_rj45_asset_closure", _missing)
    monkeypatch.setattr(FrankaRJ45InsertionEnv, "__init__", _base_init)
    env = object.__new__(FrankaRJ45PickInsertEnv)

    with pytest.raises(AssetClosureError, match=FRANKA_RJ45_ASSET_CLOSURE_ENV):
        env.__init__(cfg)
    assert base_started is False


def test_pick_native_gravity_compensation_does_not_mutate_legacy_scene_config():
    legacy_before = FrankaRJ45InsertionEnvCfg()
    pick = FrankaRJ45PickInsertEnvCfg()
    legacy_after = FrankaRJ45InsertionEnvCfg()

    assert legacy_before.scene.robot.spawn.joint_drive_props is None
    assert legacy_after.scene.robot.spawn.joint_drive_props is None
    assert isinstance(legacy_before.actions.arm_action, mdp.ResetTargetEMARelativeJointPositionActionCfg)
    assert isinstance(legacy_after.actions.arm_action, mdp.ResetTargetEMARelativeJointPositionActionCfg)
    assert isinstance(pick.scene.robot.spawn.joint_drive_props, list)
    assert len(pick.scene.robot.spawn.joint_drive_props) == 1
    assert isinstance(pick.scene.robot.spawn.joint_drive_props[0], MujocoJointCfg)
    assert pick.scene.robot.spawn.joint_drive_props[0].actuatorgravcomp is True


def test_pick_insert_contract_changes_with_learning_semantics():
    baseline = reset_dataset_digest(pick_insert_reset_dataset_task_contract(FrankaRJ45PickInsertEnvCfg()))
    mutated = []

    cfg = FrankaRJ45PickInsertEnvCfg()
    cfg.reach_orientation_reward_scale_rad *= 1.1
    mutated.append(cfg)
    cfg = FrankaRJ45PickInsertEnvCfg()
    cfg.reach_orientation_reward_weight += 0.1
    mutated.append(cfg)
    cfg = FrankaRJ45PickInsertEnvCfg()
    cfg.is_finite_horizon = False
    mutated.append(cfg)
    cfg = FrankaRJ45PickInsertEnvCfg()
    cfg.terminations.success.func = mdp.stable_insertion_success
    mutated.append(cfg)
    cfg = FrankaRJ45PickInsertEnvCfg()
    cfg.actions.gripper_action.close_position = 0.001
    mutated.append(cfg)
    cfg = FrankaRJ45PickInsertEnvCfg()
    cfg.grasp_proxy_friction = 4.4
    mutated.append(cfg)
    cfg = FrankaRJ45PickInsertEnvCfg()
    cfg.success_max_plug_speed = 0.019
    mutated.append(cfg)
    cfg = FrankaRJ45PickInsertEnvCfg()
    cfg.grasp_acquisition_axis_tolerance_rad = math.radians(14.0)
    mutated.append(cfg)
    cfg = FrankaRJ45PickInsertEnvCfg()
    cfg.grasp_retention_axis_tolerance_rad = math.radians(24.0)
    mutated.append(cfg)
    cfg = FrankaRJ45PickInsertEnvCfg()
    cfg.grasp_proxy_face_tolerance_m = 4.0e-4
    mutated.append(cfg)
    cfg = FrankaRJ45PickInsertEnvCfg()
    cfg.reset_dataset_phase_fractions = (0.29, 0.16, 0.08, 0.06, 0.06, 0.35)
    mutated.append(cfg)

    assert all(reset_dataset_digest(pick_insert_reset_dataset_task_contract(cfg)) != baseline for cfg in mutated)


@pytest.mark.parametrize("field", ("neutral_position", "default_position"))
def test_pick_insert_config_rejects_open_gripper_contract_mismatch(field: str):
    cfg = FrankaRJ45PickInsertEnvCfg()
    setattr(cfg.actions.gripper_action, field, 0.018)

    with pytest.raises(ValueError, match="exact 0.04 m open neutral/default"):
        cfg.validate_config()


def test_bilateral_grasp_proxy_contacts_are_vectorized_per_world():
    # Shapes are left/right/proxy for world zero, then left/right/proxy for world one.
    shape_world = torch.tensor([0, 0, 0, 1, 1, 1])
    left = torch.tensor([True, False, False, True, False, False])
    right = torch.tensor([False, True, False, False, True, False])
    proxy = torch.tensor([False, False, True, False, False, True])
    shape0 = torch.tensor([0, 2, 3, 4, 99])
    shape1 = torch.tensor([2, 1, 5, 0, -1])
    point0 = torch.tensor([[0.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    point1 = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

    result = _bilateral_grasp_proxy_contact_mask(
        torch.tensor([5]),
        torch.arange(5),
        shape0,
        shape1,
        point0,
        point1,
        shape_world,
        left,
        right,
        proxy,
        proxy_center=torch.zeros(3),
        proxy_half_extents=torch.ones(3),
        proxy_face_tolerance_m=0.01,
        num_envs=2,
    )

    assert result.tolist() == [True, False]


def test_bilateral_grasp_proxy_contacts_reject_zero_count_and_overflow():
    shape_world = torch.tensor([0, 0, 0])
    left = torch.tensor([True, False, False])
    right = torch.tensor([False, True, False])
    proxy = torch.tensor([False, False, True])
    shape0 = torch.tensor([0, 1])
    shape1 = torch.tensor([2, 2])
    point0 = torch.zeros((2, 3))
    point1 = torch.tensor([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    slots = torch.arange(2)

    zero = _bilateral_grasp_proxy_contact_mask(
        torch.tensor([0]),
        slots,
        shape0,
        shape1,
        point0,
        point1,
        shape_world,
        left,
        right,
        proxy,
        proxy_center=torch.zeros(3),
        proxy_half_extents=torch.ones(3),
        proxy_face_tolerance_m=0.01,
        num_envs=1,
    )
    overflow = _bilateral_grasp_proxy_contact_mask(
        torch.tensor([3]),
        slots,
        shape0,
        shape1,
        point0,
        point1,
        shape_world,
        left,
        right,
        proxy,
        proxy_center=torch.zeros(3),
        proxy_half_extents=torch.ones(3),
        proxy_face_tolerance_m=0.01,
        num_envs=1,
    )

    assert zero.tolist() == [False]
    assert overflow.tolist() == [False]


def test_bilateral_grasp_proxy_contacts_allow_finger_swap_but_reject_other_faces():
    shape_world = torch.tensor([0, 0, 0, 1, 1, 1])
    left = torch.tensor([True, False, False, True, False, False])
    right = torch.tensor([False, True, False, False, True, False])
    proxy = torch.tensor([False, False, True, False, False, True])
    shape0 = torch.tensor([0, 1, 3, 4])
    shape1 = torch.tensor([2, 2, 5, 5])
    point0 = torch.zeros((4, 3))
    point1 = torch.tensor(
        [
            # X/Z edge witnesses are still contacts on the intended X faces.
            [-1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )

    result = _bilateral_grasp_proxy_contact_mask(
        torch.tensor([4]),
        torch.arange(4),
        shape0,
        shape1,
        point0,
        point1,
        shape_world,
        left,
        right,
        proxy,
        proxy_center=torch.zeros(3),
        proxy_half_extents=torch.ones(3),
        proxy_face_tolerance_m=0.01,
        num_envs=2,
    )

    assert result.tolist() == [True, False]


def test_stage_tracker_requires_proxy_contact_for_grasp_and_loss():
    gripper = type("Gripper", (), {"bilateral_contact": torch.tensor([True, True])})()
    proxy_contact = torch.tensor([False, True])

    def grasp_contact_alignment(_self, _tolerance, *, proxy_contact_mask_out, **_kwargs):
        proxy_contact_mask_out.copy_(proxy_contact)
        return proxy_contact

    env = type(
        "FakeEnv",
        (),
        {
            "cfg": type(
                "Cfg",
                (),
                {
                    "grasp_acquisition_distance_m": 0.03,
                    "grasp_acquisition_axis_tolerance_rad": math.radians(15.0),
                    "grasp_retention_axis_tolerance_rad": math.radians(25.0),
                    "max_tcp_grasp_distance": 0.08,
                    "transport_stage_distance_m": 0.08,
                    "preinsert_stage_distance_m": 0.035,
                },
            )(),
            "action_manager": type("Actions", (), {"get_term": lambda self, _name: gripper})(),
            "tcp_pose_e": lambda self: torch.zeros((2, 7)),
            "plug_grasp_position_e": lambda self: torch.zeros((2, 3)),
            "grasp_contact_alignment_mask": grasp_contact_alignment,
            "plug_goal_translation_error_local": lambda self: torch.ones((2, 3)),
            "insertion_success_mask": lambda self: torch.tensor([False, False]),
        },
    )()
    tracker = object.__new__(mdp.PickInsertStageContext)
    tracker.ever_grasped = torch.zeros(2, dtype=torch.bool)
    tracker.new_grasp = torch.zeros(2, dtype=torch.bool)
    tracker.proxy_contact = torch.zeros(2, dtype=torch.bool)
    tracker.current_grasp = torch.zeros(2, dtype=torch.bool)
    tracker.maximum_stage = torch.zeros(2, dtype=torch.long)
    tracker.loss_count = torch.zeros(2, dtype=torch.long)
    tracker._no_termination = torch.zeros(2, dtype=torch.bool)

    tracker(env)

    assert tracker.new_grasp.tolist() == [False, True]
    assert tracker.ever_grasped.tolist() == [False, True]
    assert tracker.proxy_contact.tolist() == [False, True]
    assert tracker.current_grasp.tolist() == [False, True]
    assert tracker.loss_count.tolist() == [0, 0]

    proxy_contact[:] = False
    tracker(env)

    assert tracker.proxy_contact.tolist() == [False, False]
    assert tracker.current_grasp.tolist() == [False, False]
    assert tracker.loss_count.tolist() == [0, 1]


def test_stage_tracker_uses_wider_axis_tolerance_only_after_acquisition():
    gripper = type("Gripper", (), {"bilateral_contact": torch.tensor([True, True])})()

    def grasp_contact_alignment(_self, tolerance, *, proxy_contact_mask_out, **_kwargs):
        proxy_contact_mask_out.fill_(True)
        return tolerance >= math.radians(20.0)

    env = type(
        "FakeEnv",
        (),
        {
            "cfg": type(
                "Cfg",
                (),
                {
                    "grasp_acquisition_distance_m": 0.03,
                    "grasp_acquisition_axis_tolerance_rad": math.radians(15.0),
                    "grasp_retention_axis_tolerance_rad": math.radians(25.0),
                    "max_tcp_grasp_distance": 0.08,
                    "transport_stage_distance_m": 0.08,
                    "preinsert_stage_distance_m": 0.035,
                },
            )(),
            "action_manager": type("Actions", (), {"get_term": lambda self, _name: gripper})(),
            "tcp_pose_e": lambda self: torch.zeros((2, 7)),
            "plug_grasp_position_e": lambda self: torch.zeros((2, 3)),
            "grasp_contact_alignment_mask": grasp_contact_alignment,
            "plug_goal_translation_error_local": lambda self: torch.ones((2, 3)),
            "insertion_success_mask": lambda self: torch.tensor([False, False]),
        },
    )()
    tracker = object.__new__(mdp.PickInsertStageContext)
    tracker.ever_grasped = torch.tensor([False, True])
    tracker.new_grasp = torch.zeros(2, dtype=torch.bool)
    tracker.proxy_contact = torch.zeros(2, dtype=torch.bool)
    tracker.current_grasp = torch.zeros(2, dtype=torch.bool)
    tracker.maximum_stage = torch.tensor([0, 1])
    tracker.loss_count = torch.zeros(2, dtype=torch.long)
    tracker._no_termination = torch.zeros(2, dtype=torch.bool)

    tracker(env)

    assert tracker.current_grasp.tolist() == [False, True]
    assert tracker.ever_grasped.tolist() == [False, True]
    assert tracker.loss_count.tolist() == [0, 0]


def test_stage_tracker_does_not_declare_open_far_reset_lost():
    tracker = object.__new__(mdp.PickInsertStageContext)
    tracker.ever_grasped = torch.tensor([False, True])
    tracker.loss_count = torch.tensor([100, 2])
    fake = type(
        "FakeEnv",
        (),
        {
            "cfg": type("Cfg", (), {"grasp_loss_grace_steps": 3})(),
            "episode_length_buf": torch.tensor([20, 20]),
            "pick_insert_stage_tracker": lambda self: tracker,
        },
    )()

    assert mdp.lost_acquired_grasp(fake, minimum_episode_steps=5).tolist() == [False, False]


def test_stage_tracker_does_not_credit_insertion_without_a_physical_grasp():
    gripper = type("Gripper", (), {"bilateral_contact": torch.tensor([False])})()

    def grasp_contact_alignment(_self, _tolerance, *, proxy_contact_mask_out, **_kwargs):
        proxy_contact_mask_out.fill_(False)
        return torch.zeros(1, dtype=torch.bool)

    env = type(
        "FakeEnv",
        (),
        {
            "cfg": type(
                "Cfg",
                (),
                {
                    "grasp_acquisition_distance_m": 0.03,
                    "grasp_acquisition_axis_tolerance_rad": math.radians(15.0),
                    "grasp_retention_axis_tolerance_rad": math.radians(25.0),
                    "max_tcp_grasp_distance": 0.08,
                    "transport_stage_distance_m": 0.08,
                    "preinsert_stage_distance_m": 0.035,
                },
            )(),
            "action_manager": type("Actions", (), {"get_term": lambda self, _name: gripper})(),
            "tcp_pose_e": lambda self: torch.zeros((1, 7)),
            "plug_grasp_position_e": lambda self: torch.zeros((1, 3)),
            "grasp_contact_alignment_mask": grasp_contact_alignment,
            "plug_goal_translation_error_local": lambda self: torch.zeros((1, 3)),
            "insertion_success_mask": lambda self: torch.tensor([True]),
        },
    )()
    tracker = object.__new__(mdp.PickInsertStageContext)
    tracker.ever_grasped = torch.zeros(1, dtype=torch.bool)
    tracker.new_grasp = torch.zeros(1, dtype=torch.bool)
    tracker.proxy_contact = torch.zeros(1, dtype=torch.bool)
    tracker.current_grasp = torch.zeros(1, dtype=torch.bool)
    tracker.maximum_stage = torch.zeros(1, dtype=torch.long)
    tracker.loss_count = torch.zeros(1, dtype=torch.long)
    tracker._no_termination = torch.zeros(1, dtype=torch.bool)

    tracker(env)

    assert tracker.maximum_stage.tolist() == [0]


def test_pick_insert_success_requires_prior_physical_grasp_for_full_dwell():
    tracker = type("Tracker", (), {"ever_grasped": torch.tensor([False, True])})()
    env = type(
        "FakeEnv",
        (),
        {
            "_success_dwell_count": torch.zeros(2, dtype=torch.long),
            "_success_dwell_steps": 2,
            "episode_succeeded": torch.zeros(2, dtype=torch.bool),
            "termination_manager": type("Terminations", (), {"terminated": torch.zeros(2, dtype=torch.bool)})(),
            "pick_insert_stage_tracker": lambda self: tracker,
            "insertion_success_mask": lambda self: torch.ones(2, dtype=torch.bool),
        },
    )()

    first = mdp.stable_pick_insert_success(env)
    second = mdp.stable_pick_insert_success(env)

    assert first.tolist() == [False, False]
    assert second.tolist() == [False, True]
    assert env._success_dwell_count.tolist() == [0, 2]
    assert env.episode_succeeded.tolist() == [False, True]


def test_stage_tracker_baselines_from_reset_geometry():
    env = type(
        "FakeEnv",
        (),
        {
            "reset_dataset_row_id": torch.tensor([0, 1]),
            "_reset_dataset_states": {"starts_grasped": torch.tensor([True, False])},
            "cfg": type(
                "Cfg",
                (),
                {"transport_stage_distance_m": 0.08, "preinsert_stage_distance_m": 0.035},
            )(),
            "plug_goal_translation_error_local": lambda self: torch.tensor([[0.0, 0.02, 0.0], [0.0, 0.01, 0.0]]),
            "insertion_success_mask": lambda self: torch.tensor([False, False]),
        },
    )()
    tracker = object.__new__(mdp.PickInsertStageContext)
    tracker._env = env
    tracker.ever_grasped = torch.zeros(2, dtype=torch.bool)
    tracker.new_grasp = torch.zeros(2, dtype=torch.bool)
    tracker.proxy_contact = torch.ones(2, dtype=torch.bool)
    tracker.current_grasp = torch.ones(2, dtype=torch.bool)
    tracker.maximum_stage = torch.zeros(2, dtype=torch.long)
    tracker.loss_count = torch.zeros(2, dtype=torch.long)

    tracker.reset()

    assert tracker.maximum_stage.tolist() == [3, 0]
    assert not bool(tracker.proxy_contact.any())
    assert not bool(tracker.current_grasp.any())


def test_progress_reward_suppresses_cross_episode_baseline(monkeypatch):
    reward = object.__new__(mdp.PickInsertProgressReward)
    reward._previous = torch.tensor([5.0, 1.0])
    reward._needs_baseline = torch.tensor([True, False])
    env = type("FakeEnv", (), {"step_dt": 0.5})()
    monkeypatch.setattr(pick_insert_rewards, "pick_insert_potential", lambda _env: torch.tensor([1.0, 2.0]))

    value = reward(env)

    assert value.tolist() == [0.0, 2.0]
    assert reward._previous.tolist() == [1.0, 2.0]
    assert not bool(reward._needs_baseline.any())


def test_progress_potential_is_finite_on_terminal_nonfinite_state():
    tracker = type("Tracker", (), {"ever_grasped": torch.tensor([False])})()
    env = type(
        "FakeEnv",
        (),
        {
            "cfg": type(
                "Cfg",
                (),
                {
                    "reach_reward_scale_m": 0.08,
                    "reach_orientation_reward_scale_rad": torch.pi / 4,
                    "reach_orientation_reward_weight": 0.25,
                    "transport_reward_scale_m": 0.12,
                    "insertion_reward_scale": 0.025,
                },
            )(),
            "tcp_pose_e": lambda self: torch.tensor([[float("nan"), 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]),
            "plug_grasp_position_e": lambda self: torch.zeros((1, 3)),
            "desired_tcp_grasp_pose_e": lambda self: torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]),
            "pick_insert_stage_tracker": lambda self: tracker,
            "plug_goal_translation_error_local": lambda self: torch.tensor([[float("inf"), 0.0, 0.0]]),
            "scalar_goal_error": lambda self: torch.tensor([float("nan")]),
        },
    )()

    assert torch.isfinite(mdp.pick_insert_potential(env)).all()


def test_progress_potential_does_not_reward_unrelated_finger_deflection():
    tracker = type("Tracker", (), {"ever_grasped": torch.tensor([False, True])})()
    env = type(
        "FakeEnv",
        (),
        {
            "cfg": type(
                "Cfg",
                (),
                {
                    "reach_reward_scale_m": 0.08,
                    "reach_orientation_reward_scale_rad": torch.pi / 4,
                    "reach_orientation_reward_weight": 0.25,
                    "transport_reward_scale_m": 0.12,
                    "insertion_reward_scale": 0.025,
                },
            )(),
            "tcp_pose_e": lambda self: torch.tensor(
                [[0.08, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [0.08, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]
            ),
            "plug_grasp_position_e": lambda self: torch.zeros((2, 3)),
            "desired_tcp_grasp_pose_e": lambda self: torch.tensor(
                [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]
            ),
            "pick_insert_stage_tracker": lambda self: tracker,
            "plug_goal_translation_error_local": lambda self: torch.zeros((2, 3)),
            "scalar_goal_error": lambda self: torch.zeros(2),
        },
    )()

    potential = mdp.pick_insert_potential(env)

    expected_reach = 0.75 * torch.exp(torch.tensor(-1.0)) + 0.25
    assert torch.isclose(potential[0], expected_reach)
    assert torch.isclose(potential[1] - potential[0], torch.tensor(5.0))


def test_progress_potential_rewards_observable_pregrasp_orientation_alignment():
    tracker = type("Tracker", (), {"ever_grasped": torch.tensor([False, False])})()
    rotated = math_utils.quat_from_euler_xyz(torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([torch.pi / 2]))[0]
    tcp_pose = torch.zeros((2, 7))
    tcp_pose[:, 0] = 0.08
    tcp_pose[0, 6] = 1.0
    tcp_pose[1, 3:7] = rotated
    target_pose = torch.zeros((2, 7))
    target_pose[:, 6] = 1.0
    env = type(
        "FakeEnv",
        (),
        {
            "cfg": type(
                "Cfg",
                (),
                {
                    "reach_reward_scale_m": 0.08,
                    "reach_orientation_reward_scale_rad": torch.pi / 4,
                    "reach_orientation_reward_weight": 0.25,
                    "transport_reward_scale_m": 0.12,
                    "insertion_reward_scale": 0.025,
                },
            )(),
            "tcp_pose_e": lambda self: tcp_pose,
            "plug_grasp_position_e": lambda self: torch.zeros((2, 3)),
            "desired_tcp_grasp_pose_e": lambda self: target_pose,
            "pick_insert_stage_tracker": lambda self: tracker,
            "plug_goal_translation_error_local": lambda self: torch.zeros((2, 3)),
            "scalar_goal_error": lambda self: torch.zeros(2),
        },
    )()

    potential = mdp.pick_insert_potential(env)

    expected_difference = 0.25 * (1.0 - torch.exp(torch.tensor(-2.0)))
    assert torch.isclose(potential[0] - potential[1], expected_difference, atol=1.0e-6)


def test_pick_insert_grasp_observations_expose_proxy_current_and_latched_state():
    tracker = type(
        "Tracker",
        (),
        {
            "proxy_contact": torch.tensor([True, False]),
            "current_grasp": torch.tensor([True, False]),
            "ever_grasped": torch.tensor([True, True]),
        },
    )()
    env = type(
        "FakeEnv",
        (),
        {
            "num_envs": 2,
            "device": torch.device("cpu"),
            "termination_manager": object(),
            "pick_insert_stage_tracker": lambda self: tracker,
        },
    )()

    assert mdp.grasp_proxy_contact_obs(env).tolist() == [[1.0], [0.0]]
    assert mdp.grasp_stage_obs(env).tolist() == [[1.0, 1.0], [0.0, 1.0]]


def test_pick_insert_cable_velocity_observation_uses_socket_frame_and_is_bounded():
    indices = torch.tensor((3, 10, 18, 25, 32, 39, 47))
    velocity = torch.zeros((2, 48, 6))
    velocity[0, indices, :3] = torch.tensor((100.0, float("nan"), -float("inf")))
    velocity[1, indices, 0] = 1.0
    socket = torch.zeros((2, 7))
    socket[0, 6] = 1.0
    socket[1, 3:7] = math_utils.quat_from_euler_xyz(
        torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([torch.pi / 2])
    )[0]
    env = type(
        "FakeEnv",
        (),
        {
            "num_envs": 2,
            "cfg": type("Cfg", (), {"max_task_body_linear_speed": 20.0})(),
            "_cable_observation_body_indices": indices,
            "task_body_velocity": lambda self: velocity,
            "socket_pose_e": lambda self: socket,
        },
    )()

    observation = mdp.sampled_cable_linear_velocities_obs(env).reshape(2, 7, 3)

    assert observation.shape == (2, 7, 3)
    assert torch.isfinite(observation).all()
    assert torch.allclose(observation[0], torch.tensor((20.0, 0.0, -20.0)).repeat(7, 1))
    assert torch.allclose(
        observation[1],
        torch.tensor((0.0, -1.0, 0.0)).repeat(7, 1),
        atol=1.0e-6,
    )


def _phase_partitioned_pick_insert_curriculum(
    pool_sizes: tuple[int, ...] = (2, 7, 1, 9, 3, 4),
) -> tuple[mdp.RJ45PickInsertResetDatasetCurriculum, torch.Tensor]:
    """Build the pure sampling portion without constructing a simulation manager."""
    curriculum = object.__new__(mdp.RJ45PickInsertResetDatasetCurriculum)
    phase_by_row = torch.repeat_interleave(torch.arange(6), torch.tensor(pool_sizes))
    curriculum._row_count = int(phase_by_row.numel())
    curriculum._phase_fractions = PICK_INSERT_RESET_PHASE_FRACTIONS
    curriculum._phase_rows = tuple(torch.where(phase_by_row == phase_id)[0] for phase_id in range(6))
    curriculum._phase_credits = [0.0] * 6
    curriculum._phase_uniform_credits = [0.0] * 6
    curriculum._phase_orders = [pool.clone() for pool in curriculum._phase_rows]
    curriculum._phase_cursors = [0] * 6
    curriculum._assigned_phase_counts = [0] * 6
    curriculum._assigned_phase_uniform_counts = [0] * 6
    curriculum._assigned_phase_adaptive_counts = [0] * 6
    curriculum._sampler = _ResetDatasetSampler(
        curriculum._row_count,
        "cpu",
        ResetDatasetSamplerCfg(uniform_fraction=0.35),
    )
    curriculum._metrics_cache = {}
    curriculum._assignments_since_metrics = 0
    return curriculum, phase_by_row


@pytest.mark.parametrize(
    ("sampling_mode", "curriculum_freeze", "expected_uniform_fraction"),
    (
        ("adaptive", False, 0.35),
        ("uniform", False, 1.0),
        ("adaptive", True, 1.0),
    ),
)
def test_pick_insert_curriculum_honors_phase_fractions_in_every_sampling_mode(
    sampling_mode: str,
    curriculum_freeze: bool,
    expected_uniform_fraction: float,
):
    curriculum, phase_by_row = _phase_partitioned_pick_insert_curriculum()
    count = 10_000
    env = SimpleNamespace(
        num_envs=count,
        device=torch.device("cpu"),
        cfg=SimpleNamespace(
            curriculum_freeze=curriculum_freeze,
            reset_dataset_sampling_mode=sampling_mode,
        ),
        episode_length_buf=torch.zeros(count, dtype=torch.long),
        reset_dataset_row_id=torch.full((count,), -1, dtype=torch.long),
    )

    metrics = curriculum(env, slice(None))
    realized_phases = phase_by_row[env.reset_dataset_row_id]

    assert torch.bincount(realized_phases, minlength=6).tolist() == [3000, 1500, 800, 600, 600, 3500]
    assert metrics["sampler/uniform_replay_fraction"] == pytest.approx(expected_uniform_fraction)
    assert metrics["sampler/adaptive_sampling_fraction"] == pytest.approx(1.0 - expected_uniform_fraction)
    for phase_id, fraction in enumerate(PICK_INSERT_RESET_PHASE_FRACTIONS):
        assert metrics[f"sampler/configured_phase_{phase_id}_assignment_fraction"] == fraction
        assert metrics[f"sampler/realized_phase_{phase_id}_assignment_fraction"] == pytest.approx(fraction)
        assert metrics[f"sampler/phase_{phase_id}_uniform_replay_fraction"] == pytest.approx(expected_uniform_fraction)
        assert metrics[f"sampler/phase_{phase_id}_adaptive_sampling_fraction"] == pytest.approx(
            1.0 - expected_uniform_fraction
        )


def test_pick_insert_uniform_phase_replay_cycles_each_pool_independently():
    curriculum, phase_by_row = _phase_partitioned_pick_insert_curriculum()

    rows = curriculum._sample_uniform_rows(10_000)

    realized_phases = phase_by_row[rows]
    for phase_id, pool in enumerate(curriculum._phase_rows):
        per_row_counts = torch.bincount(rows[realized_phases == phase_id] - int(pool[0]), minlength=pool.numel())
        assert int(per_row_counts.max() - per_row_counts.min()) <= 1


def test_pick_insert_phase_allocation_carries_fractional_credit_across_async_batches():
    curriculum, _ = _phase_partitioned_pick_insert_curriculum()
    totals = [0] * 6
    chunk_sizes = (1, 2, 1, 3, 7, 13, 29) * 100
    chunk_sizes += (10_000 - sum(chunk_sizes),)

    for count in chunk_sizes:
        phase_counts = curriculum._phase_assignment_counts(count)
        assert min(phase_counts) >= 0
        assert sum(phase_counts) == count
        totals = [total + assigned for total, assigned in zip(totals, phase_counts, strict=True)]

    assert totals == [3000, 1500, 800, 600, 600, 3500]


def test_pick_insert_adaptive_phase_replay_reports_actual_assignment_mix():
    curriculum, phase_by_row = _phase_partitioned_pick_insert_curriculum()

    rows = curriculum._sample_training_rows(10_000)
    metrics = curriculum._sampling_metrics(adaptive_enabled=True)

    assert bool(((rows >= 0) & (rows < curriculum._row_count)).all())
    assert torch.bincount(phase_by_row[rows], minlength=6).tolist() == [3000, 1500, 800, 600, 600, 3500]
    assert metrics["sampler/configured_uniform_replay_fraction"] == 0.35
    assert metrics["sampler/uniform_replay_fraction"] == pytest.approx(0.35)
    assert metrics["sampler/adaptive_sampling_fraction"] == pytest.approx(0.65)
