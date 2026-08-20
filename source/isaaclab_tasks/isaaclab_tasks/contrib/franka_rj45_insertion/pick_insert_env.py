# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Randomized reset-driven Franka pickup and insertion of Newton's RJ45 cable."""

from __future__ import annotations

import json
import logging

import torch
import warp as wp

from isaaclab.utils import math as math_utils

from isaaclab_contrib.coupling import NewtonCouplerManager

from .asset_provenance import configured_franka_rj45_asset_closure, franka_rj45_asset_contract
from .franka_robot_cfg import configure_franka_rj45_external_asset
from .pick_insert_reset_dataset_io import (
    FRANKA_RJ45_PICK_INSERT_FAST_RESET_POLICY,
    FRANKA_RJ45_PICK_INSERT_FAST_RESET_VALIDATION_FORMAT,
    FRANKA_RJ45_PICK_INSERT_RESET_VALIDATION_POLICY,
    RESET_DATASET_STATE_NAMES,
    fast_reset_validation_report_validate_runtime,
    franka_rj45_validation_source_sha256,
    reset_dataset_validate_full_pick_diversity,
    reset_dataset_validate_phase_row_counts,
    reset_dataset_validate_runtime,
    reset_validation_report_validate_runtime,
)
from .rj45_env import (
    FrankaRJ45InsertionEnv,
    _resolve_reset_dataset_path,
    _resolve_reset_validation_report_path,
)
from .rj45_env_cfg import RIGID_ENTRY, RJ45_ENTRY
from .table_scene_cfg import configure_seattle_table_external_asset
from .task_success import rj45_insertion_success

logger = logging.getLogger(__name__)


def _bilateral_grasp_proxy_contact_mask(
    contact_count: torch.Tensor,
    contact_slots: torch.Tensor,
    shape0: torch.Tensor,
    shape1: torch.Tensor,
    shape_world: torch.Tensor,
    left_finger_shape: torch.Tensor,
    right_finger_shape: torch.Tensor,
    grasp_proxy_shape: torch.Tensor,
    num_envs: int,
) -> torch.Tensor:
    """Return worlds with simultaneous left- and right-finger grasp-proxy contacts."""
    capacity = contact_slots.numel()
    result = torch.zeros(num_envs, device=contact_slots.device, dtype=torch.bool)
    if capacity == 0 or shape_world.numel() == 0:
        return result

    shape0 = shape0[:capacity].long()
    shape1 = shape1[:capacity].long()
    shape_count = shape_world.numel()
    valid_shape = (shape0 >= 0) & (shape0 < shape_count) & (shape1 >= 0) & (shape1 < shape_count)
    safe_shape0 = shape0.clamp(0, shape_count - 1)
    safe_shape1 = shape1.clamp(0, shape_count - 1)
    world0 = shape_world[safe_shape0]
    world1 = shape_world[safe_shape1]
    world = torch.where(world0 >= 0, world0, world1)
    valid_world = (world >= 0) & (world < num_envs)
    safe_world = world.clamp(0, max(num_envs - 1, 0))

    active_count = contact_count.reshape(-1)[0].long().clamp(0, capacity)
    active = contact_slots < active_count
    valid_contact = active & valid_shape & valid_world
    left_contact = (left_finger_shape[safe_shape0] & grasp_proxy_shape[safe_shape1]) | (
        left_finger_shape[safe_shape1] & grasp_proxy_shape[safe_shape0]
    )
    right_contact = (right_finger_shape[safe_shape0] & grasp_proxy_shape[safe_shape1]) | (
        right_finger_shape[safe_shape1] & grasp_proxy_shape[safe_shape0]
    )
    left_count = torch.zeros(num_envs, device=contact_slots.device, dtype=torch.long)
    right_count = torch.zeros_like(left_count)
    left_count.scatter_add_(0, safe_world, (valid_contact & left_contact).long())
    right_count.scatter_add_(0, safe_world, (valid_contact & right_contact).long())
    no_overflow = contact_count.reshape(-1)[0] <= capacity
    return (left_count > 0) & (right_count > 0) & no_overflow


class FrankaRJ45PickInsertEnv(FrankaRJ45InsertionEnv):
    """Approach, grasp, transport, align, and fully seat a randomized RJ45 plug."""

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        # Match ManagerBasedEnv's partial-initialization guard before doing the
        # task-specific preflight, so a verification failure is cleanly inert.
        self._is_closed = True
        closure = configured_franka_rj45_asset_closure(required=True)
        configure_franka_rj45_external_asset(cfg.scene.robot, closure)
        configure_seattle_table_external_asset(cfg.scene.table, closure)
        self._external_asset_closure = closure
        super().__init__(cfg, render_mode, **kwargs)

    def _create_rj45_builder(self, cfg):
        from .physics import Rj45NewtonAssemblyBuilder
        from .pick_insert_env_cfg import pick_insert_topology_cfg

        return Rj45NewtonAssemblyBuilder(
            topology_cfg=pick_insert_topology_cfg(cfg),
            task_translation=cfg.task_translation,
            task_rotation_xyzw=cfg.task_rotation_xyzw,
            grasp_proxy_friction=cfg.grasp_proxy_friction,
        )

    def _add_rj45_world_to_builder(self, builder, env_id: int, position, quaternion) -> None:
        """Let the topology-aware builder compose task and replicated-world transforms."""
        self._rj45_builder.world_hook(builder, env_id, position, quaternion)

    def _bind_rj45_physics_ready(self, payload=None) -> None:
        """Bind pick-only direct-VBD cable input/alignment semantics."""
        old_handle = getattr(self, "_rj45_preserved_input_projection_handle", None)
        if old_handle is not None:
            old_handle.deregister()
        super()._bind_rj45_physics_ready(payload)
        runtime = self._ensure_rj45_runtime()
        self._rj45_preserved_input_projection_handle = (
            NewtonCouplerManager.register_vbd_preserved_input_pose_projection(
                name="franka_rj45_pick_insert_cable_alignment",
                entry_name=RJ45_ENTRY,
                body_ids=runtime.cable_preserved_input_body_ids,
                callback=runtime.align_after_step,
            )
        )

    def _prepare_rj45_substep(self, state) -> None:
        """Apply forces and synchronize anchors before the coupled solve.

        The registered coupler projection aligns cable capsules immediately
        after every solve and before state swap/contact generation.
        """
        self._ensure_rj45_runtime().prepare_step(state)

    def _align_rj45_after_step(self) -> None:
        """Skip the legacy end-of-outer-step alignment; every pick substep is projected."""

    def _clear_rj45_callbacks(self) -> None:
        """Release the pick-only projection together with inherited callbacks."""
        handle = getattr(self, "_rj45_preserved_input_projection_handle", None)
        if handle is not None:
            handle.deregister()
            self._rj45_preserved_input_projection_handle = None
        super()._clear_rj45_callbacks()

    def _bind_physics_state(self) -> None:
        super()._bind_physics_state()
        if self._task_layout.socket_body_index is None:
            raise RuntimeError("RJ45 pick-insert requires a resettable socket body.")
        self._plug_grasp_orientation = torch.as_tensor(
            self.cfg.plug_grasp_orientation_xyzw,
            device=self.device,
            dtype=torch.float32,
        ).repeat(self.num_envs, 1)
        self._bind_grasp_proxy_contacts()

    def _pose_history_selection(self, env_ids: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ordered parent-body and world IDs for selected task worlds."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        else:
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long).flatten()
        if env_ids.numel() == 0:
            raise ValueError("VBD pose-history selection must contain at least one environment.")
        out_of_range = bool(torch.any((env_ids < 0) | (env_ids >= self.num_envs)))
        if out_of_range or torch.unique(env_ids).numel() != env_ids.numel():
            raise ValueError("VBD pose-history environment IDs must be unique and in range.")
        body_ids = self._task_body_ids[env_ids].reshape(-1).to(dtype=torch.int32).contiguous()
        world_ids = env_ids.to(dtype=torch.int32).contiguous()
        return body_ids, world_ids

    def snapshot_task_pose_history_e(
        self,
        env_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Capture both VBD pose-history buffers in exact task order and local frames."""
        selected = (
            torch.arange(self.num_envs, device=self.device, dtype=torch.long)
            if env_ids is None
            else torch.as_tensor(env_ids, device=self.device, dtype=torch.long).flatten()
        )
        body_ids, world_ids = self._pose_history_selection(selected)
        previous_w, coupling_previous_w = NewtonCouplerManager.capture_vbd_pose_history(
            RJ45_ENTRY,
            wp.from_torch(body_ids, dtype=wp.int32),
            wp.from_torch(world_ids, dtype=wp.int32),
        )
        shape = (selected.numel(), self._task_layout.body_count, 7)
        previous_e = wp.to_torch(previous_w).clone().reshape(shape)
        coupling_previous_e = wp.to_torch(coupling_previous_w).clone().reshape(shape)
        origins = self.env_origins[selected, None, :]
        previous_e[..., :3] -= origins
        coupling_previous_e[..., :3] -= origins
        return previous_e, coupling_previous_e

    def restore_task_pose_history_e(
        self,
        previous_pose_e: torch.Tensor,
        coupling_previous_pose_e: torch.Tensor,
        env_ids: torch.Tensor | None = None,
    ) -> dict[str, object]:
        """Queue both VBD histories for one-shot application at the first solve."""
        selected = (
            torch.arange(self.num_envs, device=self.device, dtype=torch.long)
            if env_ids is None
            else torch.as_tensor(env_ids, device=self.device, dtype=torch.long).flatten()
        )
        body_ids, world_ids = self._pose_history_selection(selected)
        expected = (selected.numel(), self._task_layout.body_count, 7)
        previous_pose_e = torch.as_tensor(previous_pose_e, device=self.device, dtype=torch.float32)
        coupling_previous_pose_e = torch.as_tensor(
            coupling_previous_pose_e,
            device=self.device,
            dtype=torch.float32,
        )
        if tuple(previous_pose_e.shape) != expected or tuple(coupling_previous_pose_e.shape) != expected:
            raise ValueError(
                "VBD pose history must have exact selected task layout shape "
                f"{expected}, got {tuple(previous_pose_e.shape)}/{tuple(coupling_previous_pose_e.shape)}."
            )
        previous_w = previous_pose_e.clone()
        coupling_previous_w = coupling_previous_pose_e.clone()
        origins = self.env_origins[selected, None, :]
        previous_w[..., :3] += origins
        coupling_previous_w[..., :3] += origins
        request = NewtonCouplerManager.queue_vbd_pose_history_restore(
            RJ45_ENTRY,
            wp.from_torch(body_ids, dtype=wp.int32),
            wp.from_torch(world_ids, dtype=wp.int32),
            wp.from_torch(previous_w.reshape(-1, 7).contiguous(), dtype=wp.transform),
            wp.from_torch(coupling_previous_w.reshape(-1, 7).contiguous(), dtype=wp.transform),
        )
        body_order_exact = request.body_ids == tuple(int(value) for value in body_ids.detach().cpu().tolist())
        world_order_exact = request.world_ids == tuple(int(value) for value in world_ids.detach().cpu().tolist())
        body_counts_exact = request.expected_body_counts == (self._task_layout.body_count,) * selected.numel()
        pending_worlds = set(request.pending_world_ids)
        evidence: dict[str, object] = {
            "entry_name": RJ45_ENTRY,
            "body_count": self._task_layout.body_count,
            "body_order": tuple(self._task_layout.body_names),
            "generation": request.generation,
            "restore_queued": torch.full(
                (selected.numel(),),
                request.entry_name == RJ45_ENTRY and body_order_exact and world_order_exact and body_counts_exact,
                device=self.device,
                dtype=torch.bool,
            ),
            "pending_at_queue": torch.as_tensor(
                [int(world) in pending_worlds for world in request.world_ids],
                device=self.device,
                dtype=torch.bool,
            ),
            "previous_pose_queued": torch.ones(selected.numel(), device=self.device, dtype=torch.bool),
            "coupling_previous_pose_queued": torch.ones(selected.numel(), device=self.device, dtype=torch.bool),
            "body_order_exact": body_order_exact,
            "world_order_exact": world_order_exact,
            "applied_exactly_once": None,
            "failed": None,
            "superseded": None,
            "pending_after_first_solve": None,
            "application_count_delta": None,
            "expected_body_count": torch.as_tensor(
                request.expected_body_counts,
                device=self.device,
                dtype=torch.int64,
            ),
            "body_application_count_delta": None,
            "_request": request,
        }
        pending = getattr(self, "_pending_task_pose_history_restores", None)
        if pending is None:
            pending = []
            self._pending_task_pose_history_restores = pending
        pending.append(evidence)
        return evidence

    def finalize_pending_task_pose_history_restores(self, *, require_complete: bool = True) -> None:
        """Finalize queued history evidence after a real coupled solve."""
        pending = getattr(self, "_pending_task_pose_history_restores", None)
        if not pending:
            return
        still_pending: list[dict[str, object]] = []
        for evidence in pending:
            status = NewtonCouplerManager.get_vbd_pose_history_restore_status(evidence["_request"])
            pending_worlds = set(status.pending_world_ids)
            applied_worlds = set(status.applied_world_ids)
            failed_worlds = set(status.failed_world_ids)
            superseded_worlds = set(status.superseded_world_ids)
            application_counts = torch.as_tensor(
                status.application_count_deltas,
                device=self.device,
                dtype=torch.int64,
            )
            evidence["pending_after_first_solve"] = torch.as_tensor(
                [world in pending_worlds for world in status.world_ids],
                device=self.device,
                dtype=torch.bool,
            )
            evidence["failed"] = torch.as_tensor(
                [world in failed_worlds for world in status.world_ids],
                device=self.device,
                dtype=torch.bool,
            )
            evidence["superseded"] = torch.as_tensor(
                [world in superseded_worlds for world in status.world_ids],
                device=self.device,
                dtype=torch.bool,
            )
            evidence["application_count_delta"] = application_counts
            evidence["body_application_count_delta"] = torch.as_tensor(
                status.body_application_count_deltas,
                device=self.device,
                dtype=torch.int64,
            )
            evidence["applied_exactly_once"] = torch.as_tensor(
                [
                    world in applied_worlds
                    and world not in failed_worlds
                    and world not in superseded_worlds
                    and int(count) == 1
                    and int(body_count) == int(expected_body_count)
                    for world, count, body_count, expected_body_count in zip(
                        status.world_ids,
                        status.application_count_deltas,
                        status.body_application_count_deltas,
                        status.expected_body_counts,
                        strict=True,
                    )
                ],
                device=self.device,
                dtype=torch.bool,
            )
            if status.pending:
                still_pending.append(evidence)
                if require_complete:
                    raise RuntimeError(
                        "RJ45 VBD pose-history restore remained pending after the first real coupled solve: "
                        f"entry={status.entry_name!r}, generation={status.generation}, "
                        f"worlds={status.pending_world_ids}."
                    )
            elif not status.applied_exactly_once or not bool(torch.as_tensor(evidence["applied_exactly_once"]).all()):
                raise RuntimeError(
                    "RJ45 VBD pose-history restore was not applied exactly once: "
                    f"entry={status.entry_name!r}, generation={status.generation}, "
                    f"applied={status.applied_world_ids}, failed={status.failed_world_ids}, "
                    f"superseded={status.superseded_world_ids}, counts={status.application_count_deltas}, "
                    f"body_counts={status.body_application_count_deltas}/{status.expected_body_counts}."
                )
        self._pending_task_pose_history_restores = still_pending

    def _bind_grasp_proxy_contacts(self) -> None:
        contacts, destination_view, _ = NewtonCouplerManager.get_proxy_contact_data(
            RIGID_ENTRY,
            RJ45_ENTRY,
        )
        if contacts is None or contacts.rigid_contact_count is None:
            raise RuntimeError("RJ45 pick-insert requires a proxy-local rigid contact buffer.")

        shape_count = int(destination_view.shape_count)
        shape_labels = [str(label) for label in destination_view.shape_label[:shape_count]]
        body_labels = [str(label) for label in destination_view.body_label]
        if not any(label.endswith("/TableContactSurface") for label in body_labels):
            raise RuntimeError(
                "RJ45 pick-insert requires the kinematic Seattle TableContactSurface body in the VBD entry."
            )
        shape_body = wp.to_torch(destination_view.shape_body).detach().cpu().tolist()
        combined_labels = [
            f"{body_labels[body_id] if 0 <= body_id < len(body_labels) else ''} {shape_label}"
            for body_id, shape_label in zip(shape_body, shape_labels, strict=True)
        ]
        label_masks = {
            "left finger": ["panda_leftfinger" in label for label in combined_labels],
            "right finger": ["panda_rightfinger" in label for label in combined_labels],
            "grasp proxy": [label.endswith("/Plug/GraspProxy") for label in shape_labels],
        }
        missing = [name for name, mask in label_masks.items() if not any(mask)]
        if missing:
            raise RuntimeError(
                "RJ45 pick-insert contact layout is missing required shape labels: " + ", ".join(missing)
            )

        self._grasp_contact_count = wp.to_torch(contacts.rigid_contact_count)
        self._grasp_contact_shape0 = wp.to_torch(contacts.rigid_contact_shape0)
        self._grasp_contact_shape1 = wp.to_torch(contacts.rigid_contact_shape1)
        self._grasp_contact_slots = torch.arange(
            int(contacts.rigid_contact_max),
            device=self.device,
            dtype=torch.long,
        )
        self._grasp_contact_shape_world = wp.to_torch(destination_view.shape_world).long()
        self._left_finger_contact_shape = torch.as_tensor(
            label_masks["left finger"], device=self.device, dtype=torch.bool
        )
        self._right_finger_contact_shape = torch.as_tensor(
            label_masks["right finger"], device=self.device, dtype=torch.bool
        )
        self._grasp_proxy_contact_shape = torch.as_tensor(
            label_masks["grasp proxy"], device=self.device, dtype=torch.bool
        )

    def bilateral_grasp_proxy_contact_mask(self) -> torch.Tensor:
        """Return environments where both fingers contact the plug grasp proxy."""
        return _bilateral_grasp_proxy_contact_mask(
            self._grasp_contact_count,
            self._grasp_contact_slots,
            self._grasp_contact_shape0,
            self._grasp_contact_shape1,
            self._grasp_contact_shape_world,
            self._left_finger_contact_shape,
            self._right_finger_contact_shape,
            self._grasp_proxy_contact_shape,
            self.num_envs,
        )

    def _load_reset_dataset(self, configured_path: str) -> None:
        path = _resolve_reset_dataset_path(configured_path)
        payload = torch.load(path, map_location="cpu", weights_only=True)
        from .pick_insert_env_cfg import pick_insert_reset_dataset_task_contract

        task_contract = pick_insert_reset_dataset_task_contract(self.cfg)
        metadata, states, canonical_goal = reset_dataset_validate_runtime(
            payload,
            expected_content_sha256=self.cfg.reset_dataset_content_sha256,
            expected_task_contract=task_contract,
        )
        phase_counts = reset_dataset_validate_phase_row_counts(
            states["phase"],
            expected_rows_per_phase=self.cfg.reset_dataset_rows_per_phase,
        )
        diversity_evidence = reset_dataset_validate_full_pick_diversity(
            states,
            task_contract=task_contract,
        )
        report_path = _resolve_reset_validation_report_path(self.cfg.reset_validation_report_path)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        validation_kwargs = {
            "expected_content_sha256": payload["content_sha256"],
            "expected_row_count": len(states["phase"]),
            "expected_phases": states["phase"],
            "expected_task_contract": task_contract,
            "expected_asset_closure": franka_rj45_asset_contract(),
            "expected_full_pick_diversity": diversity_evidence,
        }
        fast_screened = report.get("format") == FRANKA_RJ45_PICK_INSERT_FAST_RESET_VALIDATION_FORMAT
        if fast_screened:
            fast_reset_validation_report_validate_runtime(
                report,
                expected_validation_policy=FRANKA_RJ45_PICK_INSERT_FAST_RESET_POLICY,
                expected_source_sha256=franka_rj45_validation_source_sha256(include_fast_validator=True),
                **validation_kwargs,
            )
        else:
            reset_validation_report_validate_runtime(
                report,
                expected_validation_policy=FRANKA_RJ45_PICK_INSERT_RESET_VALIDATION_POLICY,
                expected_source_sha256=franka_rj45_validation_source_sha256(),
                **validation_kwargs,
            )
        self._reset_dataset_states = {
            name: states[name].to(device=self.device, non_blocking=True) for name in RESET_DATASET_STATE_NAMES
        }
        self.canonical_goal_task_body_pose = canonical_goal["task_body_pose"].to(device=self.device, non_blocking=True)
        self.canonical_goal_task_body_previous_pose = canonical_goal["task_body_previous_pose"].to(
            device=self.device, non_blocking=True
        )
        self.canonical_goal_task_body_coupling_previous_pose = canonical_goal["task_body_coupling_previous_pose"].to(
            device=self.device, non_blocking=True
        )
        self.canonical_goal_task_body_velocity = canonical_goal["task_body_velocity"].to(
            device=self.device, non_blocking=True
        )
        # Observation terms are queried once while managers infer their output
        # dimensions, before the first curriculum/event reset has selected rows.
        # Seed every world with the canonical goal so that pre-reset manager
        # construction never reads uninitialized goal-conditioned inputs.
        self.goal_task_body_pose = self.canonical_goal_task_body_pose.unsqueeze(0).repeat(self.num_envs, 1, 1)
        logger.info(
            "Loaded %d %s six-stage RJ45 pick-insert resets from %s "
            "(phase_counts=%s, full_pick_unique=%s/%s/%s, evidence=%s, generator=%s).",
            len(states["phase"]),
            "fast-screened" if fast_screened else "physically validated",
            path,
            phase_counts,
            diversity_evidence["unique_socket_rows"],
            diversity_evidence["unique_plug_rows"],
            diversity_evidence["unique_arm_rows"],
            report_path,
            metadata.get("generator", "unknown"),
        )

    def _write_task_state(self, env_ids: torch.Tensor, rows: torch.Tensor) -> None:
        self.goal_task_body_pose[env_ids] = self._reset_dataset_states["goal_task_body_pose"][rows]
        super()._write_task_state(env_ids, rows)

    def _stage_task_pose_history_restore(self, env_ids: torch.Tensor, rows: torch.Tensor) -> None:
        """Stage the latest validated histories until the next policy step."""
        shape = (self.num_envs, self._task_layout.body_count, 7)
        if not hasattr(self, "_task_previous_pose_staging"):
            self._task_previous_pose_staging = torch.empty(shape, device=self.device, dtype=torch.float32)
            self._task_coupling_previous_pose_staging = torch.empty_like(self._task_previous_pose_staging)
            self._task_pose_history_staging_mask = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._task_previous_pose_staging[env_ids] = self._reset_dataset_states["task_body_previous_pose"][rows]
        self._task_coupling_previous_pose_staging[env_ids] = self._reset_dataset_states[
            "task_body_coupling_previous_pose"
        ][rows]
        self._task_pose_history_staging_mask[env_ids] = True

    def _queue_staged_task_pose_history_restore(self) -> dict[str, object] | None:
        """Materialize public state and queue all latest staged histories."""
        staging_mask = getattr(self, "_task_pose_history_staging_mask", None)
        if staging_mask is None or not bool(staging_mask.any()):
            return None
        env_ids = torch.where(staging_mask)[0]
        self.finalize_pending_task_pose_history_restores(require_complete=True)
        self.scene.write_data_to_sim()
        self.sim.forward()
        evidence = self.restore_task_pose_history_e(
            self._task_previous_pose_staging[env_ids],
            self._task_coupling_previous_pose_staging[env_ids],
            env_ids,
        )
        queued = (
            bool(torch.as_tensor(evidence["restore_queued"]).all())
            and bool(torch.as_tensor(evidence["pending_at_queue"]).all())
            and bool(torch.as_tensor(evidence["previous_pose_queued"]).all())
            and bool(torch.as_tensor(evidence["coupling_previous_pose_queued"]).all())
            and evidence["body_order_exact"] is True
            and evidence["world_order_exact"] is True
        )
        if not queued:
            raise RuntimeError("RJ45 reset failed to queue both VBD histories in exact task/world order.")
        staging_mask[env_ids] = False
        return evidence

    def reset_rj45_scene(self, env_ids: torch.Tensor) -> None:
        """Restore public state and stage the latest serialized VBD history."""
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long).flatten()
        # A prior policy step may have applied a ticket before same-step
        # autoreset reaches this event.  Verify it before staging the next row.
        self.finalize_pending_task_pose_history_restores(require_complete=True)
        super().reset_rj45_scene(env_ids)
        rows = self.reset_dataset_row_id[env_ids]
        self._stage_task_pose_history_restore(env_ids, rows)

    def step(self, action: torch.Tensor):
        """Queue staged reset history at the last boundary before physics."""
        queued = self._queue_staged_task_pose_history_restore()
        result = super().step(action)
        if queued is not None:
            self.finalize_pending_task_pose_history_restores(require_complete=True)
        return result

    def reset_to(
        self,
        state,
        env_ids,
        seed: int | None = None,
        is_relative: bool = False,
    ):
        """Reject public-state resets that omit the two required VBD histories."""
        del state, env_ids, seed, is_relative
        raise RuntimeError(
            "Franka RJ45 pick-insert reset_to() is unsupported because a public scene state does not include "
            "the two validated VBD pose-history buffers. Use the task reset dataset or a history-aware reset tool."
        )

    def socket_pose_e(self) -> torch.Tensor:
        """Current resettable socket pose in the environment frame."""
        assert self._socket_task_body_index is not None
        return self.task_body_pose_e()[:, self._socket_task_body_index]

    def goal_plug_pose_e(self) -> torch.Tensor:
        return self.goal_task_body_pose[:, self._plug_task_body_index]

    def plug_goal_translation_error(self) -> torch.Tensor:
        """World-frame plug error for diagnostics."""
        return self.plug_pose_e()[:, :3] - self.goal_plug_pose_e()[:, :3]

    def plug_goal_translation_error_local(self) -> torch.Tensor:
        """Plug translation error expressed in each randomized goal frame."""
        goal = self.goal_plug_pose_e()
        return math_utils.quat_apply_inverse(goal[:, 3:7], self.plug_pose_e()[:, :3] - goal[:, :3])

    def plug_goal_orientation_error_axis_angle(self) -> torch.Tensor:
        goal = self.goal_plug_pose_e()
        error = math_utils.quat_unique(
            math_utils.quat_mul(math_utils.quat_conjugate(goal[:, 3:7]), self.plug_pose_e()[:, 3:7])
        )
        return math_utils.axis_angle_from_quat(error)

    def plug_orientation_error(self) -> torch.Tensor:
        return torch.linalg.vector_norm(self.plug_goal_orientation_error_axis_angle(), dim=-1)

    def goal_error_components(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        position_error = self.plug_goal_translation_error_local()
        axial = position_error[:, 1].abs()
        radial_xz = position_error[:, (0, 2)]
        latch = self.latch_pose_e()
        goal_latch = self.goal_task_body_pose[:, self._latch_task_body_index]
        latch_error = math_utils.quat_unique(
            math_utils.quat_mul(math_utils.quat_conjugate(goal_latch[:, 3:7]), latch[:, 3:7])
        )
        latch_angle = torch.linalg.vector_norm(math_utils.axis_angle_from_quat(latch_error), dim=-1)
        return axial, radial_xz, latch_angle

    def scalar_goal_error(self) -> torch.Tensor:
        axial, radial_xz, latch_angle = self.goal_error_components()
        plug_angle = self.plug_orientation_error()
        return axial + 2.0 * torch.linalg.vector_norm(radial_xz, dim=-1) + 0.01 * plug_angle + 0.002 * latch_angle

    def insertion_success_mask(self) -> torch.Tensor:
        return rj45_insertion_success(
            self.task_body_pose_e(),
            self.task_body_velocity(),
            self.goal_task_body_pose,
            axial_tolerance=self.cfg.success_axial_tolerance,
            axial_overtravel_tolerance=self.cfg.success_axial_overtravel_tolerance,
            radial_tolerance=self.cfg.success_radial_tolerance,
            plug_angle_tolerance=self.cfg.success_plug_angle_tolerance,
            latch_angle_tolerance=self.cfg.success_latch_angle_tolerance,
            maximum_plug_spatial_speed=self.cfg.success_max_plug_speed,
            plug_body_index=self._plug_task_body_index,
            latch_body_index=self._latch_task_body_index,
        ).mask

    def desired_tcp_grasp_pose_e(self) -> torch.Tensor:
        plug = self.plug_pose_e()
        position = self.plug_grasp_position_e()
        orientation = math_utils.quat_unique(math_utils.quat_mul(plug[:, 3:7], self._plug_grasp_orientation))
        return torch.cat((position, orientation), dim=-1)

    def tcp_velocity_e(self) -> torch.Tensor:
        """TCP spatial velocity; the rigid offset correction is negligible for policy input."""
        return self._robot.data.body_link_vel_w.torch[:, self._tcp_body_idx]

    def pick_insert_stage_tracker(self):
        """Return the stateful no-termination term shared by reward, failures, and observations."""
        return self.termination_manager.get_term_cfg("stage_context").func

    def phase_progress_error(self) -> torch.Tensor:
        """Use reach error for open starts and goal error after grasp acquisition."""
        rows = self.reset_dataset_row_id.clamp_min(0)
        phase = self._reset_dataset_states["phase"][rows]
        reach = torch.linalg.vector_norm(self.tcp_pose_e()[:, :3] - self.plug_grasp_position_e(), dim=-1)
        return torch.where(phase >= 4, reach, self.scalar_goal_error())


__all__ = ["FrankaRJ45PickInsertEnv"]
