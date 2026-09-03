# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Generate the balanced dual-rack, two-ended-cable RJ45 reset bank.

The durable row schema and six-stage curriculum are shared with the production
pick-insert task.  Only the task-owned cable construction differs: every row
keeps the four free-plug anchors exact, routes all remaining authored rest
lengths to the permanently seated second plug, then admits the row through the
same batched IK and zero-step Newton collision query as the reference bank.

Canonical goal certification still uses real coupled physics.  Fast row
generation never integrates physics and never claims dynamic replay evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import generate_franka_rj45_pick_insert_reset_dataset as base
import torch
import warp as wp
from _franka_rj45_reset_tools import _RJ45ResetToolMixin, save_torch_atomic

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.utils import math as math_utils

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.contrib.franka_rj45_insertion.dual_rack_env import FrankaRJ45DualRackInsertEnv
from isaaclab_tasks.contrib.franka_rj45_insertion.dual_rack_env_cfg import (
    FrankaRJ45DualRackInsertEnvCfg,
    dual_rack_reset_dataset_task_contract,
)
from isaaclab_tasks.contrib.franka_rj45_insertion.dual_rack_workcell import (
    dual_rack_cable_body_poses_torch,
    dual_rack_cable_workcell_intersection_mask_torch,
    route_dual_rack_cable_points_torch,
)
from isaaclab_tasks.contrib.franka_rj45_insertion.pick_insert_env_cfg import (
    PICK_INSERT_CLOSED_FINGER_POSITION,
    PICK_INSERT_GRASP_PROXY_FRICTION,
)
from isaaclab_tasks.contrib.franka_rj45_insertion.pick_insert_reset_dataset_io import (
    reset_dataset_content_digest,
    reset_dataset_validate_runtime,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_RELATIVE_PATH = Path(__file__).resolve().relative_to(_REPO_ROOT).as_posix()
DEFAULT_DATASET_PATH = _REPO_ROOT / "datasets/franka_rj45_dual_rack_insert/reset_dataset.pt"
DEFAULT_VALIDATION_REPORT_PATH = (
    _REPO_ROOT / "logs/rsl_rl/franka_rj45_dual_rack_insert/validation/reset_validation.json"
)
DEFAULT_CERTIFICATE_PATH = _REPO_ROOT / "datasets/franka_rj45_dual_rack_insert/canonical_goal_certificate.pt"
DUAL_RACK_GOAL_EQUILIBRIUM_RELAX_S = 60.0
"""Warm relaxation before capturing a two-ended cable goal [s]."""


class RJ45DualRackResetToolEnv(_RJ45ResetToolMixin, FrankaRJ45DualRackInsertEnv):
    """Manager-free dual-rack environment used only by reset tooling."""

    def __init__(self, cfg, *args, **kwargs) -> None:
        self._is_closed = True
        configured = getattr(cfg, "grasp_proxy_friction", None)
        if (
            isinstance(configured, bool)
            or not isinstance(configured, int | float)
            or not math.isfinite(configured)
            or float(configured) != PICK_INSERT_GRASP_PROXY_FRICTION
        ):
            raise ValueError(
                "The dual-rack reset tool requires the exact production grasp-proxy friction "
                f"{PICK_INSERT_GRASP_PROXY_FRICTION}."
            )
        super().__init__(cfg, *args, **kwargs)

    @property
    def grasp_proxy_friction(self) -> float:
        """Return the live builder value used by the shared physical contract."""
        return float(self._rj45_builder.grasp_proxy_friction)


def _dual_rack_source_digests() -> dict[str, str]:
    """Extend the shared canonical-goal closure with this task entrypoint."""
    result = base._canonical_goal_source_digests()
    result[_SCRIPT_RELATIVE_PATH] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return dict(sorted(result.items()))


class DualRackResetDatasetGenerator(base.PickInsertResetDatasetGenerator):
    """Reuse the validated bank machinery with exact two-ended cable routing."""

    def __init__(self, env: RJ45DualRackResetToolEnv, cfg: base.GeneratorCfg) -> None:
        super().__init__(env, cfg)
        self._workcell_cfg = env._rj45_builder.workcell_cfg
        if self._workcell_cfg is None:
            raise RuntimeError("Two-ended cable generation requires one immutable workcell configuration.")
        runtime = env.rj45_runtime
        if runtime.anchored_cable_endpoint_w is None:
            raise RuntimeError("Dual-rack generation requires a bound anchored cable endpoint.")
        self._cable_prefix_point_offsets = wp.to_torch(runtime.cable_prefix_point_offsets).to(
            device=self.device,
            dtype=torch.float32,
        )
        self._cable_segment_lengths = wp.to_torch(runtime.cable_segment_lengths).to(
            device=self.device,
            dtype=torch.float32,
        )
        self._cable_prefix_rotations = wp.to_torch(runtime.cable_prefix_rotations).to(
            device=self.device,
            dtype=torch.float32,
        )
        self._anchored_endpoint_e = env.anchored_cable_target_position_e().to(
            device=self.device,
            dtype=torch.float32,
        )
        self._anchored_plug_e = env.anchored_plug_pose_e().to(device=self.device, dtype=torch.float32)
        expected = (
            (base.CABLE_KINEMATIC_COUNT + 1, 3),
            (self.layout.cable_segment_count,),
            (base.CABLE_KINEMATIC_COUNT, 4),
            (env.num_envs, 3),
        )
        actual = (
            tuple(self._cable_prefix_point_offsets.shape),
            tuple(self._cable_segment_lengths.shape),
            tuple(self._cable_prefix_rotations.shape),
            tuple(self._anchored_endpoint_e.shape),
        )
        if actual != expected:
            raise RuntimeError(f"Dual-rack cable runtime shapes changed: expected={expected}, actual={actual}.")

    def _canonical_goal_certificate_validation_kwargs(self) -> dict[str, Any]:
        """Bind certification and loading to the complete dual-rack contract."""
        validation = super()._canonical_goal_certificate_validation_kwargs()
        validation["expected_task_contract"] = dual_rack_reset_dataset_task_contract(self.env.cfg)
        validation["expected_source_sha256"] = _dual_rack_source_digests()
        return validation

    def _route_cable(self, task_q: torch.Tensor) -> torch.Tensor:
        """Route every lane from its free plug to the immutable seated end."""
        plug = task_q[:, self.plug_index]
        count = len(task_q)
        plug_rotation = plug[:, None, 3:7].expand(-1, len(self._cable_prefix_point_offsets), -1)
        offsets = self._cable_prefix_point_offsets[None].expand(count, -1, -1)
        prefix_points = plug[:, None, :3] + math_utils.quat_apply(
            plug_rotation.reshape(-1, 4),
            offsets.reshape(-1, 3),
        ).reshape_as(offsets)
        cable_points = route_dual_rack_cable_points_torch(
            prefix_points,
            self._anchored_endpoint_e[:count],
            self._cable_segment_lengths,
            fixed_suffix_points=self._anchored_cable_suffix_points(count),
        )
        self._last_route_workcell_clear = ~dual_rack_cable_workcell_intersection_mask_torch(
            cable_points,
            cable_radius_m=base.CABLE_RADIUS,
            cfg=self._workcell_cfg,
        )
        actual_lengths = torch.linalg.vector_norm(torch.diff(cable_points, dim=1), dim=-1)
        if not bool(
            torch.allclose(
                actual_lengths,
                self._cable_segment_lengths[None].expand_as(actual_lengths),
                rtol=0.0,
                atol=2.0e-6,
            )
        ):
            raise RuntimeError("Dual-rack generated cable no longer preserves every authored segment length.")
        return dual_rack_cable_body_poses_torch(
            cable_points,
            free_plug_orientation_xyzw=plug[:, 3:7],
            prefix_rotations_xyzw=self._cable_prefix_rotations,
        )

    def _anchored_cable_suffix_points(self, count: int) -> torch.Tensor:
        """Return four fixed strain-relief spans ordered toward the seated plug."""
        directions = torch.diff(self._cable_prefix_point_offsets, dim=0)
        directions /= torch.linalg.vector_norm(directions, dim=-1, keepdim=True).clamp_min(1.0e-9)
        outward_steps = directions * self._cable_segment_lengths[-base.CABLE_KINEMATIC_COUNT :].flip(0)[:, None]
        outward_offsets = torch.cat(
            (
                self._cable_prefix_point_offsets[:1],
                self._cable_prefix_point_offsets[:1] + torch.cumsum(outward_steps, dim=0),
            ),
            dim=0,
        )
        plug = self._anchored_plug_e[:count]
        rotations = plug[:, None, 3:7].expand(-1, len(outward_offsets), -1)
        offsets = outward_offsets[None].expand(count, -1, -1)
        outward = plug[:, None, :3] + math_utils.quat_apply(
            rotations.reshape(-1, 4),
            offsets.reshape(-1, 3),
        ).reshape_as(offsets)
        suffix = outward.flip(1)
        if not bool(torch.allclose(suffix[:, -1], self._anchored_endpoint_e[:count], rtol=0.0, atol=2.0e-6)):
            raise RuntimeError("Dual-rack anchored strain relief no longer ends at the static plug cable exit.")
        return suffix

    def _fast_static_checks(self, *args, **kwargs):
        """Fold exact cable/fixture separation into the shared collision gate."""
        checks, metrics = super()._fast_static_checks(*args, **kwargs)
        route_clear = getattr(self, "_last_route_workcell_clear", None)
        if not isinstance(route_clear, torch.Tensor) or tuple(route_clear.shape) != (self.env.num_envs,):
            raise RuntimeError("Dual-rack static checks did not receive same-batch cable-route evidence.")
        checks["collision_filtered"] &= route_clear
        return checks, metrics

    def _fast_task_state(
        self,
        phase: int,
        pickup_pose: torch.Tensor,
        goal_q: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Construct the shared phase pose, then solve the whole two-ended cable."""
        task_q, task_qd = super()._fast_task_state(phase, pickup_pose, goal_q)
        task_q[:, self.cable_slice] = self._route_cable(task_q)
        task_qd[:, self.cable_slice] = 0.0
        return task_q, task_qd

    @torch.inference_mode()
    def generate(
        self,
        canonical_goal_certificate: Mapping[str, Any] | None = None,
        *,
        generation_checkpoint=None,
    ) -> dict[str, Any]:
        """Generate and rebind the shared schema to the exact dual-rack contract."""
        if generation_checkpoint is not None:
            raise ValueError("Dual-rack fast generation is intentionally checkpoint-free and bounded.")
        payload = super().generate(canonical_goal_certificate)
        contract = dual_rack_reset_dataset_task_contract(self.env.cfg)
        metadata = payload["metadata"]
        metadata["generator"] = Path(__file__).name
        metadata["task_contract"] = contract
        metadata["initial_state_policy"].update(
            {
                "construction": "free-plug-phase-placement-plus-exact-two-ended-circular-cable-route-plus-batched-ik",
                "whole_cable_generated_by_coherent_rigid_transform": False,
                "whole_cable_generated_by_exact_segment_length_route": True,
                "free_end_anchor_count": base.CABLE_KINEMATIC_COUNT,
                "anchored_end": "static-seated-second-plug-with-four-pinned-strain-relief-segments",
                "route_simulation_steps": 0,
            }
        )
        metadata["goal_policy"].update(
            {
                "per_row_goal_is_rigid_socket_transform": False,
                "socket_pose_randomized_within_task_cfg": False,
                "socket_pose_fixed_to_upper_rack": True,
                "anchored_endpoint_fixed_to_lower_rack": True,
            }
        )
        payload["content_sha256"] = reset_dataset_content_digest(payload)
        reset_dataset_validate_runtime(payload, expected_task_contract=contract)
        return payload


def _generator_cfg(args: argparse.Namespace) -> base.GeneratorCfg:
    certifier = args.canonical_goal_certificate_output is not None
    return base.GeneratorCfg(
        generation_mode=base._GENERATION_MODE_PHYSICAL_ORACLE if certifier else base._GENERATION_MODE_FAST_IK,
        rows_per_phase=1 if args.quick else args.rows_per_phase,
        batch_size=1 if args.quick else args.batch_size,
        seed=args.seed,
        max_batches_per_phase=min(12, args.max_batches_per_phase) if args.quick else args.max_batches_per_phase,
        quick=args.quick,
        finger_closed_target=PICK_INSERT_CLOSED_FINGER_POSITION,
        goal_cold_equilibrium_relax_s=(
            DUAL_RACK_GOAL_EQUILIBRIUM_RELAX_S if certifier else base.GeneratorCfg.goal_cold_equilibrium_relax_s
        ),
    )


def _run(args: argparse.Namespace) -> None:
    generator_cfg = _generator_cfg(args)
    env_cfg = FrankaRJ45DualRackInsertEnvCfg()
    base._configure_generation_reset_dataset_shape(env_cfg, generator_cfg.rows_per_phase)
    env_cfg.scene.num_envs = generator_cfg.batch_size
    env_cfg.sim.device = args.device
    env_cfg.seed = generator_cfg.seed
    env_cfg.validate_config()

    with launch_simulation(env_cfg, args):
        env = RJ45DualRackResetToolEnv(env_cfg)
        try:
            if args.inspect:
                env.restore_default_task()
                env.scene.write_data_to_sim()
                env.sim.forward()
                env.scene.update(dt=0.0)
                print("[INFO] Dual-rack inspection scene is ready.", flush=True)
                while env.sim.is_headless_or_exist_active_visualizer():
                    env.sim.render()
                    time.sleep(1.0 / 60.0)
                return
            generator = DualRackResetDatasetGenerator(env, generator_cfg)
            if args.canonical_goal_certificate_output is not None:
                certificate = generator.derive_goal_certificate()
                save_torch_atomic(certificate, args.canonical_goal_certificate_output)
                print(f"[INFO] Dual-rack canonical certificate: {args.canonical_goal_certificate_output.resolve()}")
                print(f"[INFO] Content SHA-256: {certificate['content_sha256']}")
                return
            assert args.canonical_goal_certificate_input is not None
            source_snapshot = generator._canonical_goal_certificate_validation_kwargs()["expected_source_sha256"]
            payload = base._generate_and_save_reset_dataset_artifact(
                generator,
                output=args.output,
                certificate_input=args.canonical_goal_certificate_input,
                pre_environment_source_sha256=source_snapshot,
                checkpoint_path=None,
                resuming_checkpoint=False,
            )
        finally:
            env.close()

    print(f"[INFO] Wrote {len(payload['states']['phase'])} dual-rack reset rows to {args.output.resolve()}.")
    print(f"[INFO] Content SHA-256: {payload['content_sha256']}")
    if args.validate:
        command = [
            sys.executable,
            str(Path(__file__).with_name("validate_franka_rj45_pick_insert_fast_resets.py")),
            "--input",
            str(args.output),
            "--output",
            str(args.validation_report),
        ]
        subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--validation-report", type=Path, default=DEFAULT_VALIDATION_REPORT_PATH)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--rows-per-phase", type=int, default=3_334)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-batches-per-phase", type=int, default=96)
    parser.add_argument("--quick", action="store_true", help="Generate one row per phase for a smoke test.")
    parser.add_argument("--validate", action="store_true", help="Publish a source-bound zero-step report.")
    certificate = parser.add_mutually_exclusive_group(required=True)
    certificate.add_argument(
        "--canonical-goal-certificate-output",
        type=Path,
        help="Create a current dual-rack physical goal certificate and no reset bank.",
    )
    certificate.add_argument(
        "--canonical-goal-certificate-input",
        type=Path,
        help="Generate the fast bank from a current dual-rack goal certificate.",
    )
    certificate.add_argument(
        "--inspect",
        action="store_true",
        help="Open the default two-rack scene without loading or generating a reset bank.",
    )
    add_launcher_args(parser)
    args = parser.parse_args()
    if args.batch_size is None:
        if args.inspect:
            args.batch_size = 1
        else:
            args.batch_size = 4 if args.canonical_goal_certificate_output is not None else 256
    if args.canonical_goal_certificate_output is not None and args.batch_size not in (1, 4):
        parser.error("Canonical certification requires batch size 4, or 1 only with --quick.")
    if args.canonical_goal_certificate_input is not None and args.rows_per_phase < 1:
        parser.error("--rows-per-phase must be positive.")
    _run(args)


if __name__ == "__main__":
    main()
