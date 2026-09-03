# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Generate a balanced two-ended RJ45 bank over native GB300 SN2201 jacks."""

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

import generate_franka_rj45_dual_rack_reset_dataset as dual
import generate_franka_rj45_pick_insert_reset_dataset as base
import torch
from _franka_rj45_reset_tools import _RJ45ResetToolMixin, save_torch_atomic

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.utils import math as math_utils

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.contrib.franka_rj45_insertion.gb300_env import FrankaRJ45Gb300InsertEnv
from isaaclab_tasks.contrib.franka_rj45_insertion.gb300_env_cfg import (
    FrankaRJ45Gb300InsertEnvCfg,
    gb300_reset_dataset_task_contract,
)
from isaaclab_tasks.contrib.franka_rj45_insertion.gb300_workcell import (
    GB300_TARGET_TASK_TRANSLATIONS,
    GB300_TASK_ROTATION_XYZW,
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
DEFAULT_DATASET_PATH = _REPO_ROOT / "datasets/franka_rj45_gb300_insert/reset_dataset.pt"
DEFAULT_VALIDATION_REPORT_PATH = _REPO_ROOT / "logs/rsl_rl/franka_rj45_gb300_insert/validation/reset_validation.json"
DEFAULT_CERTIFICATE_PATH = _REPO_ROOT / "datasets/franka_rj45_gb300_insert/canonical_goal_certificate.pt"
GB300_GOAL_EQUILIBRIUM_RELAX_S = 70.0


class RJ45Gb300ResetToolEnv(_RJ45ResetToolMixin, FrankaRJ45Gb300InsertEnv):
    """Manager-free GB300 environment used only by reset tooling."""

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
                "The GB300 reset tool requires the exact production grasp-proxy friction "
                f"{PICK_INSERT_GRASP_PROXY_FRICTION}."
            )
        super().__init__(cfg, *args, **kwargs)

    @property
    def grasp_proxy_friction(self) -> float:
        return float(self._rj45_builder.grasp_proxy_friction)


def _gb300_source_digests() -> dict[str, str]:
    """Extend the two-ended cable certificate closure with this entrypoint."""
    result = dual._dual_rack_source_digests()
    result[_SCRIPT_RELATIVE_PATH] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return dict(sorted(result.items()))


class Gb300ResetDatasetGenerator(dual.DualRackResetDatasetGenerator):
    """Route one cable while selecting one exact SDF socket from eight anchors."""

    def _canonical_goal_certificate_validation_kwargs(self) -> dict[str, Any]:
        validation = base.PickInsertResetDatasetGenerator._canonical_goal_certificate_validation_kwargs(self)
        validation["expected_task_contract"] = gb300_reset_dataset_task_contract(self.env.cfg)
        validation["expected_source_sha256"] = _gb300_source_digests()
        return validation

    def _sample_scene(self) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.env.cfg
        candidates = torch.tensor(GB300_TARGET_TASK_TRANSLATIONS, device=self.device, dtype=torch.float32)
        candidate_ids = torch.randint(
            0,
            len(candidates),
            (self.env.num_envs,),
            device=self.device,
            generator=self.random,
        )
        assembly_position = candidates[candidate_ids]
        task_orientation = torch.tensor(
            GB300_TASK_ROTATION_XYZW,
            device=self.device,
            dtype=torch.float32,
        ).expand(self.env.num_envs, -1)
        socket_orientation = math_utils.quat_mul(task_orientation, self.socket_local_orientation)
        socket_position = assembly_position + math_utils.quat_apply(task_orientation, self.socket_local_position)
        socket_pose = torch.cat((socket_position, socket_orientation), dim=-1)

        pickup_lower = torch.tensor(cfg.pickup_position_lower, device=self.device)
        pickup_upper = torch.tensor(cfg.pickup_position_upper, device=self.device)
        pickup_position = pickup_lower + torch.rand(
            (self.env.num_envs, 3), device=self.device, generator=self.random
        ) * (pickup_upper - pickup_lower)
        for _ in range(16):
            too_close = (
                torch.linalg.vector_norm(pickup_position[:, :2] - socket_position[:, :2], dim=-1)
                < cfg.minimum_pickup_socket_distance
            )
            if not bool(too_close.any()):
                break
            resampled = pickup_lower + torch.rand((self.env.num_envs, 3), device=self.device, generator=self.random) * (
                pickup_upper - pickup_lower
            )
            pickup_position = torch.where(too_close[:, None], resampled, pickup_position)
        too_close = (
            torch.linalg.vector_norm(pickup_position[:, :2] - socket_position[:, :2], dim=-1)
            < cfg.minimum_pickup_socket_distance
        )
        if bool(too_close.any()):
            failed = torch.nonzero(too_close, as_tuple=False).flatten().tolist()
            raise RuntimeError(
                f"GB300 scene sampling exhausted the bounded pickup/socket-separation retries for lanes {failed}."
            )
        pickup_yaw = cfg.pickup_yaw_range[0] + torch.rand(
            self.env.num_envs, device=self.device, generator=self.random
        ) * (cfg.pickup_yaw_range[1] - cfg.pickup_yaw_range[0])
        pickup_quat = math_utils.quat_from_euler_xyz(
            torch.zeros_like(pickup_yaw), torch.zeros_like(pickup_yaw), pickup_yaw
        )
        return socket_pose, torch.cat((pickup_position, pickup_quat), dim=-1)

    @torch.inference_mode()
    def generate(
        self,
        canonical_goal_certificate: Mapping[str, Any] | None = None,
        *,
        generation_checkpoint=None,
    ) -> dict[str, Any]:
        if generation_checkpoint is not None:
            raise ValueError("GB300 fast generation is intentionally checkpoint-free and bounded.")
        payload = base.PickInsertResetDatasetGenerator.generate(self, canonical_goal_certificate)
        contract = gb300_reset_dataset_task_contract(self.env.cfg)
        metadata = payload["metadata"]
        metadata["generator"] = Path(__file__).name
        metadata["task_contract"] = contract
        metadata["initial_state_policy"].update(
            {
                "construction": "discrete-native-gb300-sn2201-jack-plus-exact-two-ended-cable-route-plus-batched-ik",
                "whole_cable_generated_by_coherent_rigid_transform": False,
                "whole_cable_generated_by_exact_segment_length_route": True,
                "free_end_anchor_count": base.CABLE_KINEMATIC_COUNT,
                "anchored_end": "static-seated-native-gb300-sn2201-jack-with-four-pinned-strain-relief-segments",
                "target_port_candidate_count": len(GB300_TARGET_TASK_TRANSLATIONS),
                "target_port_sampling": "uniform-discrete-per-candidate-batch-lane",
                "route_simulation_steps": 0,
            }
        )
        metadata["goal_policy"].update(
            {
                "per_row_goal_is_rigid_socket_transform": True,
                "socket_pose_randomized_within_task_cfg": False,
                "socket_pose_uniform_discrete_candidate": True,
                "anchored_endpoint_fixed_to_native_gb300_sn2201_jack": True,
            }
        )
        socket_positions = payload["states"]["task_body_pose"][:, self.socket_index, :3]
        candidate_task_positions = torch.tensor(
            GB300_TARGET_TASK_TRANSLATIONS,
            device=self.device,
            dtype=socket_positions.dtype,
        )
        candidate_task_orientations = torch.tensor(
            GB300_TASK_ROTATION_XYZW,
            device=self.device,
            dtype=socket_positions.dtype,
        ).expand(len(candidate_task_positions), -1)
        candidates = candidate_task_positions + math_utils.quat_apply(
            candidate_task_orientations,
            self.socket_local_position[0].expand(len(candidate_task_positions), -1),
        )
        candidates = candidates.cpu()
        distance = torch.cdist(socket_positions, candidates)
        nearest_distance, nearest = distance.min(dim=-1)
        if not bool((nearest_distance <= 1.0e-6).all()):
            raise RuntimeError("A generated GB300 reset row does not select one exact native SN2201 socket anchor.")
        metadata["initial_state_policy"]["accepted_target_port_counts"] = torch.bincount(
            nearest, minlength=len(candidates)
        ).tolist()
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
            GB300_GOAL_EQUILIBRIUM_RELAX_S if certifier else base.GeneratorCfg.goal_cold_equilibrium_relax_s
        ),
    )


def _run(args: argparse.Namespace) -> None:
    generator_cfg = _generator_cfg(args)
    env_cfg = FrankaRJ45Gb300InsertEnvCfg()
    base._configure_generation_reset_dataset_shape(env_cfg, generator_cfg.rows_per_phase)
    env_cfg.scene.num_envs = generator_cfg.batch_size
    env_cfg.sim.device = args.device
    env_cfg.seed = generator_cfg.seed
    env_cfg.validate_config()

    with launch_simulation(env_cfg, args):
        env = RJ45Gb300ResetToolEnv(env_cfg)
        try:
            if args.inspect:
                env.restore_default_task()
                env.scene.write_data_to_sim()
                env.sim.forward()
                env.scene.update(dt=0.0)
                print("[INFO] GB300 RJ45 inspection scene is ready.", flush=True)
                while env.sim.is_headless_or_exist_active_visualizer():
                    env.sim.render()
                    time.sleep(1.0 / 60.0)
                return
            generator = Gb300ResetDatasetGenerator(env, generator_cfg)
            if args.canonical_goal_certificate_output is not None:
                certificate = generator.derive_goal_certificate()
                save_torch_atomic(certificate, args.canonical_goal_certificate_output)
                print(f"[INFO] GB300 canonical certificate: {args.canonical_goal_certificate_output.resolve()}")
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

    print(f"[INFO] Wrote {len(payload['states']['phase'])} GB300 reset rows to {args.output.resolve()}.")
    print(f"[INFO] Content SHA-256: {payload['content_sha256']}")
    if args.validate:
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("validate_franka_rj45_pick_insert_fast_resets.py")),
                "--input",
                str(args.output),
                "--output",
                str(args.validation_report),
            ],
            check=True,
        )


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
    certificate.add_argument("--canonical-goal-certificate-output", type=Path)
    certificate.add_argument("--canonical-goal-certificate-input", type=Path)
    certificate.add_argument("--inspect", action="store_true")
    add_launcher_args(parser)
    args = parser.parse_args()
    if args.batch_size is None:
        args.batch_size = 1 if args.inspect else (4 if args.canonical_goal_certificate_output is not None else 256)
    if args.canonical_goal_certificate_output is not None and args.batch_size not in (1, 4):
        parser.error("Canonical certification requires batch size 4, or 1 only with --quick.")
    if args.canonical_goal_certificate_input is not None and args.rows_per_phase < 1:
        parser.error("--rows-per-phase must be positive.")
    _run(args)


if __name__ == "__main__":
    main()
