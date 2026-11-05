# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Publication acceptance, independently of simulation and plotting."""

import copy
import importlib
import sys
import unittest
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
publication = importlib.import_module("assemble_report")


def recorded_matrix() -> list[tuple[Path, dict]]:
    """Provide the required study matrix as JSON records, without measured or simulated data."""
    workloads = [
        ("physics", "none", 256),
        ("physics", "none", 4096),
        ("rollout", "none", 256),
        ("rollout", "none", 4096),
        ("rollout", "rgb", 64),
        ("learning", "none", 512),
        ("learning", "rgb", 512),
    ]
    runs = []
    for backend, task, workload, seed in product(("genesis", "isaaclab"), ("anymal", "cartpole"), workloads, (0, 1, 2)):
        mode, vision, count = workload
        config = dict(zip(publication.FIELDS, (backend, task, mode, vision, count, seed)))
        config.update(
            solver_profile="shared",
            solver_settings=copy.deepcopy(publication.PROFILE),
            solver_iterations=None,
            dt=0.005,
            decimation=4,
            width=64,
            height=64,
            iterations=500,
            rollout_steps=24,
            evaluation_steps=250,
            evaluation_interval=50,
        )
        metadata = {"solver_profile": "shared", "constraint_solver": "Newton", "solver_iterations": 25}
        if backend == "isaaclab":
            metadata.update(
                {
                    key: value
                    for key, value in publication.PROFILE["isaaclab"].items()
                    if key not in {"num_substeps", "use_cuda_graph"}
                }
            )
            metadata.update(
                physics_backend="Isaac Lab Newton / MuJoCo Warp",
                kit_runtime=False,
                substeps=1,
                cuda_graph=True,
                renderer="Newton Warp batched Camera",
                actual_solver={
                    "class": "newton.solvers.mujoco.SolverMuJoCo",
                    "iterations": 100,
                    "ls_iterations": 50,
                    "tolerance": 1e-6,
                    "impratio": 1.0,
                    "njmax": 300,
                    "solver_enum": 2,
                    "integrator_enum": 3,
                    "cone_enum": 0,
                },
            )
        run = {
            "schema_version": 1,
            "status": "ok",
            "config": config,
            "backend_metadata": metadata,
            "metrics": {
                "startup_s": 1,
                "physics_steps_per_s": 1,
                "env_steps_per_s": 1,
                "evaluation": {"steps": 250},
                "learning": [{"iteration": 0}, {"iteration": 500, "env_steps": 6144000}],
            },
            "provenance": {
                "harness_sha256": "test-runtime",
                "source_revisions": publication.UPSTREAM.copy(),
                "runtime": dict(
                    publication.RUNTIME, cuda_device_uuid="00000000-0000-0000-0000-000000000001", kit_modules_loaded=[]
                ),
                "hardware": {"gpu": "test, GPU-00000000-0000-0000-0000-000000000001"},
                "asset_manifest": {"source_urdf_sha256": "0" * 64, "meshes_sha256": {"mesh": "1" * 64}},
                "other_active_compute_detected": False,
                "gpu_activity_samples": [{"external_compute": []}],
            },
            "backend_diagnostics": {
                "graph_active": True,
                "fatal_capacity_flags_at_end": [0],
                "overflow_flags_at_end": [0],
                "convergence_limit_flags_at_end": [0],
                "constraint_capacity_per_world": 300,
            },
        }
        runs.append((Path(f"case-{len(runs)}.result.json"), run))
    return runs


class TestPublication(unittest.TestCase):
    """Protect study completeness and the interpretation of engine diagnostics."""

    def test_missing_duplicate_or_wrong_profile_cannot_satisfy_the_matrix(self):
        records = recorded_matrix()
        self.assertEqual(len(publication.validate_runs(records, "test-runtime")), 84)
        wrong = copy.deepcopy(records)
        wrong[0][1]["config"]["solver_profile"] = "native"
        for invalid in (records[:-1], records + [records[0]], wrong):
            with self.subTest(count=len(invalid)), self.assertRaises(ValueError):
                publication.validate_runs(invalid, "test-runtime")

    def test_convergence_caps_are_accepted_but_capacity_failures_are_rejected(self):
        records = recorded_matrix()
        diagnostics = next(run["backend_diagnostics"] for _, run in records if run["config"]["backend"] == "isaaclab")
        diagnostics.update(overflow_flags_at_end=[512], convergence_limit_flags_at_end=[512])
        self.assertEqual(len(publication.validate_runs(records, "test-runtime")), 84)
        diagnostics.update(overflow_flags_at_end=[513], fatal_capacity_flags_at_end=[1])
        with self.assertRaisesRegex(ValueError, "fatal capacity"):
            publication.validate_runs(records, "test-runtime")

    def test_physx_or_loaded_kit_cannot_be_published_as_mjwarp(self):
        records = recorded_matrix()
        target = next(run for _, run in records if run["config"]["backend"] == "isaaclab")
        target["backend_metadata"]["physics_backend"] = "Isaac Sim PhysX GPU"
        with self.assertRaisesRegex(ValueError, "Wrong main backend"):
            publication.validate_runs(records, "test-runtime")
        target["backend_metadata"]["physics_backend"] = "Isaac Lab Newton / MuJoCo Warp"
        target["provenance"]["runtime"]["kit_modules_loaded"] = ["omni.kit.app"]
        with self.assertRaisesRegex(ValueError, "Wrong main backend"):
            publication.validate_runs(records, "test-runtime")


if __name__ == "__main__":
    unittest.main()
