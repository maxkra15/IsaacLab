# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CPU checks for MuJoCo Warp diagnostic severity and retained evidence."""

import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

try:
    from mujoco_warp import OverflowType
except ModuleNotFoundError as error:
    if error.name == "mujoco_warp":
        raise unittest.SkipTest("Optional mujoco_warp package is not installed") from error
    raise

SPEC = importlib.util.spec_from_file_location(
    "simulator_mjwarp_backend", Path(__file__).parents[1] / "isaaclab_backend.py"
)
backend = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(backend)


class TestDiagnostics(unittest.TestCase):
    """Protect the severity distinction used to admit or reject benchmark results."""

    def test_convergence_limits_are_warnings_and_capacity_or_unknown_bits_are_fatal(self):
        iterations = int(OverflowType.ITERATIONS)
        line_search = int(OverflowType.LS_ITERATIONS)
        capacity = int(OverflowType.NEFC)
        unknown = 1 << 30
        raw = [0, iterations, line_search, iterations | line_search, capacity | iterations, unknown | line_search]
        overflow = Mock()
        overflow.numpy.return_value.tolist.return_value = raw.copy()
        solver = SimpleNamespace(
            mjw_data=SimpleNamespace(overflow=overflow, njmax=300),
            get_max_contact_count=Mock(return_value=128),
        )
        adapter = backend.Adapter.__new__(backend.Adapter)
        adapter._sim = SimpleNamespace(
            backend_manager=SimpleNamespace(backend=SimpleNamespace(solver=solver, graph=object()))
        )

        result = adapter.diagnostics()

        self.assertEqual(result["overflow_flags_at_end"], raw)
        self.assertEqual(result["fatal_capacity_flags_at_end"], [0, 0, 0, 0, capacity, unknown])
        self.assertEqual(
            result["convergence_limit_flags_at_end"],
            [0, iterations, line_search, iterations | line_search, iterations, line_search],
        )
        self.assertEqual(result["unknown_overflow_flags_at_end"], [0, 0, 0, 0, 0, unknown])


if __name__ == "__main__":
    unittest.main()
