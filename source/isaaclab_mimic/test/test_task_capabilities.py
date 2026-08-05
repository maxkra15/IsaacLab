# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for task-specific Mimic command-line capability checks."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from isaaclab_mimic.task_capabilities import validate_skillgen_task_support


def test_skillgen_support_is_unchanged_without_task_opt_out():
    """Tasks without an opt-out retain the existing SkillGen path."""
    validate_skillgen_task_support("Supported-Task-v0", SimpleNamespace())


def test_skillgen_opt_out_reports_task_and_reason():
    """An explicit task opt-out fails before optional planner dependencies load."""
    cfg = SimpleNamespace(skillgen_unsupported_reason="use standard MimicGen instead")

    with pytest.raises(
        ValueError,
        match=r"--use_skillgen is not supported for task 'Waterhose-v0'.*standard MimicGen",
    ):
        validate_skillgen_task_support("Waterhose-v0", cfg)


def test_waterhose_skillgen_preflight_runs_before_environment_construction():
    """Waterhose opts out and generation validates that opt-out before gym.make."""
    repo_root = Path(__file__).resolve().parents[3]
    waterhose_cfg_path = (
        repo_root / "source/isaaclab_tasks/isaaclab_tasks/contrib/waterhose/config/rby1df/mimic_env_cfg.py"
    )
    waterhose_tree = ast.parse(waterhose_cfg_path.read_text(), filename=str(waterhose_cfg_path))
    waterhose_class = next(
        node for node in waterhose_tree.body if isinstance(node, ast.ClassDef) and node.name == "WaterhoseMimicEnvCfg"
    )
    assert any(
        isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "skillgen_unsupported_reason"
        for node in waterhose_class.body
    )

    generate_script_path = repo_root / "scripts/imitation_learning/isaaclab_mimic/generate_dataset.py"
    generate_tree = ast.parse(generate_script_path.read_text(), filename=str(generate_script_path))
    main_function = next(
        node for node in generate_tree.body if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    main_source = ast.unparse(main_function)

    assert main_source.index("validate_skillgen_task_support") < main_source.index("gym.make")
