# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CPU-only registration, asset, and curtain MDP checks for the solver showcases."""

import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.contrib.textile_atelier import mdp as textile_mdp
from isaaclab_tasks.utils import resolve_task_config


@pytest.mark.parametrize(
    ("task_name", "scene_asset_name"),
    (
        ("IsaacContrib-Kinetic-Foundry", "backdrop"),
        ("IsaacContrib-Textile-Atelier-Kuka-GR1T2", "frame"),
        ("IsaacContrib-RelayJuggle-KukaAllegro-GR1T2", "gallery"),
    ),
)
def test_solver_showcase_config_references_authored_usda(task_name: str, scene_asset_name: str) -> None:
    """Each registered PPO task resolves a scene with its authored USDA present."""
    env_cfg, agent_cfg = resolve_task_config(task_name, "rsl_rl_cfg_entry_point", overrides=[])

    assert agent_cfg is not None
    usd_path = Path(getattr(env_cfg.scene, scene_asset_name).spawn.usd_path)
    assert usd_path.suffix == ".usda"
    assert usd_path.is_file(), f"{task_name} references a missing authored asset: {usd_path}"


def _curtain_env(reference_nodes: torch.Tensor, current_nodes: torch.Tensor | None = None) -> SimpleNamespace:
    """Build the small cloth interface used by the curtain MDP terms."""
    default_state = torch.cat((reference_nodes, torch.zeros_like(reference_nodes)), dim=-1)
    cloth = SimpleNamespace(
        data=SimpleNamespace(
            default_nodal_state_w=SimpleNamespace(torch=default_state),
            nodal_pos_w=SimpleNamespace(torch=current_nodes if current_nodes is not None else reference_nodes),
        ),
        write_nodal_kinematic_target_to_sim_index=Mock(),
    )
    return SimpleNamespace(scene={"cloth": cloth})


def _reference_curtain() -> torch.Tensor:
    """Return a five-column, three-row vertical sheet with known material points [m]."""
    x = torch.linspace(0.10, 1.00, 5)
    z = torch.tensor((0.70, 1.05, 1.40))
    grid_z, grid_x = torch.meshgrid(z, x, indexing="ij")
    return torch.stack((grid_x, torch.zeros_like(grid_x), grid_z), dim=-1).reshape(1, -1, 3)


def test_curtain_reset_gathers_only_pinned_top_row() -> None:
    """The reset target gathers the header while lower material points stay free."""
    reference = _reference_curtain()
    env = _curtain_env(reference)
    env_ids = torch.tensor([0])

    textile_mdp.pin_curtain_top_edge(env, env_ids)

    writer = env.scene["cloth"].write_nodal_kinematic_target_to_sim_index
    writer.assert_called_once()
    targets = writer.call_args.args[0]
    torch.testing.assert_close(targets[0, :10, :3], reference[0, :10], atol=0, rtol=0)
    torch.testing.assert_close(targets[0, :10, 3], torch.ones(10), atol=0, rtol=0)
    torch.testing.assert_close(targets[0, 10:, 3], torch.zeros(5), atol=0, rtol=0)
    top_x = targets[0, 10:, 0]
    reference_top_x = reference[0, 10:, 0]
    assert top_x.amax() - top_x.amin() < reference_top_x.amax() - reference_top_x.amin()
    assert targets[0, 10:, 1].amin() < reference[0, 10:, 1].amin()
    assert targets[0, 10:, 1].amax() > reference[0, 10:, 1].amax()
    torch.testing.assert_close(targets[0, 10:, 2], reference[0, 10:, 2], atol=0, rtol=0)


def test_curtain_deflection_profile_and_target_use_material_motion() -> None:
    """Two independently displaced patches determine the signed profile and reward."""
    reference = _reference_curtain()
    current = reference.clone()
    current[0, 6, 1] = 0.04
    current[0, 8, 1] = -0.02
    env = _curtain_env(reference, current)

    profile = textile_mdp.curtain_deflection_profile(env)
    reward = textile_mdp.curtain_deflection_target(env, target_left=0.02, target_right=-0.01, std=0.02)

    torch.testing.assert_close(profile, torch.tensor([[0.04, -0.02, -0.06]]), atol=1e-6, rtol=0)
    torch.testing.assert_close(reward, torch.tensor([math.exp(-1.5)]), atol=1e-6, rtol=0)
