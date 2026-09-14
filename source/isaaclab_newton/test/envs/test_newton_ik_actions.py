# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from isaaclab_newton.envs.mdp.actions import newton_ik_actions
from isaaclab_newton.envs.mdp.actions.newton_ik_actions_cfg import NewtonInverseKinematicsActionCfg
from isaaclab_newton.ik import NewtonIKPoseObjectiveCfg

import isaaclab.sim as sim_utils
from isaaclab.managers.action_manager import ActionTerm


class _PathResolved(Exception):
    pass


@pytest.mark.parametrize(
    ("articulation_root_prim_path", "expected_path"),
    (
        (None, "/World/envs/env_[^/]+/Robot"),
        ("/fix_base_joint", "/World/envs/env_[^/]+/Robot/fix_base_joint"),
    ),
)
def test_ik_action_resolves_configured_articulation_root(
    monkeypatch: pytest.MonkeyPatch,
    articulation_root_prim_path: str | None,
    expected_path: str,
) -> None:
    """The prototype lookup must include an explicitly configured articulation root."""
    asset = SimpleNamespace(
        is_fixed_base=True,
        cfg=SimpleNamespace(
            prim_path="/World/envs/env_[^/]+/Robot",
            articulation_root_prim_path=articulation_root_prim_path,
        ),
        find_joints=lambda *_args, **_kwargs: (SimpleNamespace(warp=()), ["joint"]),
    )

    def initialize_action_term(action, cfg, _env) -> None:
        action.cfg = cfg
        action._asset = asset

    clone_plan = object()
    path_to_source = Mock(side_effect=_PathResolved)
    monkeypatch.setattr(ActionTerm, "__init__", initialize_action_term)
    monkeypatch.setattr(
        sim_utils.SimulationContext,
        "instance",
        lambda: SimpleNamespace(get_clone_plan=lambda: clone_plan),
    )
    monkeypatch.setattr(newton_ik_actions.cloner.query, "path_to_source", path_to_source)

    cfg = NewtonInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["joint"],
        objectives=[NewtonIKPoseObjectiveCfg(body_name="hand")],
    )
    with pytest.raises(_PathResolved):
        newton_ik_actions.NewtonInverseKinematicsAction(cfg, object())

    path_to_source.assert_called_once_with(clone_plan, expected_path)
