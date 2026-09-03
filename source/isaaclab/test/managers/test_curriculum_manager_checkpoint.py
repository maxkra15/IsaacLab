# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CPU-only tests for opt-in curriculum checkpoint state."""

from types import SimpleNamespace

import pytest

from isaaclab.managers import CurriculumManager


class _StatefulTerm:
    checkpoint_state_enabled = True

    def __init__(self):
        self.state = {"count": 3}
        self.reseeded_rank = None

    def get_state(self):
        return dict(self.state)

    def set_state(self, state):
        self.state = dict(state)

    def reseed_checkpoint_generators(self, global_rank: int):
        self.reseeded_rank = global_rank


def _manager_with_terms(*terms):
    manager = object.__new__(CurriculumManager)
    manager._resolve_terms_handle = None
    manager._term_names = [name for name, _ in terms]
    manager._term_cfgs = [SimpleNamespace(func=term) for _, term in terms]
    return manager


def test_curriculum_checkpoint_state_is_opt_in_and_rank_aware():
    """Only opted-in terms restore, and non-source DDP ranks fork their RNG hooks."""
    stateful = _StatefulTerm()
    manager = _manager_with_terms(("stateful", stateful), ("legacy", object()))

    assert manager.get_checkpoint_state() == {"stateful": {"count": 3}}
    manager.set_checkpoint_state(
        {"stateful": {"count": 9}},
        source_global_rank=0,
        current_global_rank=3,
    )

    assert stateful.state == {"count": 9}
    assert stateful.reseeded_rank == 3


def test_curriculum_checkpoint_provider_names_must_match_exactly():
    """Changed provider configurations cannot silently accept partial state."""
    manager = _manager_with_terms(("stateful", _StatefulTerm()))

    with pytest.raises(ValueError, match="must match"):
        manager.set_checkpoint_state({})
