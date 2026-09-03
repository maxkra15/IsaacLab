# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CPU-only tests for stateful Isaac Lab RSL-RL checkpoints."""

from types import SimpleNamespace

import pytest
import torch

from isaaclab_rl.rsl_rl import IsaacLabOnPolicyRunner, RslRlVecEnvWrapper


class _Algorithm:
    def __init__(self):
        self.loaded = None

    def save(self):
        return {"algorithm_state": torch.tensor((1.0,))}

    def load(self, checkpoint, load_cfg, strict):
        self.loaded = (checkpoint, load_cfg, strict)
        return load_cfg is None or load_cfg.get("iteration", False)


class _Logger:
    def __init__(self):
        self.saved = []

    def save_model(self, path, iteration):
        self.saved.append((path, iteration))


class _Environment:
    num_envs = 64
    device = "cpu"

    def __init__(self, state=None):
        self.state = state or {}
        self.restored = None

    def get_checkpoint_state(self):
        return self.state

    def set_checkpoint_state(self, state, *, source_global_rank=0, current_global_rank=0):
        self.restored = (state, source_global_rank, current_global_rank)


def _runner(env, *, world_size=1, global_rank=0):
    runner = object.__new__(IsaacLabOnPolicyRunner)
    runner.env = env
    runner.alg = _Algorithm()
    runner.logger = _Logger()
    runner.current_learning_iteration = 17
    runner.device = "cpu"
    runner.gpu_world_size = world_size
    runner.gpu_global_rank = global_rank
    return runner


def test_runner_checkpoint_roundtrip_preserves_infos_and_environment_state(tmp_path):
    """A single-rank checkpoint restores opted-in state without changing user infos."""
    path = tmp_path / "model.pt"
    source = _runner(_Environment({"curriculum_manager": {"reset_sampling": {"attempts": 12}}}))
    source.save(str(path), infos={"caller": "value"})

    raw = torch.load(path, weights_only=False, map_location="cpu")
    payload = raw["infos"]["isaaclab_environment_state"]
    assert payload["source_world_size"] == 1
    assert payload["source_global_rank"] == 0
    assert payload["source_num_envs"] == 64

    target_env = _Environment()
    target = _runner(target_env)
    infos = target.load(str(path))

    assert infos == {"caller": "value"}
    assert target.current_learning_iteration == 17
    assert target_env.restored == (payload["state"], 0, 0)


def test_runner_loads_legacy_checkpoint_without_environment_restore(tmp_path):
    """Checkpoints written before the environment envelope remain unchanged."""
    path = tmp_path / "legacy.pt"
    torch.save(
        {
            "algorithm_state": torch.tensor((2.0,)),
            "iter": 5,
            "infos": {"legacy": True},
        },
        path,
    )
    target_env = _Environment()
    target = _runner(target_env)

    infos = target.load(str(path))

    assert infos == {"legacy": True}
    assert target.current_learning_iteration == 5
    assert target_env.restored is None


def test_selective_policy_load_skips_environment_state_unless_explicitly_requested(tmp_path):
    """Actor-only warm starts do not import a checkpoint's curriculum frontier."""
    path = tmp_path / "warmstart.pt"
    source = _runner(_Environment({"curriculum_manager": {"reset_sampling": {"attempts": 12}}}))
    source.save(str(path))

    actor_only_env = _Environment()
    actor_only = _runner(actor_only_env)
    assert actor_only.load(str(path), load_cfg={"actor": True}) is None
    assert actor_only_env.restored is None

    explicit_env = _Environment()
    explicit = _runner(explicit_env)
    assert explicit.load(str(path), load_cfg={"actor": True, "environment_state": True}) is None
    assert explicit_env.restored is not None


def test_distributed_load_uses_primary_state_but_forks_by_current_rank(tmp_path):
    """All ranks receive rank-0 evidence plus explicit source/current rank context."""
    path = tmp_path / "distributed.pt"
    source = _runner(_Environment({"curriculum_manager": {"reset_sampling": {"attempts": 99}}}), world_size=8)
    source.save(str(path))

    target_env = _Environment()
    target = _runner(target_env, world_size=4, global_rank=3)
    with pytest.warns(RuntimeWarning, match="world size 8"):
        infos = target.load(str(path))

    assert infos is None
    assert target_env.restored is not None
    restored_state, source_rank, current_rank = target_env.restored
    assert restored_state == {"curriculum_manager": {"reset_sampling": {"attempts": 99}}}
    assert source_rank == 0
    assert current_rank == 3


def test_vecenv_wrapper_delegates_curriculum_checkpoint_state():
    """The wrapper restores state before resampling the first resumed rollout."""
    sampled_attempts = []
    manager = SimpleNamespace(
        get_checkpoint_state=lambda: {"reset_sampling": {"attempts": 4}},
        set_checkpoint_state=lambda state, **context: setattr(manager, "restored", (state, context)),
    )
    base_env = SimpleNamespace(curriculum_manager=manager)
    wrapped_env = SimpleNamespace(
        unwrapped=base_env,
        reset=lambda: sampled_attempts.append(manager.restored[0]["reset_sampling"]["attempts"]),
    )
    wrapper = object.__new__(RslRlVecEnvWrapper)
    wrapper.env = wrapped_env

    state = wrapper.get_checkpoint_state()
    wrapper.set_checkpoint_state(state, source_global_rank=0, current_global_rank=2)

    assert state == {"curriculum_manager": {"reset_sampling": {"attempts": 4}}}
    assert manager.restored == (
        {"reset_sampling": {"attempts": 4}},
        {"source_global_rank": 0, "current_global_rank": 2},
    )
    assert sampled_attempts == [4]
