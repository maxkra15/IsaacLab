# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pure tests for optional RSL-RL environment-step checkpoint restoration."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from isaaclab_rl.entrypoints.backends import train_rsl_rl


class _RecordingCurriculumManager:
    def __init__(self, env: SimpleNamespace, calls: list[tuple[str, int]]):
        self._env = env
        self._calls = calls

    def compute_step(self) -> None:
        self._calls.append(("curriculum", self._env.common_step_counter))


class _RecordingObservationManager:
    def __init__(self, env: SimpleNamespace, calls: list[tuple[str, int]]):
        self._env = env
        self._calls = calls

    def compute(self) -> dict[str, int]:
        self._calls.append(("observation", self._env.common_step_counter))
        return {"restored_step": self._env.common_step_counter}


def _environment() -> tuple[SimpleNamespace, SimpleNamespace, list[tuple[str, int]]]:
    calls: list[tuple[str, int]] = []
    base_env = SimpleNamespace(common_step_counter=0, device="cpu", obs_buf={"restored_step": 0})
    base_env.curriculum_manager = _RecordingCurriculumManager(base_env, calls)
    base_env.observation_manager = _RecordingObservationManager(base_env, calls)
    return SimpleNamespace(unwrapped=base_env), base_env, calls


def test_optional_checkpoint_hook_binds_live_environment_step() -> None:
    env, base_env, _ = _environment()
    providers = []
    algorithm = SimpleNamespace(bind_environment_step_provider=providers.append)

    assert train_rsl_rl._bind_environment_step_checkpoint(SimpleNamespace(alg=algorithm), env)
    assert len(providers) == 1
    base_env.common_step_counter = 321
    assert providers[0]() == 321


def test_baseline_algorithm_without_checkpoint_hook_is_unchanged() -> None:
    env, base_env, calls = _environment()
    runner = SimpleNamespace(alg=SimpleNamespace())

    assert not train_rsl_rl._bind_environment_step_checkpoint(runner, env)
    assert (
        train_rsl_rl._restore_environment_step_checkpoint(
            runner,
            env,
            num_steps_per_env=100,
            distributed=False,
        )
        is None
    )
    assert base_env.common_step_counter == 0
    assert calls == []


def test_exact_step_is_applied_before_curriculum_and_observation_refresh() -> None:
    env, base_env, calls = _environment()
    runner = SimpleNamespace(
        alg=SimpleNamespace(
            restored_environment_common_step_counter=145_137,
            completed_updates=1_451,
        )
    )

    restored_step = train_rsl_rl._restore_environment_step_checkpoint(
        runner,
        env,
        num_steps_per_env=100,
        distributed=False,
    )

    assert restored_step == 145_137
    assert base_env.common_step_counter == 145_137
    assert calls == [("curriculum", 145_137), ("observation", 145_137)]
    assert base_env.obs_buf == {"restored_step": 145_137}


def test_legacy_checkpoint_derives_step_from_completed_rollouts(caplog: pytest.LogCaptureFixture) -> None:
    env, base_env, calls = _environment()
    runner = SimpleNamespace(
        alg=SimpleNamespace(
            restored_environment_common_step_counter=None,
            completed_updates=1_451,
        )
    )

    restored_step = train_rsl_rl._restore_environment_step_checkpoint(
        runner,
        env,
        num_steps_per_env=100,
        distributed=False,
    )

    assert restored_step == 145_100
    assert base_env.common_step_counter == 145_100
    assert calls == [("curriculum", 145_100), ("observation", 145_100)]
    assert "predates exact environment-step persistence" in caplog.text


def test_distributed_restore_broadcasts_and_accepts_matching_rank_zero_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    broadcast_calls = []
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda _tensor, op: None)

    def broadcast(tensor: torch.Tensor, src: int) -> None:
        broadcast_calls.append((tensor.item(), src))

    monkeypatch.setattr(torch.distributed, "broadcast", broadcast)

    assert train_rsl_rl._synchronize_environment_step(145_137, distributed=True, device="cpu") == 145_137
    assert broadcast_calls == [(145_137, 0)]


def test_distributed_restore_rejects_rank_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda _tensor, op: None)

    def broadcast_rank_zero_step(tensor: torch.Tensor, src: int) -> None:
        assert src == 0
        tensor.fill_(145_100)

    monkeypatch.setattr(torch.distributed, "broadcast", broadcast_rank_zero_step)

    with pytest.raises(RuntimeError, match="local=145137, rank_zero=145100"):
        train_rsl_rl._synchronize_environment_step(145_137, distributed=True, device="cpu")
