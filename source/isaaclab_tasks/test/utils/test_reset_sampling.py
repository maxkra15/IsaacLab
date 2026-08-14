# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Public-contract tests for task-independent reset sampling."""

import pytest
import torch

from isaaclab_tasks.utils.reset_sampling import (
    AdaptiveResetSampler,
    AdaptiveResetSamplerCfg,
    ResetStateCatalog,
    RollingOutcomeMonitor,
    RollingOutcomeMonitorCfg,
)


def test_catalog_validates_metadata_and_maps_rows_to_items() -> None:
    catalog = ResetStateCatalog(
        row_count=4,
        metadata={"phase": torch.tensor([0, 0, 1, 1]), "pose": torch.zeros(4, 7)},
        row_to_item=torch.tensor([0, 0, 1, 1]),
    )

    assert catalog.item_count == 2
    assert torch.equal(catalog.item_ids_for_rows(torch.tensor([3, 0, 2])), torch.tensor([1, 0, 1]))

    with pytest.raises(ValueError, match="leading dimension"):
        ResetStateCatalog(row_count=4, metadata={"phase": torch.zeros(3)})
    with pytest.raises(ValueError, match="contiguous"):
        ResetStateCatalog(row_count=4, row_to_item=torch.tensor([0, 0, 2, 2]))


def test_monitor_uses_target_centered_prior_for_unseen_items() -> None:
    monitor = RollingOutcomeMonitor(
        3,
        RollingOutcomeMonitorCfg(history_length=4, prior_strength=2.0),
        "cpu",
        prior_success_rate=0.4,
    )

    assert torch.allclose(monitor.success_rates, torch.full((3,), 0.4))
    monitor.record(torch.tensor([1]), torch.tensor([True]))
    assert monitor.success_rates.tolist() == pytest.approx([0.4, 0.6, 0.4])


def test_monitor_handles_repeated_ids_and_ring_rollover_in_order() -> None:
    monitor = RollingOutcomeMonitor(
        2,
        RollingOutcomeMonitorCfg(history_length=3, prior_strength=0.0),
        "cpu",
        prior_success_rate=0.5,
    )

    monitor.record(
        torch.tensor([0, 0, 1, 0]),
        torch.tensor([True, False, True, True]),
    )
    assert monitor.history_sizes.tolist() == [3, 1]
    assert monitor.success_rates.tolist() == pytest.approx([2.0 / 3.0, 1.0])

    # More than one complete ring in a single update must retain the newest three values.
    monitor.record(
        torch.tensor([0, 0, 0, 0, 0]),
        torch.tensor([True, True, False, False, False]),
    )
    assert monitor.history_sizes.tolist() == [3, 1]
    assert monitor.success_rates.tolist() == pytest.approx([0.0, 1.0])


def test_monitor_valid_mask_and_state_roundtrip() -> None:
    cfg = RollingOutcomeMonitorCfg(history_length=3, prior_strength=1.0)
    first = RollingOutcomeMonitor(3, cfg, "cpu", prior_success_rate=0.5)
    first.record(
        torch.tensor([0, 1, 1, 2]),
        torch.tensor([True, False, True, False]),
        valid=torch.tensor([True, False, True, False]),
    )

    restored = RollingOutcomeMonitor(3, cfg, "cpu", prior_success_rate=0.5)
    restored.load_state_dict(first.state_dict())
    assert torch.equal(restored.history_sizes, first.history_sizes)
    assert torch.equal(restored.success_rates, first.success_rates)

    item_ids = torch.tensor([0, 0, 1, 2, 2])
    outcomes = torch.tensor([False, True, False, True, True])
    first.record(item_ids, outcomes)
    restored.record(item_ids, outcomes)
    assert torch.equal(restored.success_rates, first.success_rates)


def test_monitor_rejects_inconsistent_partially_filled_ring_pointer() -> None:
    monitor = RollingOutcomeMonitor(
        2,
        RollingOutcomeMonitorCfg(history_length=3, prior_strength=1.0),
        "cpu",
        prior_success_rate=0.5,
    )
    monitor.record(torch.tensor([0]), torch.tensor([True]))
    state = monitor.state_dict()
    state["pointers"][0] = 0

    with pytest.raises(ValueError, match="partially filled"):
        monitor.load_state_dict(state)


def test_coverage_stream_visits_each_eligible_item_once_per_cycle() -> None:
    generator = torch.Generator(device="cpu").manual_seed(17)
    sampler = AdaptiveResetSampler(
        6,
        AdaptiveResetSamplerCfg(coverage_fraction=1.0),
        "cpu",
        eligible_mask=torch.tensor([True, False, True, True, False, True]),
        generator=generator,
    )
    rates = torch.full((6,), 0.5)

    first_cycle = torch.cat((sampler.sample(1, rates), sampler.sample(3, rates)))
    second_cycle = sampler.sample(4, rates)
    expected = torch.tensor([0, 2, 3, 5])
    assert torch.equal(first_cycle.sort().values, expected)
    assert torch.equal(second_cycle.sort().values, expected)


def test_masks_and_base_weights_define_adaptive_probabilities() -> None:
    sampler = AdaptiveResetSampler(
        4,
        AdaptiveResetSamplerCfg(kappa=0.0, coverage_fraction=0.0),
        "cpu",
        eligible_mask=torch.tensor([True, True, False, True]),
        base_weights=torch.tensor([1.0, 2.0, 0.0, 3.0]),
        generator=torch.Generator(device="cpu").manual_seed(3),
    )
    probabilities = sampler.adaptive_probabilities(torch.tensor([0.0, 0.2, 0.7, 1.0]))

    assert torch.allclose(probabilities, torch.tensor([1.0, 2.0, 0.0, 3.0]) / 6.0)
    assert not bool((sampler.sample(100, torch.full((4,), 0.5)) == 2).any())


def test_fractional_coverage_credit_is_exact_across_calls() -> None:
    sampler = AdaptiveResetSampler(
        5,
        AdaptiveResetSamplerCfg(coverage_fraction=0.25),
        "cpu",
        generator=torch.Generator(device="cpu").manual_seed(5),
    )
    rates = torch.full((5,), 0.5)

    sampler.sample(3, rates)
    assert sampler.metrics(rates)["coverage_assignments"] == 0.0
    sampler.sample(1, rates)
    assert sampler.metrics(rates)["coverage_assignments"] == 1.0
    sampler.sample(8, rates)
    metrics = sampler.metrics(rates)
    assert metrics["coverage_assignments"] == 3.0
    assert metrics["realized_coverage_fraction"] == pytest.approx(0.25)

    decimal_sampler = AdaptiveResetSampler(
        5,
        AdaptiveResetSamplerCfg(coverage_fraction=0.1),
        "cpu",
        generator=torch.Generator(device="cpu").manual_seed(6),
    )
    for _ in range(10):
        decimal_sampler.sample(1, rates)
    assert decimal_sampler.metrics(rates)["coverage_assignments"] == 1.0


def test_sampler_state_roundtrip_restores_rng_and_coverage_cycle() -> None:
    cfg = AdaptiveResetSamplerCfg(target_success_rate=0.4, kappa=2.0, coverage_fraction=0.35)
    rates = torch.tensor([0.0, 0.2, 0.4, 0.7, 1.0])
    first = AdaptiveResetSampler(
        5,
        cfg,
        "cpu",
        base_weights=torch.tensor([1.0, 2.0, 3.0, 2.0, 1.0]),
        generator=torch.Generator(device="cpu").manual_seed(42),
    )
    first.sample(13, rates)

    restored = AdaptiveResetSampler(
        5,
        cfg,
        "cpu",
        base_weights=torch.tensor([1.0, 2.0, 3.0, 2.0, 1.0]),
        generator=torch.Generator(device="cpu").manual_seed(999),
    )
    restored.load_state_dict(first.state_dict())

    assert torch.equal(first.sample(31, rates), restored.sample(31, rates))
    assert first.metrics(rates) == restored.metrics(rates)
