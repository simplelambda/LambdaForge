"""Bound live ARI feedback and joint-fit work without dropping rare OOM tails."""

from __future__ import annotations

import importlib
import itertools
import math
from dataclasses import replace

import pytest

from lambdaforge.hpo.AdaptiveResources import (
    _RESOURCE_SUPPORT_LIMIT,
    ActiveResourceCommitment,
    ActiveResourceEvidence,
    GPUResourceState,
    ResourceDemandModel,
    ResourcePrediction,
    ResourceTrajectoryAnalyzer,
    ResourceTrajectorySample,
    _coalesce_weighted_samples,
    _joint_fit_probability,
    _weighted_cdf,
    _weighted_quantile,
)

GIB = 1024**3


def prediction(samples: tuple[int, ...], weights: tuple[float, ...]) -> ResourcePrediction:
    return ResourcePrediction(
        "candidate",
        samples[0],
        min(samples),
        max(samples),
        0,
        None,
        None,
        0,
        0,
        "active",
        "poor",
        "test",
        samples,
        sample_weights=weights,
    )


def test_bounded_support_is_conservative_and_keeps_exact_endpoint_mass() -> None:
    samples = tuple(range(20_000))
    weights = tuple([0.99 / 19_999] * 19_999 + [0.01])
    bounded, masses = _coalesce_weighted_samples(samples, weights)
    assert len(bounded) <= _RESOURCE_SUPPORT_LIMIT
    assert (bounded[0], bounded[-1]) == (0, 19_999)
    assert math.isclose(masses[0], weights[0])
    assert math.isclose(masses[-1], weights[-1])
    assert math.isclose(sum(masses), 1.0)
    for limit in range(0, 20_000, 73):
        assert _weighted_cdf(bounded, masses, limit) <= sum(weights[: limit + 1]) + 1e-12
    assert _coalesce_weighted_samples(samples, weights) == (bounded, masses)
    assert _coalesce_weighted_samples((1, 2, 2, 100), (0.2, 0.3, 0.49, 0.01)) == (
        (1, 2, 100),
        (0.2, 0.79, 0.01),
    )


@pytest.mark.parametrize("free,total", [(20, 40), (30, 80), (70, 80), (90, 150)])
def test_cdf_event_sweep_matches_exact_small_support_ensemble(free: int, total: int) -> None:
    distributions = [
        ((10, 20, 80), (0.7, 0.295, 0.005)),
        ((5, 30), (0.8, 0.2)),
        ((10, 70), (0.99, 0.01)),
    ]
    residents = tuple(
        ActiveResourceCommitment(
            str(index),
            samples[0],
            current_bytes=samples[0],
            future_peak_samples=samples,
            future_peak_weights=weights,
        )
        for index, (samples, weights) in enumerate(distributions[1:])
    )
    device = GPUResourceState(0, "granted", "test", total, free, 3, residents, 8)
    independent = 0.0
    for draws in itertools.product(*(tuple(zip(*value, strict=True)) for value in distributions)):
        if 3 + sum(draw[0] for draw in draws) <= total:
            independent += math.prod(draw[1] for draw in draws)
    boundaries = {0.0, 1.0}
    for _samples, weights in distributions:
        boundaries.update(itertools.accumulate(weights))
    shared = 0.0
    ordered = sorted(boundaries)
    for left, right in zip(ordered, ordered[1:], strict=False):
        draws = [_weighted_quantile(*value, (left + right) / 2) for value in distributions]
        if 3 + sum(draws) <= total and draws[0] <= free:
            shared += right - left
    expected = min(_weighted_cdf(*distributions[0], free), (independent + shared) / 2)
    assert math.isclose(_joint_fit_probability(prediction(*distributions[0]), device), expected)


def test_large_legacy_posteriors_are_bounded_before_convolution(monkeypatch) -> None:
    module = importlib.import_module("lambdaforge.hpo.AdaptiveResources")
    original = module._compress_weighted_support
    sizes: list[int] = []

    def compressed(values):
        sizes.append(len(values))
        return original(values)

    def no_repeated_quantiles(*_args):
        raise AssertionError("joint fit must integrate CDF events, not repeatedly sort supports")

    monkeypatch.setattr(module, "_compress_weighted_support", compressed)
    monkeypatch.setattr(module, "_weighted_quantile", no_repeated_quantiles)
    samples = tuple(range(30 * GIB, 30 * GIB + 6739)) + (140 * GIB,)
    weights = tuple([0.995 / 6739] * 6739 + [0.005])
    resident = ActiveResourceCommitment(
        "resident",
        140 * GIB,
        current_bytes=30 * GIB,
        future_peak_samples=samples,
        future_peak_weights=weights,
    )
    device = GPUResourceState(
        0, "granted", "H200", 140 * GIB, 50 * GIB, 0, (resident, resident, resident), 10
    )
    probability = _joint_fit_probability(prediction(samples, weights), device)
    assert 0 < probability < 1
    # Linear input normalization, then at most bounded × bounded temporary support, regardless
    # of how large an older persisted posterior was. No elapsed-time assertion on CI machines.
    assert max(sizes) <= max(len(samples), _RESOURCE_SUPPORT_LIMIT**2)


def test_live_prediction_feedback_stays_bounded_without_terminal_runs() -> None:
    evidence = tuple(
        ActiveResourceEvidence(
            candidate_key=f"c{i}",
            compatibility_key="compatible",
            parameters={"width": i},
            hardware="H200",
            total_bytes=140 * GIB,
            current_bytes=30 * GIB,
            running_peak_bytes=30 * GIB,
            future_peak_samples=(30 * GIB, 140 * GIB),
            future_peak_weights=(0.99, 0.01),
            resource_state="PROVISIONALLY_STABLE",
            elapsed_seconds=10,
        )
        for i in range(3)
    )
    for cycle in range(70):
        model = ResourceDemandModel(active_evidence=evidence, parameter_schema={"width": [0, 1, 2]})
        updated = []
        for index, value in enumerate(evidence):
            pred = model.predict(
                candidate_key=value.candidate_key,
                compatibility_key="compatible",
                parameters=value.parameters,
                hardware="H200",
                total_bytes=140 * GIB,
            )
            peak = (30 + index) * GIB + cycle * 1024
            analysis = ResourceTrajectoryAnalyzer.analyze(
                tuple(
                    ResourceTrajectorySample(float(step), peak, step=step, phase="training")
                    for step in range(4)
                ),
                historical_residuals=pred.future_residual_samples,
                historical_residual_weights=pred.sample_weights,
                total_bytes=140 * GIB,
            )
            assert len(pred.samples) <= _RESOURCE_SUPPORT_LIMIT
            assert len(analysis.residual_samples) <= _RESOURCE_SUPPORT_LIMIT
            updated.append(
                replace(
                    value,
                    running_peak_bytes=peak,
                    future_peak_samples=tuple(
                        peak + residual for residual in analysis.residual_samples
                    ),
                    future_peak_weights=analysis.residual_weights,
                )
            )
        evidence = tuple(updated)


def test_trajectory_feedback_preserves_nonuniform_historical_tail_weights() -> None:
    samples = tuple(
        ResourceTrajectorySample(float(i), 10 * GIB, step=i, phase="training") for i in range(8)
    )
    options = dict(values=samples, historical_residuals=(GIB, 2 * GIB), total_bytes=80 * GIB)
    rare = ResourceTrajectoryAnalyzer.analyze(**options, historical_residual_weights=(0.999, 0.001))
    common = ResourceTrajectoryAnalyzer.analyze(**options, historical_residual_weights=(0.5, 0.5))
    rare_mass = dict(zip(rare.residual_samples, rare.residual_weights, strict=True))
    common_mass = dict(zip(common.residual_samples, common.residual_weights, strict=True))
    assert math.isclose(rare_mass[2 * GIB] / common_mass[2 * GIB], 0.002)
    assert math.isclose(rare_mass[70 * GIB], common_mass[70 * GIB])
