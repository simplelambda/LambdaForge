"""Finite-sample inclusive ranks and object-level resampling evidence."""

import numpy as np
import pytest

from lambdaforge.stats import EmpiricalNullCalibrator, bootstrap_quantile, bootstrap_statistic


@pytest.mark.parametrize(
    "tail,score,expected",
    [("upper", 2, 0.8), ("lower", 2, 0.8), ("upper", 4, 0.2), ("lower", 0, 0.2)],
)
def test_hand_computed_ranks_and_ties(tail, score, expected):
    report = (
        EmpiricalNullCalibrator(tail=tail, alpha=0.25).fit(np.array([1, 2, 2, 3])).evaluate(score)
    )
    assert report.p_value == expected
    assert report.ties == (2 if score == 2 else 0)
    assert report.reject == (expected <= 0.25)
    assert report.n_controls == 4 and report.resolution == 0.2


def test_resolution_cannot_produce_rejection_and_frozen_evidence():
    calibrator = EmpiricalNullCalibrator(alpha=0.01).fit(np.array([1, 2]))
    result = calibrator.evaluate(10)
    assert not result.reject and result.warnings
    with pytest.raises(TypeError):
        result.metadata["unit"] = "changed"
    with pytest.raises(RuntimeError):
        EmpiricalNullCalibrator().evaluate(1)


def test_group_maxima_not_pseudo_replicated_candidates():
    values = np.array([1, 2, 3, 4, 5, 6])
    groups = np.array(["a", "b", "b", "b", "c", "c"])
    calibrated = EmpiricalNullCalibrator().fit_groups(values, groups)
    result = calibrated.evaluate_group(np.array([3.0, 5.0]))
    assert result.n_controls == 3 and result.p_value == 0.5
    assert result.metadata["group_sizes"] == (1, 3, 2)
    lower = EmpiricalNullCalibrator(tail="lower").fit_groups(values, groups, reducer="min")
    assert lower.evaluate_group(np.array([2.0, 5.0])).p_value == 0.75


@pytest.mark.parametrize("invalid", [[], [np.nan], [np.inf], [1j], [[1]]])
def test_invalid_controls(invalid):
    with pytest.raises(ValueError):
        EmpiricalNullCalibrator().fit(invalid)


def test_bootstrap_reproducibility_units_and_strata():
    values = np.array([0.0, 0.0, 10.0, 10.0, 10.0, 10.0])
    groups = np.array([0, 0, 1, 1, 1, 1])
    kwargs = dict(group_ids=groups, n_resamples=100, seed=42)
    first = bootstrap_statistic(values, **kwargs)
    assert first == bootstrap_statistic(values, **kwargs)
    assert first.estimate == pytest.approx(40 / 6)
    assert first.interval == (0.0, 10.0)
    assert first.unit == "group" and first.n_units == 2
    stratified = bootstrap_statistic(values, strata=groups, **kwargs)
    assert stratified.diagnostics["degenerate"]
    with pytest.raises(ValueError, match="span"):
        bootstrap_statistic(values, strata=np.arange(6), **kwargs)
    quantile = bootstrap_quantile(values, 0.5, seed=7, n_resamples=100)
    assert quantile.estimate == 10 and quantile.method == "percentile"
    assert quantile.seed == 7 and quantile.n_resamples == 100
    with pytest.raises(ValueError, match="finite real scalar"):
        bootstrap_statistic(values, lambda x: float("nan"))


def test_exchangeable_negative_object_maxima_simulation():
    # Each reference/test object has the SAME predeclared candidate construction.
    rng = np.random.default_rng(123)
    rejected = []
    for _ in range(1000):
        controls = rng.normal(size=(39, 4))
        fit = EmpiricalNullCalibrator(alpha=0.1).fit_groups(
            controls.ravel(), np.repeat(np.arange(39), 4)
        )
        rejected.append(fit.evaluate_group(rng.normal(size=4)).reject)
    assert 0.065 <= np.mean(rejected) <= 0.13


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_resamples": 0},
        {"confidence_level": 1},
        {"seed": -1},
        {"group_ids": [0, float("nan")]},
        {"strata": [0]},
    ],
)
def test_bootstrap_invalid_sampling(kwargs):
    with pytest.raises(ValueError):
        bootstrap_statistic(np.array([1.0, 2.0]), **kwargs)
