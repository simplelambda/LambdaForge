"""Percentile bootstrap with explicit independent units and optional fixed strata."""

from __future__ import annotations

from collections.abc import Callable
from types import MappingProxyType

import numpy as np

from .BootstrapResult import BootstrapResult
from .EmpiricalNullCalibrator import finite_vector, grouped_indices


def bootstrap_statistic(
    values: np.ndarray,
    statistic: Callable[[np.ndarray], float] = np.mean,
    *,
    group_ids: np.ndarray | None = None,
    strata: np.ndarray | None = None,
    confidence_level: float = 0.95,
    n_resamples: int = 1000,
    seed: int | None = None,
) -> BootstrapResult:
    """Resample rows, or whole groups including all their rows, with replacement.

    Statistic is evaluated on pooled rows. Unequal group sizes therefore affect
    a pooled mean; first reduce each group to a scalar to target an equally
    weighted object statistic. Strata preserve numbers of independent units,
    not outcome guarantees; each whole group must belong to one stratum.
    This approximate percentile CI is not a null p-value or posterior interval.
    """
    data = finite_vector(values, "values")
    data.setflags(write=False)
    if isinstance(n_resamples, bool) or not isinstance(n_resamples, int) or n_resamples < 1:
        raise ValueError("n_resamples must be a positive integer.")
    if not np.isfinite(confidence_level) or not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be in (0, 1).")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int) or seed < 0):
        raise ValueError("seed must be a nonnegative integer or null.")
    groups = (
        grouped_indices(group_ids, len(data))
        if group_ids is not None
        else [np.array([i]) for i in range(len(data))]
    )
    if strata is None:
        bins = [np.arange(len(groups))]
    else:
        raw_strata = np.asarray(strata, dtype=object)
        if raw_strata.shape != data.shape:
            raise ValueError("strata must have one integer/string label per observation.")
        unit_strata = []
        for group in groups:
            if any(raw_strata[index] != raw_strata[group[0]] for index in group):
                raise ValueError("A group cannot span multiple strata.")
            unit_strata.append(raw_strata[group[0]])
        bins = grouped_indices(unit_strata, len(groups))
    rng = np.random.default_rng(seed)

    def calculate(sample: np.ndarray) -> float:
        result = statistic(sample)
        if not np.isscalar(result) or np.iscomplexobj(result) or not np.isfinite(result):
            raise ValueError("The bootstrap statistic must return a finite real scalar.")
        value = float(result)
        if not np.isfinite(value):
            raise ValueError("The bootstrap statistic must be representable as finite float64.")
        return value

    estimate = calculate(data)
    realizations = np.empty(n_resamples)
    for iteration in range(n_resamples):
        sampled_units = np.concatenate(
            [rng.choice(unit_bin, len(unit_bin), replace=True) for unit_bin in bins]
        )
        indices = np.concatenate([groups[int(unit)] for unit in sampled_units])
        realizations[iteration] = calculate(data[indices])
    tail = (1 - confidence_level) / 2
    low, high = np.quantile(realizations, [tail, 1 - tail])
    return BootstrapResult(
        estimate,
        (float(low), float(high)),
        confidence_level,
        "percentile",
        n_resamples,
        seed,
        "group" if group_ids is not None else "observation",
        len(groups),
        MappingProxyType(
            {
                "n_strata": len(bins),
                "quantile_method": "linear",
                "degenerate": bool(np.all(realizations == realizations[0])),
                "statistic_rows": "pooled; group multiplicity preserved",
            }
        ),
    )


def bootstrap_quantile(
    values: np.ndarray,
    quantile: float,
    *,
    group_ids: np.ndarray | None = None,
    strata: np.ndarray | None = None,
    confidence_level: float = 0.95,
    n_resamples: int = 1000,
    seed: int | None = None,
) -> BootstrapResult:
    """Bootstrap a scalar empirical quantile with the same explicit sampling units."""
    if not np.isfinite(quantile) or not 0 <= quantile <= 1:
        raise ValueError("quantile must be in [0, 1].")
    return bootstrap_statistic(
        values,
        lambda sample: float(np.quantile(sample, quantile)),
        group_ids=group_ids,
        strata=strata,
        confidence_level=confidence_level,
        n_resamples=n_resamples,
        seed=seed,
    )
