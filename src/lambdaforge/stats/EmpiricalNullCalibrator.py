"""Independent-control rank calibration, distinct from regression conformal intervals."""

from __future__ import annotations

from types import MappingProxyType
from typing import Any

import numpy as np

from .CalibrationResult import CalibrationResult


def finite_vector(values: Any, name: str) -> np.ndarray:
    raw = np.asarray(values)
    if (
        raw.ndim != 1
        or not len(raw)
        or np.iscomplexobj(raw)
        or not np.issubdtype(raw.dtype, np.number)
    ):
        raise ValueError(f"{name} must be a non-empty real numeric vector.")
    result = np.array(raw, dtype=float, copy=True)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite.")
    return result


def grouped_indices(group_ids: Any, n: int) -> list[np.ndarray]:
    """Stable first-occurrence grouping; identifiers are explicit integers or strings."""
    ids = np.asarray(group_ids, dtype=object)
    if ids.shape != (n,):
        raise ValueError("group_ids must have one integer/string ID per observation.")
    groups: dict[str | int, list[int]] = {}
    for index, key in enumerate(ids):
        if isinstance(key, bool) or not isinstance(key, (str, int, np.integer)):
            raise ValueError("Group IDs must be non-boolean integers or strings.")
        groups.setdefault(key if isinstance(key, str) else int(key), []).append(index)
    return [np.asarray(indices, dtype=np.int64) for indices in groups.values()]


class EmpiricalNullCalibrator:
    """Inclusive-tail add-one ranks under independent exchangeable controls.

    Fix score/candidate selection before calibration. A p-value is not a posterior,
    and per-negative-object maxima calibration does not control the false-region
    fraction in positive objects. No conditional/causal guarantee is inferred.
    """

    def __init__(self, *, tail: str = "upper", alpha: float = 0.05) -> None:
        if tail not in {"upper", "lower"}:
            raise ValueError("tail must be 'upper' or 'lower'.")
        if not np.isfinite(alpha) or not 0 < alpha < 1:
            raise ValueError("alpha must be finite and in (0, 1).")
        self.tail, self.alpha = tail, float(alpha)
        self._reference: np.ndarray | None = None
        self._metadata: dict[str, Any] = {}

    def fit(self, reference_scores: np.ndarray) -> EmpiricalNullCalibrator:
        self._reference = finite_vector(reference_scores, "reference_scores")
        self._reference.setflags(write=False)
        self._metadata = {"unit": "independent reference observation", "reducer": None}
        return self

    def fit_groups(
        self, candidate_scores: np.ndarray, group_ids: np.ndarray, *, reducer: str = "max"
    ) -> EmpiricalNullCalibrator:
        values = finite_vector(candidate_scores, "candidate_scores")
        if reducer not in {"max", "min"}:
            raise ValueError("Grouped calibration reducer must be 'max' or 'min'.")
        groups = grouped_indices(group_ids, len(values))
        function = np.max if reducer == "max" else np.min
        self.fit(np.array([function(values[group]) for group in groups]))
        self._metadata = {
            "unit": "independent object/group",
            "reducer": reducer,
            "group_sizes": tuple(len(group) for group in groups),
        }
        return self

    def evaluate(self, new_score: float) -> CalibrationResult:
        if self._reference is None:
            raise RuntimeError("Fit independent reference controls before evaluation.")
        if not np.isscalar(new_score) or np.iscomplexobj(new_score) or not np.isfinite(new_score):
            raise ValueError("new_score must be a finite real scalar.")
        score = float(new_score)
        if not np.isfinite(score):
            raise ValueError("new_score must be representable as a finite float64 scalar.")
        values = self._reference
        count = int(np.count_nonzero(values >= score if self.tail == "upper" else values <= score))
        p = (1 + count) / (len(values) + 1)
        resolution = 1 / (len(values) + 1)
        warnings = (
            ()
            if self.alpha >= resolution
            else ("alpha is below the finite-control resolution; no rejection is possible.",)
        )
        return CalibrationResult(
            score,
            p,
            p <= self.alpha,
            self.alpha,
            len(values),
            self.tail,
            int(np.count_nonzero(values == score)),
            resolution,
            float(
                np.quantile(
                    values,
                    1 - self.alpha if self.tail == "upper" else self.alpha,
                    method="higher" if self.tail == "upper" else "lower",
                )
            ),
            MappingProxyType(dict(self._metadata)),
            warnings,
        )

    def evaluate_group(self, candidate_scores: np.ndarray) -> CalibrationResult:
        """Apply exactly the statistic used by fit_groups to a new object's candidates."""
        reducer = self._metadata.get("reducer")
        if reducer is None:
            raise RuntimeError("evaluate_group requires fit_groups first.")
        values = finite_vector(candidate_scores, "candidate_scores")
        return self.evaluate(float(np.max(values) if reducer == "max" else np.min(values)))
