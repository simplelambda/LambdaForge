"""Balanced q=2 FGW through the optional, established Python Optimal Transport solver."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from .FGWResult import FGWResult
from .StructuredMeasure import StructuredMeasure


class FusedGromovWassersteinDistance:
    r"""Compare whole weighted attributed objects using Vayer et al. (ICML 2019).

    Minimize ``(1-alpha) sum(d_feature(i,j)^2 T[i,j]) +
    alpha sum((C1[i,k]-C2[j,l])^2 T[i,j] T[k,l])`` subject to probability
    marginals. alpha=0 is feature-only Wasserstein squared; alpha=1 is structural GW.
    Structures need not have the same number of points or coordinate dimensions.
    Features must share a numeric space, or supply a callable metric / cross-distances.

    No automatic feature/structure rescaling is performed. Relative units matter.
    A conditional-gradient local solution is returned, not a certified global minimum
    or a guaranteed metric for arbitrary dissimilarities. No barycenter/clustering is run.
    """

    def __init__(
        self,
        *,
        alpha: float = 0.5,
        feature_metric: str | Callable[[Any, Any], float] = "euclidean",
        max_iter: int = 200,
        tol_rel: float = 1e-9,
        tol_abs: float = 1e-9,
        max_memory_bytes: int = 512 * 1024**2,
    ) -> None:
        if isinstance(alpha, bool) or not np.isfinite(alpha) or not 0 <= alpha <= 1:
            raise ValueError("alpha must be finite and in [0,1].")
        if isinstance(max_iter, bool) or not isinstance(max_iter, int) or max_iter < 1:
            raise ValueError("max_iter must be a positive integer.")
        if any(not np.isfinite(value) or value <= 0 for value in (tol_rel, tol_abs)):
            raise ValueError("tol_rel and tol_abs must be finite and positive.")
        if (
            isinstance(max_memory_bytes, bool)
            or not isinstance(max_memory_bytes, int)
            or max_memory_bytes <= 0
        ):
            raise ValueError("max_memory_bytes must be a positive integer.")
        if not isinstance(feature_metric, str) and not callable(feature_metric):
            raise TypeError("feature_metric must be a SciPy metric name or callable distance.")
        self.alpha, self.feature_metric = float(alpha), feature_metric
        self.max_iter, self.tol_rel, self.tol_abs = max_iter, float(tol_rel), float(tol_abs)
        self.max_memory_bytes = max_memory_bytes

    def __call__(
        self, source: StructuredMeasure, target: StructuredMeasure, **kwargs: Any
    ) -> float:
        """Return the unrooted FGW objective; use solve for coupling/diagnostics."""
        return self.solve(source, target, **kwargs).value

    def solve(
        self,
        source: StructuredMeasure,
        target: StructuredMeasure,
        *,
        feature_distances: ArrayLike | None = None,
        initial_coupling: ArrayLike | None = None,
    ) -> FGWResult:
        """Solve one comparison. Precomputed/callable feature distances are squared once.

        ``feature_distances`` has shape [N,M] and permits arbitrary domain attributes,
        without parsing their internal representation. ``initial_coupling`` is optional
        feasible [N,M] state for an explicit warm start; there is no hidden random seed.
        Dense memory is conservatively estimated before cross-distance allocation, but
        third-party solver peak memory is not a hard process-memory limit.
        """
        if not isinstance(source, StructuredMeasure) or not isinstance(target, StructuredMeasure):
            raise TypeError("FGW compares StructuredMeasure objects, not clustering features.")
        n, m = len(source.structure), len(target.structure)
        estimate = 8 * (4 * (n * n + m * m) + 12 * n * m)
        if estimate > self.max_memory_bytes:
            raise MemoryError("Dense FGW workspace estimate exceeds max_memory_bytes.")
        # Leave room for sums/products in the quadratic objective, not just C**2.
        safe_scale = np.sqrt(np.finfo(np.float64).max / 16)
        if max(np.max(source.structure), np.max(target.structure)) > safe_scale:
            raise ValueError("Structural distances overflow squared costs; choose explicit units.")
        try:
            from ot.gromov import fused_gromov_wasserstein
        except ImportError as error:
            raise ImportError("Install lambdaforge[optimal-transport] to calculate FGW.") from error
        cost = self._feature_cost(source, target, feature_distances)
        initial = None
        if initial_coupling is not None:
            initial = np.array(initial_coupling, dtype=np.float64, copy=True)
            self._check_coupling(initial, source, target)
        elif source is target:
            # A known exact feasible zero-cost self-coupling, not an invented zero score.
            initial = np.diag(source.weights)
        transport, log = fused_gromov_wasserstein(
            cost,
            source.structure.copy(),
            target.structure.copy(),
            source.weights.copy(),
            target.weights.copy(),
            loss_fun="square_loss",
            symmetric=True,
            alpha=self.alpha,
            G0=initial,
            log=True,
            max_iter=self.max_iter,
            tol_rel=self.tol_rel,
            tol_abs=self.tol_abs,
        )
        transport = np.asarray(transport, dtype=np.float64)
        if log.get("warning"):
            raise RuntimeError(f"FGW transport solver failed: {log['warning']}")
        marginal_error = self._check_coupling(transport, source, target)
        history = tuple(float(value) for value in log["loss"])
        if not history or not np.isfinite(history).all():
            raise RuntimeError("FGW backend returned an invalid objective history.")
        feature = float(np.sum(cost * transport))
        # q=2 contraction: never allocate the [N,M,N,M] structural-loss tensor.
        structural = float(
            source.weights @ (source.structure**2) @ source.weights
            + target.weights @ (target.structure**2) @ target.weights
            - 2 * np.sum((source.structure @ transport @ target.structure.T) * transport)
        )
        if structural < -1e-8 * max(
            1.0, float(np.max(source.structure)) ** 2, float(np.max(target.structure)) ** 2
        ):
            raise RuntimeError("FGW backend produced a negative structural discrepancy.")
        structural = max(0.0, structural)  # Round-off only in squared-distance cancellation.
        value = (1 - self.alpha) * feature + self.alpha * structural
        delta = abs(history[-1] - history[-2]) if len(history) > 1 else float("inf")
        converged = delta < self.tol_abs or delta < self.tol_rel * abs(history[-1])
        return FGWResult(
            value,
            transport,
            feature,
            structural,
            self.alpha,
            converged,
            len(history) - 1,
            history,
            marginal_error,
        )

    def _feature_cost(
        self,
        source: StructuredMeasure,
        target: StructuredMeasure,
        distances: ArrayLike | None,
    ) -> np.ndarray:
        shape = (len(source.structure), len(target.structure))
        if distances is not None:
            values = np.array(distances, dtype=np.float64, copy=True)
        elif self.alpha == 1:
            return np.zeros(shape, dtype=np.float64)
        elif source.features is None or target.features is None:
            raise ValueError("FGW needs features or feature_distances when alpha < 1.")
        elif callable(self.feature_metric):
            values = np.array(
                [
                    [self.feature_metric(left, right) for right in target.features]
                    for left in source.features
                ],
                dtype=np.float64,
            )
        else:
            from scipy.spatial.distance import cdist

            left, right = (np.asarray(obj.features, dtype=np.float64) for obj in (source, target))
            if left.ndim == 1:
                left = left[:, None]
            if right.ndim == 1:
                right = right[:, None]
            if (
                left.ndim != 2
                or right.ndim != 2
                or left.shape[1] != right.shape[1]
                or left.shape[1] == 0
                or not np.isfinite(left).all()
                or not np.isfinite(right).all()
            ):
                raise ValueError("Numeric features must be finite [N,F]/[M,F] in a shared space.")
            values = cdist(left, right, metric=self.feature_metric)
        if values.shape != shape or not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError("feature_distances must be finite nonnegative [N,M] distances.")
        with np.errstate(over="raise", invalid="raise"):
            try:
                return np.square(values)
            except FloatingPointError as error:
                raise ValueError(
                    "Feature distances overflow when squared; choose explicit units."
                ) from error

    @staticmethod
    def _check_coupling(
        coupling: np.ndarray,
        source: StructuredMeasure,
        target: StructuredMeasure,
    ) -> float:
        if (
            coupling.shape != (len(source.structure), len(target.structure))
            or not np.isfinite(coupling).all()
            or np.any(coupling < 0)
        ):
            raise ValueError("Coupling must be a finite nonnegative [N,M] matrix.")
        error = max(
            float(np.max(np.abs(coupling.sum(axis=1) - source.weights))),
            float(np.max(np.abs(coupling.sum(axis=0) - target.weights))),
        )
        if error > 1e-7:
            raise ValueError("Coupling does not satisfy the source/target probability marginals.")
        return error

    def pairwise(
        self, objects: Sequence[StructuredMeasure], *, symmetric: bool = False
    ) -> np.ndarray:
        """Return [K,K] object discrepancies, not node distances or cluster labels.

        By default each orientation is solved independently (local optima may differ).
        ``symmetric=True`` explicitly assumes a symmetric feature metric and mirrors the
        upper triangle, saving solves. The diagonal uses a known feasible self-coupling.
        """
        count = len(objects)
        if any(not isinstance(obj, StructuredMeasure) for obj in objects):
            raise TypeError("pairwise expects StructuredMeasure objects.")
        output_bytes = count * count * 8
        if output_bytes > self.max_memory_bytes:
            raise MemoryError("Pairwise output exceeds max_memory_bytes.")
        largest = max((len(obj.structure) for obj in objects), default=0)
        if output_bytes + 160 * largest * largest > self.max_memory_bytes:
            raise MemoryError("Pairwise output plus dense FGW workspace exceeds max_memory_bytes.")
        result = np.empty((count, count), dtype=np.float64)
        for i, left in enumerate(objects):
            for j in range(i if symmetric else 0, count):
                result[i, j] = self(left, objects[j])
                if symmetric:
                    result[j, i] = result[i, j]
        return result
