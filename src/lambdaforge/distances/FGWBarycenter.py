"""Balanced numeric-feature FGW barycenters using the real optional POT optimizer."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import replace
from importlib.metadata import version
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from threadpoolctl import threadpool_limits

from .FGWBarycenterResult import FGWBarycenterResult
from .FusedGromovWassersteinDistance import FusedGromovWassersteinDistance
from .StructuredMeasure import StructuredMeasure


class FGWBarycenter:
    """Optimize a synthetic relational prototype, not a medoid or Euclidean centroid.

    Only finite numeric features [N,F] in one shared space and squared-Euclidean
    feature cost are supported. alpha is strictly between zero and one. Object
    and support weights are explicitly normalized; support weights must be >0
    because POT's barycenter updates divide by them. No structure/feature rescaling
    or zeroing the diagonal is performed. n_support is an authored hyperparameter.
    Multistart keeps the smallest weighted FINAL-prototype postcheck objective.
    """

    def __init__(
        self,
        n_support: int,
        *,
        alpha: float = 0.5,
        max_iter: int = 100,
        tol: float = 1e-7,
        seed: int | None = None,
        n_init: int = 1,
        max_memory_bytes: int = 512 * 1024**2,
        threads: int = 1,
        transport_max_iter: int = 200,
        transport_tol: float = 1e-9,
    ) -> None:
        for name, value in (
            ("n_support", n_support),
            ("max_iter", max_iter),
            ("n_init", n_init),
            ("max_memory_bytes", max_memory_bytes),
            ("threads", threads),
            ("transport_max_iter", transport_max_iter),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if isinstance(alpha, bool) or not np.isfinite(alpha) or not 0 < alpha < 1:
            raise ValueError("FGW barycenters currently require alpha strictly in (0,1).")
        if any(not np.isfinite(t) or t <= 0 for t in (tol, transport_tol)):
            raise ValueError("tol and transport_tol must be finite and positive.")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int) or seed < 0):
            raise ValueError("seed must be a nonnegative integer or null.")
        self.n_support, self.alpha, self.max_iter, self.tol = (
            n_support,
            float(alpha),
            max_iter,
            float(tol),
        )
        self.seed, self.n_init, self.max_memory_bytes = seed, n_init, max_memory_bytes
        self.threads, self.transport_max_iter, self.transport_tol = (
            threads,
            transport_max_iter,
            transport_tol,
        )

    @staticmethod
    def _weights(weights: ArrayLike | None, count: int, *, positive: bool = False) -> np.ndarray:
        raw = np.ones(count) if weights is None else np.asarray(weights)
        if (
            raw.shape != (count,)
            or np.iscomplexobj(raw)
            or not np.issubdtype(raw.dtype, np.number)
            or not np.isfinite(raw).all()
            or np.any(raw < 0)
        ):
            raise ValueError("Weights must be finite nonnegative real values with matching count.")
        values = np.array(raw, dtype=float, copy=True)
        total = float(np.sum(values, dtype=np.longdouble))
        if not np.isfinite(total) or total <= 0 or (positive and np.any(values <= 0)):
            raise ValueError(
                "Weights need positive finite total; support weights must be strictly positive."
            )
        normalized = values / total
        if positive and np.any(normalized <= 0):
            raise ValueError(
                "Normalized support weights underflow float64; revise the mass ratios."
            )
        return normalized

    def fit(
        self,
        objects: Sequence[StructuredMeasure],
        *,
        object_weights: ArrayLike | None = None,
        support_weights: ArrayLike | None = None,
        init_structure: ArrayLike | None = None,
        init_features: ArrayLike | None = None,
    ) -> FGWBarycenterResult:
        """Return the best valid local start, retaining nonconvergence and warnings.

        Explicit C/X initializations apply to each start. If both are given,
        restarts may be identical; no unrequested perturbation is introduced.
        Memory estimate includes POT's per-iteration transport logging.
        """
        items = tuple(objects)
        if not items or any(not isinstance(obj, StructuredMeasure) for obj in items):
            raise ValueError("Barycenters require nonempty StructuredMeasure objects.")
        sizes = [len(obj.structure) for obj in items]
        feature_count = None
        for obj in items:
            features = np.asarray(obj.features)
            if (
                features.ndim != 2
                or features.shape[0] != len(obj.structure)
                or not features.shape[1]
                or np.iscomplexobj(features)
                or not np.issubdtype(features.dtype, np.number)
                or not np.isfinite(features).all()
            ):
                raise ValueError("Barycenter features must be finite real numeric [N,F] matrices.")
            if feature_count is not None and features.shape[1] != feature_count:
                raise ValueError("All barycenter features must share the same dimension.")
            feature_count = features.shape[1]
        n = self.n_support
        estimate = 8 * (
            8 * sum(s * s for s in sizes)
            + 8 * n * n
            + 24 * n * sum(sizes)
            + 8 * (sum(sizes) + n) * int(feature_count or 0)
            + self.max_iter * n * sum(sizes)
        )
        if estimate > self.max_memory_bytes:
            raise MemoryError("FGW barycenter workspace/history estimate exceeds max_memory_bytes.")
        lambdas = self._weights(object_weights, len(items))
        p = self._weights(support_weights, n, positive=True)
        initial_c = None
        if init_structure is not None:
            if np.shape(init_structure) != (n, n):
                raise ValueError("init_structure must match [n_support,n_support].")
            initial_c = StructuredMeasure(init_structure, weights=p, relational=True).structure
        initial_x = None
        if init_features is not None:
            raw_x = np.asarray(init_features)
            if (
                raw_x.shape != (n, feature_count)
                or np.iscomplexobj(raw_x)
                or not np.issubdtype(raw_x.dtype, np.number)
                or not np.isfinite(raw_x).all()
            ):
                raise ValueError("init_features must be finite real [n_support,F].")
            initial_x = np.array(raw_x, dtype=float, copy=True)
        safe_scale = np.sqrt(np.finfo(float).max / 64 / max(1, int(feature_count or 0)))
        for obj in items:
            if np.max(obj.structure) > safe_scale or np.max(np.abs(obj.features)) > safe_scale:
                raise ValueError("Barycenter squared costs would overflow; choose explicit units.")
        if (initial_c is not None and np.max(initial_c) > safe_scale) or (
            initial_x is not None and np.max(np.abs(initial_x)) > safe_scale
        ):
            raise ValueError(
                "Initial barycenter squared costs would overflow; choose explicit units."
            )
        try:
            from ot.gromov import fgw_barycenters
        except ImportError as error:
            raise ImportError(
                "Install lambdaforge[optimal-transport] for FGW barycenters."
            ) from error
        rng = np.random.default_rng(self.seed)
        best: FGWBarycenterResult | None = None
        starts: list[dict[str, Any]] = []
        for index in range(self.n_init):
            start_seed = int(rng.integers(0, 2**31 - 1))
            try:
                with (
                    warnings.catch_warnings(record=True) as captured,
                    threadpool_limits(limits=self.threads),
                ):
                    warnings.simplefilter("always")
                    x, c, log = fgw_barycenters(
                        n,
                        [np.array(obj.features, dtype=float, copy=True) for obj in items],
                        [obj.structure.copy() for obj in items],
                        ps=[obj.weights.copy() for obj in items],
                        lambdas=lambdas.copy(),
                        p=p.copy(),
                        alpha=self.alpha,
                        loss_fun="square_loss",
                        max_iter=self.max_iter,
                        tol=self.tol,
                        stop_criterion="barycenter",
                        symmetric=True,
                        warmstartT=True,
                        log=True,
                        random_state=start_seed,
                        init_C=None if initial_c is None else initial_c.copy(),
                        init_X=None if initial_x is None else initial_x.copy(),
                    )
                    prototype = StructuredMeasure(
                        c, features=np.asarray(x), weights=p, relational=True
                    )
                    if (
                        np.asarray(x).shape != (n, feature_count)
                        or not np.isfinite(x).all()
                        or prototype.structure.shape != (n, n)
                    ):
                        raise RuntimeError(
                            "POT returned invalid barycenter features/structure dimensions."
                        )
                    feature_errors = tuple(float(e) for e in log["err_feature"])
                    structure_errors = tuple(float(e) for e in log["err_structure"])
                    if (
                        not feature_errors
                        or len(feature_errors) != len(structure_errors)
                        or not np.isfinite(feature_errors + structure_errors).all()
                        or min(feature_errors + structure_errors) < 0
                    ):
                        raise RuntimeError("POT returned invalid barycenter change history.")
                    # Free per-iteration transports before allocating postchecks.
                    del log
                    distance = FusedGromovWassersteinDistance(
                        alpha=self.alpha,
                        max_iter=self.transport_max_iter,
                        tol_rel=self.transport_tol,
                        tol_abs=self.transport_tol,
                        max_memory_bytes=self.max_memory_bytes,
                    )
                    comparisons = tuple(distance.solve(prototype, obj) for obj in items)
                objective = float(np.dot(lambdas, [result.value for result in comparisons]))
                backend_both = feature_errors[-1] <= self.tol and structure_errors[-1] <= self.tol
                notices = [str(w.message) for w in captured]
                if not backend_both:
                    notices.append(
                        "Both parameter-change criteria were not met; "
                        "POT may stop when only one is small."
                    )
                if any(not result.converged for result in comparisons):
                    notices.append(
                        "At least one final-prototype postcheck FGW solve did not converge."
                    )
                candidate = FGWBarycenterResult(
                    prototype,
                    objective,
                    backend_both and all(r.converged for r in comparisons),
                    len(feature_errors),
                    self.seed,
                    start_seed,
                    lambdas,
                    feature_errors,
                    structure_errors,
                    comparisons,
                    (),
                    tuple(notices),
                    MappingProxyType(
                        {
                            "backend": "POT fgw_barycenters",
                            "version": version("POT"),
                            "alpha": self.alpha,
                            "n_support": n,
                            "start": index,
                            "init_structure": initial_c is not None,
                            "init_features": initial_x is not None,
                            "objective_source": (
                                "weighted post-optimization FGW solves at final prototype"
                            ),
                            "backend_objective": None,
                            "backend_both_changes_small": backend_both,
                            "backend_criterion": (
                                "barycenter: POT OR; wrapper requires BOTH "
                                "plus postcheck convergence"
                            ),
                            "max_iter": self.max_iter,
                            "tol": self.tol,
                            "threads": self.threads,
                        }
                    ),
                )
            except (ValueError, RuntimeError, ArithmeticError) as error:
                starts.append(
                    {
                        "start": index,
                        "seed": start_seed,
                        "valid": False,
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
                continue
            starts.append(
                {
                    "start": index,
                    "seed": start_seed,
                    "valid": True,
                    "objective": objective,
                    "converged": candidate.converged,
                }
            )
            if best is None or candidate.objective < best.objective:
                best = candidate
        if best is None:
            raise RuntimeError(f"No valid FGW barycenter initialization: {starts}")
        return replace(best, restarts=tuple(MappingProxyType(start) for start in starts))
